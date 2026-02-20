# train.py

import os
import sys
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import logging

from dataset import AMOS_Dataset
from models.model import DIF_Net
from utils import convert_cuda, save_visualization_3view, simple_eval, gpu_slice_volume, GPUDailyScanSimulator, ElasticDeformation, simple_eval_metric, compute_gradient



# ==========================================
#  配置区域
# ==========================================
class Config:
    name = 'dif_amos_roi_v2' # 建议改名以区分旧实验
    # 指向你刚才预处理后的数据盘路径
    data_root = r'/root/autodl-tmp/Proj/data/amos_mri_npy'
    # 新增：指向你生成的 ROI JSON 文件夹
    label_root = r'/root/autodl-tmp/Proj/data/amos_mri_label_npy'

    gpu_id = 0
    num_workers = 22 # 配合数据盘读取，不需要设置过大
    preload = False # 如果内存不够（系统盘爆过），建议设为 False
    batch_size = 1
    epoch = 400
    lr = 1e-3
    num_views = 3
    out_res = (256, 256, 128)
    num_points = 100000 # 配合 ROI 采样，10w 点就能达到很好的效果
    combine = 'attention'
    eval_freq = 10
    save_freq = 50
    gamma = 0.95
    sigma = (0.02, 0.02, 0.08)


def worker_init_fn(worker_id):
    np.random.seed((worker_id + torch.initial_seed()) % np.iinfo(np.int32).max)

def setup_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 文件输出
    fh = logging.FileHandler(log_file, mode='a')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)

    # 控制台输出
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)

    return logger

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = str(Config.gpu_id)
    save_dir = f'./logs/{Config.name}'
    os.makedirs(save_dir, exist_ok=True)

    logger = setup_logger(os.path.join(save_dir, 'train_log.txt'))
    logger.info(f"Start training: {Config.name}")
    logger.info(f"Config: Batch={Config.batch_size}, LR={Config.lr}, Sigma={Config.sigma}")

    train_dst = AMOS_Dataset(
        data_root=Config.data_root,
        label_root=Config.label_root,
        split='train',
        npoint=Config.num_points,
        out_res=Config.out_res
    )
    # 如果 preload=False，建议设置 num_workers 开启多线程读取
    train_loader = DataLoader(
        train_dst,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    val_dst = AMOS_Dataset(
        data_root=Config.data_root,
        label_root=Config.label_root,
        split='eval',
        npoint=50000, # 评估时采样点可以少一点
        out_res=Config.out_res
    )

    eval_loader = DataLoader(val_dst, batch_size=1, shuffle=False)

    # Model
    model = DIF_Net(num_views=Config.num_views, combine=Config.combine).cuda()

    # Simulator (Data Augmentation)
    # 噪声生成器
    train_simulator = GPUDailyScanSimulator(
        noise_level=0.0,
        blur_sigma=0.0
    ).cuda()

    eval_simulator = GPUDailyScanSimulator(
        noise_level=0.0,
        blur_sigma=0.0
    ).cuda()

    deformer = ElasticDeformation(grid_size=8, sigma=Config.sigma).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr, weight_decay=0)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=Config.gamma)

    logger.info("Start Training Loop...")
    epoch = 0
    while epoch <= Config.epoch:
        loss_list = []
        model.train()

        with tqdm(train_loader, desc=f'Epoch {epoch}/{Config.epoch}', ncols=120, unit='img') as pbar:
            for item in pbar:
                optimizer.zero_grad()
                item = convert_cuda(item)

                with torch.no_grad():
                    # 🟢 逻辑翻转：源图像作为 Prior，形变产生 Target
                    prior_vol = item['image']
                    target_vol = deformer(prior_vol, mode='bilinear')

                    # 🟢 从 Target 中切片并更新输入
                    item['projs'] = gpu_slice_volume(target_vol)
                    item['prior'] = prior_vol

                    # 🟢 动态重新采样 GT，因为真实的物理状态 (Target) 已经改变
                    uv = item['points']
                    uv_sampling = uv[..., [2, 1, 0]].reshape(uv.shape[0], 1, 1, uv.shape[1], 3)
                    gt = F.grid_sample(target_vol, uv_sampling, align_corners=True)[:, :, 0, 0, :]
                    item['p_gt'] = gt

                pred_val, delta_coords = model(item)

                # 取 Loss 最大的前 hard_ratio的点
                loss_pixel = F.l1_loss(pred_val, gt, reduction='none')
                loss_flat = loss_pixel.view(-1)
                hard_ratio = 1
                k = int(loss_flat.numel() * hard_ratio)
                topk_loss, _ = torch.topk(loss_flat, k)
                loss_recon = topk_loss.mean()

                loss_reg = torch.mean(delta_coords ** 2)

                # 总 Loss
                w_reg = 0.02
                loss = loss_recon + w_reg * loss_reg
                loss_list.append(loss.item())
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                current_lr = optimizer.param_groups[0]["lr"]
                pbar.set_postfix({'loss': f'{np.mean(loss_list):.6f}', 'lr': f'{current_lr:.6f}'})

        logger.info(f"Epoch {epoch} | Train Loss: {np.mean(loss_list):.6f} | LR: {current_lr:.2e}")


        if epoch > 0 and epoch % Config.save_freq == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f'ep_{epoch}.pth'))

        if epoch == 0 or epoch % Config.eval_freq == 0:
            print(f" --> Running Evaluation at Epoch {epoch}...")

            eval_start_time = time.time()

            save_visualization_3view(
                model, val_dst, epoch,
                save_dir=os.path.join(save_dir, 'vis'),
                simulator=eval_simulator,
                prior_deformer=deformer  # 传入修改后的形变器
            )

            model.eval()
            psnrs, ssims = [], []
            inference_times = []

            with torch.no_grad():
                for i, v_item in enumerate(eval_loader):
                    if i>=5: break
                    v_item = convert_cuda(v_item)

                    # 🟢 评估阶段逻辑翻转
                    prior_vol = v_item['image']
                    target_vol = deformer(prior_vol, mode='bilinear')

                    v_item['projs'] = gpu_slice_volume(target_vol)
                    v_item['prior'] = prior_vol

                    torch.cuda.synchronize()
                    t_start = time.time()

                    pred, _ = model(v_item, is_eval=True, eval_npoint=50000)

                    torch.cuda.synchronize()
                    t_end = time.time()
                    inference_times.append(t_end - t_start)

                    pred = pred[0, 0].cpu().numpy().reshape(v_item['image'].shape[2:])
                    # 🟢 SSIM 的基准对象必须是 Target
                    gt_img = target_vol.cpu().numpy()[0, 0]

                    # 背景滤除
                    # pred[pred < 0.05] = 0

                    p, s = simple_eval_metric(gt_img, pred)
                    psnrs.append(p)
                    ssims.append(s)


            avg_psnr = np.mean(psnrs)
            avg_ssim = np.mean(ssims)
            avg_time = np.mean(inference_times)
            eval_msg = f"     [Eval Result] Epoch {epoch}: PSNR = {avg_psnr:.4f} | SSIM = {avg_ssim:.4f}"
            print(f"Inference Time = {avg_time:.4f}s")
            logger.info(eval_msg)

        lr_scheduler.step()
        epoch += 1

