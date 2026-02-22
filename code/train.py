# train.py

import os
import sys
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

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
    label_root = r'/root/autodl-tmp/Proj/data/amos_mri_label_npy'
    # resume_path = r'/root/autodl-tmp/Proj/code/logs/dif_amos_roi_v2/ep_100.pth'
    resume_path = None
    gpu_id = 0
    num_workers = 22 # 配合数据盘读取，不需要设置过大
    preload = False # 如果内存不够（系统盘爆过），建议设为 False
    batch_size = 1
    epoch = 400
    lr = 5e-4
    num_views = 3
    out_res = (256, 256, 128)
    num_points = 100000 # 配合 ROI 采样，10w 点就能达到很好的效果
    combine = 'attention'
    eval_freq = 10
    save_freq = 50
    gamma = 0.95
    sigma = (0.02, 0.02, 0.08)
    # sigma = (0.0, 0.0, 0.0)



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

    # 1. 实例化模型与优化器
    model = DIF_Net(num_views=Config.num_views, combine=Config.combine).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr, weight_decay=0)
    for group in optimizer.param_groups:
        group.setdefault('initial_lr', Config.lr)

    # 2. 🟢 解析断点续训 (优先于调度器初始化)
    start_epoch = 0
    if hasattr(Config, 'resume_path') and Config.resume_path and os.path.exists(Config.resume_path):
        logger.info(f"==> [Resume] 发现预训练权重，正在加载: {Config.resume_path}")
        checkpoint = torch.load(Config.resume_path, map_location='cuda')
        model.load_state_dict(checkpoint)

        try:
            base_name = os.path.basename(Config.resume_path)
            start_epoch = int(base_name.split('_')[1].split('.')[0]) + 1
            logger.info(f"==> [Resume] 解析到基础轮次，将从 Epoch {start_epoch} 开始训练。")
        except Exception as e:
            logger.warning(f"==> [Resume] 无法从文件名解析 Epoch，默认从 Epoch 0 重新计数。错误: {e}")

    # 3. 🟢 优雅初始化调度器 (消除 UserWarning)
    # 直接传入 last_epoch=start_epoch-1，让 PyTorch 自己算好当前应该处于什么学习率
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=Config.gamma, last_epoch=start_epoch - 1
    )

    # 4. 🟢 补回丢失的形变器与模拟器 (消除 NameError)
    train_simulator = GPUDailyScanSimulator(noise_level=0.0, blur_sigma=0.0).cuda()
    eval_simulator = GPUDailyScanSimulator(noise_level=0.0, blur_sigma=0.0).cuda()
    deformer = ElasticDeformation(grid_size=8, sigma=Config.sigma).cuda()

    logger.info("Start Training Loop...")
    epoch = start_epoch

    while epoch <= Config.epoch:
        loss_list = []
        model.train()

        with tqdm(train_loader, desc=f'Epoch {epoch}/{Config.epoch}', ncols=120, unit='img') as pbar:
            for item in pbar:
                optimizer.zero_grad()
                item = convert_cuda(item)

                with torch.no_grad():
                    prior_vol = item['image']
                    prior_mask = (item['mask'] == 6).float() # 🟢 把 Mask 取出来

                    # 1. 让图像和 Mask 同步发生物理形变
                    combined_vol = torch.cat([prior_vol, prior_mask], dim=1)
                    warped_combined = deformer(combined_vol, mode='bilinear')

                    target_vol = warped_combined[:, 0:1]
                    target_mask = warped_combined[:, 1:2] # 🟢 获取形变后的真实器官位置

                    # 2. 从 Target 中切片并更新输入
                    item['projs'] = gpu_slice_volume(target_vol)
                    item['prior'] = prior_vol

                    # 3. 动态重新采样 GT 和 GT_Mask
                    uv = item['points']
                    uv_sampling = uv[..., [2, 1, 0]].reshape(uv.shape[0], 1, 1, uv.shape[1], 3)

                    # 同时采样像素值和掩码值
                    gt = F.grid_sample(target_vol, uv_sampling, align_corners=True)[:, :, 0, 0, :]
                    gt_mask_sampled = F.grid_sample(target_mask, uv_sampling, align_corners=True)[:, :, 0, 0, :]
                    item['p_gt'] = gt

                # ==========================================
                # 🟢 阶段 3: 纯粹靶区 Loss 引擎 (暴力聚焦版)
                # ==========================================
                pred_val, delta_coords = model(item)

                # 1. 计算基础 L1 误差
                # 此时所有的 10 万个采样点，已经在 dataset 层面被物理锁死在了膨胀靶区内
                # 直接求均值，不需要任何空间权重，保证肝脏与缓冲带梯度的平滑过渡
                loss_recon = F.l1_loss(pred_val, gt, reduction='mean')

                # 2. 轻微的位移正则 (防止边缘缓冲带的形变场发散)
                loss_reg = torch.mean(delta_coords ** 2)

                # 3. 总 Loss
                w_reg = 0.02
                loss = loss_recon + w_reg * loss_reg

                loss_list.append(loss.item())
                loss.backward()
                # ==========================================

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
                    if i >= 5: break
                    v_item = convert_cuda(v_item)

                    prior_vol = v_item['image']

                    # 🟢 1. 记录当前狂野的随机宇宙状态（保护训练的随机性）
                    cpu_rng_state = torch.get_rng_state()
                    gpu_rng_state = torch.cuda.get_rng_state()

                    # 🟢 2. 时间静止：为当前样本注入绝对固定的命运 (Seed)
                    # 保证 amos_507 每次 eval 遭遇的形变场连小数点后 6 位都一模一样！
                    fixed_seed = 2026 + i
                    torch.manual_seed(fixed_seed)
                    torch.cuda.manual_seed(fixed_seed)

                    # 🟢 3. 宿命形变：生成永远一致的 Target
                    if 'mask' in v_item: # 如果你用了 BBox 评测，把 mask 也带上
                        prior_mask = (v_item['mask'] == 6).float()
                        combined_eval = torch.cat([prior_vol, prior_mask], dim=1)
                        warped_eval = deformer(combined_eval, mode='bilinear')
                        target_vol = warped_eval[:, 0:1]
                        target_mask = warped_eval[:, 1:2]
                    else:
                        target_vol = deformer(prior_vol, mode='bilinear')
                        target_mask = None # 视你当前用的哪种 eval 逻辑而定

                    # 🟢 4. 恢复时间的流动：把随机状态还给系统
                    torch.set_rng_state(cpu_rng_state)
                    torch.cuda.set_rng_state(gpu_rng_state)

                    # --- 后续的切片、模型推理、PSNR/SSIM 计算完全保持不变 ---
                    v_item['projs'] = gpu_slice_volume(target_vol)
                    v_item['prior'] = prior_vol

                    torch.cuda.synchronize()
                    t_start = time.time()

                    pred, _ = model(v_item, is_eval=True, eval_npoint=50000)

                    torch.cuda.synchronize()
                    t_end = time.time()
                    inference_times.append(t_end - t_start)

                    # ==========================================
                    # 🟢 数据解构与终极测谎
                    # ==========================================
                    pred_np = pred[0, 0].cpu().numpy().reshape(v_item['image'].shape[2:])
                    gt_img_np = target_vol.cpu().numpy()[0, 0]

                    # 🔴 修复 Bug 1：必须使用形变后的 target_mask 来定位！
                    gt_mask_np = (target_mask.cpu().numpy()[0, 0] > 0.5).astype(np.float32)

                    # 提取肝脏的 3D 物理边界框 (BBox)
                    coords = np.argwhere(gt_mask_np > 0.5)
                    if len(coords) > 0:
                        x_min, y_min, z_min = coords.min(axis=0)
                        x_max, y_max, z_max = coords.max(axis=0)

                        # 为了给 SSIM 滑动窗口留下一点计算空间，并检验网络对边界脂肪的拟合
                        # 我们加上 10 个体素的评估 Margin（不超过训练时的 15）
                        margin = 10
                        x_min = max(0, x_min - margin)
                        y_min = max(0, y_min - margin)
                        z_min = max(0, z_min - margin)
                        x_max = min(gt_img_np.shape[0]-1, x_max + margin)
                        y_max = min(gt_img_np.shape[1]-1, y_max + margin)
                        z_max = min(gt_img_np.shape[2]-1, z_max + margin)

                        # 强行裁切出包含器官和极其微小缓冲带的干净长方体！
                        gt_roi = gt_img_np[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
                        pred_roi = pred_np[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]

                        # 🔴 避开陷阱 2：绝对不乘 mask_roi！直接评估这个“带肉”的长方体！
                        p, s = simple_eval_metric(gt_roi, pred_roi)
                        psnrs.append(p)
                        ssims.append(s)
                    else:
                        print(f"  [Warning] {v_item['name'][0]} 提取 BBox 失败。")


            avg_psnr = np.mean(psnrs)
            avg_ssim = np.mean(ssims)
            avg_time = np.mean(inference_times)
            eval_msg = f"     [Eval Result] Epoch {epoch}: PSNR = {avg_psnr:.4f} | SSIM = {avg_ssim:.4f}"
            print(f"Inference Time = {avg_time:.4f}s")
            logger.info(eval_msg)

        lr_scheduler.step()
        epoch += 1