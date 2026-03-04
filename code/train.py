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
from monai.metrics import compute_hausdorff_distance, compute_dice
from scipy.ndimage import center_of_mass, shift

# 🟢 导入你的新 Dataset (注意你上一轮起的名字叫 AMOS_Dataset)
from dataset import AMOS_Dataset as Prostate_Dataset
from models.model import DIF_Net

# 导入你原有的 utils 工具包
from utils import compute_com_error, save_visualization_3view


# ==========================================
#  配置区域 (专为前列腺 128 尺寸重构)
# ==========================================
class Config:
    name = 'prostate_shape_completion_pel10'
    data_root = r'/root/autodl-tmp/Proj/data/prostate_158_128' # 指向你的前列腺数据
    resume_path = None
    # resume_path = r'/root/autodl-tmp/Proj/code/logs/prostate_shape_completion_v1/ep_100.pth'
    gpu_id = 0
    num_workers = 4
    preload = False
    batch_size = 4
    epoch = 300
    lr = 2e-4
    num_views = 3
    out_res = (128, 128, 128)
    num_points = 80000
    combine = 'attention'
    eval_freq = 5
    save_freq = 20
    gamma = 0.95


def worker_init_fn(worker_id):
    np.random.seed((worker_id + torch.initial_seed()) % np.iinfo(np.int32).max)

def setup_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file, mode='a')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)

    return logger

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = str(Config.gpu_id)
    save_dir = f'./logs/{Config.name}'
    os.makedirs(save_dir, exist_ok=True)

    logger = setup_logger(os.path.join(save_dir, 'train_log.txt'))
    logger.info(f"Start training on Prostate: {Config.name}")
    logger.info(f"Config: Batch={Config.batch_size}, LR={Config.lr}, Points={Config.num_points}")

    # ==========================================
    # 数据加载
    # ==========================================
    # 注意：你的 dataset 需要 label_root，假设结构是在 data_root 的 label 子文件夹下
    train_dst = Prostate_Dataset(data_root=os.path.join(Config.data_root, 'train', 'image'),
                                 label_root=os.path.join(Config.data_root, 'train', 'label'),
                                 split='train', npoint=Config.num_points, out_res=Config.out_res)
    train_loader = DataLoader(
        train_dst,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    val_dst = Prostate_Dataset(data_root=os.path.join(Config.data_root, 'valid', 'image'),
                               label_root=os.path.join(Config.data_root, 'valid', 'label'),
                               split='eval', npoint=50000, out_res=Config.out_res)
    eval_loader = DataLoader(val_dst, batch_size=1, shuffle=False)

    # ==========================================
    # 实例化模型与优化器
    # ==========================================
    model = DIF_Net(num_views=Config.num_views, combine=Config.combine).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr, weight_decay=0)
    for group in optimizer.param_groups:
        group.setdefault('initial_lr', Config.lr)

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

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=Config.gamma, last_epoch=start_epoch - 1
    )

    logger.info("Start Training Loop...")
    best_val_dice = 0.0
    epoch = start_epoch

    while epoch <= Config.epoch:
        loss_list = []
        loss_bce_list = []
        loss_dice_list = []
        model.train()

        with tqdm(train_loader, desc=f'Epoch {epoch}/{Config.epoch}', ncols=120, unit='img') as pbar:
            for item in pbar:
                optimizer.zero_grad()

                # 转换到 CUDA
                for key in item.keys():
                    if key not in ['name']: item[key] = item[key].float().cuda(non_blocking=True)

                # ==========================================
                # 🟢 极简前向推理：模型直接输出 Logits
                # ==========================================
                pred_logits = model(item) # [B, 1, N]
                gt_points = item['p_gt']  # [B, 1, N] 真实的 0/1 占据值

                # ------------------------------------------
                # 1. 点级二分类损失 (BCEWithLogitsLoss 自带 Sigmoid，数值极度稳定)
                # ------------------------------------------
                loss_bce = F.binary_cross_entropy_with_logits(pred_logits, gt_points)

                # ------------------------------------------
                # 2. 体积级软对齐损失 (Soft Dice Loss)
                # ------------------------------------------
                pred_probs = torch.sigmoid(pred_logits)
                intersection = (pred_probs * gt_points).sum(dim=2)
                union = pred_probs.sum(dim=2) + gt_points.sum(dim=2)
                loss_dice = 1.0 - (2.0 * intersection + 1e-5) / (union + 1e-5)
                loss_dice = loss_dice.mean()

                # 损失融合 (无需正则项，因为没有位移场了)
                loss = loss_bce + loss_dice

                loss_list.append(loss.item())
                loss_bce_list.append(loss_bce.item())
                loss_dice_list.append(loss_dice.item())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                pbar.set_postfix({
                    'L': f'{loss.item():.4f}',
                    'L_BCE': f'{loss_bce.item():.4f}',
                    'L_Dice': f'{loss_dice.item():.4f}'
                })

            current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"Epoch {epoch} | Loss: {np.mean(loss_list):.4f} (BCE:{np.mean(loss_bce_list):.4f}, Dice:{np.mean(loss_dice_list):.4f}) | LR: {current_lr:.2e}")

        # 保存逻辑
        if epoch > 0 and epoch % Config.save_freq == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f'ep_{epoch}.pth'))

        # ==========================================
        # 🟢 评估阶段：不再生成图片纹理，直接还原 3D Mask
        # ==========================================
        if epoch == 0 or epoch % Config.eval_freq == 0:
            print(f" --> Running Evaluation at Epoch {epoch}...")
            eval_start_time = time.time()

            model.eval()
            init_coms, recon_coms = [], []
            init_dices, recon_dices = [], []
            init_hd95s, recon_hd95s = [], []
            inference_times = []
            aligned_dices = []

            with torch.no_grad():
                for i, v_item in enumerate(eval_loader):
                    if i >= 5: break # 评估前5个加速

                    for key in v_item.keys():
                        if key not in ['name']: v_item[key] = v_item[key].float().cuda(non_blocking=True)

                    torch.cuda.synchronize()
                    t_start = time.time()

                    # 推理输出 Logits [B, 1, N_total]
                    pred_logits = model(v_item, is_eval=True, eval_npoint=50000)

                    torch.cuda.synchronize()
                    t_end = time.time()
                    inference_times.append(t_end - t_start)

                    # ==========================================
                    # 从点云还原回 3D 矩阵 (128x128x128)
                    # ==========================================
                    res = Config.out_res
                    pred_probs = torch.sigmoid(pred_logits)

                    # 极其优雅的还原：因为点的坐标就是按网格展平的，直接 view 即可恢复 3D 拓扑
                    pred_mask_tensor = (pred_probs > 0.5).float().view(1, 1, res[0], res[1], res[2])
                    gt_mask_tensor = v_item['p_gt'].view(1, 1, res[0], res[1], res[2])
                    prior_mask_tensor = v_item['prior_mask'] # [1, 1, 128, 128, 128]

                    pred_mask_np = pred_mask_tensor[0, 0].cpu().numpy()
                    gt_mask_np = gt_mask_tensor[0, 0].cpu().numpy()
                    prior_mask_np = prior_mask_tensor[0, 0].cpu().numpy()

                    # ==========================================
                    # 🟢 新增：计算质心对齐后的 Baseline Dice (CoM-aligned Dice)
                    # ==========================================
                    try:
                        com_gt = np.array(center_of_mass(gt_mask_np))
                        com_prior = np.array(center_of_mass(prior_mask_np))
                        shift_vec = com_gt - com_prior

                        # 使用 order=0 (最近邻插值) 保证 Mask 依然是干净的 0/1
                        aligned_prior_np = shift(prior_mask_np, shift_vec, order=0)
                        aligned_prior_tensor = torch.from_numpy(aligned_prior_np).unsqueeze(0).unsqueeze(0).cuda()
                    except:
                        # 极端兜底情况：如果 Mask 为空
                        aligned_prior_tensor = prior_mask_tensor

                    # 🟢 1. 计算质心误差 CoM
                    init_com_err = compute_com_error(gt_mask_np, prior_mask_np, spacing=(1.0, 1.0, 1.0))
                    recon_com_err = compute_com_error(gt_mask_np, pred_mask_np, spacing=(1.0, 1.0, 1.0))
                    init_coms.append(init_com_err)
                    recon_coms.append(recon_com_err)

                    # 🟢 2. 计算临床指标 Dice & HD95
                    gt_b = gt_mask_tensor
                    pred_b = pred_mask_tensor
                    prior_b = prior_mask_tensor

                    i_dice = compute_dice(prior_b, gt_b, include_background=False).item()
                    r_dice = compute_dice(pred_b, gt_b, include_background=False).item()
                    # 🟢 计算对齐后的 Dice
                    a_dice = compute_dice(aligned_prior_tensor, gt_b, include_background=False).item()

                    try: i_hd95 = compute_hausdorff_distance(prior_b, gt_b, include_background=False, percentile=95).item()
                    except: i_hd95 = 99.0

                    try: r_hd95 = compute_hausdorff_distance(pred_b, gt_b, include_background=False, percentile=95).item()
                    except: r_hd95 = 99.0

                    init_dices.append(i_dice); recon_dices.append(r_dice)
                    aligned_dices.append(a_dice) # 🟢 记录
                    init_hd95s.append(i_hd95); recon_hd95s.append(r_hd95)

                    # 🟢 修改打印日志：加入 Aligned Dice 的对比
                    print(f"  [{v_item['name'][0]}] CoM: {init_com_err:.2f}->{recon_com_err:.2f}mm | Dice(Init/Align/Pred): {i_dice:.3f}/{a_dice:.3f}/{r_dice:.3f} | HD95: {i_hd95:.2f}->{r_hd95:.2f}mm")

                    if i == 0:
                        vis_save_path = os.path.join(save_dir, 'vis', f"ep_{epoch}_{v_item['name'][0]}.png")
                        gt_img_np = v_item['target_image'][0, 0].cpu().numpy()

                        save_visualization_3view(
                            img_np=gt_img_np,
                            prior_mask=prior_mask_np,
                            aligned_prior=aligned_prior_np,  # 🟢 新增：把刚才算出来的质心对齐 Baseline 传进去
                            gt_mask=gt_mask_np,
                            pred_mask=pred_mask_np,
                            save_path=vis_save_path,
                            case_name=v_item['name'][0],
                            epoch=epoch
                        )

            # 循环结束后的平均统计与总览打印
            avg_init_com = np.mean(init_coms)
            avg_recon_com = np.mean(recon_coms)

            eval_msg = f"     [Eval Result] Epoch {epoch}:\n" \
                       f"     - CoM Error: {avg_init_com:.2f} -> {avg_recon_com:.2f} mm\n" \
                       f"     - Dice Score: Init {np.mean(init_dices):.4f} | Aligned {np.mean(aligned_dices):.4f} | Pred {np.mean(recon_dices):.4f}\n" \
                       f"     - HD95: {np.mean(init_hd95s):.2f} -> {np.mean(recon_hd95s):.2f} mm\n" \
                       f"     - Speed: {np.mean(inference_times):.3f} s/vol"
            logger.info(eval_msg)

            current_mean_dice = np.mean(recon_dices)
            if current_mean_dice > best_val_dice:
                best_val_dice = current_mean_dice
                best_model_path = os.path.join(save_dir, 'model_best.pth')
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"     🔥 [Checkpoint] 发现新的最优 Pred Dice: {best_val_dice:.4f}，权重已覆盖保存至 model_best.pth")

        lr_scheduler.step()
        epoch += 1