# train_baseline.py

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from monai.metrics import compute_hausdorff_distance, compute_dice
from monai.losses import DiceCELoss
from scipy.ndimage import center_of_mass, shift

# 导入 Dataset 和外部接口
from dataset import Prostate_Dataset
from utils import compute_com_error, save_visualization_3view

# 🟢 导入我们在 baseline_models.py 中写好的模型
from models.baseline_models import Baseline_3DUNet, Baseline_SwinUNETR

# ==========================================
# 🟢 核心配置区域 (预留了模型切换接口)
# ==========================================
class Config:
    # 🔴 模型切换开关：支持 '3dunet' 或 'swin_unetr'
    model_type = 'swin_unetr'

    # 实验名称自动根据选定的模型生成，防止日志和权重相互覆盖
    name = f'baseline_{model_type}_sparse_amp'

    data_root = r'/root/autodl-tmp/Proj/data/prostate_158_128'
    resume_path = None
    gpu_id = 0
    num_workers = 22
    batch_size = 4        # ⚠️ 提示：3D 网络极其吃显存，如果显存大于 24G 可以尝试 8 或 16，32 极易 OOM
    epoch = 300           # CNN 收敛较快，300 轮通常足以见顶
    lr = 2e-4
    out_res = (128, 128, 128)
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
    logger.info(f"Start Baseline Training: {Config.name} (with AMP Acceleration)")
    logger.info(f"Selected Model Architecture: {Config.model_type.upper()}")

    # ==========================================
    # 🚀 极限加速：数据加载管线构建
    # ==========================================
    train_dst = Prostate_Dataset(data_root=os.path.join(Config.data_root, 'train', 'image'),
                                 label_root=os.path.join(Config.data_root, 'train', 'label'),
                                 npoint=1024,
                                 split='train', out_res=Config.out_res)

    train_loader = DataLoader(
        train_dst,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=True, # 保持 worker 存活，切断进程反复创建的 CPU 开销
        prefetch_factor=2        # 提前将下一个 Batch 载入内存
    )

    val_dst = Prostate_Dataset(data_root=os.path.join(Config.data_root, 'valid', 'image'),
                               label_root=os.path.join(Config.data_root, 'valid', 'label'),
                               npoint=1024,
                               split='eval', out_res=Config.out_res)

    eval_loader = DataLoader(
        val_dst,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    # ==========================================
    # 🔴 实例化模型：根据 Config 动态路由
    # ==========================================
    if Config.model_type == '3dunet':
        model = Baseline_3DUNet().cuda()
    elif Config.model_type == 'swin_unetr':
        model = Baseline_SwinUNETR().cuda()
    else:
        raise ValueError(f"Unsupported model_type: {Config.model_type}")

    # 使用医学图像标准的 DiceCELoss
    loss_function = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.lr, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=Config.gamma)

    # 🚀 极限加速：初始化 AMP 混合精度缩放器
    scaler = torch.cuda.amp.GradScaler()

    best_val_dice = 0.0
    epoch = 0

    # ==========================================
    # 主训练循环
    # ==========================================
    while epoch <= Config.epoch:
        loss_list = []
        model.train()

        with tqdm(train_loader, desc=f'Epoch {epoch}/{Config.epoch}', ncols=120, unit='img') as pbar:
            for item in pbar:
                # 优化点：用 set_to_none 替代默认的 zero_grad，降低显存波峰
                optimizer.zero_grad(set_to_none=True)

                # 严密对齐 item，杜绝作用域混淆
                full_target_img = item['target_image'].float().cuda(non_blocking=True)
                full_prior_img = item['prior_image'].float().cuda(non_blocking=True)
                prior_mask = item['prior_mask'].float().cuda(non_blocking=True)
                gt_3d_mask = item['target_mask'].float().cuda(non_blocking=True)
                coords = item['center_coords'].cuda(non_blocking=True)

                # 🚀 极限加速：在 AMP 上下文中执行前向推理和 Loss 计算
                with torch.cuda.amp.autocast():
                    pred_logits = model(full_target_img, full_prior_img, prior_mask, coords)
                    loss = loss_function(pred_logits, gt_3d_mask)

                loss_list.append(loss.item())

                # 🚀 极限加速：使用 scaler 缩放梯度并反向传播
                scaler.scale(loss).backward()

                # 梯度裁剪前必须先取消缩放
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()

                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

        logger.info(f"Epoch {epoch} | Loss: {np.mean(loss_list):.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        if epoch > 0 and epoch % Config.save_freq == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f'ep_{epoch}.pth'))

        # ==========================================
        # 验证集评估阶段
        # ==========================================
        if epoch == 0 or epoch % Config.eval_freq == 0:
            model.eval()
            init_dices, recon_dices = [], []
            init_hd95s, recon_hd95s = [], []
            inference_times = []  # 🔴 已修复：补充耗时记录器初始化

            with torch.no_grad():
                for i, v_item in enumerate(eval_loader):
                    if i >= 5: break

                    # 严密对齐 v_item，杜绝作用域混淆
                    full_target_img = v_item['target_image'].float().cuda()
                    full_prior_img = v_item['prior_image'].float().cuda()
                    prior_mask_tensor = v_item['prior_mask'].float().cuda()
                    gt_mask_tensor = v_item['target_mask'].float().cuda()
                    coords = v_item['center_coords'].cuda()

                    torch.cuda.synchronize()
                    t_start = time.time()

                    # 🚀 极限加速：验证阶段同样开启混合精度推理
                    with torch.cuda.amp.autocast():
                        pred_logits = model(full_target_img, full_prior_img, prior_mask_tensor, coords)

                    torch.cuda.synchronize()
                    inference_times.append(time.time() - t_start)

                    pred_probs = torch.sigmoid(pred_logits)
                    pred_mask_tensor = (pred_probs > 0.5).float()

                    # --- 数据转回 CPU numpy 用于算分和可视化 ---
                    prior_mask_np = prior_mask_tensor[0, 0].cpu().numpy()
                    pred_mask_np = pred_mask_tensor[0, 0].cpu().numpy()
                    gt_mask_np = gt_mask_tensor[0, 0].cpu().numpy()

                    i_dice = compute_dice(prior_mask_tensor, gt_mask_tensor, include_background=False).item()
                    r_dice = compute_dice(pred_mask_tensor, gt_mask_tensor, include_background=False).item()

                    try: i_hd95 = compute_hausdorff_distance(prior_mask_tensor, gt_mask_tensor, include_background=False, percentile=95).item()
                    except: i_hd95 = 99.0
                    try: r_hd95 = compute_hausdorff_distance(pred_mask_tensor, gt_mask_tensor, include_background=False, percentile=95).item()
                    except: r_hd95 = 99.0

                    init_dices.append(i_dice); recon_dices.append(r_dice)
                    init_hd95s.append(i_hd95); recon_hd95s.append(r_hd95)

                    print(f"  [{v_item['name'][0]}] Dice: {i_dice:.3f}->{r_dice:.3f} | HD95: {i_hd95:.2f}->{r_hd95:.2f}mm")

                    # 🔴 保留第一张图的可视化，作为论文图表的素材库
                    if i == 0:
                        vis_save_dir = os.path.join(save_dir, 'vis')
                        os.makedirs(vis_save_dir, exist_ok=True)
                        vis_save_path = os.path.join(vis_save_dir, f"ep_{epoch}_{v_item['name'][0]}.png")
                        gt_img_np = full_target_img[0, 0].cpu().numpy()

                        save_visualization_3view(
                            img_np=gt_img_np,
                            prior_mask=prior_mask_np,
                            aligned_prior=prior_mask_np, # baseline 这里没有多阶段对齐概念，直接传相同的
                            gt_mask=gt_mask_np,
                            pred_mask=pred_mask_np,
                            save_path=vis_save_path,
                            case_name=v_item['name'][0],
                            epoch=epoch
                        )

            eval_msg = f"      [Eval] Dice: Init {np.mean(init_dices):.4f} -> Pred {np.mean(recon_dices):.4f} | HD95: {np.mean(init_hd95s):.2f} -> {np.mean(recon_hd95s):.2f}"
            logger.info(eval_msg)

            current_mean_dice = np.mean(recon_dices)
            if current_mean_dice > best_val_dice:
                best_val_dice = current_mean_dice
                torch.save(model.state_dict(), os.path.join(save_dir, 'model_best.pth'))
                logger.info(f"      🔥 [Checkpoint] 更新最优 Pred Dice: {best_val_dice:.4f}")

        lr_scheduler.step()
        epoch += 1