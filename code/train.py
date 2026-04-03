# train.py

import os
import sys
# 限制底层数学库的线程数，防止 DataLoader 多进程读取时发生 CPU 线程死锁
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

# 导入自定义组件
from dataset import Prostate_Dataset
from models.model import DIF_Net
from utils import compute_com_error, save_visualization_3view

# ==========================================
#  核心配置区域 (Hyperparameters & Config)
# ==========================================
class Config:
    name = 'prostate_new_2'           # 实验名称，对应 logs 下的文件夹名
    data_root = r'/root/autodl-tmp/Proj/data/prostate_158_128' # 数据集根目录
    # resume_path = None                # 预训练权重路径，设置为 None 表示从零训练
    resume_path = r'/root/autodl-tmp/Proj/code/logs/prostate_new_2/model_best.pth'
    gpu_id = 0                        # 使用的 GPU 序号
    num_workers = 4                   # DataLoader 进程数
    preload = False                   # 是否将所有数据预加载到内存（数据量大时易爆内存）
    batch_size = 4                    # 批大小
    epoch = 600                       # 总训练轮次
    lr = 1.4e-4                         # 初始学习率
    num_views = 3                     # 输入的稀疏切片数量 (Axial, Coronal, Sagittal)
    out_res = (128, 128, 128)         # 3D 物理体素重构分辨率
    num_points = 80000                # 每次前向传播在 3D 空间中采样的点云数量
    combine = 'attention'             # 稀疏视图特征融合策略（三平面注意力机制 Tri-Att）
    eval_freq = 5                     # 验证集评估频率 (每 N 轮评估一次)
    save_freq = 20                    # 权重常规保存频率 (每 N 轮保存一次)
    gamma = 0.95                      # 学习率衰减率 (StepLR)

# 固定 DataLoader 的随机种子，确保多进程数据加载的随机性是可复现的
def worker_init_fn(worker_id):
    np.random.seed((worker_id + torch.initial_seed()) % np.iinfo(np.int32).max)

# 日志记录器：同时输出到控制台和 train_log.txt 文件
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
    # 1. 环境与日志初始化
    os.environ['CUDA_VISIBLE_DEVICES'] = str(Config.gpu_id)
    save_dir = f'./logs/{Config.name}'
    os.makedirs(save_dir, exist_ok=True)

    logger = setup_logger(os.path.join(save_dir, 'train_log.txt'))
    logger.info(f"Start training on Prostate: {Config.name}")
    logger.info(f"Config: Batch={Config.batch_size}, LR={Config.lr}, Points={Config.num_points}")

    # ==========================================
    # 2. 数据加载管线构建
    # 核心创新：Dataset 内部已实施“数学刚性预对齐”与“双轨非对称采样”
    # ==========================================
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
    # 3. 实例化模型与优化器
    # ==========================================
    model = DIF_Net(num_views=Config.num_views, combine=Config.combine).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr, weight_decay=0)
    for group in optimizer.param_groups:
        group.setdefault('initial_lr', Config.lr)

    # 4. 断点续训逻辑 (Resume)
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

    # 学习率调度器：每 10 轮衰减一次
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=Config.gamma, last_epoch=start_epoch - 1
    )

    logger.info("Start Training Loop...")
    best_val_dice = 0.0
    epoch = start_epoch

    # ==========================================
    # 5. 主训练循环
    # ==========================================
    while epoch <= Config.epoch:
        loss_list, loss_bce_list, loss_dice_list = [], [], []
        model.train()

        with tqdm(train_loader, desc=f'Epoch {epoch}/{Config.epoch}', ncols=120, unit='img') as pbar:
            for item in pbar:
                optimizer.zero_grad()

                # 数据迁移至 GPU
                for key in item.keys():
                    if key not in ['name']: item[key] = item[key].float().cuda(non_blocking=True)

                # ------------------------------------------
                # 5.1 前向推理与隐式空间输出
                # ------------------------------------------
                pred_logits = model(item) # 网络输出对数几率 (Logits) [B, 1, N]
                gt_points = item['p_gt']  # 真实的 0/1 空间占据值 [B, 1, N]
                is_lock = item['is_lock'] # 物理锚点锁定标签 (1为切片截面点，0为盲区游离点) [B, 1, N]

                # ------------------------------------------
                # 5.2 带有狄利克雷物理边界约束的 BCE Loss (核心创新点)
                # 学术叙事：Spatially-Weighted Implicit Representation
                # ------------------------------------------
                # A. 计算不求均值的基础分类损失 (Point-wise)
                bce_unreduced = F.binary_cross_entropy_with_logits(pred_logits, gt_points, reduction='none')

                # B. 构建非对称权重矩阵 (Asymmetric Weight Mask)
                # 切片区（已知边界）权重杠杆为 10.0；盲区（拓扑推演区）权重为 1.0
                lambda_lock = 10.0
                weight_mask = 1.0 + (lambda_lock - 1.0) * is_lock

                # C. 施加物理劫持并求均值，迫使网络优先咬合正交切片轮廓
                loss_bce = (bce_unreduced * weight_mask).mean()

                # ------------------------------------------
                # 5.3 体积级软对齐损失 (Soft Dice Loss)
                # 用于约束全局宏观体积的交并比
                # ------------------------------------------
                pred_probs = torch.sigmoid(pred_logits)
                intersection = (pred_probs * gt_points).sum(dim=2)
                union = pred_probs.sum(dim=2) + gt_points.sum(dim=2)
                loss_dice = 1.0 - (2.0 * intersection + 1e-5) / (union + 1e-5)
                loss_dice = loss_dice.mean()

                # 总损失融合
                loss = loss_bce + loss_dice

                loss_list.append(loss.item())
                loss_bce_list.append(loss_bce.item())
                loss_dice_list.append(loss_dice.item())

                # 梯度回传与参数更新
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 防止梯度爆炸
                optimizer.step()

                pbar.set_postfix({
                    'L': f'{loss.item():.4f}',
                    'L_BCE': f'{loss_bce.item():.4f}',
                    'L_Dice': f'{loss_dice.item():.4f}'
                })

            current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"Epoch {epoch} | Loss: {np.mean(loss_list):.4f} (BCE:{np.mean(loss_bce_list):.4f}, Dice:{np.mean(loss_dice_list):.4f}) | LR: {current_lr:.2e}")

        # 常规权重保存
        if epoch > 0 and epoch % Config.save_freq == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f'ep_{epoch}.pth'))

        # ==========================================
        # 6. 周期性验证集评估阶段 (Evaluation)
        # ==========================================
        if epoch == 0 or epoch % Config.eval_freq == 0:
            print(f" --> Running Evaluation at Epoch {epoch}...")
            model.eval()
            init_coms, recon_coms = [], []
            init_dices, recon_dices, aligned_dices = [], [], []
            init_hd95s, recon_hd95s = [], []
            inference_times = []

            with torch.no_grad():
                for i, v_item in enumerate(eval_loader):
                    if i >= 5: break # 验证集前5个病例加速评估，以窥探收敛趋势

                    for key in v_item.keys():
                        if key not in ['name']: v_item[key] = v_item[key].float().cuda(non_blocking=True)

                    torch.cuda.synchronize()
                    t_start = time.time()

                    # 推理输出 Logits [B, 1, 128*128*128]
                    pred_logits = model(v_item, is_eval=True, eval_npoint=50000)

                    torch.cuda.synchronize()
                    t_end = time.time()
                    inference_times.append(t_end - t_start)

                    # ------------------------------------------
                    # 6.1 将点云展平数组还原为 3D 物理空间矩阵
                    # ------------------------------------------
                    res = Config.out_res
                    pred_probs = torch.sigmoid(pred_logits)
                    pred_mask_tensor = (pred_probs > 0.5).float().view(1, 1, res[0], res[1], res[2])
                    gt_mask_tensor = v_item['p_gt'].view(1, 1, res[0], res[1], res[2])
                    prior_mask_tensor = v_item['prior_mask'] # 注意：此处的 Prior 已经是 Dataset 中刚性预对齐过的

                    pred_mask_np = pred_mask_tensor[0, 0].cpu().numpy()
                    gt_mask_np = gt_mask_tensor[0, 0].cpu().numpy()
                    prior_mask_np = prior_mask_tensor[0, 0].cpu().numpy()

                    # ------------------------------------------
                    # 6.2 计算质心对齐 Baseline
                    # (由于 Dataset 已经对 Prior 做了平移，此处的 shift_vec 理论上极小，近似为0)
                    # ------------------------------------------
                    try:
                        com_gt = np.array(center_of_mass(gt_mask_np))
                        com_prior = np.array(center_of_mass(prior_mask_np))
                        shift_vec = com_gt - com_prior
                        aligned_prior_np = shift(prior_mask_np, shift_vec, order=0)
                        aligned_prior_tensor = torch.from_numpy(aligned_prior_np).unsqueeze(0).unsqueeze(0).cuda()
                    except:
                        aligned_prior_tensor = prior_mask_tensor

                    # ------------------------------------------
                    # 6.3 计算物理空间量化指标 (CoM, Dice, HD95)
                    # ------------------------------------------
                    init_com_err = compute_com_error(gt_mask_np, prior_mask_np, spacing=(1.0, 1.0, 1.0))
                    recon_com_err = compute_com_error(gt_mask_np, pred_mask_np, spacing=(1.0, 1.0, 1.0))
                    init_coms.append(init_com_err); recon_coms.append(recon_com_err)

                    gt_b = gt_mask_tensor
                    pred_b = pred_mask_tensor
                    prior_b = prior_mask_tensor

                    i_dice = compute_dice(prior_b, gt_b, include_background=False).item()
                    r_dice = compute_dice(pred_b, gt_b, include_background=False).item()
                    a_dice = compute_dice(aligned_prior_tensor, gt_b, include_background=False).item()

                    try: i_hd95 = compute_hausdorff_distance(prior_b, gt_b, include_background=False, percentile=95).item()
                    except: i_hd95 = 99.0
                    try: r_hd95 = compute_hausdorff_distance(pred_b, gt_b, include_background=False, percentile=95).item()
                    except: r_hd95 = 99.0

                    init_dices.append(i_dice); recon_dices.append(r_dice); aligned_dices.append(a_dice)
                    init_hd95s.append(i_hd95); recon_hd95s.append(r_hd95)

                    print(f"  [{v_item['name'][0]}] CoM: {init_com_err:.2f}->{recon_com_err:.2f}mm | Dice(Init/Align/Pred): {i_dice:.3f}/{a_dice:.3f}/{r_dice:.3f} | HD95: {i_hd95:.2f}->{r_hd95:.2f}mm")

                    # 保存第一张验证数据的 2D 轮廓叠加图用于视觉检查
                    if i == 0:
                        vis_save_path = os.path.join(save_dir, 'vis', f"ep_{epoch}_{v_item['name'][0]}.png")
                        gt_img_np = v_item['target_image'][0, 0].cpu().numpy()
                        save_visualization_3view(
                            img_np=gt_img_np, prior_mask=prior_mask_np, aligned_prior=aligned_prior_np,
                            gt_mask=gt_mask_np, pred_mask=pred_mask_np, save_path=vis_save_path,
                            case_name=v_item['name'][0], epoch=epoch
                        )

            # 6.4 评估结果汇总与最优模型保存
            avg_init_com, avg_recon_com = np.mean(init_coms), np.mean(recon_coms)
            eval_msg = f"      [Eval Result] Epoch {epoch}:\n" \
                       f"      - CoM Error: {avg_init_com:.2f} -> {avg_recon_com:.2f} mm\n" \
                       f"      - Dice Score: Init {np.mean(init_dices):.4f} | Aligned {np.mean(aligned_dices):.4f} | Pred {np.mean(recon_dices):.4f}\n" \
                       f"      - HD95: {np.mean(init_hd95s):.2f} -> {np.mean(recon_hd95s):.2f} mm\n" \
                       f"      - Speed: {np.mean(inference_times):.3f} s/vol"
            logger.info(eval_msg)

            current_mean_dice = np.mean(recon_dices)
            if current_mean_dice > best_val_dice:
                best_val_dice = current_mean_dice
                best_model_path = os.path.join(save_dir, 'model_best.pth')
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"      🔥 [Checkpoint] 发现新的最优 Pred Dice: {best_val_dice:.4f}，权重已覆盖保存至 model_best.pth")

        lr_scheduler.step()
        epoch += 1