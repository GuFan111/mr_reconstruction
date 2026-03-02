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
# from chaos_dataset import CHAOS_Dataset
from models.model import DIF_Net, index_3d_deform_local
from utils import convert_cuda, save_visualization_3view, simple_eval, gpu_slice_volume, GPUDailyScanSimulator, simple_eval_metric, compute_gradient, strict_masked_eval, ElasticDeformation, HybridReconLoss, compute_com_error, MR_Linac_SyntheticDeformer



# ==========================================
#  配置区域
# ==========================================
class Config:
    name = 'amos_v4' # 建议改名以区分旧实验
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
    lr = 2e-4
    num_views = 3
    out_res = (256, 256, 128)
    num_points = 100000 # 配合 ROI 采样，10w 点就能达到很好的效果
    combine = 'attention'
    eval_freq = 5
    save_freq = 20
    gamma = 0.95
    sigma = (0.01, 0.01, 0.05)
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
        optimizer, step_size=10, gamma=Config.gamma, last_epoch=start_epoch - 1
    )

    # 4. 🟢 补回丢失的形变器与模拟器 (消除 NameError)
    train_simulator = GPUDailyScanSimulator(noise_level=0.0, blur_sigma=0.0).cuda()
    eval_simulator = GPUDailyScanSimulator(noise_level=0.0, blur_sigma=0.0).cuda()
    # deformer = PCARespiratoryDeformation(grid_size=4, amp_xyz=(0.01, 0.04, 0.15)).cuda()
    # deformer = ElasticDeformation(grid_size=8, sigma=Config.sigma).cuda()
    # deformer = HybridRespiratoryDeformation(grid_size=6, global_z_amp=0.30, local_sigma=(0.01, 0.02, 0.03)).cuda()
    deformer = MR_Linac_SyntheticDeformer(
        grid_size=5,       # 决定形变的平滑度 (越小越平滑)
        max_disp=0.04,     # 最大形变率 4% (控制 Loss 下降的难度)
        z_multiplier=1.0,  # 模拟剧烈的呼吸 Z 轴滑动
        center_focus=1.5   # 强迫大部分形变发生在切片可视的交汇中心！
    ).to('cuda')

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
                    prior_mask = (item['mask'] == 1).float()

                    # 1. 让图像和 Mask 同步发生物理形变
                    combined_vol = torch.cat([prior_vol, prior_mask], dim=1)
                    # warped_combined = deformer(combined_vol, mode='bilinear')
                    warped_combined = deformer(combined_vol)

                    target_vol = warped_combined[:, 0:1]
                    target_mask = warped_combined[:, 1:2]

                    # 2. 从 Target 中切片并更新输入
                    item['projs'] = gpu_slice_volume(target_vol)
                    item['prior_projs'] = gpu_slice_volume(prior_vol)
                    item['prior'] = prior_vol

                    # ==========================================================
                    # 🟢 物理引擎 1：随机有限差分对 (Stochastic Point Pairing)
                    # 彻底绕开 PyTorch 双重求导报错，同时完美保留 ROI 采样权重
                    # ==========================================================
                    B, N, _ = item['points'].shape
                    half_N = N // 2

                    # 1. 提取一半的锚点，并加入亚体素抖动打破离散网格 (1/192 约等于 0.01)
                    base_pts = item['points'][:, :half_N, :].to(target_vol.device)
                    voxel_jitter = (torch.rand_like(base_pts) - 0.5) * 0.01
                    anchors = torch.clamp(base_pts + voxel_jitter, -1.0, 1.0)

                    # 2. 生成一半的“影子点” (距离锚点极小的 epsilon)
                    epsilon = 2e-3  # 在 [-1,1] 空间内约等于 0.2mm 的物理极小距离
                    noise = torch.randn_like(anchors)
                    noise = F.normalize(noise, dim=-1) * epsilon
                    perturbed = torch.clamp(anchors + noise, -1.0, 1.0)

                    # 3. 拼合为完整的 N 个点
                    continuous_pts = torch.cat([anchors, perturbed], dim=1)
                    item['points'] = continuous_pts

                    # 4. 同步修复 is_fg 权重 (影子点和锚点的组织属性完全一致)
                    is_fg_base = item['point_is_fg'][:, :half_N].to(target_vol.device)
                    item['point_is_fg'] = torch.cat([is_fg_base, is_fg_base], dim=1)

                    # 5. 必须同步更新 2D 投影坐标，保持严丝合缝的物理映射！
                    item['proj_points'] = torch.stack([
                        continuous_pts[..., [0, 1]], # Axial (XY)
                        continuous_pts[..., [0, 2]], # Coronal (XZ)
                        continuous_pts[..., [1, 2]]  # Sagittal (YZ)
                    ], dim=1)

                    # 3. 动态重新采样 GT 和 GT_Mask (这里的 uv 变成了连续浮点)
                    uv = item['points']
                    uv_sampling = uv[..., [2, 1, 0]].reshape(uv.shape[0], 1, 1, uv.shape[1], 3)

                    # F.grid_sample 强大的插值能力会给出精准的亚像素 GT
                    gt = F.grid_sample(target_vol, uv_sampling, align_corners=True)[:, :, 0, 0, :]
                    gt_mask_sampled = F.grid_sample(target_mask, uv_sampling, align_corners=True)[:, :, 0, 0, :]
                    item['p_gt'] = gt

                # ==========================================
                # 阶段 3: 前向推理与分项误差计算
                # ==========================================
                pred_val, delta_coords = model(item)

                # 获取 ROI 标志位
                is_fg = item['point_is_fg'].unsqueeze(1).to(pred_val.device)

                # ------------------------------------------
                # 1. 基础图像误差 (Charbonnier, 保留高频边缘对齐)
                # ------------------------------------------
                diff_sq = (pred_val - gt) ** 2
                base_loss = torch.sqrt(diff_sq + 1e-6)
                weight_mask = torch.where(is_fg > 0.5,
                                          torch.tensor(10.0, device=pred_val.device),
                                          torch.tensor(1.0, device=pred_val.device))
                loss_recon = torch.mean(base_loss * weight_mask)

                # ------------------------------------------
                # 2. 软掩码误差 (Mask Loss, 提供宏观推力)
                # ------------------------------------------
                prior_mask_tensor = (item['mask'] == 1).float().to(pred_val.device)
                target_mask_tensor = (target_mask == 1).float().to(pred_val.device)

                # 大核池化融化边界，构建梯度盆地
                blur_kernel = 31
                padding = blur_kernel // 2
                soft_prior_mask = F.avg_pool3d(prior_mask_tensor, kernel_size=blur_kernel, stride=1, padding=padding)
                soft_target_mask = F.avg_pool3d(target_mask_tensor, kernel_size=blur_kernel, stride=1, padding=padding)

                # 坐标形变与 3D 软采样
                corrected_coords = item['points'] + delta_coords.transpose(1, 2)
                B, N, _ = corrected_coords.shape

                uv_sampling = corrected_coords[..., [2, 1, 0]].reshape(B, 1, 1, N, 3)
                pred_mask_soft = F.grid_sample(soft_prior_mask, uv_sampling, align_corners=True, padding_mode='border')[:, :, 0, 0, :]

                uv_target = item['points'][..., [2, 1, 0]].reshape(B, 1, 1, N, 3)
                gt_mask_soft = F.grid_sample(soft_target_mask, uv_target, align_corners=True, padding_mode='border')[:, :, 0, 0, :]

                loss_reg = torch.tensor(0.0, device=delta_coords.device)
                loss_mask = torch.mean((pred_mask_soft - gt_mask_soft) ** 2)

                # ==========================================
                # 🟢 回滚 2：恢复简单的局部平滑，不干扰宏观大局
                # ==========================================
                half_N = N // 2
                delta_anchors = delta_coords[:, :, :half_N]
                delta_perturbed = delta_coords[:, :, half_N:]
                loss_smooth = torch.mean((delta_anchors - delta_perturbed) ** 2)

                # ==========================================
                # 🔴 新增防线：防止作弊的“原点橡皮筋”
                # ==========================================
                loss_reg = torch.mean(delta_coords ** 2)

                # ==========================================
                # 终极物理聚合：带天眼的受控推土机
                # ==========================================
                # 1. 恢复灰度重构，让网络找回重建能力
                w_recon = 1.0

                # 2. 宏观推力：因为 MSE 算出来的数值很小（约 0.01 级别），
                # 我们需要极大的权重把它放大成强劲的推力。
                w_mask = 20.0

                # 3. 基础平滑：防止微观撕裂，但绝不干扰大局
                w_smooth = 100000.0

                # 4. 🔴 防作弊底线：一旦坐标位移过大，平方惩罚就会爆炸
                # 这逼迫网络必须在原地老老实实地对齐，而不是把点扔出画外
                w_reg = 0.5

                loss = (w_recon * loss_recon) + (w_mask * loss_mask) + (w_smooth * loss_smooth) + (w_reg * loss_reg)

                loss_list.append(loss.item())

                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                current_lr = optimizer.param_groups[0]["lr"]
                # 进度条打印平滑损失，观察网络的屈服过程
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    # 'L_mask': f'{loss_mask.item():.4f}',
                    'lr': f'{current_lr:.6f}',
                    # 'w_rec': w_recon_dynamic,
                    # 'w_mask': w_mask_dynamic,
                    # 'L_rec': f'{loss_recon.item():.4f}',
                    # 'L_sm': f'{loss_smooth.item():.5f}',
                    # 'sm_wt': w_smooth_dynamic,
                })

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
            # 🔴 新增：用于记录初始和重建的质心误差
            init_coms, recon_coms = [], []
            inference_times = []

            with torch.no_grad():
                for i, v_item in enumerate(eval_loader):
                    if i >= 5: break
                    v_item = convert_cuda(v_item)

                    prior_vol = v_item['image']

                    cpu_rng_state = torch.get_rng_state()
                    gpu_rng_state = torch.cuda.get_rng_state()

                    fixed_seed = 2026 + i
                    torch.manual_seed(fixed_seed)
                    torch.cuda.manual_seed(fixed_seed)

                    if 'mask' in v_item:
                        prior_mask = (v_item['mask'] == 1).float()
                        combined_eval = torch.cat([prior_vol, prior_mask], dim=1)

                        warped_eval = deformer(combined_eval)

                        target_vol = warped_eval[:, 0:1]
                        target_mask = warped_eval[:, 1:2]
                    else:
                        raise ValueError("计算 CoM 误差必须输入 Mask！")

                    torch.set_rng_state(cpu_rng_state)
                    torch.cuda.set_rng_state(gpu_rng_state)

                    v_item['projs'] = gpu_slice_volume(target_vol)
                    v_item['prior_projs'] = gpu_slice_volume(prior_vol)
                    v_item['prior'] = prior_vol

                    torch.cuda.synchronize()
                    t_start = time.time()

                    pred, delta_coords = model(v_item, is_eval=True, eval_npoint=50000)

                    torch.cuda.synchronize()
                    t_end = time.time()
                    inference_times.append(t_end - t_start)

                    # ==========================================
                    # 🟢 获取预测的 Mask 并计算 Recon CoM
                    # ==========================================
                    shape = v_item['image'].shape[2:] # 获取 3D 尺寸，例如 [256, 256, 128]

                    # 1. 生成标准的 [-1, 1] 基础空间网格 (Base Grid)
                    # affine_grid 是生成 3D 坐标网格最严谨的方法，默认匹配 grid_sample 的 (x,y,z) 顺序
                    theta = torch.eye(4, device=pred.device).unsqueeze(0)[:, :3, :]
                    base_grid = F.affine_grid(theta, [1, 1, shape[0], shape[1], shape[2]], align_corners=True)

                    # 2. 将网络输出的位移场 (1, 3, N) reshape 为 (1, X, Y, Z, 3)
                    delta_grid = delta_coords.view(1, 3, shape[0], shape[1], shape[2]).permute(0, 2, 3, 4, 1)

                    # 3. 基础坐标叠加位移场，获得最终采样网格
                    sample_grid = base_grid + delta_grid

                    # 4. 扭曲先验掩码 (必须使用 'nearest' 最邻近插值，保证掩码依然是 0/1 的二值状态)
                    pred_mask_tensor = F.grid_sample(prior_mask, sample_grid, mode='nearest', padding_mode='zeros', align_corners=True)

                    # 转换为 numpy 供后续使用
                    pred_mask_np = pred_mask_tensor[0, 0].cpu().numpy()
                    pred_np = pred[0, 0].cpu().numpy().reshape(shape)
                    gt_img_np = target_vol.cpu().numpy()[0, 0]
                    gt_mask_np = (target_mask.cpu().numpy()[0, 0] > 0.5).astype(np.float32)
                    prior_mask_np = prior_mask.cpu().numpy()[0, 0]

                    # 🔴 计算 Initial 和 Recon 的质心误差 (假设 spacing 为 1.5mm，请核实你的真实 spacing)
                    init_com_err = compute_com_error(gt_mask_np, prior_mask_np, spacing=(1.5, 1.5, 1.5))
                    recon_com_err = compute_com_error(gt_mask_np, pred_mask_np, spacing=(1.5, 1.5, 1.5))

                    init_coms.append(init_com_err)
                    recon_coms.append(recon_com_err)

                    # ==========================================
                    # 🟢 提取 ROI 并计算 SSIM (严苛掩码版)
                    # ==========================================
                    coords = np.argwhere(gt_mask_np > 0.5)
                    if len(coords) > 0:
                        x_min, y_min, z_min = coords.min(axis=0)
                        x_max, y_max, z_max = coords.max(axis=0)

                        eval_margin = 5
                        x_min = max(0, x_min - eval_margin)
                        y_min = max(0, y_min - eval_margin)
                        z_min = max(0, z_min - eval_margin)
                        x_max = min(gt_img_np.shape[0]-1, x_max + eval_margin)
                        y_max = min(gt_img_np.shape[1]-1, y_max + eval_margin)
                        z_max = min(gt_img_np.shape[2]-1, z_max + eval_margin)

                        gt_roi = gt_img_np[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
                        pred_roi = pred_np[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
                        mask_roi = gt_mask_np[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]

                        p, s = strict_masked_eval(gt_roi, pred_roi, mask_roi)

                        prior_np = prior_vol.cpu().numpy()[0, 0]
                        prior_roi = prior_np[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]

                        p_init, s_init = strict_masked_eval(gt_roi, prior_roi, mask_roi)

                        # 🔴 全维度的论文级打印输出
                        print(f"  [Metric] SSIM: {s_init:.4f} -> {s:.4f} | CoM Error: {init_com_err:.2f} mm -> {recon_com_err:.2f} mm")

                        psnrs.append(p)
                        ssims.append(s)
                    else:
                        print(f"  [Warning] {v_item['name'][0]} 提取 BBox 失败。")

            # 循环结束后的平均统计
            avg_psnr = np.mean(psnrs)
            avg_ssim = np.mean(ssims)
            avg_init_com = np.mean(init_coms)
            avg_recon_com = np.mean(recon_coms)

            eval_msg = f"     [Eval Result] Epoch {epoch}: PSNR = {avg_psnr:.4f} | SSIM = {avg_ssim:.4f} | CoM = {avg_init_com:.2f} -> {avg_recon_com:.2f} mm"
            logger.info(eval_msg)

        lr_scheduler.step()
        epoch += 1