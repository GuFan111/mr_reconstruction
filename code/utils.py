# utils.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from monai.losses import SSIMLoss
from scipy.ndimage import center_of_mass

def convert_cuda(item):
    for key in item.keys():
        if key not in ['name', 'dst_name']:
            item[key] = item[key].float().cuda(non_blocking=True)
    return item


def save_nifti(image, path):
    out = sitk.GetImageFromArray(image)
    sitk.WriteImage(out, path)


class HybridReconLoss(nn.Module):
    def __init__(self, alpha=0.85):
        """
        alpha: SSIM 的权重。通常 SSIM 占主导 (0.8~0.85)，L1 用于维持绝对亮度。
        """
        super().__init__()
        self.alpha = alpha
        self.l1_loss = nn.L1Loss()
        # MONAI 的 SSIMLoss 完美支持 3D 张量 (B, C, D, H, W)
        self.ssim_loss = SSIMLoss(spatial_dims=3, data_range=1.0)

    def forward(self, pred, target):
        # 计算 L1 损失
        l1 = self.l1_loss(pred, target)
        # 计算 SSIM 损失 (SSIMLoss 返回的是 1 - SSIM，所以越小越好)
        ssim = self.ssim_loss(pred, target)

        # 混合 Loss
        total_loss = self.alpha * ssim + (1 - self.alpha) * l1
        return total_loss, l1, ssim


def gpu_slice_volume(volume, slice_idx=None):
    # volume: [B, 1, X, Y, Z]
    B, C, X, Y, Z = volume.shape
    res_max = max(X, Y, Z)

    # 自动计算中心索引
    if slice_idx is None:
        idx_x, idx_y, idx_z = X // 2, Y // 2, Z // 2
    else:
        # 兼容标量或三维元组输入
        idx_x = slice_idx[0] if isinstance(slice_idx, (tuple, list)) else X // 2
        idx_y = slice_idx[1] if isinstance(slice_idx, (tuple, list)) else Y // 2
        idx_z = slice_idx[2] if isinstance(slice_idx, (tuple, list)) else Z // 2

    # 1. Axial (XY平面): 固定 Z 轴 (dim 4)
    slice_ax = volume[:, :, :, :, idx_z]   # [B, 1, X, Y]
    # 2. Coronal (XZ平面): 固定 Y 轴 (dim 3)
    slice_cor = volume[:, :, :, idx_y, :]  # [B, 1, X, Z]
    # 3. Sagittal (YZ平面): 固定 X 轴 (dim 2)
    slice_sag = volume[:, :, idx_x, :, :]  # [B, 1, Y, Z]

    # 为了让模型能够并行处理不同尺寸的切片，Resize 到统一的正方形
    slice_ax = F.interpolate(slice_ax, size=(res_max, res_max), mode='bilinear', align_corners=True)
    slice_cor = F.interpolate(slice_cor, size=(res_max, res_max), mode='bilinear', align_corners=True)
    slice_sag = F.interpolate(slice_sag, size=(res_max, res_max), mode='bilinear', align_corners=True)

    # 堆叠成 [B, 3, 1, res_max, res_max]
    return torch.stack([slice_ax, slice_cor, slice_sag], dim=1)

def save_visualization_3view_l2r(model, dataset, epoch, device='cuda', save_dir='vis_results', simulator=None, prior_deformer=None):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    # ==========================================
    # 🔴 核心修复：优先加载数据，动态读取真实体素分辨率
    # ==========================================
    data_item = dataset[0]
    name = data_item['name']
    vol_np = data_item['image'] # Shape: [1, X, Y, Z]

    # 动态解析当前病人的真实 X, Y, Z 尺寸
    _, res_x, res_y, res_z = vol_np.shape
    idx_x, idx_y, idx_z = res_x // 2, res_y // 2, res_z // 2
    fix_x, fix_y, fix_z = idx_x/(res_x-1), idx_y/(res_y-1), idx_z/(res_z-1)

    # 2. 生成隐式查询网格 (生成切片所在的十字坐标)
    ax_x, ax_y, ax_z = np.linspace(0, 1, res_x), np.linspace(0, 1, res_y), np.linspace(0, 1, res_z)
    u_ax, v_ax = np.meshgrid(ax_x, ax_y, indexing='ij')
    pts_ax = np.stack([u_ax, v_ax, np.ones_like(u_ax)*fix_z], axis=-1).reshape(-1, 3) # XY
    u_co, v_co = np.meshgrid(ax_x, ax_z, indexing='ij')
    pts_co = np.stack([u_co, np.ones_like(u_co)*fix_y, v_co], axis=-1).reshape(-1, 3) # XZ
    u_sa, v_sa = np.meshgrid(ax_y, ax_z, indexing='ij')
    pts_sa = np.stack([np.ones_like(u_sa)*fix_x, u_sa, v_sa], axis=-1).reshape(-1, 3) # YZ

    n_list = [len(pts_ax), len(pts_co), len(pts_sa)]
    all_points = np.concatenate([pts_ax, pts_co, pts_sa], axis=0).astype(np.float32)

    # 3. 准备数据与推理 (兼容真实的 Prior 和 Target)
    vol_cuda = torch.from_numpy(vol_np).unsqueeze(0).to(device)

    with torch.no_grad():
        # L2R 数据集中已经包含了真实的 prior
        prior_vol = torch.from_numpy(data_item['prior']).unsqueeze(0).to(device)
        prior_projs = gpu_slice_volume(prior_vol)

        # 🔴 L2R 是真实对齐数据，彻底弃用 deformer (形变器) 的时间静止魔法
        target_vol = vol_cuda
        noisy_vol = simulator(target_vol) if simulator else target_vol
        projs = gpu_slice_volume(noisy_vol)

        points_ts = torch.from_numpy((all_points - 0.5) * 2).unsqueeze(0).to(device)
        proj_ts = torch.stack([points_ts[..., [0, 1]], points_ts[..., [0, 2]], points_ts[..., [1, 2]]], dim=1)

        input_dict = {'projs': projs, 'points': points_ts, 'proj_points': proj_ts, 'prior': prior_vol}

        # DIF-Net 隐式推断
        preds, deltas = model(input_dict, is_eval=True, eval_npoint=50000)

    # 4. 数据解构
    preds_raw = preds[0, 0].cpu().numpy()
    deltas_raw = deltas[0].cpu().numpy()
    deform_mag = np.linalg.norm(deltas_raw, axis=0)

    view_resolutions = [(res_x, res_y), (res_x, res_z), (res_y, res_z)]
    imgs_pred, imgs_deform, imgs_delta_v = [], [], []
    curr = 0
    for i, n in enumerate(n_list):
        h_v, w_v = view_resolutions[i]
        imgs_pred.append(preds_raw[curr:curr+n].reshape(h_v, w_v))
        imgs_deform.append(deform_mag[curr:curr+n].reshape(h_v, w_v))
        imgs_delta_v.append(deltas_raw[:, curr:curr+n].reshape(3, h_v, w_v))
        curr += n

    # 5. 提取 GT 并处理 Prior 缩放
    # 🟢 修正：GT 切片必须从 target_vol 中提取
    target_vol_np = target_vol[0, 0].cpu().numpy()
    gt_slices = [target_vol_np[:, :, idx_z], target_vol_np[:, idx_y, :], target_vol_np[idx_x, :, :]]

    raw_prior_np = prior_projs[0, :, 0].cpu().numpy()

    imgs_prior = []
    for i in range(3):
        h, w = gt_slices[i].shape
        import cv2
        imgs_prior.append(cv2.resize(raw_prior_np[i], (w, h), interpolation=cv2.INTER_LINEAR))

    # 6. 绘图 (完全保留你引入的自动对比度拉伸)
    fig, axes = plt.subplots(3, 5, figsize=(22, 12))
    titles = ['Axial', 'Coronal', 'Sagittal']
    col_titles = ["GT/Input", "Prior", "Recon", "Diff (x5)", "Deform Flow"]

    for i in range(3):
        # 动态计算位移上限
        local_vmax = max(0.01, np.percentile(imgs_deform[i], 98))
        h_img, w_img = gt_slices[i].shape

        # --- 对比度增强核心逻辑 ---
        # 即使数据已经是 0-1 归一化的，由于 MRI 的特性，软组织往往偏暗。
        # 我们计算第 2% 和第 98% 分位数，将其拉伸到 0-1 范围。
        def enhance_contrast(img):
            p2, p98 = np.percentile(img, [2, 98])
            # 防止分母为 0
            img_adj = np.clip((img - p2) / (p98 - p2 + 1e-8), 0, 1)
            # 加上一点伽马校正，让暗部细节更深邃
            return np.power(img_adj, 0.9)

        im_list = [
            enhance_contrast(gt_slices[i]),            # 增强后的 GT
            enhance_contrast(imgs_prior[i]),           # 增强后的 Prior
            enhance_contrast(np.clip(imgs_pred[i], 0, 1)), # 增强后的 Recon
            np.abs(gt_slices[i] - np.clip(imgs_pred[i], 0, 1)) # Diff 保持原始比例
        ]

        # 绘制前 4 列
        for j in range(4):
            data_to_show = im_list[j].T
            # 因为已经在上面 enhance_contrast 过了，这里 vmax 统一用 1.0 即可
            v_max = 1.0 if j < 3 else 0.2
            cmap = 'gray' if j < 3 else 'inferno'
            axes[i, j].imshow(data_to_show, cmap=cmap, vmin=0, vmax=v_max, origin='lower', aspect='auto')

            if i == 0:
                axes[i, j].set_title(col_titles[j], fontsize=14, fontweight='bold')
            axes[i, j].axis('off')

        # 第 5 列 (索引 4)：Deform Flow
        ax = axes[i, 4]
        extent = [0, w_img, 0, h_img]
        # 背景叠加
        ax.imshow(gt_slices[i].T, cmap='gray', alpha=0.8, origin='lower', aspect='auto', extent=extent)
        ax.imshow(imgs_deform[i].T, cmap='jet', alpha=0.5, vmin=0, vmax=local_vmax, origin='lower', aspect='auto', extent=extent)

        # 绘制稀疏矢量箭头
        step = 16
        y, x = np.mgrid[step//2:h_img:step, step//2:w_img:step]

        if i == 0: # Axial
            u, v = imgs_delta_v[i][0, ::step, ::step], imgs_delta_v[i][1, ::step, ::step]
        elif i == 1: # Coronal
            u, v = imgs_delta_v[i][0, ::step, ::step], imgs_delta_v[i][2, ::step, ::step]
        else: # Sagittal
            u, v = imgs_delta_v[i][1, ::step, ::step], imgs_delta_v[i][2, ::step, ::step]

        ax.quiver(x, y, u.T, v.T, color='white', scale=1.0, width=0.005, alpha=0.9, pivot='mid')

        if i == 0:
            ax.set_title(col_titles[4], fontsize=14, fontweight='bold')
        ax.axis('off')

        # 修复：优化行标题偏移量，防止遮挡
        axes[i, 0].text(-0.15, 0.5, titles[i], transform=axes[i, 0].transAxes,
                        rotation=90, va='center', ha='center', fontsize=14, fontweight='bold')

    # 修复：预留左侧 5% 的空间给行标题文字
    plt.tight_layout(rect=[0.05, 0, 1, 1])
    save_path = os.path.join(save_dir, f'vis_ep{epoch}_{name}.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Merged visualization saved to {save_path}")


def save_visualization_3view_chaos(model, dataset, epoch, device='cuda', save_dir='vis_results', simulator=None, prior_deformer=None):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    # 1. 优先提取数据，动态获取真实分辨率 (消除 out_res 报错)
    data_item = dataset[0]
    name = data_item['name']
    vol_np = data_item['image'] # Shape: [1, X, Y, Z]
    _, res_x, res_y, res_z = vol_np.shape

    # 2. 锁定绝对解剖中心（与切片逻辑 100% 对齐）
    cx, cy, cz = res_x // 2, res_y // 2, res_z // 2

    # 3. 直接构造完美的 [-1, 1] 物理空间标尺
    x_coords = np.linspace(-1, 1, res_x)
    y_coords = np.linspace(-1, 1, res_y)
    z_coords = np.linspace(-1, 1, res_z)

    # 4. 生成三个正交面的 3D 坐标网格 (indexing='ij' 防止长宽转置)
    u_ax, v_ax = np.meshgrid(x_coords, y_coords, indexing='ij')
    pts_ax = np.stack([u_ax, v_ax, np.full_like(u_ax, z_coords[cz])], axis=-1).reshape(-1, 3) # XY

    u_co, v_co = np.meshgrid(x_coords, z_coords, indexing='ij')
    pts_co = np.stack([u_co, np.full_like(u_co, y_coords[cy]), v_co], axis=-1).reshape(-1, 3) # XZ

    u_sa, v_sa = np.meshgrid(y_coords, z_coords, indexing='ij')
    pts_sa = np.stack([np.full_like(u_sa, x_coords[cx]), u_sa, v_sa], axis=-1).reshape(-1, 3) # YZ

    n_list = [len(pts_ax), len(pts_co), len(pts_sa)]
    all_points = np.concatenate([pts_ax, pts_co, pts_sa], axis=0).astype(np.float32)

    # 5. 准备数据与推理
    vol_cuda = torch.from_numpy(vol_np).unsqueeze(0).to(device)

    with torch.no_grad():
        prior_vol = vol_cuda
        # 注意：若 gpu_slice_volume 还没在这个文件 import，记得在顶部引入
        prior_projs = gpu_slice_volume(prior_vol)

        # 植入时间静止魔法：保证可视化和 Eval 用的是同一个形变场
        if prior_deformer:
            cpu_rng_state = torch.get_rng_state()
            gpu_rng_state = torch.cuda.get_rng_state()

            fixed_seed = 2026
            torch.manual_seed(fixed_seed)
            torch.cuda.manual_seed(fixed_seed)

            target_vol = prior_deformer(prior_vol)

            torch.set_rng_state(cpu_rng_state)
            torch.cuda.set_rng_state(gpu_rng_state)
        else:
            target_vol = prior_vol

        noisy_vol = simulator(target_vol) if simulator else target_vol
        projs = gpu_slice_volume(noisy_vol)

        # 🟢 核心修正：直接使用 all_points，不再需要 (points - 0.5)*2
        points_ts = torch.from_numpy(all_points).unsqueeze(0).to(device)
        proj_ts = torch.stack([
            points_ts[..., [0, 1]],
            points_ts[..., [0, 2]],
            points_ts[..., [1, 2]]
        ], dim=1)

        input_dict = {'projs': projs, 'points': points_ts, 'proj_points': proj_ts, 'prior': prior_vol}
        preds, deltas = model(input_dict, is_eval=True, eval_npoint=50000)

    # 6. 数据解构
    preds_raw = preds[0, 0].cpu().numpy()
    deltas_raw = deltas[0].cpu().numpy()
    deform_mag = np.linalg.norm(deltas_raw, axis=0)

    view_resolutions = [(res_x, res_y), (res_x, res_z), (res_y, res_z)]
    imgs_pred, imgs_deform, imgs_delta_v = [], [], []
    curr = 0
    for i, n in enumerate(n_list):
        h_v, w_v = view_resolutions[i]
        imgs_pred.append(preds_raw[curr:curr+n].reshape(h_v, w_v))
        imgs_deform.append(deform_mag[curr:curr+n].reshape(h_v, w_v))
        imgs_delta_v.append(deltas_raw[:, curr:curr+n].reshape(3, h_v, w_v))
        curr += n

    # 7. 提取 GT 并处理 Prior 缩放
    target_vol_np = target_vol[0, 0].cpu().numpy()
    gt_slices = [target_vol_np[:, :, cz], target_vol_np[:, cy, :], target_vol_np[cx, :, :]]

    raw_prior_np = prior_projs[0, :, 0].cpu().numpy()

    imgs_prior = []
    for i in range(3):
        h, w = gt_slices[i].shape
        import cv2
        imgs_prior.append(cv2.resize(raw_prior_np[i], (w, h), interpolation=cv2.INTER_LINEAR))

    # 8. 绘图与对比度拉伸
    fig, axes = plt.subplots(3, 5, figsize=(22, 12))
    titles = ['Axial', 'Coronal', 'Sagittal']
    col_titles = ["GT/Input", "Prior", "Recon", "Diff (x5)", "Deform Flow"]

    def enhance_contrast(img):
        p2, p98 = np.percentile(img, [2, 98])
        img_adj = np.clip((img - p2) / (p98 - p2 + 1e-8), 0, 1)
        return np.power(img_adj, 0.9)

    for i in range(3):
        local_vmax = max(0.01, np.percentile(imgs_deform[i], 98))
        h_img, w_img = gt_slices[i].shape

        im_list = [
            enhance_contrast(gt_slices[i]),
            enhance_contrast(imgs_prior[i]),
            enhance_contrast(np.clip(imgs_pred[i], 0, 1)),
            np.abs(gt_slices[i] - np.clip(imgs_pred[i], 0, 1))
        ]

        for j in range(4):
            data_to_show = im_list[j].T
            v_max = 1.0 if j < 3 else 0.2
            cmap = 'gray' if j < 3 else 'inferno'
            axes[i, j].imshow(data_to_show, cmap=cmap, vmin=0, vmax=v_max, origin='lower', aspect='auto')

            if i == 0:
                axes[i, j].set_title(col_titles[j], fontsize=14, fontweight='bold')
            axes[i, j].axis('off')

        # 第 5 列 (Deform Flow)
        ax = axes[i, 4]
        extent = [0, w_img, 0, h_img]
        ax.imshow(gt_slices[i].T, cmap='gray', alpha=0.8, origin='lower', aspect='auto', extent=extent)
        ax.imshow(imgs_deform[i].T, cmap='jet', alpha=0.5, vmin=0, vmax=local_vmax, origin='lower', aspect='auto', extent=extent)

        step = 16
        y, x = np.mgrid[step//2:h_img:step, step//2:w_img:step]

        if i == 0:
            u, v = imgs_delta_v[i][0, ::step, ::step], imgs_delta_v[i][1, ::step, ::step]
        elif i == 1:
            u, v = imgs_delta_v[i][0, ::step, ::step], imgs_delta_v[i][2, ::step, ::step]
        else:
            u, v = imgs_delta_v[i][1, ::step, ::step], imgs_delta_v[i][2, ::step, ::step]

        ax.quiver(x, y, u.T, v.T, color='white', scale=1.0, width=0.005, alpha=0.9, pivot='mid')

        if i == 0:
            ax.set_title(col_titles[4], fontsize=14, fontweight='bold')
        ax.axis('off')

        axes[i, 0].text(-0.15, 0.5, titles[i], transform=axes[i, 0].transAxes,
                        rotation=90, va='center', ha='center', fontsize=14, fontweight='bold')

    plt.tight_layout(rect=[0.05, 0, 1, 1])
    save_path = os.path.join(save_dir, f'vis_ep{epoch}_{name}.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Merged visualization saved to {save_path}")


def save_visualization_3view(model, dataset, epoch, device='cuda', save_dir='vis_results', simulator=None, prior_deformer=None):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    # 1. 获取非立方体分辨率信息
    res_x, res_y, res_z = dataset.out_res
    idx_x, idx_y, idx_z = res_x // 2, res_y // 2, res_z // 2
    fix_x, fix_y, fix_z = idx_x/(res_x-1), idx_y/(res_y-1), idx_z/(res_z-1)

    # 2. 生成采样网格
    ax_x, ax_y, ax_z = np.linspace(0, 1, res_x), np.linspace(0, 1, res_y), np.linspace(0, 1, res_z)
    u_ax, v_ax = np.meshgrid(ax_x, ax_y, indexing='ij')
    pts_ax = np.stack([u_ax, v_ax, np.ones_like(u_ax)*fix_z], axis=-1).reshape(-1, 3) # XY
    u_co, v_co = np.meshgrid(ax_x, ax_z, indexing='ij')
    pts_co = np.stack([u_co, np.ones_like(u_co)*fix_y, v_co], axis=-1).reshape(-1, 3) # XZ
    u_sa, v_sa = np.meshgrid(ax_y, ax_z, indexing='ij')
    pts_sa = np.stack([np.ones_like(u_sa)*fix_x, u_sa, v_sa], axis=-1).reshape(-1, 3) # YZ

    n_list = [len(pts_ax), len(pts_co), len(pts_sa)]
    all_points = np.concatenate([pts_ax, pts_co, pts_sa], axis=0).astype(np.float32)

    # 3. 准备数据与推理
    data_item = dataset[0]
    name = data_item['name']
    vol_np = data_item['image'] # [1, X, Y, Z]
    vol_cuda = torch.from_numpy(vol_np).unsqueeze(0).to(device)

    with torch.no_grad():
        prior_vol = vol_cuda
        prior_projs = gpu_slice_volume(prior_vol)

        # ==========================================
        # 🟢 植入时间静止魔法：保证可视化和 Eval 算指标用的是同一个形变场！
        # ==========================================
        if prior_deformer:
            cpu_rng_state = torch.get_rng_state()
            gpu_rng_state = torch.cuda.get_rng_state()

            # 因为可视化固定画的是 dataset[0] (也就是 i=0)，所以 seed 必须是 2026 + 0
            fixed_seed = 2026
            torch.manual_seed(fixed_seed)
            torch.cuda.manual_seed(fixed_seed)

            # 宿命形变
            # target_vol = prior_deformer(prior_vol, mode='bilinear', fixed_phase=1.5708)
            target_vol = prior_deformer(prior_vol)

            # 恢复时间的流动
            torch.set_rng_state(cpu_rng_state)
            torch.cuda.set_rng_state(gpu_rng_state)
        else:
            target_vol = prior_vol
        # ==========================================

        noisy_vol = simulator(target_vol) if simulator else target_vol
        projs = gpu_slice_volume(noisy_vol)

        points_ts = torch.from_numpy((all_points - 0.5) * 2).unsqueeze(0).to(device)
        proj_ts = torch.stack([points_ts[..., [0, 1]], points_ts[..., [0, 2]], points_ts[..., [1, 2]]], dim=1)

        input_dict = {
            'projs': projs,
            'prior_projs': prior_projs,  # 补上缺失的键值
            'points': points_ts,
            'proj_points': proj_ts,
            'prior': prior_vol
        }
        # 获取预测值和位移场
        preds, deltas = model(input_dict, is_eval=True, eval_npoint=50000)

    # 4. 数据解构
    preds_raw = preds[0, 0].cpu().numpy()
    deltas_raw = deltas[0].cpu().numpy()
    deform_mag = np.linalg.norm(deltas_raw, axis=0)

    view_resolutions = [(res_x, res_y), (res_x, res_z), (res_y, res_z)]
    imgs_pred, imgs_deform, imgs_delta_v = [], [], []
    curr = 0
    for i, n in enumerate(n_list):
        h_v, w_v = view_resolutions[i]
        imgs_pred.append(preds_raw[curr:curr+n].reshape(h_v, w_v))
        imgs_deform.append(deform_mag[curr:curr+n].reshape(h_v, w_v))
        imgs_delta_v.append(deltas_raw[:, curr:curr+n].reshape(3, h_v, w_v))
        curr += n

    # 5. 提取 GT 并处理 Prior 缩放
    # 🟢 修正：GT 切片必须从 target_vol 中提取
    target_vol_np = target_vol[0, 0].cpu().numpy()
    gt_slices = [target_vol_np[:, :, idx_z], target_vol_np[:, idx_y, :], target_vol_np[idx_x, :, :]]

    raw_prior_np = prior_projs[0, :, 0].cpu().numpy()

    imgs_prior = []
    for i in range(3):
        h, w = gt_slices[i].shape
        import cv2
        imgs_prior.append(cv2.resize(raw_prior_np[i], (w, h), interpolation=cv2.INTER_LINEAR))

    # 6. 绘图 (完全保留你引入的自动对比度拉伸)
    fig, axes = plt.subplots(3, 5, figsize=(22, 12))
    titles = ['Axial', 'Coronal', 'Sagittal']
    col_titles = ["GT/Input", "Prior", "Recon", "Diff (x5)", "Deform Flow"]

    for i in range(3):
        # 动态计算位移上限
        local_vmax = max(0.01, np.percentile(imgs_deform[i], 98))
        h_img, w_img = gt_slices[i].shape

        # --- 对比度增强核心逻辑 ---
        # 即使数据已经是 0-1 归一化的，由于 MRI 的特性，软组织往往偏暗。
        # 我们计算第 2% 和第 98% 分位数，将其拉伸到 0-1 范围。
        def enhance_contrast(img):
            p2, p98 = np.percentile(img, [2, 98])
            # 防止分母为 0
            img_adj = np.clip((img - p2) / (p98 - p2 + 1e-8), 0, 1)
            # 加上一点伽马校正，让暗部细节更深邃
            return np.power(img_adj, 0.9)

        im_list = [
            enhance_contrast(gt_slices[i]),            # 增强后的 GT
            enhance_contrast(imgs_prior[i]),           # 增强后的 Prior
            enhance_contrast(np.clip(imgs_pred[i], 0, 1)), # 增强后的 Recon
            np.abs(gt_slices[i] - np.clip(imgs_pred[i], 0, 1)) # Diff 保持原始比例
        ]

        # 绘制前 4 列
        for j in range(4):
            data_to_show = im_list[j].T
            # 因为已经在上面 enhance_contrast 过了，这里 vmax 统一用 1.0 即可
            v_max = 1.0 if j < 3 else 0.2
            cmap = 'gray' if j < 3 else 'inferno'
            axes[i, j].imshow(data_to_show, cmap=cmap, vmin=0, vmax=v_max, origin='lower', aspect='auto')

            if i == 0:
                axes[i, j].set_title(col_titles[j], fontsize=14, fontweight='bold')
            axes[i, j].axis('off')

        # 第 5 列 (索引 4)：Deform Flow
        ax = axes[i, 4]
        extent = [0, w_img, 0, h_img]
        # 背景叠加
        ax.imshow(gt_slices[i].T, cmap='gray', alpha=0.8, origin='lower', aspect='auto', extent=extent)
        ax.imshow(imgs_deform[i].T, cmap='jet', alpha=0.5, vmin=0, vmax=local_vmax, origin='lower', aspect='auto', extent=extent)

        # 绘制稀疏矢量箭头
        step = 16
        y, x = np.mgrid[step//2:h_img:step, step//2:w_img:step]

        if i == 0: # Axial
            u, v = imgs_delta_v[i][0, ::step, ::step], imgs_delta_v[i][1, ::step, ::step]
        elif i == 1: # Coronal
            u, v = imgs_delta_v[i][0, ::step, ::step], imgs_delta_v[i][2, ::step, ::step]
        else: # Sagittal
            u, v = imgs_delta_v[i][1, ::step, ::step], imgs_delta_v[i][2, ::step, ::step]

        ax.quiver(x, y, u.T, v.T, color='white', scale=1.0, width=0.005, alpha=0.9, pivot='mid')

        if i == 0:
            ax.set_title(col_titles[4], fontsize=14, fontweight='bold')
        ax.axis('off')

        # 修复：优化行标题偏移量，防止遮挡
        axes[i, 0].text(-0.15, 0.5, titles[i], transform=axes[i, 0].transAxes,
                        rotation=90, va='center', ha='center', fontsize=14, fontweight='bold')

    # 修复：预留左侧 5% 的空间给行标题文字
    plt.tight_layout(rect=[0.05, 0, 1, 1])
    save_path = os.path.join(save_dir, f'vis_ep{epoch}_{name}.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Merged visualization saved to {save_path}")


def compute_com_error(mask_a, mask_b, spacing=(1.0, 1.0, 1.0)):
    """计算两个 3D 掩码的质心物理误差 (毫米)"""
    com_a = np.array(center_of_mass(mask_a > 0.5))
    com_b = np.array(center_of_mass(mask_b > 0.5))

    if np.any(np.isnan(com_a)) or np.any(np.isnan(com_b)):
        return 999.0 # 异常惩罚

    physical_diff = (com_a - com_b) * np.array(spacing)
    return np.linalg.norm(physical_diff)


def simple_eval(model, loader, npoint=50000):  # 每次评估每批计算50000个点
    model.eval()
    psnr_list, ssim_list = [], []

    with torch.no_grad():
        for i, item in enumerate(loader):
            if i >= 5: break  # 只评估前5个样本以节省时间
            item = convert_cuda(item)

            if torch.sum(item['projs']) == 0:
                item['projs'] = gpu_slice_volume(item['image'])
            image_gt = item['image'].cpu().numpy()[0, 0] # [X, Y, Z]

            # 全图推理
            pred = model(item, is_eval=True, eval_npoint=npoint)
            pred = pred[0, 0].cpu().numpy().reshape(image_gt.shape)

            p = peak_signal_noise_ratio(image_gt, pred, data_range=1.0)
            s = structural_similarity(image_gt, pred, data_range=1.0)
            psnr_list.append(p)
            ssim_list.append(s)

    return np.mean(psnr_list), np.mean(ssim_list)


# 模拟噪声
class GPUDailyScanSimulator(nn.Module):
    def __init__(self, noise_level=0.05, blur_sigma=0.5, kernel_size=5):
        super().__init__()
        self.noise_level = noise_level
        self.blur_sigma = blur_sigma
        self.kernel_size = kernel_size

    def get_gaussian_kernel(self, sigma, channels=1):
        # 生成 1D 核
        k_size = self.kernel_size
        x = torch.arange(k_size, device='cuda') - k_size // 2
        kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()

        # 生成 3D 核: k1 * k2 * k3
        k1 = kernel_1d.view(1, 1, -1, 1, 1)
        k2 = kernel_1d.view(1, 1, 1, -1, 1)
        k3 = kernel_1d.view(1, 1, 1, 1, -1)

        return k1, k2, k3

    def forward(self, volume):
        # volume: [B, 1, D, H, W]
        if self.blur_sigma <= 0: return volume

        B = volume.shape[0]

        # 1. 模拟模糊 (Blurring)
        sigma = np.random.uniform(0.1, self.blur_sigma)
        k1, k2, k3 = self.get_gaussian_kernel(sigma)

        pad = self.kernel_size // 2
        x = F.conv3d(volume, k1.repeat(1,1,1,1,1), padding=(pad, 0, 0))
        x = F.conv3d(x, k2.repeat(1,1,1,1,1), padding=(0, pad, 0))
        x = F.conv3d(x, k3.repeat(1,1,1,1,1), padding=(0, 0, pad))

        # 2. 模拟噪声 (Gaussian Noise)
        noise = torch.randn_like(x)
        amplitude = np.random.uniform(0, self.noise_level)
        x = x + noise * amplitude

        # 3. 截断
        x = torch.clamp(x, 0, 1)
        return x


# 模拟形变
class ElasticDeformation(nn.Module):
    def __init__(self, grid_size=8, sigma=(0.02, 0.02, 0.08)):
        super().__init__()
        self.grid_size = grid_size # 形变频率
        # sigma 支持 float (各向同性) 或 tuple (sigma_x, sigma_y, sigma_z) (各向异性)
        self.sigma = sigma

    def forward(self, x, mode='bilinear'):
        # x: [B, C, D, H, W] -> 对应物理空间的 [B, C, X, Y, Z]
        B, C, D, H, W = x.shape
        device = x.device

        # 1. 生成低分辨率的随机位移场
        if isinstance(self.sigma, (list, tuple)):
            assert len(self.sigma) == 3, "Sigma must be a sequence of 3 floats: (sigma_x, sigma_y, sigma_z)"
            sigma_x, sigma_y, sigma_z = self.sigma

            # 🟢 物理映射对齐：
            # grid_sample 需要的顺序是 (W, H, D) -> 对应 (Z, Y, X)
            # 所以 Channel 0 修改 Z 轴, Channel 1 修改 Y 轴, Channel 2 修改 X 轴
            flow_z = torch.randn(B, 1, self.grid_size, self.grid_size, self.grid_size, device=device) * sigma_z
            flow_y = torch.randn(B, 1, self.grid_size, self.grid_size, self.grid_size, device=device) * sigma_y
            flow_x = torch.randn(B, 1, self.grid_size, self.grid_size, self.grid_size, device=device) * sigma_x
            flow_coarse = torch.cat([flow_z, flow_y, flow_x], dim=1)
        else:
            flow_coarse = torch.randn(B, 3, self.grid_size, self.grid_size, self.grid_size, device=device) * self.sigma

        # 2. 上采样到全分辨率
        flow = F.interpolate(flow_coarse, size=(D, H, W), mode='trilinear', align_corners=True)
        # flow shape: [B, 3, D, H, W] -> permute to [B, D, H, W, 3] for grid_sample
        flow = flow.permute(0, 2, 3, 4, 1)

        # 3. 生成基础网格
        d = torch.linspace(-1, 1, D, device=device)
        h = torch.linspace(-1, 1, H, device=device)
        w = torch.linspace(-1, 1, W, device=device)
        grid_d, grid_h, grid_w = torch.meshgrid(d, h, w, indexing='ij')

        # 🟢 注意这里的 stack 顺序: W(Z), H(Y), D(X)
        base_grid = torch.stack([grid_w, grid_h, grid_d], dim=-1).unsqueeze(0)

        # 4. 叠加形变
        final_grid = base_grid + flow

        # 5. 采样得到变形后的 Volume (兼容 Mask 的 nearest 插值)
        deformed_x = F.grid_sample(x, final_grid, mode=mode, padding_mode='reflection', align_corners=True)

        return deformed_x


class MR_Linac_SyntheticDeformer(nn.Module):
    """
    专为极稀疏 MR-Linac 切片重建设计的合成弹性形变器。
    具备：低通平滑约束、呼吸 Z 轴主导、等中心（切片交汇处）振幅增强。
    """
    def __init__(self, grid_size=5, max_disp=0.08, z_multiplier=3.0, center_focus=1.5):
        super().__init__()
        # 1. 物理低通滤波器：极低的分辨率网格，保证形变像水波一样绝对平滑，消除微观撕裂
        self.grid_size = grid_size

        # 2. 最大位移幅度：0.08 在 [-1, 1] 坐标系中代表约 8% 的图像宽度形变
        # 配合 192 的矩阵，大约是 15 个像素的位移 (约 22mm，极其符合真实呼吸幅度)
        self.max_disp = max_disp

        # 3. 呼吸主轴倍率：Z 轴(头脚方向)的形变必须远大于 X/Y 轴
        self.z_multiplier = z_multiplier

        # 4. 等中心增强系数：靶区(切片中心)的形变振幅是边缘的几倍
        self.center_focus = center_focus

    def forward(self, x):
        """
        x: [B, C, D, H, W] 的 3D 图像张量 (D=Z, H=Y, W=X)
        """
        B, C, D, H, W = x.shape
        device = x.device

        # ==========================================
        # 阶段 1：生成低频随机形变场 (Low-Frequency Sub-grid)
        # ==========================================
        # 生成 [B, 3, G, G, G] 的随机向量，范围 [-1, 1]
        disp_low_res = (torch.rand(B, 3, self.grid_size, self.grid_size, self.grid_size, device=device) * 2 - 1)

        # 施加呼吸主导法则：强行放大 Z 轴 (在 F.grid_sample 坐标系中，Z轴是 channel 2)
        # disp_low_res[:, 2, :, :, :] *= self.z_multiplier
        # disp_low_res[:, 1, :, :, :] *= self.z_multiplier
        disp_low_res[:, 0, :, :, :] *= self.z_multiplier

        # ==========================================
        # 阶段 2：等中心靶区增强 (Isocenter Modulation)
        # ==========================================
        # 构造一个 3D 高斯权重图，中心点权重为 center_focus，边缘衰减为 1.0
        lin = torch.linspace(-1, 1, self.grid_size, device=device)
        zz, yy, xx = torch.meshgrid(lin, lin, lin, indexing='ij')
        dist_sq = xx**2 + yy**2 + zz**2

        # 半径参数 0.3 控制了增强区域的范围 (约占中心 1/3 区域)
        weight_mask = 1.0 + (self.center_focus - 1.0) * torch.exp(-dist_sq / 0.3)

        # 将靶区权重施加到形变场上 [B, 3, G, G, G]
        disp_low_res = disp_low_res * weight_mask.unsqueeze(0).unsqueeze(0)

        # ==========================================
        # 阶段 3：无级平滑插值与绝对幅度缩放
        # ==========================================
        # 利用 C++ 底层的三线性插值，将 5x5x5 的粗糙网格，瞬间平滑放大到真实的 D x H x W
        disp_dense = F.interpolate(disp_low_res, size=(D, H, W), mode='trilinear', align_corners=True)

        # 将随机场限制在我们设定的物理最大位移范围内
        disp_dense = disp_dense * self.max_disp

        # ==========================================
        # 阶段 4：坐标映射与扭曲 (Warping)
        # ==========================================
        # 生成标准的 [-1, 1] 绝对物理坐标系
        lin_z = torch.linspace(-1, 1, D, device=device)
        lin_y = torch.linspace(-1, 1, H, device=device)
        lin_x = torch.linspace(-1, 1, W, device=device)

        # meshgrid 默认是 'ij'，对应 D(z), H(y), W(x)
        grid_z, grid_y, grid_x = torch.meshgrid(lin_z, lin_y, lin_x, indexing='ij')

        # grid_sample 极度反人类的规则：最后维度的坐标必须是 (x, y, z) 即 (W, H, D)
        identity_grid = torch.stack([grid_x, grid_y, grid_z], dim=-1) # [D, H, W, 3]
        identity_grid = identity_grid.unsqueeze(0).expand(B, -1, -1, -1, -1)

        # 形变场相加 (将 disp_dense 的通道维移到最后)
        sample_grid = identity_grid + disp_dense.permute(0, 2, 3, 4, 1)

        # 🔴 关键防御：使用 border 模式。如果形变把图像扯到了边缘外，用边缘像素延伸，绝不允许出现纯黑的背景断层
        warped_x = F.grid_sample(x, sample_grid, mode='bilinear', padding_mode='border', align_corners=True)

        return warped_x


class PCARespiratoryDeformation(nn.Module):
    def __init__(self, grid_size=4, amp_xyz=(0.01, 0.04, 0.15)):
        super().__init__()
        self.grid_size = grid_size
        self.amp_x, self.amp_y, self.amp_z = amp_xyz

    def forward(self, x, mode='bilinear', fixed_phase=None):
        B, C, D, H, W = x.shape
        device = x.device

        # 1. 呼吸相位控制
        if fixed_phase is not None:
            # Eval 阶段：强制指定处于呼吸的哪个阶段 (例如: 0, pi/2, pi 等)
            phase = torch.full((B, 1, 1, 1, 1), fixed_phase, device=device, dtype=torch.float32)
        else:
            # Train 阶段：随机相位，覆盖整个呼吸周期
            phase = torch.rand(B, 1, 1, 1, 1, device=device) * 2 * 3.1415926

        # 2. 极低频本底噪声 (继续受 RNG 种子控制，保证微观可重复性)
        noise_z = torch.randn(B, 1, self.grid_size, self.grid_size, self.grid_size, device=device) * 0.2
        noise_y = torch.randn(B, 1, self.grid_size, self.grid_size, self.grid_size, device=device) * 0.2
        noise_x = torch.randn(B, 1, self.grid_size, self.grid_size, self.grid_size, device=device) * 0.3

        # 3. 构造伪 PCA 运动学方程
        flow_z = (torch.sin(phase) + noise_z) * self.amp_z
        flow_y = (torch.sin(phase + 3.1415926/4) + noise_y) * self.amp_y
        flow_x = noise_x * self.amp_x

        flow_coarse = torch.cat([flow_z, flow_y, flow_x], dim=1)

        # 4. 上采样与网格生成保持不变
        flow = F.interpolate(flow_coarse, size=(D, H, W), mode='trilinear', align_corners=True)
        flow = flow.permute(0, 2, 3, 4, 1)

        d = torch.linspace(-1, 1, D, device=device)
        h = torch.linspace(-1, 1, H, device=device)
        w = torch.linspace(-1, 1, W, device=device)
        grid_d, grid_h, grid_w = torch.meshgrid(d, h, w, indexing='ij')

        base_grid = torch.stack([grid_w, grid_h, grid_d], dim=-1).unsqueeze(0)
        final_grid = base_grid + flow

        deformed_x = F.grid_sample(x, final_grid, mode=mode, padding_mode='reflection', align_corners=True)

        return deformed_x

class HybridRespiratoryDeformation(nn.Module):
    def __init__(self, grid_size=6, global_z_amp=0.15, local_sigma=(0.01, 0.02, 0.03)):
        super().__init__()
        self.grid_size = grid_size
        self.global_z_amp = global_z_amp
        self.local_sigma = local_sigma

    def forward(self, x):
        B, C, D, H, W = x.shape
        device = x.device

        # 1. 重新映射平移场顺序
        # 根据 grid_sample 规则：Channel 0->W, 1->H, 2->D
        # 对应你的数据维度：Channel 0->Patient Z, 1->Patient Y, 2->Patient X
        z_shift = (torch.rand(B, 1, 1, 1, 1, device=device) * 2 - 1) * self.global_z_amp
        y_shift = (torch.rand(B, 1, 1, 1, 1, device=device) * 2 - 1) * (self.global_z_amp * 0.2)
        x_shift = (torch.rand(B, 1, 1, 1, 1, device=device) * 2 - 1) * (self.global_z_amp * 0.1)

        # 🟢 关键修正：将 z_shift 放至第 1 位
        global_flow = torch.cat([
            z_shift.expand(-1, 1, D, H, W), # 控制 W 轴 (Patient Z)
            y_shift.expand(-1, 1, D, H, W), # 控制 H 轴 (Patient Y)
            x_shift.expand(-1, 1, D, H, W)  # 控制 D 轴 (Patient X)
        ], dim=1)

        # 2. 弹性噪声也需要同步调整顺序
        # 假设 local_sigma 对应的是 (X, Y, Z)
        noise_x = torch.randn(B, 1, self.grid_size, self.grid_size, self.grid_size, device=device) * self.local_sigma[0]
        noise_y = torch.randn(B, 1, self.grid_size, self.grid_size, self.grid_size, device=device) * self.local_sigma[1]
        noise_z = torch.randn(B, 1, self.grid_size, self.grid_size, self.grid_size, device=device) * self.local_sigma[2]

        # 🟢 关键修正：与 global_flow 通道顺序对齐 (Z, Y, X)
        noise_coarse = torch.cat([noise_z, noise_y, noise_x], dim=1)
        local_flow = F.interpolate(noise_coarse, size=(D, H, W), mode='trilinear', align_corners=True)

        # 3. 场耦合与重采样
        flow = global_flow + local_flow
        flow = flow.permute(0, 2, 3, 4, 1) # [B, D, H, W, 3] -> (x, y, z) coords

        # 基础网格保持不变，因为 grid_w 对应 grid[..., 0] 依然控制 Width
        d = torch.linspace(-1, 1, D, device=device)
        h = torch.linspace(-1, 1, H, device=device)
        w = torch.linspace(-1, 1, W, device=device)
        grid_d, grid_h, grid_w = torch.meshgrid(d, h, w, indexing='ij')
        base_grid = torch.stack([grid_w, grid_h, grid_d], dim=-1).unsqueeze(0)

        final_grid = base_grid + flow
        deformed_x = F.grid_sample(x, final_grid, mode='bilinear', padding_mode='border', align_corners=True)

        return deformed_x

def strict_masked_eval(gt_roi, pred_roi, mask_roi):
    """
    最保守的掩码级指标计算
    gt_roi, pred_roi, mask_roi 尺寸必须完全一致
    """
    # 1. 确保 mask 是 0/1 的二值矩阵
    mask_bin = (mask_roi > 0.5).astype(np.float32)
    valid_pixels = np.sum(mask_bin)

    if valid_pixels == 0:
        return 0.0, 0.0

    # 2. Masked MSE & PSNR
    # 只计算 mask 内部像素的均方误差
    mse = np.sum(((gt_roi - pred_roi) * mask_bin) ** 2) / valid_pixels
    if mse == 0:
        p = 100.0 # 完美对齐
    else:
        p = 10 * np.log10(1.0 / mse) # 假设 data_range 为 1.0

    # 3. Masked SSIM
    # 设置 full=True，返回一张与图像等大的 SSIM 逐像素得分图
    _, ssim_map = structural_similarity(gt_roi, pred_roi, data_range=1.0, full=True)

    # 仅对 mask 内部的 SSIM 得分进行平均
    s = np.sum(ssim_map * mask_bin) / valid_pixels

    return p, s


def simple_eval_metric(gt, pred):
    # 计算 PSNR
    p = peak_signal_noise_ratio(gt, pred, data_range=1.0)
    # 计算 SSIM
    s = structural_similarity(gt, pred, data_range=1.0)
    return p, s


def compute_gradient(img):
    # 计算 x 方向和 y 方向的梯度
    grad_x = img[..., 1:, :] - img[..., :-1, :]
    grad_y = img[..., :, 1:] - img[..., :, :-1]
    return grad_x, grad_y