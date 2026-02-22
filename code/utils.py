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



def convert_cuda(item):
    for key in item.keys():
        if key not in ['name', 'dst_name']:
            item[key] = item[key].float().cuda(non_blocking=True)
    return item


def save_nifti(image, path):
    out = sitk.GetImageFromArray(image)
    sitk.WriteImage(out, path)


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
            target_vol = prior_deformer(prior_vol, mode='bilinear')

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

        input_dict = {'projs': projs, 'points': points_ts, 'proj_points': proj_ts, 'prior': prior_vol}
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