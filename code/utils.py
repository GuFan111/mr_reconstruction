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

    # --- 严格按照 dataset.py 的投影定义提取 ---
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

    # 1. Get non-cubic resolution
    res_x, res_y, res_z = dataset.out_res
    idx_x, idx_y, idx_z = res_x // 2, res_y // 2, res_z // 2

    # Normalize coordinate fixed values
    fix_x, fix_y, fix_z = idx_x/(res_x-1), idx_y/(res_y-1), idx_z/(res_z-1)

    # Generate sampling grids (Must strictly match training logic)
    ax_x, ax_y, ax_z = np.linspace(0, 1, res_x), np.linspace(0, 1, res_y), np.linspace(0, 1, res_z)
    u_ax, v_ax = np.meshgrid(ax_x, ax_y, indexing='ij')
    pts_ax = np.stack([u_ax, v_ax, np.ones_like(u_ax)*fix_z], axis=-1).reshape(-1, 3) # XY
    u_co, v_co = np.meshgrid(ax_x, ax_z, indexing='ij')
    pts_co = np.stack([u_co, np.ones_like(u_co)*fix_y, v_co], axis=-1).reshape(-1, 3) # XZ
    u_sa, v_sa = np.meshgrid(ax_y, ax_z, indexing='ij')
    pts_sa = np.stack([np.ones_like(u_sa)*fix_x, u_sa, v_sa], axis=-1).reshape(-1, 3) # YZ

    n_list = [len(pts_ax), len(pts_co), len(pts_sa)]
    all_points = np.concatenate([pts_ax, pts_co, pts_sa], axis=0).astype(np.float32)

    # 2. Prepare data
    data_item = dataset[0]
    name = data_item['name']
    vol_np = data_item['image'] # [1, X, Y, Z]
    vol_cuda = torch.from_numpy(vol_np).unsqueeze(0).to(device)

    with torch.no_grad():
        noisy_vol = simulator(vol_cuda) if simulator else vol_cuda
        projs = gpu_slice_volume(noisy_vol)
        prior_vol = prior_deformer(vol_cuda) if prior_deformer else vol_cuda
        prior_projs = gpu_slice_volume(prior_vol)

        # 3. Inference
    points_ts = torch.from_numpy((all_points - 0.5) * 2).unsqueeze(0).to(device)
    proj_ts = torch.stack([points_ts[..., [0, 1]], points_ts[..., [0, 2]], points_ts[..., [1, 2]]], dim=1)

    with torch.no_grad():
        input_dict = {'projs': projs, 'points': points_ts, 'proj_points': proj_ts, 'prior': prior_vol}
        preds = model(input_dict, is_eval=True, eval_npoint=50000)

        # Extract prediction results
    preds_raw = preds[0, 0].cpu().numpy()
    imgs_pred = [
        preds_raw[:n_list[0]].reshape(res_x, res_y),
        preds_raw[n_list[0] : n_list[0]+n_list[1]].reshape(res_x, res_z),
        preds_raw[n_list[0]+n_list[1] :].reshape(res_y, res_z)
    ]

    # 4. Extract GT and restore Input/Prior scale
    vol = vol_np[0]
    gt_slices = [vol[:, :, idx_z], vol[:, idx_y, :], vol[idx_x, :, :]]

    raw_input_np = projs[0, :, 0].cpu().numpy() # [3, res_max, res_max]
    raw_prior_np = prior_projs[0, :, 0].cpu().numpy()

    imgs_input, imgs_prior = [], []
    for i in range(3):
        h, w = gt_slices[i].shape
        # Resize the square inputs back to the GT's actual height and width
        imgs_input.append(cv2.resize(raw_input_np[i], (w, h), interpolation=cv2.INTER_LINEAR))
        imgs_prior.append(cv2.resize(raw_prior_np[i], (w, h), interpolation=cv2.INTER_LINEAR))

    # 5. Drawing
    fig, axes = plt.subplots(3, 5, figsize=(15, 10))
    titles = ['Axial', 'Coronal', 'Sagittal']
    col_titles = ["GT", "Input", "Prior", "Recon", "Diff (x5)"]

    for i in range(3):
        # We must use 'auto' aspect to prevent Matplotlib from forcing square pixels
        im_list = [
            gt_slices[i],
            imgs_input[i],
            imgs_prior[i],
            np.clip(imgs_pred[i], 0, 1),
            np.abs(gt_slices[i] - np.clip(imgs_pred[i], 0, 1))
        ]

        for j in range(5):
            data_to_show = im_list[j].T
            # Force vmin/vmax for Diff differently
            v_max = 1.0 if j < 4 else 0.2
            cmap = 'gray' if j < 4 else 'inferno'

            # CRITICAL: aspect='auto' ensures the image fills the subplot regardless of ratio
            axes[i, j].imshow(data_to_show, cmap=cmap, vmin=0, vmax=v_max, origin='lower', aspect='auto')

            if i == 0:
                axes[i, j].set_title(col_titles[j])
            axes[i, j].axis('off')

        axes[i, 0].set_ylabel(titles[i])
        # Re-enable the label since axis('off') hides it
        axes[i, 0].text(-0.2, 0.5, titles[i], transform=axes[i, 0].transAxes,
                        rotation=90, va='center', ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'vis_ep{epoch}_{name}.png')
    plt.savefig(save_path, dpi=150)
    plt.close()


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
    def __init__(self, grid_size=8, sigma=0.05):
        super().__init__()
        self.grid_size = grid_size # 形变频率
        self.sigma = sigma         # 形变幅度 

    def forward(self, x):
        # x: [B, C, D, H, W]
        B, C, D, H, W = x.shape
        device = x.device

        # 1. 生成低分辨率的随机位移场
        # shape: [B, 3, grid, grid, grid]
        flow_coarse = torch.randn(B, 3, self.grid_size, self.grid_size, self.grid_size, device=device) * self.sigma

        # 2. 上采样到全分辨率
        # grid_sample 需要的 flow 必须与 input 尺寸一致
        flow = F.interpolate(flow_coarse, size=(D, H, W), mode='trilinear', align_corners=True)
        # flow shape: [B, 3, D, H, W] -> permute to [B, D, H, W, 3] for grid_sample
        flow = flow.permute(0, 2, 3, 4, 1)

        # 3. 生成基础网格
        # grid: [1, D, H, W, 3]
        d = torch.linspace(-1, 1, D, device=device)
        h = torch.linspace(-1, 1, H, device=device)
        w = torch.linspace(-1, 1, W, device=device)
        grid_d, grid_h, grid_w = torch.meshgrid(d, h, w, indexing='ij')
        base_grid = torch.stack([grid_w, grid_h, grid_d], dim=-1).unsqueeze(0) # 注意: grid_sample 顺序是 x,y,z (W,H,D)

        # 4. 叠加形变
        # base_grid + flow
        final_grid = base_grid + flow

        # 5. 采样得到变形后的 Volume
        deformed_x = F.grid_sample(x, final_grid, mode='bilinear', padding_mode='reflection', align_corners=True)

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

