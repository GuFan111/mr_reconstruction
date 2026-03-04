# utils.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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

def save_visualization_3view(img_np, prior_mask, aligned_prior, gt_mask, pred_mask, save_path, case_name, epoch):
    """
    针对 3D 形状补全任务的 3 视图切片轮廓可视化。
    融入了最新的学术逻辑：加入了质心对齐后的 Prior，用于直观对比弹性形变与平滑修复。
    """
    # 找到 Ground Truth 的质心，用于提取最具代表性的十字正交切片
    coords = np.argwhere(gt_mask > 0.5)
    if len(coords) > 0:
        cx, cy, cz = coords.mean(axis=0).astype(int)
    else:
        cx, cy, cz = [s // 2 for s in gt_mask.shape]

    # 提取三个正交切片
    slices = [
        (img_np[:, :, cz], prior_mask[:, :, cz], aligned_prior[:, :, cz], gt_mask[:, :, cz], pred_mask[:, :, cz], "Axial (XY)"),
        (img_np[:, cy, :], prior_mask[:, cy, :], aligned_prior[:, cy, :], gt_mask[:, cy, :], pred_mask[:, cy, :], "Coronal (XZ)"),
        (img_np[cx, :, :], prior_mask[cx, :, :], aligned_prior[cx, :, :], gt_mask[cx, :, :], pred_mask[cx, :, :], "Sagittal (YZ)")
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6)) # 稍微加宽以适应图例
    fig.suptitle(f"Case: {case_name} | Epoch: {epoch}", fontsize=16)

    for i, (img_slice, prior_slc, aligned_slc, gt_slc, pred_slc, title) in enumerate(slices):
        ax = axes[i]
        # 显示 MRI 灰度底图
        ax.imshow(img_slice, cmap='gray', origin='lower')

        # 绘制轮廓线 (防御性编程：确保切片内有前景像素才画等高线)
        if prior_slc.sum() > 0:
            ax.contour(prior_slc, levels=[0.5], colors='cyan', linestyles='dashed', linewidths=1.5, alpha=0.7)
        if aligned_slc.sum() > 0:
            ax.contour(aligned_slc, levels=[0.5], colors='lime', linestyles='dotted', linewidths=2.0, alpha=0.9)
        if gt_slc.sum() > 0:
            ax.contour(gt_slc, levels=[0.5], colors='red', linestyles='solid', linewidths=2.0)
        if pred_slc.sum() > 0:
            ax.contour(pred_slc, levels=[0.5], colors='yellow', linestyles='solid', linewidths=2.0)

        ax.set_title(title)
        ax.axis('off')

    # 添加自定义图例
    custom_lines = [
        Line2D([0], [0], color='cyan', lw=2, linestyle='dashed'),
        Line2D([0], [0], color='lime', lw=2, linestyle='dotted'),
        Line2D([0], [0], color='red', lw=2),
        Line2D([0], [0], color='yellow', lw=2)
    ]
    fig.legend(custom_lines,
               ['Original Prior (Init Pos)', 'Aligned Prior (Rigid Baseline)', 'Ground Truth (Target)', 'Prediction (Our Model)'],
               loc='lower center', ncol=4, fontsize=12)

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

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

class BladderBalloonDeformer(nn.Module):
    """
    肉眼可见的底部锚定气球模拟器
    基于严格的后向映射数学法则：产生明显的向外膨胀视觉效果。
    """
    def __init__(self, max_disp=0.25, radius=3.0):
        super().__init__()
        self.max_disp = max_disp
        self.radius = radius

    def forward(self, x, is_eval=False):
        B, C, D, H, W = x.shape
        device = x.device

        lin_Z = torch.linspace(-1, 1, D, device=device)
        lin_Y = torch.linspace(-1, 1, H, device=device)
        lin_X = torch.linspace(-1, 1, W, device=device)
        gZ, gY, gX = torch.meshgrid(lin_Z, lin_Y, lin_X, indexing='ij')

        gZ_5d = gZ.unsqueeze(0).unsqueeze(0)
        gY_5d = gY.unsqueeze(0).unsqueeze(0)
        gX_5d = gX.unsqueeze(0).unsqueeze(0)

        # 1. 设定形变中心（稍微偏向一侧，模拟从底部向上的不对称充盈）
        center_Z, center_Y, center_X = 0.3, 0.0, 0.0

        # 2. 计算空间每个点到中心的距离向量
        vec_Z = gZ_5d - center_Z
        vec_Y = gY_5d - center_Y
        vec_X = gX_5d - center_X

        dist_sq = vec_X**2 + vec_Y**2 + vec_Z**2

        # 3. 高斯平滑衰减，确保只在局部（膀胱及周边）发生形变
        weight = torch.exp(-dist_sq * self.radius)

        if is_eval:
            amp = torch.full((B, 1, 1, 1, 1), self.max_disp, device=device)
        else:
            amp = torch.rand(B, 1, 1, 1, 1, device=device) * self.max_disp + 0.05

        # 🟢 核心数学：后向映射中，要让视觉上膨胀，采样网格必须向中心收缩！
        # 所以位移方向必须等于 -vec
        disp_Z = -vec_Z * weight * amp
        disp_Y = -vec_Y * weight * amp
        disp_X = -vec_X * weight * amp

        disp_dense = torch.cat([disp_X, disp_Y, disp_Z], dim=1)

        identity_grid = torch.cat([gX_5d, gY_5d, gZ_5d], dim=1).expand(B, -1, -1, -1, -1)
        sample_grid = identity_grid + disp_dense
        sample_grid = sample_grid.permute(0, 2, 3, 4, 1)

        warped_x = F.grid_sample(x, sample_grid, mode='bilinear', padding_mode='border', align_corners=True)
        return warped_x

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