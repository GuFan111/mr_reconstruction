# utils.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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

def gpu_slice_volume(volume, slice_idx=None):  # 切片
    B, C, D, H, W = volume.shape
    
    if slice_idx is None:
        idx_d = D // 2
        idx_h = H // 2
        idx_w = W // 2
    else:
        idx_d = idx_h = idx_w = slice_idx

    slice_ax = volume[..., idx_w] 
    slice_cor = volume[..., idx_h, :]
    slice_sag = volume[:, :, idx_d, :, :]

    slice_ax  = slice_ax.unsqueeze(2)
    slice_cor = slice_cor.unsqueeze(2)
    slice_sag = slice_sag.unsqueeze(2)

    projs = torch.cat([slice_ax, slice_cor, slice_sag], dim=1)
    
    return projs


def save_visualization_3view(model, dataset, epoch, device='cuda', save_dir='vis_results', simulator=None, prior_deformer=None):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    res = dataset.out_res

    target_idx = res // 2  # 64
    # 转换为 [0, 1] 范围的坐标: idx / (res - 1)
    fix_val = target_idx / (res - 1) 

    # 生成坐标网格
    axis = np.linspace(0, 1, res)
    u, v = np.meshgrid(axis, axis, indexing='ij')
    
    # 使用修正后的固定值
    fixed = np.ones_like(u) * fix_val 
    
    # 组合坐标 (x, y, z)
    pts_ax = np.stack([u, v, fixed], axis=-1).reshape(-1, 3)  # Axial: z固定的xy平面
    pts_co = np.stack([u, fixed, v], axis=-1).reshape(-1, 3)  # Coronal: y固定的xz平面
    pts_sa = np.stack([fixed, u, v], axis=-1).reshape(-1, 3)  # Sagittal: x固定的yz平面
    
    all_points = np.concatenate([pts_ax, pts_co, pts_sa], axis=0).astype(np.float32)

    # 1. 准备数据
    data_item = dataset[0] 
    name = data_item['name']
    
    # 获取干净的完整体积 [1, X, Y, Z]
    vol_np = data_item['image'] 
    vol_cuda = torch.from_numpy(vol_np).unsqueeze(0).to(device) # [1, 1, X, Y, Z]

    with torch.no_grad():
        # A. 处理 Projections (Input 1)
        if simulator is not None:
            noisy_vol = simulator(vol_cuda)
        else:
            noisy_vol = vol_cuda
        projs = gpu_slice_volume(noisy_vol) # [1, 3, 1, H, W]

        # B. 处理 Prior (Input 2)
        if prior_deformer is not None:
            # 使用传入的形变器生成 Imperfect Prior
            prior_vol = prior_deformer(vol_cuda)
        else:
            # 如果没传形变器，就直接用 GT 作为 Prior (Ideal Prior)
            prior_vol = vol_cuda

    # 准备推理用的坐标
    points_ts = torch.from_numpy((all_points - 0.5) * 2).unsqueeze(0).to(device)
    proj_ts = []
    
    for i in range(3):
        # 简单正交投影: 
        # View 0 (Axial, XY): x,y
        # View 1 (Coronal, XZ): x,z
        # View 2 (Sagittal, YZ): y,z (注意 dataset 定义)
        # 对应 dataset.py 的 OrthogonalGeometry.project
        # 你的 dataset 定义: 0->(0,1), 1->(0,2), 2->(1,2)
        if i == 0: uv = points_ts[..., [0, 1]]
        elif i == 1: uv = points_ts[..., [0, 2]]
        elif i == 2: uv = points_ts[..., [1, 2]]
        proj_ts.append(uv)
    proj_ts = torch.stack(proj_ts, dim=1) # [1, 3, N, 2]

    # 2. 推理
    with torch.no_grad():
        input_dict = {
            'projs': projs, 
            'points': points_ts, 
            'proj_points': proj_ts,
            'prior': prior_vol # 传入 Prior
        }
        preds = model(input_dict, is_eval=True, eval_npoint=50000) 

    imgs_pred = preds[0, 0].cpu().numpy().reshape(3, res, res) 
    
    # 准备显示用的 Input (取切片)
    # projs 是 [1, 3, 1, H, W] -> [3, H, W]
    imgs_input = projs[0, :, 0].cpu().numpy()
    
    # 准备显示用的 Prior (取切片)
    # prior_vol: [1, 1, D, H, W]
    # 使用同样的切片逻辑
    prior_slices = gpu_slice_volume(prior_vol)[0, :, 0].cpu().numpy()

    # 3. 获取 GT 切片
    vol = vol_np[0] # [X, Y, Z]
    cx, cy, cz = np.array(vol.shape) // 2
    # 注意：这里的切片顺序要和 projs 对应
    # projs 顺序是 [Axial(D), Coronal(H), Sagittal(W)] (我们之前修正过的)
    # Axial: 切 D(z) -> XY平面 -> vol[:, :, cz]
    # Coronal: 切 H(y) -> XZ平面 -> vol[:, cy, :]
    # Sagittal: 切 W(x) -> YZ平面 -> vol[cx, :, :]
    gt_slices = [
        vol[:, :, cz],    # Axial
        vol[:, cy, :],    # Coronal
        vol[cx, :, :]     # Sagittal
    ]

    # 4. 绘图
    fig, axes = plt.subplots(3, 5, figsize=(15, 10)) 
    titles = ['Axial', 'Coronal', 'Sagittal']
    col_titles = ["GT", "Input (Slices)", "Prior (Deformed)", "Recon", "Diff (x5)"]

    for i in range(3):
        # GT
        axes[i, 0].imshow(gt_slices[i].T, cmap='gray', vmin=0, vmax=1, origin='lower')
        if i==0: axes[i, 0].set_title(col_titles[0])
        axes[i, 0].set_ylabel(titles[i])
        
        # Input (Slices)
        axes[i, 1].imshow(imgs_input[i].T, cmap='gray', vmin=0, vmax=1, origin='lower')
        if i==0: axes[i, 1].set_title(col_titles[1])
        
        # Prior (Deformed) - 新增
        axes[i, 2].imshow(prior_slices[i].T, cmap='gray', vmin=0, vmax=1, origin='lower')
        if i==0: axes[i, 2].set_title(col_titles[2])

        # Recon
        p = np.clip(imgs_pred[i], 0, 1)
        axes[i, 3].imshow(p.T, cmap='gray', vmin=0, vmax=1, origin='lower')
        if i==0: axes[i, 3].set_title(col_titles[3])

        # Diff
        diff = np.abs(gt_slices[i] - p)
        axes[i, 4].imshow(diff.T, cmap='inferno', vmin=0, vmax=0.2, origin='lower')
        if i==0: axes[i, 4].set_title(col_titles[4])

        for ax in axes[i]: ax.axis('off')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'vis_ep{epoch}_{name}.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Visualization saved: {save_path}")


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


    