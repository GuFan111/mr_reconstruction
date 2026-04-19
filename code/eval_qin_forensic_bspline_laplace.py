# eval_qin_forensic_bspline_laplace.py

import os
# 限制底层数学库的线程数，防止多进程评估时的 CPU 资源抢占
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import time
import numpy as np
import torch
import torch.nn.functional as F
import SimpleITK as sitk
from torch.utils.data import Dataset
from monai.metrics import compute_hausdorff_distance
import scipy.ndimage as ndimage
from tqdm import tqdm
import nibabel as nib
import logging

from utils import save_visualization_3view

# ==========================================
# 🟢 物理桥梁与扩散算子
# ==========================================
def mask_to_sdf_logits(binary_mask_np, scale=2.0):
    """
    桥梁：将 B-spline 生成的二值掩膜转换为连续的类 Logit 符号距离场 (SDF)
    供 Laplace 扩散使用。
    """
    dist_out = ndimage.distance_transform_edt(binary_mask_np == 0)
    dist_in = ndimage.distance_transform_edt(binary_mask_np > 0)
    sdf = dist_out - dist_in 
    logits_np = -sdf * scale
    return logits_np

def dirichlet_harmonic_diffusion(prior_3d_logits, target_slices_2d, cx, cy, cz, num_iterations=100):
    """3D 拉普拉斯扩散算子"""
    target_logits_2d = torch.where(target_slices_2d > 0.5, 10.0, -10.0).float()
    
    is_boundary = torch.zeros_like(prior_3d_logits, dtype=torch.bool)
    is_boundary[:, :, :, :, cz] = True  
    is_boundary[:, :, :, cy, :] = True  
    is_boundary[:, :, cx, :, :] = True  
    
    known_residual = torch.zeros_like(prior_3d_logits)
    known_residual[:, :, :, :, cz] = target_logits_2d[0] - prior_3d_logits[:, :, :, :, cz]
    known_residual[:, :, :, cy, :] = target_logits_2d[1] - prior_3d_logits[:, :, :, cy, :]
    known_residual[:, :, cx, :, :] = target_logits_2d[2] - prior_3d_logits[:, :, cx, :, :]
    
    delta_field = torch.zeros_like(prior_3d_logits)
    delta_field[is_boundary] = known_residual[is_boundary]
    
    kernel = torch.ones(1, 1, 3, 3, 3, device=prior_3d_logits.device)
    kernel[0, 0, 1, 1, 1] = 0
    kernel = kernel / kernel.sum()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            new_delta = F.conv3d(delta_field, kernel, padding=1)
            delta_field = torch.where(is_boundary, known_residual, new_delta)
            
    final_3d_probs = torch.sigmoid(prior_3d_logits + delta_field)
    return final_3d_probs

# ==========================================
# 🟢 法医级算分与日志引擎
# ==========================================
def setup_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    return logger

def save_nifti(array_np, save_path, is_mask=True):
    dtype = np.uint8 if is_mask else np.float32
    nii_img = nib.Nifti1Image(array_np.astype(dtype), np.eye(4))
    nib.save(nii_img, save_path)

def keep_largest_connected_component_3d(mask_3d):
    mask_bool = mask_3d > 0.5
    if not mask_bool.any(): return mask_3d 
    labeled_mask, num_features = ndimage.label(mask_bool)
    if num_features <= 1: return mask_3d
    component_sizes = np.bincount(labeled_mask.ravel())
    component_sizes[0] = 0 
    return (labeled_mask == component_sizes.argmax()).astype(mask_3d.dtype)

def smooth_binary_mask(mask_np, sigma=1.0, threshold=0.5):
    mask_float = mask_np.astype(np.float32)
    smoothed = ndimage.gaussian_filter(mask_float, sigma=sigma)
    return (smoothed > threshold).astype(np.uint8)

def compute_band_dice(pred_mask, gt_mask, band_mask):
    p = pred_mask[band_mask].astype(bool)
    g = gt_mask[band_mask].astype(bool)
    union = p.sum() + g.sum()
    if union == 0: return np.nan
    return 2.0 * np.logical_and(p, g).sum() / union

def safe_global_hd95(pred_tensor, gt_tensor):
    if pred_tensor.sum() == 0 or gt_tensor.sum() == 0: return 99.0
    try:
        return compute_hausdorff_distance(pred_tensor, gt_tensor, include_background=False, percentile=95).item()
    except:
        return 99.0

# ==========================================
# 🟢 QIN 盲测数据集读取
# ==========================================
class QinBlindTestDataset(Dataset):
    def __init__(self, data_root):
        self.patient_dirs = sorted([os.path.join(data_root, d) for d in os.listdir(data_root) 
                                    if os.path.isdir(os.path.join(data_root, d)) and not d.startswith('.')])
    def __len__(self): return len(self.patient_dirs)
    def __getitem__(self, idx):
        p_dir = self.patient_dirs[idx]
        return {
            'name': os.path.basename(p_dir),
            'prior_image': np.load(os.path.join(p_dir, 'prior_image.npy')),
            'prior_mask': np.load(os.path.join(p_dir, 'prior_mask.npy')),
            'target_image': np.load(os.path.join(p_dir, 'target_image.npy')),
            'target_mask': np.load(os.path.join(p_dir, 'target_mask.npy'))
        }

# ==========================================
# 🟢 B-spline 稀疏配准核心引擎
# ==========================================
def bspline_sparse_registration(fixed_img_np, moving_img_np, coords, grid_physical_spacing=30.0):
    fixed_image = sitk.GetImageFromArray(fixed_img_np.astype(np.float32))
    moving_image = sitk.GetImageFromArray(moving_img_np.astype(np.float32))

    cx, cy, cz = int(coords[0]), int(coords[1]), int(coords[2])
    mask_np = np.zeros_like(fixed_img_np, dtype=np.uint8)
    mask_np[cx, :, :] = 1; mask_np[:, cy, :] = 1; mask_np[:, :, cz] = 1
    metric_mask = sitk.GetImageFromArray(mask_np)
    metric_mask.CopyInformation(fixed_image)

    transformDomainMeshSize = [max(1, int(dim / grid_physical_spacing)) for dim in fixed_image.GetSize()]
    initial_transform = sitk.BSplineTransformInitializer(fixed_image, transformDomainMeshSize)

    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMeanSquares() 
    registration_method.SetMetricFixedMask(metric_mask) 
    registration_method.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=100, maximumNumberOfCorrections=5)
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    registration_method.SetInterpolator(sitk.sitkLinear)
    return registration_method.Execute(fixed_image, moving_image)

# ==========================================
# 🟢 测试配置区域
# ==========================================
class QinBsplineConfig:
    data_root = r'/root/autodl-tmp/Proj/data/qin_testset_npy'
    out_res = (128, 128, 128)
    use_laplace = True  # 🔴 恶魔开关：是否启动 B-spline + Laplace 融合体
    
    tag = "bspline_laplace" if use_laplace else "bspline"
    save_dir = f'./logs/qin_forensic__411_finetuning_{tag}'
    save_vis = True
    save_nii = True
    gpu_id = 0

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = str(QinBsplineConfig.gpu_id)
    os.makedirs(QinBsplineConfig.save_dir, exist_ok=True)
    if QinBsplineConfig.save_vis: os.makedirs(os.path.join(QinBsplineConfig.save_dir, 'vis'), exist_ok=True)
    if QinBsplineConfig.save_nii: os.makedirs(os.path.join(QinBsplineConfig.save_dir, 'vis_nii'), exist_ok=True)

    logger = setup_logger(os.path.join(QinBsplineConfig.save_dir, 'forensic_bspline_log.txt'))
    
    mode_name = "B-SPLINE + LAPLACE" if QinBsplineConfig.use_laplace else "B-SPLINE ONLY"
    logger.info(f"\n🚀 开始执行 QIN 盲测 ({mode_name})...")
    
    val_dst = QinBlindTestDataset(data_root=QinBsplineConfig.data_root)

    metrics = {
        'purged_global': [], 'band_1_3': [], 'band_4_7': [], 'band_gt_7': [], 'glo_hd95': [],
        'oracle_purged': [], 'oracle_b1_3': [], 'oracle_b4_7': [], 'oracle_gt_7': [], 'oracle_hd95': [],
        't_bspline': [], 't_laplace': []
    }

    for idx in tqdm(range(len(val_dst)), ncols=100, desc="Forensic Autopsy"):
        item = val_dst[idx]
        name = item['name']
        prior_image, prior_mask = item['prior_image'], item['prior_mask']
        target_image, target_mask = item['target_image'], item['target_mask']

        nz_prior = np.argwhere(prior_mask > 0)
        px, py, pz = nz_prior.mean(axis=0).astype(int) if len(nz_prior) > 0 else (64, 64, 64)
        nz_target = np.argwhere(target_mask > 0)
        gx, gy, gz = nz_target.mean(axis=0).astype(int) if len(nz_target) > 0 else (64, 64, 64)

        oracle_shift_vec = (gx - px, gy - py, gz - pz)
        oracle_aligned_prior_mask = ndimage.shift(prior_mask, oracle_shift_vec, order=0)
        oracle_aligned_prior_image = ndimage.shift(prior_image, oracle_shift_vec, order=1)

        # ==========================================
        # 1. 运行 B-spline 稀疏优化
        # ==========================================
        t_start = time.time()
        try:
            final_transform = bspline_sparse_registration(target_image, oracle_aligned_prior_image, coords=(gx, gy, gz))
        except Exception as e:
            logger.error(f"[{name}] B-spline Optimization Failed: {e}")
            continue

        prior_mask_sitk = sitk.GetImageFromArray(oracle_aligned_prior_mask.astype(np.float32))
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(prior_mask_sitk)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(final_transform)
        
        pred_mask_sitk = resampler.Execute(prior_mask_sitk)
        pred_mask_np = sitk.GetArrayFromImage(pred_mask_sitk)
        pred_mask_np = (pred_mask_np > 0.5).astype(np.float32)
        metrics['t_bspline'].append(time.time() - t_start)

        # ==========================================
        # 2. 拉普拉斯热传导缝合 (The Frankenstein Module)
        # ==========================================
        t_laplace = 0.0
        if QinBsplineConfig.use_laplace:
            torch.cuda.synchronize()
            t_l_start = time.time()
            
            bspline_logits = mask_to_sdf_logits(pred_mask_np, scale=2.0)
            prior_3d_logits = torch.from_numpy(bspline_logits).view(1, 1, 128, 128, 128).float().cuda()
            
            target_slices_2d = torch.from_numpy(np.stack([
                target_mask[:, :, gz], 
                target_mask[:, gy, :], 
                target_mask[gx, :, :]
            ])).float().cuda()

            final_3d_probs = dirichlet_harmonic_diffusion(
                prior_3d_logits=prior_3d_logits, target_slices_2d=target_slices_2d,
                cx=gx, cy=gy, cz=gz, num_iterations=50
            )
            pred_mask_np = (final_3d_probs.cpu().numpy().squeeze() > 0.5).astype(np.float32)
            
            torch.cuda.synchronize()
            t_laplace = time.time() - t_l_start
        
        metrics['t_laplace'].append(t_laplace)
        pred_mask_np = keep_largest_connected_component_3d(pred_mask_np)

        # ==========================================
        # 3. 隔离算分与统计
        # ==========================================
        M_anchor = np.zeros((128, 128, 128), dtype=bool)
        M_anchor[gx, :, :] = True; M_anchor[:, gy, :] = True; M_anchor[:, :, gz] = True
        mask_purged = ~M_anchor
        dist_map = ndimage.distance_transform_edt(~M_anchor)
        mask_b1 = (dist_map > 0) & (dist_map <= 3)
        mask_b2 = (dist_map > 3) & (dist_map <= 7)
        mask_b3 = (dist_map > 7)

        gt_tensor = torch.from_numpy(smooth_binary_mask(target_mask)).view(1, 1, 128, 128, 128).float().cuda()
        pred_tensor = torch.from_numpy(smooth_binary_mask(pred_mask_np)).view(1, 1, 128, 128, 128).float().cuda()
        oracle_tensor = torch.from_numpy(smooth_binary_mask(oracle_aligned_prior_mask)).view(1, 1, 128, 128, 128).float().cuda()

        results_p_purged = compute_band_dice(pred_mask_np, target_mask, mask_purged)
        results_p_b1 = compute_band_dice(pred_mask_np, target_mask, mask_b1)
        results_p_b2 = compute_band_dice(pred_mask_np, target_mask, mask_b2)
        results_p_b3 = compute_band_dice(pred_mask_np, target_mask, mask_b3)
        results_p_hd95 = safe_global_hd95(pred_tensor, gt_tensor)
        
        metrics['purged_global'].append(results_p_purged)
        if not np.isnan(results_p_b1): metrics['band_1_3'].append(results_p_b1)
        if not np.isnan(results_p_b2): metrics['band_4_7'].append(results_p_b2)
        if not np.isnan(results_p_b3): metrics['band_gt_7'].append(results_p_b3)
        metrics['glo_hd95'].append(results_p_hd95)

        results_o_purged = compute_band_dice(oracle_aligned_prior_mask, target_mask, mask_purged)
        results_o_b1 = compute_band_dice(oracle_aligned_prior_mask, target_mask, mask_b1)
        results_o_b2 = compute_band_dice(oracle_aligned_prior_mask, target_mask, mask_b2)
        results_o_b3 = compute_band_dice(oracle_aligned_prior_mask, target_mask, mask_b3)
        results_o_hd95 = safe_global_hd95(oracle_tensor, gt_tensor)

        metrics['oracle_purged'].append(results_o_purged)
        if not np.isnan(results_o_b1): metrics['oracle_b1_3'].append(results_o_b1)
        if not np.isnan(results_o_b2): metrics['oracle_b4_7'].append(results_o_b2)
        if not np.isnan(results_o_b3): metrics['oracle_gt_7'].append(results_o_b3)
        metrics['oracle_hd95'].append(results_o_hd95)

        logger.info(f"  [{name}] Purged Dice (Oracle/Pred): {results_o_purged:.3f} / {results_p_purged:.3f} | Glo HD95: {results_p_hd95:.2f}mm")

        # ==========================================
        # 4. 导出可视化与 NIfTI
        # ==========================================
        if QinBsplineConfig.save_vis:
            vis_save_path = os.path.join(QinBsplineConfig.save_dir, 'vis', f"{name}_pred_{results_p_purged:.3f}.png")
            save_visualization_3view(
                img_np=target_image, prior_mask=prior_mask, aligned_prior=oracle_aligned_prior_mask,  
                gt_mask=target_mask, pred_mask=pred_mask_np, save_path=vis_save_path, 
                case_name=name, epoch=QinBsplineConfig.tag.upper()
            )

        if QinBsplineConfig.save_nii:
            nii_dir = os.path.join(QinBsplineConfig.save_dir, 'vis_nii')
            save_nifti(target_image, os.path.join(nii_dir, f"{name}_TARGET_IMG.nii.gz"), is_mask=False)
            save_nifti(target_mask, os.path.join(nii_dir, f"{name}_GT.nii.gz"))
            save_nifti(oracle_aligned_prior_mask, os.path.join(nii_dir, f"{name}_ORACLE_PRIOR.nii.gz"))
            save_nifti(pred_mask_np, os.path.join(nii_dir, f"{name}_PRED_{QinBsplineConfig.tag.upper()}.nii.gz"))

    # ==========================================
    # 5. 打印最终结果
    # ==========================================
    logger.info("\n" + "="*80)
    logger.info(f"🔥 FORENSIC AUTOPSY SUMMARY: {mode_name} 🔥")
    logger.info("="*80)
    logger.info(f"[Macroscopic] Isolated Global Dice : Oracle {np.nanmean(metrics['oracle_purged']):.4f} -> Pred {np.nanmean(metrics['purged_global']):.4f}")
    logger.info(f"[Macroscopic] Global HD95 (mm)     : Oracle {np.nanmean(metrics['oracle_hd95']):.2f} -> Pred {np.nanmean(metrics['glo_hd95']):.2f}")
    logger.info("-" * 80)
    logger.info(f"[Distance-Stratified] Band 1-3 v.  : Oracle {np.nanmean(metrics['oracle_b1_3']):.4f} -> Pred {np.nanmean(metrics['band_1_3']):.4f}")
    logger.info(f"[Distance-Stratified] Band 4-7 v.  : Oracle {np.nanmean(metrics['oracle_b4_7']):.4f} -> Pred {np.nanmean(metrics['band_4_7']):.4f}")
    logger.info(f"[Distance-Stratified] Band >7 v.   : Oracle {np.nanmean(metrics['oracle_gt_7']):.4f} -> Pred {np.nanmean(metrics['band_gt_7']):.4f}")
    logger.info("-" * 80)
    logger.info(f"[Temporal Profiling] T_bspline (CPU): {np.mean(metrics['t_bspline']):.2f} s/vol")
    if QinBsplineConfig.use_laplace:
        logger.info(f"[Temporal Profiling] T_laplace (GPU): {np.mean(metrics['t_laplace']):.3f} s/vol")
    logger.info("="*80)