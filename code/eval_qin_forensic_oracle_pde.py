# eval_qin_forensic_oracle_pde.py

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import time
import logging
import numpy as np
import torch
import torch.nn.functional as F
import scipy.ndimage as ndimage
from tqdm import tqdm
import nibabel as nib

from monai.metrics import compute_hausdorff_distance
from torch.utils.data import Dataset
from utils import save_visualization_3view

# ==========================================
# 🟢 核心算法：SDF 转换与拉普拉斯扩散
# ==========================================
def mask_to_sdf_logits(binary_mask_np, scale=2.0):
    """
    将二值掩膜 (0和1) 转换为平滑的类 Logit 符号距离场 (SDF)
    - 内部距离越大，Logit 越高 (概率越趋近 1)
    - 外部距离越大，Logit 越低 (概率越趋近 0)
    - 边界处 Logit 接近 0 (概率约 0.5)
    """
    # 提取内外距离
    dist_out = ndimage.distance_transform_edt(binary_mask_np == 0)
    dist_in = ndimage.distance_transform_edt(binary_mask_np > 0)

    # 构建 SDF: 内部为负，外部为正
    sdf = dist_out - dist_in

    # 转换为 Logit 空间 (-SDF 使得内部为正 Logit)
    logits_np = -sdf * scale
    return logits_np

def dirichlet_harmonic_diffusion(prior_3d_logits, target_slices_2d, cx, cy, cz, num_iterations=100):
    """
    基于狄利克雷硬约束的 3D 拉普拉斯扩散 (全局无玻璃墙版)
    """
    target_logits_2d = torch.where(target_slices_2d > 0.5, 10.0, -10.0).float()

    is_boundary = torch.zeros_like(prior_3d_logits, dtype=torch.bool)
    is_boundary[:, :, :, :, cz] = True  # Axial
    is_boundary[:, :, :, cy, :] = True  # Coronal
    is_boundary[:, :, cx, :, :] = True  # Sagittal

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

    final_3d_logits = prior_3d_logits + delta_field
    final_3d_probs = torch.sigmoid(final_3d_logits)
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
    largest_component_idx = component_sizes.argmax()
    return (labeled_mask == largest_component_idx).astype(mask_3d.dtype)

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
# 🟢 QIN 数据集加载器
# ==========================================
class QinBlindTestDataset(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
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
# 🟢 主程序：古典物理重构流水线
# ==========================================
if __name__ == '__main__':
    gpu_id = 0
    data_root = '/root/autodl-tmp/Proj/data/qin_testset_npy'
    save_dir = './logs/qin_oracle_pde_baseline'

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'vis'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'vis_nii'), exist_ok=True)

    logger = setup_logger(os.path.join(save_dir, 'forensic_log.txt'))
    logger.info("🚀 [PURE CLASSICAL BASELINE] 启动: Oracle SDF Prior + Laplace Diffusion")
    logger.info("⚡ 模型状态: DIF-Net 已被物理拔除。全流程无深度学习参与。")

    val_dst = QinBlindTestDataset(data_root=data_root)

    metrics = {
        'purged_global': [], 'band_1_3': [], 'band_4_7': [], 'band_gt_7': [], 'glo_hd95': [],
        'oracle_purged': [], 'oracle_b1_3': [], 'oracle_b4_7': [], 'oracle_gt_7': [], 'oracle_hd95': [],
        't_sdf': [], 't_pde': []
    }

    for idx in tqdm(range(len(val_dst)), ncols=100, desc="Classical Autopsy"):
        item = val_dst[idx]
        case_name = item['name']
        prior_image, prior_mask = item['prior_image'], item['prior_mask']
        target_image, target_mask = item['target_image'], item['target_mask']

        # 获取质心并对齐 (模拟上一帧的 Oracle 平移)
        nz_prior = np.argwhere(prior_mask > 0)
        px, py, pz = nz_prior.mean(axis=0).astype(int) if len(nz_prior) > 0 else (64, 64, 64)
        nz_target = np.argwhere(target_mask > 0)
        gx, gy, gz = nz_target.mean(axis=0).astype(int) if len(nz_target) > 0 else (64, 64, 64)
        feed_cx, feed_cy, feed_cz = gx, gy, gz

        # 刚性平移 Prior Mask 得到 Oracle Aligned Mask
        oracle_shift_vec = (gx - px, gy - py, gz - pz)
        oracle_aligned_prior_mask = ndimage.shift(prior_mask, oracle_shift_vec, order=0)

        # 将目标靶区也中心化，以便进行 PDE
        shift_t2c = (64 - feed_cx, 64 - feed_cy, 64 - feed_cz)
        centered_target_image = ndimage.shift(target_image, shift_t2c, order=1)
        centered_target_mask = ndimage.shift(target_mask, shift_t2c, order=0)

        slice_idx = 64

        # -----------------------------------------------------
        # ⚡ 步骤 1: 提取纯物理的 SDF Logit 场 (替代 DIF-Net)
        # -----------------------------------------------------
        t0 = time.time()
        # 注意我们要对齐到目标中心，所以用 shift 后的 oracle mask
        centered_oracle_mask = ndimage.shift(oracle_aligned_prior_mask, shift_t2c, order=0)

        # 计算 SDF 并转为 Logit 空间
        oracle_logits_np = mask_to_sdf_logits(centered_oracle_mask, scale=2.0)

        # 送入 GPU
        prior_3d_logits = torch.from_numpy(oracle_logits_np).view(1, 1, 128, 128, 128).float().cuda()
        t_sdf = time.time() - t0

        # -----------------------------------------------------
        # ⚡ 步骤 2: 提取 3 张 GT 2D 切片作为狄利克雷边界
        # -----------------------------------------------------
        target_slices_2d = torch.from_numpy(np.stack([
            centered_target_mask[:, :, slice_idx],
            centered_target_mask[:, slice_idx, :],
            centered_target_mask[slice_idx, :, :]
        ])).float().cuda()

        # -----------------------------------------------------
        # ⚡ 步骤 3: 瞬时拉普拉斯热传导
        # -----------------------------------------------------
        torch.cuda.synchronize()
        t1 = time.time()
        final_3d_probs = dirichlet_harmonic_diffusion(
            prior_3d_logits=prior_3d_logits,
            target_slices_2d=target_slices_2d,
            cx=slice_idx, cy=slice_idx, cz=slice_idx,
            num_iterations=100
        )
        pred_mask_centered = (final_3d_probs.cpu().numpy().squeeze() > 0.5).astype(np.uint8)
        torch.cuda.synchronize()
        t_pde = time.time() - t1

        # 还原坐标并清理拓扑
        pred_mask_np = ndimage.shift(pred_mask_centered, (feed_cx - 64, feed_cy - 64, feed_cz - 64), order=0)
        pred_mask_np = keep_largest_connected_component_3d(pred_mask_np)

        # 算分 (分段)
        M_anchor = np.zeros((128, 128, 128), dtype=bool)
        M_anchor[feed_cx, :, :] = True; M_anchor[:, feed_cy, :] = True; M_anchor[:, :, feed_cz] = True
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

        metrics['t_sdf'].append(t_sdf)
        metrics['t_pde'].append(t_pde)

        logger.info(f"[{case_name}] Purged Dice(Oracle/Pred): {results_o_purged:.3f}/{results_p_purged:.3f} | Glo HD95: {results_o_hd95:.2f}/{results_p_hd95:.2f}mm")

        # 导出定性分析
        vis_save_path = os.path.join(save_dir, 'vis', f"{case_name}_pred_{results_p_purged:.3f}.png")
        save_visualization_3view(
            img_np=target_image,
            prior_mask=prior_mask,
            aligned_prior=oracle_aligned_prior_mask,
            gt_mask=target_mask,
            pred_mask=pred_mask_np,
            save_path=vis_save_path,
            case_name=case_name,
            epoch="ORACLE_PDE"
        )

        nii_dir = os.path.join(save_dir, 'vis_nii')
        save_nifti(target_image, os.path.join(nii_dir, f"{case_name}_TARGET_IMG.nii.gz"), is_mask=False)
        save_nifti(target_mask, os.path.join(nii_dir, f"{case_name}_GT.nii.gz"))
        save_nifti(oracle_aligned_prior_mask, os.path.join(nii_dir, f"{case_name}_ORACLE_PRIOR.nii.gz"))
        save_nifti(pred_mask_np, os.path.join(nii_dir, f"{case_name}_PRED_ORACLE_PDE.nii.gz"))

    logger.info("\n" + "="*80)
    logger.info("🔥 QIN FORENSIC AUTOPSY SUMMARY: PURE ORACLE + PDE (NO AI) 🔥")
    logger.info("="*80)
    logger.info(f"[Macroscopic] Isolated Global Dice : Oracle {np.nanmean(metrics['oracle_purged']):.4f} -> Pred {np.nanmean(metrics['purged_global']):.4f}")
    logger.info(f"[Macroscopic] Global HD95 (mm)     : Oracle {np.nanmean(metrics['oracle_hd95']):.2f} -> Pred {np.nanmean(metrics['glo_hd95']):.2f}")
    logger.info("-" * 80)
    logger.info(f"[Distance-Stratified] Band 1-3 v.  : Oracle {np.nanmean(metrics['oracle_b1_3']):.4f} -> Pred {np.nanmean(metrics['band_1_3']):.4f}")
    logger.info(f"[Distance-Stratified] Band 4-7 v.  : Oracle {np.nanmean(metrics['oracle_b4_7']):.4f} -> Pred {np.nanmean(metrics['band_4_7']):.4f}")
    logger.info(f"[Distance-Stratified] Band >7 v.   : Oracle {np.nanmean(metrics['oracle_gt_7']):.4f} -> Pred {np.nanmean(metrics['band_gt_7']):.4f}")
    logger.info("-" * 80)
    logger.info(f"[Temporal Profiling] T_SDF (CPU)   : {np.mean(metrics['t_sdf']):.3f} s/vol")
    logger.info(f"[Temporal Profiling] T_PDE (GPU)   : {np.mean(metrics['t_pde']):.3f} s/vol")
    logger.info(f"[Temporal Profiling] Total Latency : {(np.mean(metrics['t_sdf']) + np.mean(metrics['t_pde'])):.3f} s/vol")
    logger.info("="*80)