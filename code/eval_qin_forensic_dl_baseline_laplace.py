# eval_qin_forensic_dl_baseline_laplace.py

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import sys
import time
import copy
import logging
import numpy as np
import torch
import torch.nn.functional as F
import scipy.ndimage as ndimage
from skimage.measure import label, regionprops
from tqdm import tqdm
import nibabel as nib

from monai.metrics import compute_hausdorff_distance
from torch.utils.data import Dataset

from models.unet import UNet
from models.model import DIF_Net
from models.baseline_models import Baseline_3DUNet, Baseline_SwinUNETR
from dataset import OrthogonalGeometry
from utils import save_visualization_3view

# ==========================================
# 🟢 狄利克雷拉普拉斯插值算子
# ==========================================
def dirichlet_harmonic_diffusion(prior_3d_logits, target_slices_2d, cx, cy, cz, num_iterations=100):
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
# 🟢 法医级日志与 IO 引擎
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

# ==========================================
# 🟢 物理隔离与频带算分算子
# ==========================================
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

class QinBlindTestDataset(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        self.patient_dirs = sorted([os.path.join(data_root, d) for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d)) and not d.startswith('.')])
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
# 🟢 评测配置
# ==========================================
class QinForensicConfig:
    # 🔴 支持: 'difnet_laplace', 'difnet_no_tto', '3dunet', '3dunet_laplace', 'swin_unetr', 'swin_unetr_laplace'
    model_type = '3dunet' 
    name = f'qin_4_11_finetuning_{model_type}'
    gpu_id = 0
    data_root = '/root/autodl-tmp/Proj/data/qin_testset_npy'
    
    model_weights = {
        'difnet_laplace': '/root/autodl-tmp/Proj/code/logs/prostate_4_8_mlp/model_best.pth',
        'difnet_no_tto': '/root/autodl-tmp/Proj/code/logs/prostate_4_8_mlp/model_best.pth',
        '3dunet': '/root/autodl-tmp/Proj/code/logs/baseline_3dunet_sparse_amp_4_8/model_best.pth',
        '3dunet_laplace': '/root/autodl-tmp/Proj/code/logs/baseline_3dunet_sparse_amp_4_8/model_best.pth',
        'swin_unetr': '/root/autodl-tmp/Proj/code/logs/baseline_swin_unetr_sparse_amp_4_8/model_best.pth',
        'swin_unetr_laplace': '/root/autodl-tmp/Proj/code/logs/baseline_swin_unetr_sparse_amp_4_8/model_best.pth'
    }
    out_res = (128, 128, 128)
    laplace_iters = 50
    save_vis = True  
    save_nii = True

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = str(QinForensicConfig.gpu_id)
    save_dir = f'./logs/{QinForensicConfig.name}'
    os.makedirs(save_dir, exist_ok=True)
    if QinForensicConfig.save_vis: os.makedirs(os.path.join(save_dir, 'vis'), exist_ok=True)
    if QinForensicConfig.save_nii: os.makedirs(os.path.join(save_dir, 'vis_nii'), exist_ok=True)
    
    logger = setup_logger(os.path.join(save_dir, 'forensic_log.txt'))
    logger.info(f"🚀 [QIN FORENSIC PROTOCOL] 启动模型: {QinForensicConfig.model_type.upper()}")

    if 'difnet' in QinForensicConfig.model_type:
        model = DIF_Net(num_views=3, combine='mlp').cuda()
    elif '3dunet' in QinForensicConfig.model_type:
        model = Baseline_3DUNet().cuda()
    elif 'swin_unetr' in QinForensicConfig.model_type:
        model = Baseline_SwinUNETR().cuda()
        
    checkpoint = torch.load(QinForensicConfig.model_weights[QinForensicConfig.model_type], map_location='cuda')
    if 'net.weight' in list(checkpoint.keys())[0] and 'difnet' not in QinForensicConfig.model_type:
        model.load_state_dict(checkpoint, strict=False)
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    val_dst = QinBlindTestDataset(data_root=QinForensicConfig.data_root)
    geo = OrthogonalGeometry()

    metrics = {
        'purged_global': [], 'band_1_3': [], 'band_4_7': [], 'band_gt_7': [], 'glo_hd95': [],
        'oracle_purged': [], 'oracle_b1_3': [], 'oracle_b4_7': [], 'oracle_gt_7': [], 'oracle_hd95': [],
        't_pre': [], 't_intra': []
    }

    for idx in tqdm(range(len(val_dst)), ncols=100, desc="Forensic Autopsy"):
        item = val_dst[idx]
        case_name = item['name']
        prior_image, prior_mask = item['prior_image'], item['prior_mask']
        target_image, target_mask = item['target_image'], item['target_mask']

        nz_prior = np.argwhere(prior_mask > 0)
        px, py, pz = nz_prior.mean(axis=0).astype(int) if len(nz_prior) > 0 else (64, 64, 64)
        nz_target = np.argwhere(target_mask > 0)
        gx, gy, gz = nz_target.mean(axis=0).astype(int) if len(nz_target) > 0 else (64, 64, 64)
        feed_cx, feed_cy, feed_cz = gx, gy, gz

        oracle_shift_vec = (gx - px, gy - py, gz - pz)
        oracle_aligned_prior_mask = ndimage.shift(prior_mask, oracle_shift_vec, order=0)
        oracle_aligned_prior_image = ndimage.shift(prior_image, oracle_shift_vec, order=1)

        shift_t2c = (64 - feed_cx, 64 - feed_cy, 64 - feed_cz)
        centered_target_image = ndimage.shift(target_image, shift_t2c, order=1)
        centered_target_mask = ndimage.shift(target_mask, shift_t2c, order=0)
        
        shift_p2c = (64 - px, 64 - py, 64 - pz)
        centered_prior_image = ndimage.shift(prior_image, shift_p2c, order=1)
        centered_prior_mask = ndimage.shift(prior_mask, shift_p2c, order=0)

        t_pre = 0.0
        t_intra = 0.0
        pred_mask_centered = None
        pred_mask_np = None

        if 'difnet' in QinForensicConfig.model_type:
            slice_idx = 64
            projs = np.zeros((3, 1, 128, 128), dtype=np.float32)
            projs[0, 0] = centered_target_image[:, :, slice_idx]
            projs[1, 0] = centered_target_image[:, slice_idx, :]
            projs[2, 0] = centered_target_image[slice_idx, :, :]
            prior_projs = np.zeros((3, 1, 128, 128), dtype=np.float32)
            prior_projs[0, 0] = centered_prior_image[:, :, slice_idx]
            prior_projs[1, 0] = centered_prior_image[:, slice_idx, :]
            prior_projs[2, 0] = centered_prior_image[slice_idx, :, :]
            
            res = 128
            grid = np.mgrid[:res, :res, :res].reshape(3, -1).transpose(1, 0)
            points_norm_full = ((grid.astype(np.float32) / (np.array([res, res, res], dtype=np.float32) - 1)) - 0.5) * 2
            proj_points_full = np.stack([geo.project(points_norm_full, 0), geo.project(points_norm_full, 1), geo.project(points_norm_full, 2)], axis=0)

            torch.cuda.synchronize()
            t1 = time.time()
            with torch.no_grad():
                pred_logits_prior = model({
                    'projs': torch.from_numpy(projs).unsqueeze(0).cuda(),
                    'prior_projs': torch.from_numpy(prior_projs).unsqueeze(0).cuda(),
                    'prior_mask': torch.from_numpy(centered_prior_mask).view(1, 1, 128, 128, 128).float().cuda(),
                    'points': torch.from_numpy(points_norm_full).unsqueeze(0).cuda(),
                    'proj_points': torch.from_numpy(proj_points_full).unsqueeze(0).cuda()
                }, is_eval=True, eval_npoint=50000)
                
                if 'laplace' in QinForensicConfig.model_type:
                    prior_3d_logits = pred_logits_prior.view(1, 1, 128, 128, 128)
                    target_slices_2d = torch.from_numpy(np.stack([
                        centered_target_mask[:, :, slice_idx], 
                        centered_target_mask[:, slice_idx, :], 
                        centered_target_mask[slice_idx, :, :]
                    ])).float().cuda()

                    final_3d_probs = dirichlet_harmonic_diffusion(
                        prior_3d_logits=prior_3d_logits,
                        target_slices_2d=target_slices_2d,
                        cx=slice_idx, cy=slice_idx, cz=slice_idx,
                        num_iterations=QinForensicConfig.laplace_iters
                    )
                    pred_mask_centered = (final_3d_probs.cpu().numpy().squeeze() > 0.5).astype(np.uint8)
                else:
                    pred_mask_centered = (torch.sigmoid(pred_logits_prior).view(128, 128, 128).cpu().numpy() > 0.5).astype(np.uint8)
                    
            torch.cuda.synchronize()
            t_intra = time.time() - t1

            # 还原坐标
            pred_mask_np = ndimage.shift(pred_mask_centered, (feed_cx - 64, feed_cy - 64, feed_cz - 64), order=0)

        elif '3dunet' in QinForensicConfig.model_type or 'swin_unetr' in QinForensicConfig.model_type:
            target_img_tensor = torch.from_numpy(target_image).view(1, 1, 128, 128, 128).float().cuda()
            prior_img_tensor = torch.from_numpy(oracle_aligned_prior_image).view(1, 1, 128, 128, 128).float().cuda()
            prior_mask_tensor = torch.from_numpy(oracle_aligned_prior_mask).view(1, 1, 128, 128, 128).float().cuda()
            coords_tensor = torch.tensor([[feed_cx, feed_cy, feed_cz]], dtype=torch.long).cuda()
            
            torch.cuda.synchronize()
            t1 = time.time()
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    pred_logits = model(target_img_tensor, prior_img_tensor, prior_mask_tensor, coords_tensor)
                
                if 'laplace' in QinForensicConfig.model_type:
                    # 对于 Baseline，由于推理是在原始空间进行，切片抽取必须基于真实靶区质心坐标
                    target_slices_2d = torch.from_numpy(np.stack([
                        target_mask[:, :, feed_cz], 
                        target_mask[:, feed_cy, :], 
                        target_mask[feed_cx, :, :]
                    ])).float().cuda()
                    
                    prior_3d_logits = pred_logits.float()
                    
                    final_3d_probs = dirichlet_harmonic_diffusion(
                        prior_3d_logits=prior_3d_logits,
                        target_slices_2d=target_slices_2d,
                        cx=feed_cx, cy=feed_cy, cz=feed_cz,
                        num_iterations=QinForensicConfig.laplace_iters
                    )
                    pred_mask_np = (final_3d_probs.cpu().numpy()[0, 0] > 0.5).astype(np.uint8)
                else:
                    prob_network = torch.sigmoid(pred_logits.float())
                    pred_mask_np = (prob_network > 0.5).cpu().numpy()[0, 0].astype(np.uint8)
                    
            torch.cuda.synchronize()
            t_intra = time.time() - t1

        pred_mask_np = keep_largest_connected_component_3d(pred_mask_np)

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
        
        metrics['t_pre'].append(t_pre)
        metrics['t_intra'].append(t_intra)

        logger.info(f"[{case_name}] Purged Dice(Oracle/Pred): {results_o_purged:.3f}/{results_p_purged:.3f} | Glo HD95: {results_o_hd95:.2f}/{results_p_hd95:.2f}mm")

        if QinForensicConfig.save_vis:
            vis_save_path = os.path.join(save_dir, 'vis', f"{case_name}_pred_{results_p_purged:.3f}.png")
            save_visualization_3view(
                img_np=target_image, 
                prior_mask=prior_mask, 
                aligned_prior=oracle_aligned_prior_mask,  
                gt_mask=target_mask, 
                pred_mask=pred_mask_np, 
                save_path=vis_save_path, 
                case_name=case_name, 
                epoch=QinForensicConfig.model_type.upper()
            )

        if QinForensicConfig.save_nii:
            nii_dir = os.path.join(save_dir, 'vis_nii')
            save_nifti(target_image, os.path.join(nii_dir, f"{case_name}_TARGET_IMG.nii.gz"), is_mask=False)
            save_nifti(target_mask, os.path.join(nii_dir, f"{case_name}_GT.nii.gz"))
            save_nifti(oracle_aligned_prior_mask, os.path.join(nii_dir, f"{case_name}_ORACLE_PRIOR.nii.gz"))
            save_nifti(pred_mask_np, os.path.join(nii_dir, f"{case_name}_PRED_{QinForensicConfig.model_type.upper()}.nii.gz"))

    logger.info("\n" + "="*80)
    logger.info(f"🔥 QIN FORENSIC AUTOPSY SUMMARY: {QinForensicConfig.model_type.upper()} 🔥")
    logger.info("="*80)
    logger.info(f"[Macroscopic] Isolated Global Dice : Oracle {np.nanmean(metrics['oracle_purged']):.4f} -> Pred {np.nanmean(metrics['purged_global']):.4f}")
    logger.info(f"[Macroscopic] Global HD95 (mm)     : Oracle {np.nanmean(metrics['oracle_hd95']):.2f} -> Pred {np.nanmean(metrics['glo_hd95']):.2f}")
    logger.info("-" * 80)
    logger.info(f"[Distance-Stratified] Band 1-3 v.  : Oracle {np.nanmean(metrics['oracle_b1_3']):.4f} -> Pred {np.nanmean(metrics['band_1_3']):.4f}")
    logger.info(f"[Distance-Stratified] Band 4-7 v.  : Oracle {np.nanmean(metrics['oracle_b4_7']):.4f} -> Pred {np.nanmean(metrics['band_4_7']):.4f}")
    logger.info(f"[Distance-Stratified] Band >7 v.   : Oracle {np.nanmean(metrics['oracle_gt_7']):.4f} -> Pred {np.nanmean(metrics['band_gt_7']):.4f}")
    logger.info("-" * 80)
    logger.info(f"[Temporal Profiling] T_intra (infer) : {np.mean(metrics['t_intra']):.3f} s/vol")
    logger.info("="*80)