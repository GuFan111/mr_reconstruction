# eval_qin_forensic_dl_laplace_auto_poison.py

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import sys
import time
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import scipy.ndimage as ndimage
from tqdm import tqdm
import nibabel as nib

from monai.metrics import compute_hausdorff_distance
from torch.utils.data import Dataset

from models.model import DIF_Net
from models.baseline_models import Baseline_SwinUNETR
from dataset import OrthogonalGeometry

# ==========================================
# 🟢 升级版：学术界标准复合掩膜误差仿真器
# ==========================================
def academic_mask_noise_generator(mask_2d, severity_level, noise_type='elastic'):
    if severity_level == 0:
        return mask_2d.copy().astype(np.float32)

    mask_noisy = mask_2d.copy().astype(float)

    def apply_morph(m, s):
        action = np.random.choice(['dilate', 'erode'])
        return ndimage.binary_dilation(m, iterations=s) if action == 'dilate' else ndimage.binary_erosion(m, iterations=s)

    def apply_elastic(m, s):
        shape = m.shape
        alpha = s * 25.0
        sigma = 4.0
        dx = ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        dy = ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        return ndimage.map_coordinates(m, indices, order=0, mode='constant').reshape(shape) > 0.5

    def apply_dropout(m, s):
        m_out = m.copy()
        cx, cy = np.random.randint(40, 88, 2)
        r = int(s * 2.5)  # 当 s=6 (新 severity 5) 时，将产生半径 15 的大洞
        y, x = np.ogrid[:128, :128]
        hole = (x - cx)**2 + (y - cy)**2 <= r**2
        m_out[hole] = 0
        if s >= 2:
            cx2, cy2 = np.random.randint(20, 108, 2)
            r2 = int(s * 2.0) # 当 s=6 (新 severity 5) 时，将产生半径 12 的错误飞地
            island = (x - cx2)**2 + (y - cy2)**2 <= r2**2
            m_out[island] = 1
        return m_out

    if noise_type == 'morphology':
        mask_noisy = apply_morph(mask_noisy, severity_level)
    elif noise_type == 'elastic':
        mask_noisy = apply_elastic(mask_noisy, severity_level)
    elif noise_type == 'dropout':
        # 🔴 核心修改：删掉原先的 1，原 2~5 前移，并补充新的 5（内部对应原先的 6）
        mask_noisy = apply_dropout(mask_noisy, severity_level + 1)
    elif noise_type == 'mixed':
        mask_noisy = apply_elastic(mask_noisy, severity_level)
        mask_noisy = apply_morph(mask_noisy, max(1, severity_level // 2))
        mask_noisy = apply_dropout(mask_noisy, max(1, severity_level - 1))

    return mask_noisy.astype(np.float32)

# ==========================================
# 🟢 拉普拉斯算子与其他工具函数
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

    with torch.no_grad(), torch.amp.autocast('cuda'):
        for _ in range(num_iterations):
            new_delta = F.conv3d(delta_field, kernel, padding=1)
            delta_field = torch.where(is_boundary, known_residual, new_delta)

    return torch.sigmoid(prior_3d_logits + delta_field)

def setup_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    return logger

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
    p, g = pred_mask[band_mask].astype(bool), gt_mask[band_mask].astype(bool)
    union = p.sum() + g.sum()
    return np.nan if union == 0 else 2.0 * np.logical_and(p, g).sum() / union

def safe_global_hd95(pred_tensor, gt_tensor):
    if pred_tensor.sum() == 0 or gt_tensor.sum() == 0: return 99.0
    try: return compute_hausdorff_distance(pred_tensor, gt_tensor, include_background=False, percentile=95).item()
    except: return 99.0

class QinBlindTestDataset(Dataset):
    def __init__(self, data_root):
        self.patient_dirs = sorted([os.path.join(data_root, d) for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d)) and not d.startswith('.')])
    def __len__(self): return len(self.patient_dirs)
    def __getitem__(self, idx):
        p_dir = self.patient_dirs[idx]
        return {'prior_image': np.load(os.path.join(p_dir, 'prior_image.npy')), 'prior_mask': np.load(os.path.join(p_dir, 'prior_mask.npy')),
                'target_image': np.load(os.path.join(p_dir, 'target_image.npy')), 'target_mask': np.load(os.path.join(p_dir, 'target_mask.npy'))}

# ==========================================
# 🟢 主入口
# ==========================================
if __name__ == '__main__':
    gpu_id = 0
    data_root = '/root/autodl-tmp/Proj/data/qin_testset_npy'
    difnet_weights = '/root/autodl-tmp/Proj/code/logs/prostate_4_8_attention/model_best.pth'
    swin_weights = '/root/autodl-tmp/Proj/code/logs/baseline_swin_unetr_sparse_amp_4_8/model_best.pth'
    name = 'qin_laplace_robustness_metrics_full_v2'

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    save_dir = f'./logs/{name}'
    os.makedirs(save_dir, exist_ok=True)

    logger = setup_logger(os.path.join(save_dir, 'auto_poison_full_v2_log.txt'))
    logger.info(f"🚀 [ROBUSTNESS FULL PROTOCOL] 启动带全量指标的联合投毒测试 (ATT vs SWIN-UNETR)")

    # 1. 加载 ATT 模型
    model_att = DIF_Net(num_views=3, combine='attention').cuda()
    model_att.load_state_dict(torch.load(difnet_weights, map_location='cuda'), strict=False)
    model_att.eval()

    # 2. 加载 SWIN-UNETR 模型
    model_swin = Baseline_SwinUNETR().cuda()
    model_swin.load_state_dict(torch.load(swin_weights, map_location='cuda'), strict=False)
    model_swin.eval()

    val_dst = QinBlindTestDataset(data_root=data_root)
    geo = OrthogonalGeometry()

    noise_types = ['morphology', 'elastic', 'dropout', 'mixed']
    severities = [1, 2, 3, 4, 5]
    final_csv_results = []

    for n_type in noise_types:
        for sev in severities:
            metrics = {
                'noisy_2d_dice': [],
                'att_glo': [], 'att_hd95': [], 'att_gt7': [], 'att_t_intra': [],
                'swin_glo': [], 'swin_hd95': [], 'swin_gt7': [], 'swin_t_intra': []
            }

            for idx in tqdm(range(len(val_dst)), ncols=100, desc=f"{n_type}-{sev}"):
                item = val_dst[idx]
                prior_image, prior_mask = item['prior_image'], item['prior_mask']
                target_image, target_mask = item['target_image'], item['target_mask']

                nz_prior = np.argwhere(prior_mask > 0)
                px, py, pz = nz_prior.mean(axis=0).astype(int) if len(nz_prior) > 0 else (64, 64, 64)
                nz_target = np.argwhere(target_mask > 0)
                feed_cx, feed_cy, feed_cz = nz_target.mean(axis=0).astype(int) if len(nz_target) > 0 else (64, 64, 64)

                # Oracle对齐与居中
                shift_t2c = (64 - feed_cx, 64 - feed_cy, 64 - feed_cz)
                centered_target_image = ndimage.shift(target_image, shift_t2c, order=1)
                centered_target_mask = ndimage.shift(target_mask, shift_t2c, order=0)

                shift_p2c = (64 - px, 64 - py, 64 - pz)
                centered_prior_image = ndimage.shift(prior_image, shift_p2c, order=1)
                centered_prior_mask = ndimage.shift(prior_mask, shift_p2c, order=0)
                oracle_aligned_prior_mask = ndimage.shift(prior_mask, (feed_cx - px, feed_cy - py, feed_cz - pz), order=0)
                oracle_aligned_prior_image = ndimage.shift(prior_image, (feed_cx - px, feed_cy - py, feed_cz - pz), order=1)

                slice_idx = 64
                prior_projs = np.zeros((3, 1, 128, 128), dtype=np.float32)
                prior_projs[0, 0] = centered_prior_image[:, :, slice_idx]
                prior_projs[1, 0] = centered_prior_image[:, slice_idx, :]
                prior_projs[2, 0] = centered_prior_image[slice_idx, :, :]

                res = 128
                grid = np.mgrid[:res, :res, :res].reshape(3, -1).transpose(1, 0)
                points_norm_full = ((grid.astype(np.float32) / (res - 1)) - 0.5) * 2
                proj_points_full = np.stack([geo.project(points_norm_full, 0), geo.project(points_norm_full, 1), geo.project(points_norm_full, 2)], axis=0)

                # ----------------------------------------------------
                # ☣️ 制造联合投毒的公共毒药
                # ----------------------------------------------------
                gt_slices_np = np.stack([centered_target_mask[:, :, slice_idx], centered_target_mask[:, slice_idx, :], centered_target_mask[slice_idx, :, :]])
                noisy_slices_np, slice_dices = np.zeros_like(gt_slices_np), []

                np.random.seed(idx * 100) # 保证所有模型每次面临的毒性完全一致
                for i in range(3):
                    noisy_slices_np[i] = academic_mask_noise_generator(gt_slices_np[i], sev, n_type)
                    inter = np.sum(noisy_slices_np[i] * gt_slices_np[i])
                    slice_dices.append(2.0 * inter / (np.sum(noisy_slices_np[i]) + np.sum(gt_slices_np[i]) + 1e-5))
                metrics['noisy_2d_dice'].append(np.mean(slice_dices))
                target_slices_2d = torch.from_numpy(noisy_slices_np).float().cuda()

                # ----------------------------------------------------
                # 🟢 评测 ATT
                # ----------------------------------------------------
                torch.cuda.synchronize(); t1 = time.time()
                with torch.no_grad():
                    pred_logits_att = model_att({'projs': torch.from_numpy(prior_projs).unsqueeze(0).cuda(),
                                                 'prior_projs': torch.from_numpy(prior_projs).unsqueeze(0).cuda(),
                                                 'prior_mask': torch.from_numpy(centered_prior_mask).view(1,1,128,128,128).float().cuda(),
                                                 'points': torch.from_numpy(points_norm_full).unsqueeze(0).cuda(),
                                                 'proj_points': torch.from_numpy(proj_points_full).unsqueeze(0).cuda()}, is_eval=True, eval_npoint=50000)
                    final_3d_probs_att = dirichlet_harmonic_diffusion(pred_logits_att.view(1,1,128,128,128).contiguous(), target_slices_2d, slice_idx, slice_idx, slice_idx, 50)
                    pred_mask_centered_att = (final_3d_probs_att.cpu().numpy().squeeze() > 0.5).astype(np.uint8)
                torch.cuda.synchronize()
                metrics['att_t_intra'].append(time.time() - t1)
                pred_mask_np_att = keep_largest_connected_component_3d(ndimage.shift(pred_mask_centered_att, (feed_cx-64, feed_cy-64, feed_cz-64), order=0))

                # ----------------------------------------------------
                # 🔵 评测 SWIN-UNETR
                # ----------------------------------------------------
                torch.cuda.synchronize(); t2 = time.time()
                with torch.no_grad():
                    with torch.amp.autocast('cuda'):
                        pred_logits_swin = model_swin(
                            torch.from_numpy(target_image).view(1,1,128,128,128).float().cuda(),
                            torch.from_numpy(oracle_aligned_prior_image).view(1,1,128,128,128).float().cuda(),
                            torch.from_numpy(oracle_aligned_prior_mask).view(1,1,128,128,128).float().cuda(),
                            torch.tensor([[feed_cx, feed_cy, feed_cz]], dtype=torch.long).cuda()
                        )
                    # 将 SWIN 的预测拉到原点做 Laplace，使用相同 target_slices_2d
                    centered_logits_swin = ndimage.shift(pred_logits_swin[0,0].float().cpu().numpy(), shift_t2c, order=1)
                    centered_logits_swin = torch.from_numpy(centered_logits_swin).cuda().view(1,1,128,128,128)
                    final_probs_swin = dirichlet_harmonic_diffusion(centered_logits_swin, target_slices_2d, slice_idx, slice_idx, slice_idx, 50)
                    pred_mask_centered_swin = (final_probs_swin.cpu().numpy().squeeze() > 0.5).astype(np.uint8)
                torch.cuda.synchronize()
                metrics['swin_t_intra'].append(time.time() - t2)
                pred_mask_np_swin = keep_largest_connected_component_3d(ndimage.shift(pred_mask_centered_swin, (feed_cx-64, feed_cy-64, feed_cz-64), order=0))

                # ----------------------------------------------------
                # 📐 频带掩膜与算分
                # ----------------------------------------------------
                M_anchor = np.zeros((128, 128, 128), dtype=bool)
                M_anchor[feed_cx, :, :] = True; M_anchor[:, feed_cy, :] = True; M_anchor[:, :, feed_cz] = True
                mask_purged = ~M_anchor
                mask_gt7 = (ndimage.distance_transform_edt(mask_purged) > 7)
                gt_tensor = torch.from_numpy(smooth_binary_mask(target_mask)).view(1,1,128,128,128).float().cuda()

                tensor_att = torch.from_numpy(smooth_binary_mask(pred_mask_np_att)).view(1,1,128,128,128).float().cuda()
                tensor_swin = torch.from_numpy(smooth_binary_mask(pred_mask_np_swin)).view(1,1,128,128,128).float().cuda()

                metrics['att_glo'].append(compute_band_dice(pred_mask_np_att, target_mask, mask_purged))
                metrics['att_gt7'].append(compute_band_dice(pred_mask_np_att, target_mask, mask_gt7))
                metrics['att_hd95'].append(safe_global_hd95(tensor_att, gt_tensor))

                metrics['swin_glo'].append(compute_band_dice(pred_mask_np_swin, target_mask, mask_purged))
                metrics['swin_gt7'].append(compute_band_dice(pred_mask_np_swin, target_mask, mask_gt7))
                metrics['swin_hd95'].append(safe_global_hd95(tensor_swin, gt_tensor))

            # -----------------------------------------
            # 📝 日志打印与数据收集
            # -----------------------------------------
            logger.info(f"\n================================================================================")
            logger.info(f"🔥 NOISE AUTOPSY SUMMARY: [{n_type.upper()}] Severity-{sev}  |  2D Input DSC: {np.nanmean(metrics['noisy_2d_dice']):.4f}")
            logger.info(f"================================================================================")
            logger.info(f"{'Method':<20} | {'Global DSC':<12} | {'Dark (>7v)':<12} | {'HD95 (mm)':<12}")
            logger.info(f"--------------------------------------------------------------------------------")
            logger.info(f"{'Swin-UNETR':<20} | {np.nanmean(metrics['swin_glo']):<12.4f} | {np.nanmean(metrics['swin_gt7']):<12.4f} | {np.nanmean(metrics['swin_hd95']):<12.2f}")
            logger.info(f"{'Proposed (ATT)':<20} | {np.nanmean(metrics['att_glo']):<12.4f} | {np.nanmean(metrics['att_gt7']):<12.4f} | {np.nanmean(metrics['att_hd95']):<12.2f}")
            logger.info(f"================================================================================\n")

            final_csv_results.append({
                'Noise_Type': n_type,
                'Severity': sev,
                '2D_Input_Dice': np.nanmean(metrics['noisy_2d_dice']),
                'Swin_Global_DSC': np.nanmean(metrics['swin_glo']),
                'Swin_GT7_DSC': np.nanmean(metrics['swin_gt7']),
                'Swin_HD95': np.nanmean(metrics['swin_hd95']),
                'ATT_Global_DSC': np.nanmean(metrics['att_glo']),
                'ATT_GT7_DSC': np.nanmean(metrics['att_gt7']),
                'ATT_HD95': np.nanmean(metrics['att_hd95'])
            })

    # ==========================================
    # 🏁 全部完成，保存全量指标 CSV
    # ==========================================
    df = pd.DataFrame(final_csv_results)
    csv_path = os.path.join(save_dir, "robustness_swin_vs_att.csv")
    df.to_csv(csv_path, index=False)

    logger.info("🏁"*40)
    logger.info(f"✅ 全量对比数据收集完毕！数据表已落盘: {csv_path}")
    logger.info("💡 你可以直接把这个 csv 里的列数据复制到画 Figure 2 雷达/阴影图的 Python 脚本中！")
    logger.info("🏁"*40)