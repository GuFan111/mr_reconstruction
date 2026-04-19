# eval_qin_forensic_missing_view_pvalue.py

import os
import warnings
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore')  # 忽略小样本由于数值完全一致可能带来的 tie 警告

import sys
import time
import logging
import numpy as np
import torch
import scipy.ndimage as ndimage
from tqdm import tqdm
from scipy.stats import wilcoxon

from monai.metrics import compute_hausdorff_distance
from torch.utils.data import Dataset
from models.model import DIF_Net
from dataset import OrthogonalGeometry

# ==========================================
# 🟢 杂项工具函数
# ==========================================
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
    return (ndimage.gaussian_filter(mask_np.astype(np.float32), sigma=sigma) > threshold).astype(np.uint8)

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
        return {
            'name': os.path.basename(p_dir),
            'prior_image': np.load(os.path.join(p_dir, 'prior_image.npy')),
            'prior_mask': np.load(os.path.join(p_dir, 'prior_mask.npy')),
            'target_image': np.load(os.path.join(p_dir, 'target_image.npy')),
            'target_mask': np.load(os.path.join(p_dir, 'target_mask.npy'))
        }

# ==========================================
# 🟢 全景缺失评估配置
# ==========================================
class QinForensicConfig:
    gpu_id = 0
    data_root = '/root/autodl-tmp/Proj/data/qin_testset_npy'

    models_to_test = ['difnet_att', 'difnet_mlp']
    missing_views = [0, 1, 2] # 0: Axial, 1: Coronal, 2: Sagittal
    view_names = {0: 'AXIAL', 1: 'CORONAL', 2: 'SAGITTAL'}

    model_weights = {
        'difnet_att': '/root/autodl-tmp/Proj/code/logs/prostate_4_8_attention/model_best.pth',
        'difnet_mlp': '/root/autodl-tmp/Proj/code/logs/prostate_4_8_mlp/model_best.pth'
    }

# ==========================================
# 🟢 模块化单次缺失测试函数 (支持返回全量数组以便计算 P 值)
# ==========================================
def run_missing_view_test(model_type, missing_idx, dataset, geo):
    combine_mode = 'attention' if 'att' in model_type else 'mlp'
    model = DIF_Net(num_views=3, combine=combine_mode).cuda()
    model.load_state_dict(torch.load(QinForensicConfig.model_weights[model_type], map_location='cuda'), strict=False)
    model.eval()

    metrics = {'purged_global': [], 'band_gt_7': [], 'glo_hd95': []}
    desc_str = f"BLIND {QinForensicConfig.view_names[missing_idx]}" if missing_idx is not None else "FULL 3-VIEW"

    for idx in tqdm(range(len(dataset)), ncols=90, desc=f"[{model_type.upper()}] {desc_str}"):
        item = dataset[idx]
        prior_image, prior_mask = item['prior_image'], item['prior_mask']
        target_image, target_mask = item['target_image'], item['target_mask']

        nz_prior = np.argwhere(prior_mask > 0)
        px, py, pz = nz_prior.mean(axis=0).astype(int) if len(nz_prior) > 0 else (64, 64, 64)
        nz_target = np.argwhere(target_mask > 0)
        feed_cx, feed_cy, feed_cz = nz_target.mean(axis=0).astype(int) if len(nz_target) > 0 else (64, 64, 64)

        shift_t2c = (64 - feed_cx, 64 - feed_cy, 64 - feed_cz)
        centered_target_image = ndimage.shift(target_image, shift_t2c, order=1)
        centered_target_mask = ndimage.shift(target_mask, shift_t2c, order=0)

        shift_p2c = (64 - px, 64 - py, 64 - pz)
        centered_prior_image = ndimage.shift(prior_image, shift_p2c, order=1)
        centered_prior_mask = ndimage.shift(prior_mask, shift_p2c, order=0)

        slice_idx = 64
        projs = np.zeros((3, 1, 128, 128), dtype=np.float32)
        projs[0, 0] = centered_target_image[:, :, slice_idx]
        projs[1, 0] = centered_target_image[:, slice_idx, :]
        projs[2, 0] = centered_target_image[slice_idx, :, :]

        prior_projs = np.zeros((3, 1, 128, 128), dtype=np.float32)
        prior_projs[0, 0] = centered_prior_image[:, :, slice_idx]
        prior_projs[1, 0] = centered_prior_image[:, slice_idx, :]
        prior_projs[2, 0] = centered_prior_image[slice_idx, :, :]

        # 🩸 核心破坏：直接致盲目标视角 (如果 missing_idx 存在)
        if missing_idx is not None:
            projs[missing_idx, 0] = 0.0
            prior_projs[missing_idx, 0] = 0.0

        res = 128
        grid = np.mgrid[:res, :res, :res].reshape(3, -1).transpose(1, 0)
        points_norm = ((grid.astype(np.float32) / (res - 1)) - 0.5) * 2
        proj_points = np.stack([geo.project(points_norm, 0), geo.project(points_norm, 1), geo.project(points_norm, 2)], axis=0)

        with torch.no_grad():
            pred_logits = model({
                'projs': torch.from_numpy(projs).unsqueeze(0).cuda(),
                'prior_projs': torch.from_numpy(prior_projs).unsqueeze(0).cuda(),
                'prior_mask': torch.from_numpy(centered_prior_mask).view(1, 1, 128, 128, 128).float().cuda(),
                'points': torch.from_numpy(points_norm).unsqueeze(0).cuda(),
                'proj_points': torch.from_numpy(proj_points).unsqueeze(0).cuda()
            }, is_eval=True, eval_npoint=50000)

            pred_mask_centered = (torch.sigmoid(pred_logits).view(128, 128, 128).cpu().numpy() > 0.5).astype(np.uint8)

        pred_mask_np = keep_largest_connected_component_3d(ndimage.shift(pred_mask_centered, (feed_cx - 64, feed_cy - 64, feed_cz - 64), order=0))

        M_anchor = np.zeros((128, 128, 128), dtype=bool)
        M_anchor[feed_cx, :, :] = True; M_anchor[:, feed_cy, :] = True; M_anchor[:, :, feed_cz] = True
        mask_purged = ~M_anchor
        dist_map = ndimage.distance_transform_edt(mask_purged)
        mask_b3 = (dist_map > 7)

        gt_tensor = torch.from_numpy(smooth_binary_mask(target_mask)).view(1, 1, 128, 128, 128).float().cuda()
        pred_tensor = torch.from_numpy(smooth_binary_mask(pred_mask_np)).view(1, 1, 128, 128, 128).float().cuda()

        metrics['purged_global'].append(compute_band_dice(pred_mask_np, target_mask, mask_purged))
        metrics['band_gt_7'].append(compute_band_dice(pred_mask_np, target_mask, mask_b3))
        metrics['glo_hd95'].append(safe_global_hd95(pred_tensor, gt_tensor))

    return metrics

# 辅助排版函数
def format_pvalue_diff(base_val, new_val, p_val, is_hd95=False):
    diff = new_val - base_val
    # 针对 HD95，下降(差值<0)是变好；针对 Dice，上升(差值>0)是变好。
    is_better = (diff < 0) if is_hd95 else (diff > 0)

    arrow = "↑" if diff > 0 else "↓" if diff < 0 else "-"

    if p_val < 0.05:
        marker = "★ SURGE" if is_better else "▼ DROP"
        return f"{new_val:.4f} ({arrow}) | p={p_val:.4f} [{marker}]"
    else:
        return f"{new_val:.4f} ({arrow}) | p={p_val:.4f} [n.s.]"


# ==========================================
# 🟢 主控程序
# ==========================================
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = str(QinForensicConfig.gpu_id)
    save_dir = f'./logs/qin_missing_view_pvalue'
    os.makedirs(save_dir, exist_ok=True)

    logger = setup_logger(os.path.join(save_dir, 'panoramic_missing_view_sig.txt'))
    logger.info("🚀 [MISSING VIEW SIGNIFICANCE ENGINE] 启动全景致盲双侧统计学检验...")
    logger.info("目标：寻找是否发生 p < 0.05 的反常指标飙升 (★ SURGE)！\n")

    val_dst = QinBlindTestDataset(data_root=QinForensicConfig.data_root)
    geo = OrthogonalGeometry()

    results_matrix = {}

    for model_type in QinForensicConfig.models_to_test:
        logger.info(f"\n==========================================")
        logger.info(f"⚡ 开始评估模型: {model_type.upper()}")
        results_matrix[model_type] = {}

        # 1. 跑一遍完整的 3 视角作为基准
        full_metrics = run_missing_view_test(model_type, None, val_dst, geo)
        results_matrix[model_type]['FULL'] = full_metrics

        # 2. 依次跑致盲视角
        for v_idx in QinForensicConfig.missing_views:
            v_name = QinForensicConfig.view_names[v_idx]
            blind_metrics = run_missing_view_test(model_type, v_idx, val_dst, geo)
            results_matrix[model_type][v_name] = blind_metrics

    # ==========================================
    # 🟢 打印终极矩阵显著性大表
    # ==========================================
    logger.info("\n\n" + "="*110)
    logger.info("🔥 PANORAMIC MISSING VIEW SIGNIFICANCE MATRIX (TWO-SIDED PAIRED WILCOXON) 🔥")
    logger.info("="*110)

    for model_type in QinForensicConfig.models_to_test:
        logger.info(f"【 Architecture: {model_type.upper()} 】")

        # 提取 Full View 均值
        full_glo = np.mean(results_matrix[model_type]['FULL']['purged_global'])
        full_far = np.mean(results_matrix[model_type]['FULL']['band_gt_7'])
        full_hd95 = np.mean(results_matrix[model_type]['FULL']['glo_hd95'])

        logger.info(f"   [ FULL 3-VIEW ] -> Glo: {full_glo:.4f} | >7v: {full_far:.4f} | HD95: {full_hd95:.2f}mm")
        logger.info("-" * 110)

        for v_idx in QinForensicConfig.missing_views:
            v_name = QinForensicConfig.view_names[v_idx]

            # 计算盲视图下的双侧 p-value
            blind_m = results_matrix[model_type][v_name]

            # Global
            b_glo_mean = np.mean(blind_m['purged_global'])
            _, p_glo = wilcoxon(results_matrix[model_type]['FULL']['purged_global'], blind_m['purged_global'], alternative='two-sided')

            # Deep Void
            b_far_mean = np.mean(blind_m['band_gt_7'])
            _, p_far = wilcoxon(results_matrix[model_type]['FULL']['band_gt_7'], blind_m['band_gt_7'], alternative='two-sided')

            # HD95
            b_hd95_mean = np.mean(blind_m['glo_hd95'])
            _, p_hd95 = wilcoxon(results_matrix[model_type]['FULL']['glo_hd95'], blind_m['glo_hd95'], alternative='two-sided')

            # 格式化打印
            str_glo = format_pvalue_diff(full_glo, b_glo_mean, p_glo)
            str_far = format_pvalue_diff(full_far, b_far_mean, p_far)
            str_hd95 = format_pvalue_diff(full_hd95, b_hd95_mean, p_hd95, is_hd95=True)

            logger.info(f"   [ BLIND {v_name:<8} ] ")
            logger.info(f"        Glo Dice : {str_glo}")
            logger.info(f"        >7v Dice : {str_far}")
            logger.info(f"        HD95     : {str_hd95}")
        logger.info("="*110)

    logger.info("💡 临床解读指引：")
    logger.info("寻找 DIFNET_MLP 下，是否出现了 `★ SURGE` (显著性飙升)！")
    logger.info("如果出现，这将被写进论文，作为『拼接融合导致信息毒性』的铁证！")