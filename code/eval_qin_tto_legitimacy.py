import os
import sys
# 限制底层数学库的线程数
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.ndimage as ndimage
from skimage.measure import label, regionprops
from tqdm import tqdm
import copy

from torch.utils.data import Dataset

from models.unet import UNet
from models.model import DIF_Net
from dataset import OrthogonalGeometry

# ==========================================
# 🟢 鲁棒后处理与分层算分工具
# ==========================================
def get_2d_com_robust(mask_2d):
    mask_bool = mask_2d > 0.5
    if not mask_bool.any(): return None
    labeled_mask = label(mask_bool)
    regions = regionprops(labeled_mask)
    if not regions: return None
    largest_region = max(regions, key=lambda r: r.area)
    if largest_region.area < 20: return None
    return np.array(largest_region.centroid)

def keep_largest_connected_component_3d(mask_3d):
    mask_bool = mask_3d > 0.5
    if not mask_bool.any(): return mask_3d
    labeled_mask, num_features = ndimage.label(mask_bool)
    if num_features <= 1: return mask_3d
    component_sizes = np.bincount(labeled_mask.ravel())
    component_sizes[0] = 0
    largest_component_idx = component_sizes.argmax()
    cleaned_mask = (labeled_mask == largest_component_idx).astype(mask_3d.dtype)
    return cleaned_mask

def compute_band_dice(pred, gt, mask_band):
    """
    计算特定空间距离频带内的 Dice
    """
    p = pred[mask_band].astype(bool)
    g = gt[mask_band].astype(bool)
    intersection = np.logical_and(p, g).sum()
    union = p.sum() + g.sum()

    if union == 0:
        return np.nan # 该频带内既没有预测也没有真实标签，算作无效值，不计入均值
    return 2.0 * intersection / union

class QinLegitimacyConfig:
    gpu_id = 0
    data_root = '/root/autodl-tmp/Proj/data/qin_testset_npy'

    unet_weights = '/root/autodl-tmp/Proj/code/logs/prostate_2d_unet_prior_guided/unet_best.pth'
    difnet_weights = '/root/autodl-tmp/Proj/code/logs/prostate_new_2/model_best.pth'

    tto_iters = 30
    tto_lr = 1e-4

# ==========================================
# 🟢 数据加载
# ==========================================
class QinBlindTestDataset(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        self.patient_dirs = sorted([
            os.path.join(data_root, d) for d in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, d)) and not d.startswith('.')
        ])
        print(f"[Dataset] 成功肃清幽灵目录，发现 {len(self.patient_dirs)} 例真实 QIN 盲测数据。")

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

class TriViewUNet(nn.Module):
    def __init__(self, base_c=16):
        super().__init__()
        self.unet_ax = UNet(n_channels=2, n_classes=1, bilinear=False)
        self.unet_co = UNet(n_channels=2, n_classes=1, bilinear=False)
        self.unet_sa = UNet(n_channels=2, n_classes=1, bilinear=False)

    def forward(self, unet_input):
        return torch.stack([self.unet_ax(unet_input[:, 0]), self.unet_co(unet_input[:, 1]), self.unet_sa(unet_input[:, 2])], dim=1)

# ==========================================
# 🟢 主测试流程
# ==========================================
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = str(QinLegitimacyConfig.gpu_id)

    print("Loading Models...")
    unet = TriViewUNet().cuda()
    unet.load_state_dict(torch.load(QinLegitimacyConfig.unet_weights, map_location='cuda'))
    unet.eval()

    model = DIF_Net(num_views=3, combine='attention').cuda()
    model.load_state_dict(torch.load(QinLegitimacyConfig.difnet_weights, map_location='cuda'))
    model.eval()

    val_dst = QinBlindTestDataset(data_root=QinLegitimacyConfig.data_root)
    geo = OrthogonalGeometry()

    print(f"\n🚀 开始 QIN TTO 合法性物理校验 (引入 Oracle 强对峙)...")

    # 统计指标仓库 (双轨)
    results = {
        'purged_global': [], 'band_1_3': [], 'band_4_7': [], 'band_gt_7': [],
        'oracle_global': [], 'oracle_b1_3': [], 'oracle_b4_7': [], 'oracle_gt_7': []
    }

    for idx in tqdm(range(len(val_dst)), ncols=120):
        item = val_dst[idx]
        case_name = item['name']

        prior_image, prior_mask = item['prior_image'], item['prior_mask']
        target_image, target_mask = item['target_image'], item['target_mask']

        nz_prior = np.argwhere(prior_mask > 0)
        px, py, pz = nz_prior.mean(axis=0).astype(int) if len(nz_prior) > 0 else (64, 64, 64)
        nz_target = np.argwhere(target_mask > 0)
        gx, gy, gz = nz_target.mean(axis=0).astype(int) if len(nz_target) > 0 else (64, 64, 64)

        with torch.no_grad():
            img_ax, img_co, img_sa = target_image[:, :, pz], target_image[:, py, :], target_image[px, :, :]
            prior_ax, prior_co, prior_sa = prior_mask[:, :, pz], prior_mask[:, py, :], prior_mask[px, :, :]

            unet_input = torch.from_numpy(np.stack([np.stack([img_ax, img_co, img_sa]), np.stack([prior_ax, prior_co, prior_sa])], axis=1)).unsqueeze(0).float().cuda()
            pred_2d_masks = (torch.sigmoid(unet(unet_input)) > 0.5).cpu().numpy()[0, :, 0]

            cx_list, cy_list, cz_list = [], [], []
            com_ax, com_co, com_sa = get_2d_com_robust(pred_2d_masks[0]), get_2d_com_robust(pred_2d_masks[1]), get_2d_com_robust(pred_2d_masks[2])
            if com_ax is not None: cx_list.append(com_ax[0]); cy_list.append(com_ax[1])
            if com_co is not None: cx_list.append(com_co[0]); cz_list.append(com_co[1])
            if com_sa is not None: cy_list.append(com_sa[0]); cz_list.append(com_sa[1])

            feed_cx, feed_cy, feed_cz = gx, gy, gz # 强行使用 Oracle 作为 TTO 输入坐标

            # 🔴 构建 Oracle 对照组的刚性 3D Mask
            oracle_shift_vec = (gx - px, gy - py, gz - pz)
            oracle_aligned_prior_mask = ndimage.shift(prior_mask, oracle_shift_vec, order=0)

            shift_p2c = (64 - px, 64 - py, 64 - pz)
            centered_prior_image = ndimage.shift(prior_image, shift_p2c, order=1)
            centered_prior_mask = ndimage.shift(prior_mask, shift_p2c, order=0)

            shift_t2c = (64 - feed_cx, 64 - feed_cy, 64 - feed_cz)
            centered_target_image = ndimage.shift(target_image, shift_t2c, order=1)

            slice_idx = 64
            projs = np.zeros((3, 1, 128, 128), dtype=np.float32)
            projs[0, 0] = centered_target_image[:, :, slice_idx]
            projs[1, 0] = centered_target_image[:, slice_idx, :]
            projs[2, 0] = centered_target_image[slice_idx, :, :]

            prior_projs = np.zeros((3, 1, 128, 128), dtype=np.float32)
            prior_projs[0, 0] = centered_prior_image[:, :, slice_idx]
            prior_projs[1, 0] = centered_prior_image[:, slice_idx, :]
            prior_projs[2, 0] = centered_prior_image[slice_idx, :, :]

        # ==========================================
        # 🟢 阶段 2：TTO 降维闪电微调
        # ==========================================
        original_state_dict = copy.deepcopy(model.state_dict())

        centered_target_mask = ndimage.shift(target_mask, shift_t2c, order=0)
        pseudo_gt_2d = torch.from_numpy(np.stack([centered_target_mask[:, :, slice_idx], centered_target_mask[:, slice_idx, :], centered_target_mask[slice_idx, :, :]])).float().cuda()

        res = 128
        grid_1d = np.arange(res)

        xx_ax, yy_ax = np.meshgrid(grid_1d, grid_1d, indexing='ij')
        pts_ax = np.stack([xx_ax, yy_ax, np.full_like(xx_ax, slice_idx)], axis=-1).reshape(-1, 3)
        xx_co, zz_co = np.meshgrid(grid_1d, grid_1d, indexing='ij')
        pts_co = np.stack([xx_co, np.full_like(xx_co, slice_idx), zz_co], axis=-1).reshape(-1, 3)
        yy_sa, zz_sa = np.meshgrid(grid_1d, grid_1d, indexing='ij')
        pts_sa = np.stack([np.full_like(yy_sa, slice_idx), yy_sa, zz_sa], axis=-1).reshape(-1, 3)

        points_norm_tto = ((np.concatenate([pts_ax, pts_co, pts_sa], axis=0).astype(np.float32) / (res - 1)) - 0.5) * 2
        proj_points_tto = np.stack([geo.project(points_norm_tto, 0), geo.project(points_norm_tto, 1), geo.project(points_norm_tto, 2)], axis=0)

        dif_input_tto = {
            'projs': torch.from_numpy(projs).unsqueeze(0).cuda(),
            'prior_projs': torch.from_numpy(prior_projs).unsqueeze(0).cuda(),
            'prior_mask': torch.from_numpy(centered_prior_mask).view(1, 1, 128, 128, 128).float().cuda(),
            'points': torch.from_numpy(points_norm_tto).unsqueeze(0).cuda(),
            'proj_points': torch.from_numpy(proj_points_tto).unsqueeze(0).cuda()
        }

        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=QinLegitimacyConfig.tto_lr)

        with torch.enable_grad():
            for tto_step in range(QinLegitimacyConfig.tto_iters):
                optimizer.zero_grad()
                pred_logits_tto = model(dif_input_tto, eval_npoint=None)
                prob_1d = torch.sigmoid(pred_logits_tto).squeeze()

                pred_2d_slices = torch.stack([prob_1d[0:16384].view(128, 128), prob_1d[16384:32768].view(128, 128), prob_1d[32768:49152].view(128, 128)])

                loss_bce = F.binary_cross_entropy(pred_2d_slices, pseudo_gt_2d)
                intersection = (pred_2d_slices * pseudo_gt_2d).sum()
                loss_dice = 1.0 - (2. * intersection + 1e-5) / (pred_2d_slices.sum() + pseudo_gt_2d.sum() + 1e-5)

                (loss_bce + loss_dice).backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            grid = np.mgrid[:res, :res, :res].reshape(3, -1).transpose(1, 0)
            points_norm_full = ((grid.astype(np.float32) / (np.array([res, res, res], dtype=np.float32) - 1)) - 0.5) * 2
            proj_points_full = np.stack([geo.project(points_norm_full, 0), geo.project(points_norm_full, 1), geo.project(points_norm_full, 2)], axis=0)

            pred_logits_final = model({
                'projs': torch.from_numpy(projs).unsqueeze(0).cuda(),
                'prior_projs': torch.from_numpy(prior_projs).unsqueeze(0).cuda(),
                'prior_mask': torch.from_numpy(centered_prior_mask).view(1, 1, 128, 128, 128).float().cuda(),
                'points': torch.from_numpy(points_norm_full).unsqueeze(0).cuda(),
                'proj_points': torch.from_numpy(proj_points_full).unsqueeze(0).cuda()
            }, is_eval=True, eval_npoint=50000)

            # 裸奔判定
            pred_mask_centered = (torch.sigmoid(pred_logits_final).view(128, 128, 128).cpu().numpy() > 0.5).astype(np.uint8)
            pred_mask_np = keep_largest_connected_component_3d(ndimage.shift(pred_mask_centered, (feed_cx - 64, feed_cy - 64, feed_cz - 64), order=0))

            # 恢复记忆
            model.load_state_dict(original_state_dict)

            # ==========================================
            # 🔴 阶段 3：空间法医鉴定 (双轨对峙)
            # ==========================================
            M_anchor = np.zeros((128, 128, 128), dtype=bool)
            M_anchor[feed_cx, :, :] = True
            M_anchor[:, feed_cy, :] = True
            M_anchor[:, :, feed_cz] = True

            mask_purged = ~M_anchor

            # 计算欧氏距离场 (Distance Field)
            dist_map = ndimage.distance_transform_edt(~M_anchor)

            # 划分物理盲区频带
            mask_b1 = (dist_map > 0) & (dist_map <= 3)
            mask_b2 = (dist_map > 3) & (dist_map <= 7)
            mask_b3 = (dist_map > 7)

            # --- 轨迹 1：TTO 动态推演 ---
            d_purged = compute_band_dice(pred_mask_np, target_mask, mask_purged)
            d_b1 = compute_band_dice(pred_mask_np, target_mask, mask_b1)
            d_b2 = compute_band_dice(pred_mask_np, target_mask, mask_b2)
            d_b3 = compute_band_dice(pred_mask_np, target_mask, mask_b3)

            results['purged_global'].append(d_purged)
            if not np.isnan(d_b1): results['band_1_3'].append(d_b1)
            if not np.isnan(d_b2): results['band_4_7'].append(d_b2)
            if not np.isnan(d_b3): results['band_gt_7'].append(d_b3)

            # --- 轨迹 2：Oracle 刚性平移 ---
            d_o_purged = compute_band_dice(oracle_aligned_prior_mask, target_mask, mask_purged)
            d_o_b1 = compute_band_dice(oracle_aligned_prior_mask, target_mask, mask_b1)
            d_o_b2 = compute_band_dice(oracle_aligned_prior_mask, target_mask, mask_b2)
            d_o_b3 = compute_band_dice(oracle_aligned_prior_mask, target_mask, mask_b3)

            results['oracle_global'].append(d_o_purged)
            if not np.isnan(d_o_b1): results['oracle_b1_3'].append(d_o_b1)
            if not np.isnan(d_o_b2): results['oracle_b4_7'].append(d_o_b2)
            if not np.isnan(d_o_b3): results['oracle_gt_7'].append(d_o_b3)

            print(f"[{case_name}]")
            print(f"  TTO    -> Purged: {d_purged:.3f} | B(1-3): {d_b1:.3f} | B(4-7): {d_b2:.3f} | B(>7): {d_b3:.3f}")
            print(f"  Oracle -> Purged: {d_o_purged:.3f} | B(1-3): {d_o_b1:.3f} | B(4-7): {d_o_b2:.3f} | B(>7): {d_o_b3:.3f}")

    print("\n" + "="*80)
    print(f"🔥 FINAL RESULTS: 距离场空间衰减剖析 (TTO vs Oracle) 🔥")
    print("="*80)

    print(f"【全局剔除锚点 Dice (Global Purged)】")
    print(f"   TTO    : {np.nanmean(results['purged_global']):.4f}")
    print(f"   Oracle : {np.nanmean(results['oracle_global']):.4f}\n")

    print(f"【近场平滑传导区 Dice (1 - 3 voxel)】")
    print(f"   TTO    : {np.nanmean(results['band_1_3']):.4f}")
    print(f"   Oracle : {np.nanmean(results['oracle_b1_3']):.4f}\n")

    print(f"【中场拉扯博弈区 Dice (4 - 7 voxel)】")
    print(f"   TTO    : {np.nanmean(results['band_4_7']):.4f}")
    print(f"   Oracle : {np.nanmean(results['oracle_b4_7']):.4f}\n")

    print(f"【绝对黑暗深水区 Dice ( > 7 voxel )】")
    print(f"   TTO    : {np.nanmean(results['band_gt_7']):.4f}")
    print(f"   Oracle : {np.nanmean(results['oracle_gt_7']):.4f}")
    print("="*80)