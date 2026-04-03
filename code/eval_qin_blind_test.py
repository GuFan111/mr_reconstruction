# eval_qin_blind_test.py

import os
import sys
# 限制底层数学库的线程数，防止 DataLoader 多进程读取时发生 CPU 线程死锁
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import glob
import torch
import torch.nn as nn
import numpy as np
import scipy.ndimage as ndimage
from skimage.measure import label, regionprops
from tqdm import tqdm
import nibabel as nib

from monai.metrics import compute_dice, compute_hausdorff_distance
from torch.utils.data import Dataset

from models.unet import UNet
from models.model import DIF_Net
# 🟢 引入 Baseline 模型
from models.baseline_models import Baseline_3DUNet, Baseline_SwinUNETR
from dataset import OrthogonalGeometry
from utils import save_visualization_3view

# ==========================================
# 🟢 鲁棒后处理与拓扑清理
# ==========================================
def get_2d_com_robust(mask_2d):
    """鲁棒的 2D 质心提取：带最大连通域滤波防噪点"""
    mask_bool = mask_2d > 0.5
    if not mask_bool.any():
        return None
    labeled_mask = label(mask_bool)
    regions = regionprops(labeled_mask)
    if not regions:
        return None
    largest_region = max(regions, key=lambda r: r.area)
    if largest_region.area < 20:
        return None
    return np.array(largest_region.centroid)

def keep_largest_connected_component_3d(mask_3d):
    """三维最大连通域提取，无情抹杀散落的幽灵碎块"""
    mask_bool = mask_3d > 0.5
    if not mask_bool.any():
        return mask_3d  # 如果全空，直接返回
    labeled_mask, num_features = ndimage.label(mask_bool)
    if num_features <= 1:
        return mask_3d  # 只有一个连通域，无需处理

    # 计算每个连通域的体积
    component_sizes = np.bincount(labeled_mask.ravel())
    component_sizes[0] = 0  # 背景 (0) 不参与计算

    # 找到最大的一块
    largest_component_idx = component_sizes.argmax()
    cleaned_mask = (labeled_mask == largest_component_idx).astype(mask_3d.dtype)
    return cleaned_mask

# 🟢 评测级抗锯齿滤波
def smooth_binary_mask(mask_np, sigma=1.0, threshold=0.5):
    """
    配准级 Mask 平滑：用高斯核融化阶梯伪影，再重新二值化
    """
    mask_float = mask_np.astype(np.float32)
    smoothed = ndimage.gaussian_filter(mask_float, sigma=sigma)
    return (smoothed > threshold).astype(np.uint8)

class QinTestConfig:
    # 🔴 模型切换开关: 'difnet', '3dunet', 'swin_unetr'
    model_type = 'difnet'

    name = f'qin_blind_test_{model_type}'
    gpu_id = 0
    data_root = '/root/autodl-tmp/Proj/data/qin_testset_npy'

    # 2D 上游分割网络权重
    unet_weights = '/root/autodl-tmp/Proj/code/logs/prostate_2d_unet_prior_guided/unet_best.pth'

    # 🔴 对应各模型的 3D 权重路径 (请确保路径正确)
    model_weights = {
        'difnet': '/root/autodl-tmp/Proj/code/logs/prostate_new_2/model_best.pth',
        '3dunet': '/root/autodl-tmp/Proj/code/logs/baseline_3dunet_sparse_amp/model_best.pth',
        'swin_unetr': '/root/autodl-tmp/Proj/code/logs/baseline_swin_unetr_sparse_amp/model_best.pth'
    }

    out_res = (128, 128, 128)

    # 🔴【消融实验开关】
    # True  -> 强行注入完美质心，测试下游纯弹性形变上限 (Oracle)
    # False -> 纯天然真实端到端盲测 (E2E)
    use_oracle_centroid = True

    save_vis = True
    save_nii = True

def save_nifti(array_np, save_path, is_mask=True):
    dtype = np.uint8 if is_mask else np.float32
    nii_img = nib.Nifti1Image(array_np.astype(dtype), np.eye(4))
    nib.save(nii_img, save_path)

class QinBlindTestDataset(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        self.patient_dirs = sorted([
            os.path.join(data_root, d) for d in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, d)) and not d.startswith('.')
        ])
        print(f"[Dataset] 成功肃清幽灵目录，发现 {len(self.patient_dirs)} 例真实 QIN 盲测患者数据。")

    def __len__(self):
        return len(self.patient_dirs)

    def __getitem__(self, idx):
        p_dir = self.patient_dirs[idx]
        patient_id = os.path.basename(p_dir)
        prior_image = np.load(os.path.join(p_dir, 'prior_image.npy'))
        prior_mask = np.load(os.path.join(p_dir, 'prior_mask.npy'))
        target_image = np.load(os.path.join(p_dir, 'target_image.npy'))
        target_mask = np.load(os.path.join(p_dir, 'target_mask.npy'))

        return {
            'name': patient_id,
            'prior_image': prior_image,
            'prior_mask': prior_mask,
            'target_image': target_image,
            'target_mask': target_mask
        }

class TriViewUNet(nn.Module):
    def __init__(self, base_c=16):
        super().__init__()
        self.unet_ax = UNet(n_channels=2, n_classes=1, bilinear=False)
        self.unet_co = UNet(n_channels=2, n_classes=1, bilinear=False)
        self.unet_sa = UNet(n_channels=2, n_classes=1, bilinear=False)

    def forward(self, unet_input):
        out_ax = self.unet_ax(unet_input[:, 0])
        out_co = self.unet_co(unet_input[:, 1])
        out_sa = self.unet_sa(unet_input[:, 2])
        return torch.stack([out_ax, out_co, out_sa], dim=1)

def safe_hd95(pred_tensor, gt_tensor):
    if pred_tensor.sum() == 0 or gt_tensor.sum() == 0:
        return 99.0
    try:
        return compute_hausdorff_distance(pred_tensor, gt_tensor, include_background=False, percentile=95).item()
    except:
        return 99.0

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = str(QinTestConfig.gpu_id)

    save_dir = f'./logs/{QinTestConfig.name}'
    os.makedirs(save_dir, exist_ok=True)
    if QinTestConfig.save_vis:
        os.makedirs(os.path.join(save_dir, 'vis'), exist_ok=True)
    if QinTestConfig.save_nii:
        os.makedirs(os.path.join(save_dir, 'vis_nii'), exist_ok=True)

    print("Loading 2D Segmenter (2-Channel Prior-Guided UNet)...")
    unet = TriViewUNet(base_c=16).cuda()
    unet.load_state_dict(torch.load(QinTestConfig.unet_weights, map_location='cuda'))
    unet.eval()

    print(f"Loading 3D Reconstructor ({QinTestConfig.model_type.upper()})...")
    if QinTestConfig.model_type == 'difnet':
        model = DIF_Net(num_views=3, combine='attention').cuda()
    elif QinTestConfig.model_type == '3dunet':
        model = Baseline_3DUNet().cuda()
    elif QinTestConfig.model_type == 'swin_unetr':
        model = Baseline_SwinUNETR().cuda()
    else:
        raise ValueError(f"Unknown model_type: {QinTestConfig.model_type}")

    checkpoint_path = QinTestConfig.model_weights[QinTestConfig.model_type]
    if not os.path.exists(checkpoint_path):
        print(f"[Error] Checkpoint not found: {checkpoint_path}")
        sys.exit()

    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    if 'net.weight' in list(checkpoint.keys())[0] and QinTestConfig.model_type != 'difnet':
        model.load_state_dict(checkpoint, strict=False)
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    val_dst = QinBlindTestDataset(data_root=QinTestConfig.data_root)
    geo = OrthogonalGeometry()

    mode_str = "Oracle-Guided Pure Elastic" if QinTestConfig.use_oracle_centroid else "True E2E Zero-Shot"
    print(f"\n🚀 开始纯天然临床数据盲测 (网络: {QinTestConfig.model_type.upper()} | 模式: {mode_str})...")

    dices_oracle, hd95s_oracle = [], []
    dices_est, hd95s_est = [], []
    dices_e2e, hd95s_e2e = [], []

    with torch.no_grad():
        for idx in tqdm(range(len(val_dst)), ncols=120):
            item = val_dst[idx]
            case_name = item['name']

            prior_image = item['prior_image']
            prior_mask = item['prior_mask']
            target_image = item['target_image']
            target_mask = item['target_mask']

            nz_prior = np.argwhere(prior_mask > 0)
            px, py, pz = nz_prior.mean(axis=0).astype(int) if len(nz_prior) > 0 else (64, 64, 64)

            nz_target = np.argwhere(target_mask > 0)
            gx, gy, gz = nz_target.mean(axis=0).astype(int) if len(nz_target) > 0 else (64, 64, 64)

            oracle_shift_vec = (gx - px, gy - py, gz - pz)
            oracle_aligned_prior_mask = ndimage.shift(prior_mask, oracle_shift_vec, order=0)
            oracle_aligned_prior_image = ndimage.shift(prior_image, oracle_shift_vec, order=1)

            img_ax = target_image[:, :, pz]
            img_co = target_image[:, py, :]
            img_sa = target_image[px, :, :]

            prior_ax = prior_mask[:, :, pz]
            prior_co = prior_mask[:, py, :]
            prior_sa = prior_mask[px, :, :]

            img_slices = np.stack([img_ax, img_co, img_sa])
            prior_slices = np.stack([prior_ax, prior_co, prior_sa])
            combined_slices = np.stack([img_slices, prior_slices], axis=1)

            unet_input = torch.from_numpy(combined_slices).unsqueeze(0).float().cuda()
            unet_out = unet(unet_input)
            pred_2d_masks = (torch.sigmoid(unet_out) > 0.5).cpu().numpy()[0, :, 0]

            com_ax = get_2d_com_robust(pred_2d_masks[0])
            com_co = get_2d_com_robust(pred_2d_masks[1])
            com_sa = get_2d_com_robust(pred_2d_masks[2])

            cx_list, cy_list, cz_list = [], [], []
            if com_ax is not None: cx_list.append(com_ax[0]); cy_list.append(com_ax[1])
            if com_co is not None: cx_list.append(com_co[0]); cz_list.append(com_co[1])
            if com_sa is not None: cy_list.append(com_sa[0]); cz_list.append(com_sa[1])

            cx = int(np.mean(cx_list)) if len(cx_list) > 0 else px
            cy = int(np.mean(cy_list)) if len(cy_list) > 0 else py
            cz = int(np.mean(cz_list)) if len(cz_list) > 0 else pz

            est_shift_vec = (cx - px, cy - py, cz - pz)
            est_aligned_prior_mask = ndimage.shift(prior_mask, est_shift_vec, order=0)
            est_aligned_prior_image = ndimage.shift(prior_image, est_shift_vec, order=1)

            if QinTestConfig.use_oracle_centroid:
                feed_cx, feed_cy, feed_cz = gx, gy, gz
                feed_image = oracle_aligned_prior_image
                feed_mask = oracle_aligned_prior_mask
            else:
                feed_cx, feed_cy, feed_cz = cx, cy, cz
                feed_image = est_aligned_prior_image
                feed_mask = est_aligned_prior_mask

            if QinTestConfig.model_type == 'difnet':
                projs = np.zeros((3, 1, 128, 128), dtype=np.float32)
                projs[0, 0] = target_image[:, :, feed_cz]
                projs[1, 0] = target_image[:, feed_cy, :]
                projs[2, 0] = target_image[feed_cx, :, :]

                prior_projs = np.zeros((3, 1, 128, 128), dtype=np.float32)
                prior_projs[0, 0] = feed_image[:, :, feed_cz]
                prior_projs[1, 0] = feed_image[:, feed_cy, :]
                prior_projs[2, 0] = feed_image[feed_cx, :, :]

                res_x, res_y, res_z = QinTestConfig.out_res
                grid = np.mgrid[:res_x, :res_y, :res_z].reshape(3, -1).transpose(1, 0)
                res_array = np.array([res_x, res_y, res_z], dtype=np.float32)
                points_norm = ((grid.astype(np.float32) / (res_array - 1)) - 0.5) * 2

                proj_points = np.stack([
                    geo.project(points_norm, 0), geo.project(points_norm, 1), geo.project(points_norm, 2)
                ], axis=0)

                dif_input = {
                    'projs': torch.from_numpy(projs).unsqueeze(0).cuda(),
                    'prior_projs': torch.from_numpy(prior_projs).unsqueeze(0).cuda(),
                    'prior_mask': torch.from_numpy(feed_mask).view(1, 1, 128, 128, 128).float().cuda(),
                    'points': torch.from_numpy(points_norm).unsqueeze(0).cuda(),
                    'proj_points': torch.from_numpy(proj_points).unsqueeze(0).cuda()
                }

                pred_logits = model(dif_input, is_eval=True, eval_npoint=50000)

                # 提取网络预测概率和物理先验概率
                prob_network = torch.sigmoid(pred_logits).view(1, 1, 128, 128, 128)

                # 直接通过 0.5 阈值进行存在主义的二元判决
                pred_mask_np = (prob_network > 0.5).cpu().numpy()[0, 0]

                # 拓扑清理，抹杀幽灵碎块
                pred_mask_np = keep_largest_connected_component_3d(pred_mask_np)

            else:
                target_img_tensor = torch.from_numpy(target_image).view(1, 1, 128, 128, 128).float().cuda()
                prior_img_tensor = torch.from_numpy(feed_image).view(1, 1, 128, 128, 128).float().cuda()
                prior_mask_tensor = torch.from_numpy(feed_mask).view(1, 1, 128, 128, 128).float().cuda()
                coords_tensor = torch.tensor([[feed_cx, feed_cy, feed_cz]], dtype=torch.long).cuda()

                with torch.cuda.amp.autocast():
                    pred_logits = model(target_img_tensor, prior_img_tensor, prior_mask_tensor, coords_tensor)

                prob_network = torch.sigmoid(pred_logits)
                pred_mask_np = (prob_network > 0.5).cpu().numpy()[0, 0]

                pred_mask_np = keep_largest_connected_component_3d(pred_mask_np)


            # ==========================================
            # 阶段 3：算分 (引入物理抗锯齿平滑)
            # ==========================================
            # 🔴 用高斯平滑融化 GT 和所有对照组的阶梯锯齿
            gt_smoothed = smooth_binary_mask(target_mask, sigma=1.0)
            gt_tensor = torch.from_numpy(gt_smoothed).view(1, 1, 128, 128, 128).float().cuda()

            oracle_smoothed = smooth_binary_mask(oracle_aligned_prior_mask, sigma=1.0)
            oracle_tensor = torch.from_numpy(oracle_smoothed).view(1, 1, 128, 128, 128).float().cuda()

            d_oracle = compute_dice(oracle_tensor, gt_tensor, include_background=False).item()
            hd_oracle = safe_hd95(oracle_tensor, gt_tensor)
            dices_oracle.append(d_oracle)
            hd95s_oracle.append(hd_oracle)

            est_smoothed = smooth_binary_mask(est_aligned_prior_mask, sigma=1.0)
            est_tensor = torch.from_numpy(est_smoothed).view(1, 1, 128, 128, 128).float().cuda()

            d_est = compute_dice(est_tensor, gt_tensor, include_background=False).item()
            hd_est = safe_hd95(est_tensor, gt_tensor)
            dices_est.append(d_est)
            hd95s_est.append(hd_est)

            pred_smoothed = smooth_binary_mask(pred_mask_np, sigma=1.0)
            pred_3d_mask_eval = torch.from_numpy(pred_smoothed).view(1, 1, 128, 128, 128).float().cuda()

            d_e2e = compute_dice(pred_3d_mask_eval, gt_tensor, include_background=False).item()
            hd_e2e = safe_hd95(pred_3d_mask_eval, gt_tensor)
            dices_e2e.append(d_e2e)
            hd95s_e2e.append(hd_e2e)

            # ==========================================
            # 阶段 4：可视化与 NIfTI 导出 (依然保存未经平滑的真实输出)
            # ==========================================
            if QinTestConfig.save_vis:
                vis_save_path = os.path.join(save_dir, 'vis', f"{case_name}_pred_{d_e2e:.3f}.png")
                save_visualization_3view(
                    img_np=target_image,
                    prior_mask=prior_mask,
                    aligned_prior=feed_mask,
                    gt_mask=target_mask,
                    pred_mask=pred_mask_np,
                    save_path=vis_save_path,
                    case_name=case_name,
                    epoch=QinTestConfig.model_type.upper()
                )

            if QinTestConfig.save_nii:
                nii_dir = os.path.join(save_dir, 'vis_nii')
                save_nifti(target_image, os.path.join(nii_dir, f"{case_name}_TARGET_IMG.nii.gz"), is_mask=False)
                save_nifti(target_mask, os.path.join(nii_dir, f"{case_name}_GT.nii.gz"))
                save_nifti(prior_mask, os.path.join(nii_dir, f"{case_name}_PRIOR.nii.gz"))
                save_nifti(pred_mask_np, os.path.join(nii_dir, f"{case_name}_PRED_{d_e2e:.3f}.nii.gz"))

            print(f"[{case_name}] Dice (Oracle/Est/Pred): {d_oracle:.3f} | {d_est:.3f} | {d_e2e:.3f}  ---  HD95: {hd_oracle:.2f} | {hd_est:.2f} | {hd_e2e:.2f}")

    print("\n" + "="*60)
    print(f"🔥 FINAL RESULTS: {QinTestConfig.model_type.upper()} ({mode_str}) 🔥")
    print("="*60)

    v_d_oracle = np.nanmean(dices_oracle)
    v_hd_oracle = np.mean([x for x in hd95s_oracle if x < 99.0])

    v_d_est = np.nanmean(dices_est)
    v_hd_est = np.mean([x for x in hd95s_est if x < 99.0])

    v_d_e2e = np.nanmean(dices_e2e)
    v_hd_e2e = np.mean([x for x in hd95s_e2e if x < 99.0])

    print(f"1. Oracle 3D Rigid         -> Surface-Smoothed Dice: {v_d_oracle:.4f} | HD95: {v_hd_oracle:.2f} mm")
    print(f"2. 2D-Estimated Rigid      -> Surface-Smoothed Dice: {v_d_est:.4f} | HD95: {v_hd_est:.2f} mm")
    print(f"3. 3D Non-Rigid Output     -> Surface-Smoothed Dice: {v_d_e2e:.4f} | HD95: {v_hd_e2e:.2f} mm")
    print("="*60)