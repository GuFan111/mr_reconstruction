# eval_qin_blind_test_bspline.py

import os
# 限制底层数学库的线程数，防止多进程评估时的 CPU 资源抢占
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import time
import numpy as np
import torch
import SimpleITK as sitk
from torch.utils.data import Dataset
from monai.metrics import compute_hausdorff_distance, compute_dice
import scipy.ndimage as ndimage
from tqdm import tqdm
import nibabel as nib

from utils import save_visualization_3view

# ==========================================
# 🟢 测试配置区域
# ==========================================
class QinBsplineConfig:
    data_root = r'/root/autodl-tmp/Proj/data/qin_testset_npy'
    out_res = (128, 128, 128)
    save_dir = './logs/qin_blind_test_bspline'
    save_vis = True
    save_nii = True

def save_nifti(array_np, save_path, is_mask=True):
    dtype = np.uint8 if is_mask else np.float32
    nii_img = nib.Nifti1Image(array_np.astype(dtype), np.eye(4))
    nib.save(nii_img, save_path)

# ==========================================
# 🟢 QIN 盲测数据集读取 (同 DL 脚本)
# ==========================================
class QinBlindTestDataset(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        self.patient_dirs = sorted([os.path.join(data_root, d) for d in os.listdir(data_root)
                                    if os.path.isdir(os.path.join(data_root, d))])
        print(f"[Dataset] 发现 {len(self.patient_dirs)} 例 QIN 盲测患者数据。")

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

# ==========================================
# 🟢 B-spline 稀疏配准核心引擎
# ==========================================
def bspline_sparse_registration(fixed_img_np, moving_img_np, coords, grid_physical_spacing=30.0):
    """
    核心配准引擎：带有稀疏度量掩码的 B-spline 形变
    fixed_img_np: 今天的真实形变图 (目标, Target)
    moving_img_np: 昨天的预对齐图 (先验, Prior)
    coords: [cx, cy, cz] 今天的切片相交中心
    """
    fixed_image = sitk.GetImageFromArray(fixed_img_np.astype(np.float32))
    moving_image = sitk.GetImageFromArray(moving_img_np.astype(np.float32))

    # 构造极其关键的度量掩码 (Metric Mask)
    cx, cy, cz = int(coords[0]), int(coords[1]), int(coords[2])
    mask_np = np.zeros_like(fixed_img_np, dtype=np.uint8)
    mask_np[cx, :, :] = 1
    mask_np[:, cy, :] = 1
    mask_np[:, :, cz] = 1
    metric_mask = sitk.GetImageFromArray(mask_np)
    metric_mask.CopyInformation(fixed_image)

    # 初始化 B-spline 网格
    transformDomainMeshSize = [max(1, int(dim / grid_physical_spacing)) for dim in fixed_image.GetSize()]
    initial_transform = sitk.BSplineTransformInitializer(fixed_image, transformDomainMeshSize)

    # 配置配准方法
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMeanSquares()
    registration_method.SetMetricFixedMask(metric_mask) # 🔴 挂载掩码，实现稀疏配准！

    registration_method.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5,
                                             numberOfIterations=100,
                                             maximumNumberOfCorrections=5)

    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    registration_method.SetInterpolator(sitk.sitkLinear)

    # 执行优化计算
    final_transform = registration_method.Execute(fixed_image, moving_image)
    return final_transform

def safe_hd95(pred_tensor, gt_tensor):
    if pred_tensor.sum() == 0 or gt_tensor.sum() == 0:
        return 99.0
    try:
        return compute_hausdorff_distance(pred_tensor, gt_tensor, include_background=False, percentile=95).item()
    except:
        return 99.0


if __name__ == '__main__':
    os.makedirs(QinBsplineConfig.save_dir, exist_ok=True)
    if QinBsplineConfig.save_vis:
        os.makedirs(os.path.join(QinBsplineConfig.save_dir, 'vis'), exist_ok=True)
    if QinBsplineConfig.save_nii:
        os.makedirs(os.path.join(QinBsplineConfig.save_dir, 'vis_nii'), exist_ok=True)

    val_dst = QinBlindTestDataset(data_root=QinBsplineConfig.data_root)

    print("\n🚀 开始执行 B-spline QIN 盲测 (Oracle-Guided Pure Elastic)...")

    dices_oracle, hd95s_oracle = [], []
    dices_bspline, hd95s_bspline = [], []
    inference_times = []

    for idx in tqdm(range(len(val_dst)), ncols=100):
        item = val_dst[idx]
        name = item['name']

        prior_image = item['prior_image']
        prior_mask = item['prior_mask']
        target_image = item['target_image']
        target_mask = item['target_mask']

        # ==========================================
        # 1. 理论上限基准 (Oracle Centroid Shift)
        # ==========================================
        nz_prior = np.argwhere(prior_mask > 0)
        px, py, pz = nz_prior.mean(axis=0).astype(int) if len(nz_prior) > 0 else (64, 64, 64)

        nz_target = np.argwhere(target_mask > 0)
        gx, gy, gz = nz_target.mean(axis=0).astype(int) if len(nz_target) > 0 else (64, 64, 64)

        oracle_shift_vec = (gx - px, gy - py, gz - pz)
        oracle_aligned_prior_mask = ndimage.shift(prior_mask, oracle_shift_vec, order=0)
        oracle_aligned_prior_image = ndimage.shift(prior_image, oracle_shift_vec, order=1)

        # ==========================================
        # 2. 运行 B-spline 稀疏优化
        # ==========================================
        t_start = time.time()

        try:
            # 传入 Target 图，对齐后的 Prior 图，以及 Target 真实质心作为掩码交叉点
            final_transform = bspline_sparse_registration(target_image, oracle_aligned_prior_image, coords=(gx, gy, gz))
        except Exception as e:
            print(f"[{name}] B-spline Optimization Failed: {e}")
            continue

        # 将求得的形变场应用到 先验 Mask (已对齐) 上
        prior_mask_sitk = sitk.GetImageFromArray(oracle_aligned_prior_mask.astype(np.float32))
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(prior_mask_sitk)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(final_transform)

        pred_mask_sitk = resampler.Execute(prior_mask_sitk)
        pred_mask_np = sitk.GetArrayFromImage(pred_mask_sitk)
        pred_mask_np = (pred_mask_np > 0.5).astype(np.float32)

        inference_times.append(time.time() - t_start)

        # ==========================================
        # 3. 算分与统计
        # ==========================================
        gt_tensor = torch.from_numpy(target_mask).unsqueeze(0).unsqueeze(0).float()

        oracle_tensor = torch.from_numpy(oracle_aligned_prior_mask).unsqueeze(0).unsqueeze(0).float()
        d_oracle = compute_dice(oracle_tensor, gt_tensor, include_background=False).item()
        hd_oracle = safe_hd95(oracle_tensor, gt_tensor)
        dices_oracle.append(d_oracle)
        hd95s_oracle.append(hd_oracle)

        pred_tensor = torch.from_numpy(pred_mask_np).unsqueeze(0).unsqueeze(0).float()
        d_bspline = compute_dice(pred_tensor, gt_tensor, include_background=False).item()
        hd_bspline = safe_hd95(pred_tensor, gt_tensor)
        dices_bspline.append(d_bspline)
        hd95s_bspline.append(hd_bspline)

        print(f"  [{name}] Oracle: {d_oracle:.3f} -> B-spline: {d_bspline:.3f} | Time: {inference_times[-1]:.2f}s")

        # ==========================================
        # 4. 导出可视化与 NIfTI
        # ==========================================
        if QinBsplineConfig.save_vis:
            vis_save_path = os.path.join(QinBsplineConfig.save_dir, 'vis', f"{name}_pred_{d_bspline:.3f}.png")
            save_visualization_3view(
                img_np=target_image,
                prior_mask=prior_mask,
                aligned_prior=oracle_aligned_prior_mask,
                gt_mask=target_mask,
                pred_mask=pred_mask_np,
                save_path=vis_save_path,
                case_name=name,
                epoch="BSPLINE"
            )

        if QinBsplineConfig.save_nii:
            nii_dir = os.path.join(QinBsplineConfig.save_dir, 'vis_nii')
            save_nifti(target_image, os.path.join(nii_dir, f"{name}_TARGET_IMG.nii.gz"), is_mask=False)
            save_nifti(target_mask, os.path.join(nii_dir, f"{name}_GT.nii.gz"))
            save_nifti(oracle_aligned_prior_mask, os.path.join(nii_dir, f"{name}_ORACLE_PRIOR.nii.gz"))
            save_nifti(pred_mask_np, os.path.join(nii_dir, f"{name}_PRED_BSPLINE_{d_bspline:.3f}.nii.gz"))

    # ==========================================
    # 5. 打印最终结果
    # ==========================================
    print("\n" + "="*60)
    print("🔥 FINAL RESULTS: B-SPLINE (QIN Blind Test) 🔥")
    print("="*60)
    v_d_oracle = np.nanmean(dices_oracle)
    v_hd_oracle = np.mean([x for x in hd95s_oracle if x < 99.0])

    v_d_bspline = np.nanmean(dices_bspline)
    v_hd_bspline = np.mean([x for x in hd95s_bspline if x < 99.0])

    print(f"1. Oracle 3D Rigid         -> Dice: {v_d_oracle:.4f} | HD95: {v_hd_oracle:.2f} mm")
    print(f"2. B-spline Non-Rigid      -> Dice: {v_d_bspline:.4f} | HD95: {v_hd_bspline:.2f} mm")
    print(f"3. Avg Inference Time      -> {np.mean(inference_times):.2f} Seconds / Volume")
    print("="*60)