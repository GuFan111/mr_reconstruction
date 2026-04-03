# eval_bspline.py

import os
import time
import numpy as np
import torch
import SimpleITK as sitk
from torch.utils.data import DataLoader
from monai.metrics import compute_hausdorff_distance, compute_dice
from tqdm import tqdm
import hashlib  # 🔴 引入哈希库用于锁定种子

from dataset import Prostate_Dataset
from utils import compute_com_error, save_visualization_3view

class Config:
    data_root = r'/root/autodl-tmp/Proj/data/prostate_158_128'
    out_res = (128, 128, 128)
    save_dir = './logs/baseline_bspline'

def bspline_sparse_registration(fixed_img_np, moving_img_np, coords, grid_physical_spacing=30.0):
    """
    核心配准引擎：带有稀疏度量掩码的 B-spline 形变
    fixed_img_np: 今天的真实形变图 (目标, Target)
    moving_img_np: 昨天的预对齐图 (先验, Prior)
    coords: [cx, cy, cz] 今天的切片相交中心
    """
    # 1. 转换为 SimpleITK Image
    fixed_image = sitk.GetImageFromArray(fixed_img_np.astype(np.float32))
    moving_image = sitk.GetImageFromArray(moving_img_np.astype(np.float32))

    # 2. 构造极其关键的度量掩码 (Metric Mask)
    # 作用：告诉配准算法只在这三张切片上计算 Loss，忽略 97% 的空气
    cx, cy, cz = int(coords[0]), int(coords[1]), int(coords[2])
    mask_np = np.zeros_like(fixed_img_np, dtype=np.uint8)
    mask_np[cx, :, :] = 1
    mask_np[:, cy, :] = 1
    mask_np[:, :, cz] = 1
    metric_mask = sitk.GetImageFromArray(mask_np)
    metric_mask.CopyInformation(fixed_image)

    # 3. 初始化 B-spline 网格
    transformDomainMeshSize = [max(1, int(dim / grid_physical_spacing)) for dim in fixed_image.GetSize()]
    initial_transform = sitk.BSplineTransformInitializer(fixed_image, transformDomainMeshSize)

    # 4. 配置配准方法
    registration_method = sitk.ImageRegistrationMethod()

    # 采用均方误差 (MeanSquares) 或 互信息 (MattesMutualInformation)
    registration_method.SetMetricAsMeanSquares()
    registration_method.SetMetricFixedMask(metric_mask) # 🔴 挂载掩码，实现稀疏配准！

    registration_method.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5,
                                             numberOfIterations=100,
                                             maximumNumberOfCorrections=5)

    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    registration_method.SetInterpolator(sitk.sitkLinear)

    # 5. 执行优化计算
    final_transform = registration_method.Execute(fixed_image, moving_image)
    return final_transform

if __name__ == '__main__':
    os.makedirs(Config.save_dir, exist_ok=True)
    os.makedirs(os.path.join(Config.save_dir, 'vis'), exist_ok=True)

    print("Loading Test Dataset...")
    val_dst = Prostate_Dataset(data_root=os.path.join(Config.data_root, 'test', 'image'),
                               label_root=os.path.join(Config.data_root, 'test', 'label'),
                               npoint=1024,
                               split='test', out_res=Config.out_res) # 🔴 修正：由于读的是 test，这里的 split 最好写成 'test' 保持语义一致
    eval_loader = DataLoader(val_dst, batch_size=1, shuffle=False)

    init_dices, recon_dices = [], []
    init_hd95s, recon_hd95s = [], []
    inference_times = []

    print("Starting B-spline Sparse Registration Baseline...")

    for i, item in enumerate(tqdm(eval_loader)):
        name = item['name'][0]

        # ==========================================
        # 🔴 强制锁定形变种子 (Deterministic Perturbation)
        # ==========================================
        # 利用病例名的哈希值作为种子，确保不同脚本测同一个病例时形变完全一致
        seed_str = str(name) + "_fixed_seed"
        seed_val = int(hashlib.md5(seed_str.encode('utf-8')).hexdigest(), 16) % (2**32)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)

        # 此时从 Dataloader 取出的 numpy 数据就是经过确定性扰动的了
        target_img_np = item['target_image'][0, 0].numpy()
        prior_img_np = item['prior_image'][0, 0].numpy()
        prior_mask_np = item['prior_mask'][0, 0].numpy()
        target_mask_np = item['target_mask'][0, 0].numpy()
        coords = item['center_coords'][0].numpy()

        # 用完种子后立刻释放，防止影响其他库的随机性
        np.random.seed()
        torch.seed()
        # ==========================================

        t_start = time.time()

        # 1. 运行优化器，获得形变场
        try:
            final_transform = bspline_sparse_registration(target_img_np, prior_img_np, coords)
        except Exception as e:
            print(f"[{name}] B-spline Optimization Failed: {e}")
            continue

        # 2. 将求得的形变场应用到 先验 Mask 上
        prior_mask_sitk = sitk.GetImageFromArray(prior_mask_np.astype(np.float32))
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(prior_mask_sitk)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(final_transform)

        pred_mask_sitk = resampler.Execute(prior_mask_sitk)
        pred_mask_np = sitk.GetArrayFromImage(pred_mask_sitk)

        # 提取 0.5 等值面恢复二进制 Mask
        pred_mask_np = (pred_mask_np > 0.5).astype(np.float32)

        inference_times.append(time.time() - t_start)

        # 3. 转换回 Tensor 算分 (与深度学习评测管线绝对对齐)
        prior_mask_tensor = torch.from_numpy(prior_mask_np).unsqueeze(0).unsqueeze(0)
        pred_mask_tensor = torch.from_numpy(pred_mask_np).unsqueeze(0).unsqueeze(0)
        gt_mask_tensor = torch.from_numpy(target_mask_np).unsqueeze(0).unsqueeze(0)

        i_dice = compute_dice(prior_mask_tensor, gt_mask_tensor, include_background=False).item()
        r_dice = compute_dice(pred_mask_tensor, gt_mask_tensor, include_background=False).item()

        try: i_hd95 = compute_hausdorff_distance(prior_mask_tensor, gt_mask_tensor, include_background=False, percentile=95).item()
        except: i_hd95 = 99.0
        try: r_hd95 = compute_hausdorff_distance(pred_mask_tensor, gt_mask_tensor, include_background=False, percentile=95).item()
        except: r_hd95 = 99.0

        init_dices.append(i_dice); recon_dices.append(r_dice)
        init_hd95s.append(i_hd95); recon_hd95s.append(r_hd95)

        print(f"  [{name}] Dice: {i_dice:.3f}->{r_dice:.3f} | HD95: {i_hd95:.2f}->{r_hd95:.2f}mm")

        if i == 0:
            vis_save_path = os.path.join(Config.save_dir, 'vis', f"bspline_{name}.png")
            save_visualization_3view(
                img_np=target_img_np, prior_mask=prior_mask_np, aligned_prior=prior_mask_np,
                gt_mask=target_mask_np, pred_mask=pred_mask_np, save_path=vis_save_path,
                case_name=name, epoch='Bspline'
            )

    print(f"\n[B-Spline Result] Dice: Init {np.mean(init_dices):.4f} -> Pred {np.mean(recon_dices):.4f} | HD95: {np.mean(init_hd95s):.2f} -> {np.mean(recon_hd95s):.2f}")
    print(f"Average Inference Time per Vol: {np.mean(inference_times):.2f} s")