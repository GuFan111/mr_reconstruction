import os
import sys
import torch
import numpy as np
import SimpleITK as sitk
import time
import nibabel as nib
import torch.nn.functional as F
# 确保安装: pip install TotalSegmentator
from totalsegmentator.python_api import totalsegmentator

from dataset import AMOS_Dataset
from models.model import DIF_Net
from utils import convert_cuda, gpu_slice_volume, PCARespiratoryDeformation

# ==========================================
#  推理配置
# ==========================================
class InferenceConfig:
    model_path = r'/root/autodl-tmp/Proj/code/logs/dif_amos_roi_v2/ep_100.pth'
    data_root = r'/root/autodl-tmp/Proj/data/amos_mri_npy'
    save_dir = r'./inference_results/fusion_recon'
    label_root = r'/root/autodl-tmp/Proj/data/amos_mri_label_npy'

    gpu_id = 0
    mid_ch = 128
    num_views = 3
    out_res = (256, 256, 128)
    combine = 'mlp'
    num_points = 200000,
    sigma = 0.05
    bbox_padding = 5  # 建议稍微留一点 padding，防止形变把器官扯出框外

# ==========================================
#  辅助函数
# ==========================================
def save_nifti(image_np, path):
    """鲁棒的 NIfTI 保存函数"""
    if torch.is_tensor(image_np):
        image_np = image_np.detach().cpu().numpy()
    img_cleaned = np.squeeze(image_np)
    if img_cleaned.ndim == 3:
        out_img = img_cleaned.transpose(2, 1, 0)
    else:
        out_img = img_cleaned
    img_itk = sitk.GetImageFromArray(out_img)
    sitk.WriteImage(img_itk, path)
    print(f"Saved: {path}")

def get_roi_mask_ts(image_np):
    """使用 TotalSegmentator 获取 Mask"""
    affine = np.eye(4)
    img_nib = nib.Nifti1Image(image_np.transpose(2, 1, 0), affine)
    mask_img = totalsegmentator(img_nib, task="total_mr", ml=True)
    return mask_img.get_fdata().transpose(2, 1, 0)

# ==========================================
#  主程序
# ==========================================
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = str(InferenceConfig.gpu_id)
    device = torch.device('cuda')
    os.makedirs(InferenceConfig.save_dir, exist_ok=True)

    res_x, res_y, res_z = InferenceConfig.out_res
    print(f"[1/6] Loading Model...")
    model = DIF_Net(num_views=InferenceConfig.num_views, combine=InferenceConfig.combine, mid_ch=InferenceConfig.mid_ch).to(device)
    model.load_state_dict(torch.load(InferenceConfig.model_path, map_location=device))
    model.eval()

    # 2. 加载数据
    val_dst = AMOS_Dataset(
        data_root=InferenceConfig.data_root,
        label_root=InferenceConfig.label_root,
        split='eval',
        npoint=InferenceConfig.num_points,
        out_res=InferenceConfig.out_res
    )
    item = val_dst[0]
    sample_name = item['name']
    prior_vol_np = item['image'] # [1, X, Y, Z]
    prior_vol = torch.from_numpy(prior_vol_np).unsqueeze(0).to(device)

    # 3. 提取 BBox ROI
    print("[2/6] Extracting Bounding Box via TotalSegmentator...")
    raw_mask = get_roi_mask_ts(prior_vol_np[0])
    mask_indices = torch.nonzero(torch.from_numpy(raw_mask).to(device))

    if mask_indices.shape[0] > 0:
        mins = mask_indices.min(dim=0)[0]
        maxs = mask_indices.max(dim=0)[0]

        # 修复切片越界 Bug: 必须 +1 才能包含 max 索引本身
        x_start = max(0, mins[0].item() - InferenceConfig.bbox_padding)
        x_stop  = min(res_x, maxs[0].item() + 1 + InferenceConfig.bbox_padding)
        y_start = max(0, mins[1].item() - InferenceConfig.bbox_padding)
        y_stop  = min(res_y, maxs[1].item() + 1 + InferenceConfig.bbox_padding)
        z_start = max(0, mins[2].item() - InferenceConfig.bbox_padding)
        z_stop  = min(res_z, maxs[2].item() + 1 + InferenceConfig.bbox_padding)

        # 记录局部网格的三维尺寸
        roi_shape = (x_stop - x_start, y_stop - y_start, z_stop - z_start)

        xx, yy, zz = torch.meshgrid(
            torch.arange(x_start, x_stop, device=device),
            torch.arange(y_start, y_stop, device=device),
            torch.arange(z_start, z_stop, device=device),
            indexing='ij'
        )
        roi_indices = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
    else:
        roi_shape = InferenceConfig.out_res
        roi_indices = torch.nonzero(torch.ones(InferenceConfig.out_res, device=device))

    # 4. 模拟实时场景
    print("[3/6] Simulating real-time deformation...")
    deformer = PCARespiratoryDeformation(grid_size=4, amp_xyz=(0.01, 0.04, 0.15)).cuda()
    with torch.no_grad():
        rt_vol = deformer(prior_vol)
        projs = gpu_slice_volume(rt_vol)

        res_tensor = torch.tensor([res_x, res_y, res_z], device=device).float()
        points_norm = ((roi_indices.float() / (res_tensor - 1)) - 0.5) * 2
        points_tensor = points_norm.unsqueeze(0).to(device)
        proj_pts_list = [points_tensor[..., [0, 1]], points_tensor[..., [0, 2]], points_tensor[..., [1, 2]]]
        proj_pts_tensor = torch.stack(proj_pts_list, dim=1)

    # 将 roi_shape 塞进 input_data，告知模型当前处理的网格长宽
    input_data = {
        'prior': prior_vol,
        'projs': projs,
        'points': points_tensor,
        'proj_points': proj_pts_tensor,
        'grid_shape': roi_shape
    }

    # 5. 推理
    print(f"[4/6] Reconstructing {roi_indices.shape[0]} points in BBox...")
    with torch.no_grad():
        pred_roi_vals, _ = model(input_data, is_eval=True, eval_npoint=500000)

    # 6. 背景无缝融合
    print("[5/6] Finalizing with Seamless Background Fusion...")

    # 使用 Prior 图像的副本作为底图
    full_recon_np = prior_vol_np[0].copy()

    # 准备 ROI 数据
    pred_vals_np = pred_roi_vals.squeeze().cpu().numpy()
    idx_np = roi_indices.cpu().numpy()

    # 🟢 纯净回填：直接将预测的医学纹理放回空间，不再加任何光度偏移
    full_recon_np[idx_np[:, 0], idx_np[:, 1], idx_np[:, 2]] = pred_vals_np

    # 保存图像
    save_nifti(prior_vol, os.path.join(InferenceConfig.save_dir, f'{sample_name}_0_Prior.nii.gz'))
    save_nifti(rt_vol,    os.path.join(InferenceConfig.save_dir, f'{sample_name}_1_GT.nii.gz'))
    save_nifti(full_recon_np, os.path.join(InferenceConfig.save_dir, f'{sample_name}_2_Recon_Fusion.nii.gz'))

    print("\n[6/6] All Done! Check the 'Fusion' NIfTI file.")