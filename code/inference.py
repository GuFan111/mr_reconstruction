# inference.py

import os
import sys
import torch
import numpy as np
import SimpleITK as sitk
import time
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataset import BraTS_Dataset
from models.model import DIF_Net
from utils import convert_cuda, gpu_slice_volume, ElasticDeformation



# ==========================================
#  推理配置
# ==========================================
class InferenceConfig:
    # 模型路径
    model_path = r'/root/autodl-tmp/Proj/code/logs/dif_brats_prior/ep_1.pth'
    # 数据路径 (验证集)
    data_root = r'/root/autodl-tmp/Proj/data/BraTS_128_Ready'
    # 输出保存路径
    save_dir = r'./inference_results/ep_250'

    gpu_id = 0
    mid_ch = 128
    num_views = 3
    out_res = 128
    combine = 'attention'
    sigma = 0.1 


def inference_accelerated(model, data, chunk_size=200000):
    model.eval()
    with torch.no_grad():
        pred = model(data, is_eval=True, eval_npoint=chunk_size)
    return pred

def save_nifti(image_np, path):
    """保存 numpy 数组为 nifti"""
    if image_np.ndim == 3:
        img_itk = sitk.GetImageFromArray(image_np)
    else:
        img_itk = sitk.GetImageFromArray(image_np.squeeze())
    sitk.WriteImage(img_itk, path)
    print(f"Saved: {path}")

# ==========================================
#  主程序
# ==========================================
if __name__ == '__main__':
    # 1. 环境设置
    os.environ['CUDA_VISIBLE_DEVICES'] = str(InferenceConfig.gpu_id)
    device = torch.device('cuda')
    os.makedirs(InferenceConfig.save_dir, exist_ok=True)
    
    print(f"[1/5] Loading model from {InferenceConfig.model_path} ...")
    
    # 2. 加载模型
    # 必须与训练时的参数一致
    model = DIF_Net(
        num_views=InferenceConfig.num_views, 
        combine=InferenceConfig.combine, 
        mid_ch=InferenceConfig.mid_ch
    ).to(device)
    
    # 加载权重
    try:
        ckpt = torch.load(InferenceConfig.model_path, map_location=device)
        model.load_state_dict(ckpt)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    model.eval()

    # 3. 准备数据
    print("[2/5] Loading test data...")
    val_dst = BraTS_Dataset(
        data_root=InferenceConfig.data_root, 
        split='eval', 
        npoint=10000, 
        out_res=InferenceConfig.out_res, 
        preload=False
    )
    
    # 随机取一个样本，或者指定 index
    sample_idx = 0 
    item = val_dst[sample_idx] 
    sample_name = item['name']
    print(f"Target Sample: {sample_name}")
    
    # 转 Tensor 并上 GPU
    prior_vol = torch.from_numpy(item['image']).unsqueeze(0).to(device)
    
    # 4. 现场制造"测试题" (模拟实时场景)
    print("[3/5] Simulating real-time deformation...")
    # 初始化形变器
    deformer = ElasticDeformation(grid_size=8, sigma=InferenceConfig.sigma).to(device)
    
    with torch.no_grad():
        # A. 生成 Ground Truth (变形后的实时图像)
        rt_vol = deformer(prior_vol) # [1, 1, D, H, W]
        
        # B. 生成实时切片 (Projections)
        projs = gpu_slice_volume(rt_vol) # [1, 3, 1, H, W]
        
        # C. 准备全图采样点 (用于生成完整 3D 结果)
        res = InferenceConfig.out_res
        grid = np.mgrid[:res, :res, :res].astype(np.float32) / (res - 1)
        grid = grid.reshape(3, -1).transpose(1, 0) # [N, 3]
        points_norm = (grid - 0.5) * 2
        
        points_tensor = torch.from_numpy(points_norm).unsqueeze(0).to(device) # [1, N, 3]
        
        # D. 计算投影坐标
        # 简单正交投影逻辑 (Axial, Coronal, Sagittal)
        # View 0 (Axial): x,y -> [0, 1]
        # View 1 (Coronal): x,z -> [0, 2]
        # View 2 (Sagittal): y,z -> [1, 2]
        proj_pts_list = [
            points_tensor[..., [0, 1]], 
            points_tensor[..., [0, 2]], 
            points_tensor[..., [1, 2]]
        ]
        proj_pts_tensor = torch.stack(proj_pts_list, dim=1) # [1, 3, N, 2]

    # 构造输入字典
    input_data = {
        'prior': prior_vol,
        'projs': projs,
        'points': points_tensor,
        'proj_points': proj_pts_tensor
    }

    # 5. 执行推理
    print("[4/5] Running inference...")
    start_time = time.time()
    
    with torch.no_grad():
        # 使用模型的 eval 模式自动分块推理
        pred = model(input_data, is_eval=True, eval_npoint=100000)
        
    end_time = time.time()
    print(f"Inference done in {end_time - start_time:.4f}s")

    # 6. 保存结果
    print("[5/5] Saving results...")
    
    # Reshape
    pred_vol_np = pred[0, 0].cpu().numpy().reshape(res, res, res)
    prior_vol_np = prior_vol[0, 0].cpu().numpy()
    rt_vol_np = rt_vol[0, 0].cpu().numpy()
    
    # 保存
    save_nifti(prior_vol_np, os.path.join(InferenceConfig.save_dir, f'{sample_name}_0_Prior.nii.gz'))
    save_nifti(rt_vol_np,    os.path.join(InferenceConfig.save_dir, f'{sample_name}_1_GT_Realtime.nii.gz'))
    save_nifti(pred_vol_np,  os.path.join(InferenceConfig.save_dir, f'{sample_name}_2_Recon.nii.gz'))
    
    # 计算简单的误差图并保存
    diff_vol_np = np.abs(rt_vol_np - pred_vol_np)
    save_nifti(diff_vol_np,  os.path.join(InferenceConfig.save_dir, f'{sample_name}_3_Diff.nii.gz'))

    print("\nAll Done! Please check folder:", InferenceConfig.save_dir)