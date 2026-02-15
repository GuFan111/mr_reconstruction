import os
os.environ['OMP_NUM_THREADS'] = '1'
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset_seg import AMOS_Seg_Dataset
from models.mednext.create_mednext_v1 import create_mednext_v1

# --- 新增引用 ---
from monai.inferers import sliding_window_inference
from monai.transforms import KeepLargestConnectedComponent
from monai.data import decollate_batch

import scipy.ndimage as ndimage

# ================= 配置区域 =================
class Config:
    img_root = '/root/autodl-tmp/Proj/data/amos_mri_npy'
    label_root = '/root/autodl-tmp/Proj/data/amos_mri_label_npy'
    
    # 填入你训练好的最佳模型路径
    checkpoint_path = './logs/mednext_seg_v1/best_metric_model.pth' 
    
    num_classes = 16
    crop_size = (160, 160, 96) # 必须与训练时的 patch size 一致

def show_slices(img_np, gt_np, pred_np=None, save_name='debug_vis_fixed.png'):
    """
    修正后的可视化：正确对应 Axial, Coronal, Sagittal
    输入形状: [X, Y, Z] (RAS)
    """
    w, h, d = img_np.shape
    sx, sy, sz = w//2, h//2, d//2
    
    # --- 修正切片逻辑与旋转 ---
    # 1. Axial (轴状面): 切 Z 轴，看 XY 平面。
    #    通常需要旋转 90 度才能正过来 (Anterior在上, Right在左)
    slice_ax_img = np.rot90(img_np[:, :, sz]) 
    slice_ax_gt  = np.rot90(gt_np[:, :, sz])
    
    # 2. Coronal (冠状面): 切 Y 轴，看 XZ 平面。
    #    通常也需要旋转 90 度 (Superior在上)
    slice_co_img = np.rot90(img_np[:, sy, :])
    slice_co_gt  = np.rot90(gt_np[:, sy, :])
    
    # 3. Sagittal (矢状面): 切 X 轴，看 YZ 平面。
    #    通常也需要旋转 90 度 (Superior在上)
    slice_sa_img = np.rot90(img_np[sx, :, :])
    slice_sa_gt  = np.rot90(gt_np[sx, :, :])

    slices_img = [slice_ax_img, slice_co_img, slice_sa_img]
    slices_gt  = [slice_ax_gt,  slice_co_gt,  slice_sa_gt]
    
    if pred_np is not None:
        slice_ax_pred = np.rot90(pred_np[:, :, sz])
        slice_co_pred = np.rot90(pred_np[:, sy, :])
        slice_sa_pred = np.rot90(pred_np[sx, :, :])
        slices_pred = [slice_ax_pred, slice_co_pred, slice_sa_pred]
    
    # 绘图配置
    rows = 3 if pred_np is not None else 2
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    
    titles = ['Axial (XY)', 'Coronal (XZ)', 'Sagittal (YZ)']
    
    for i in range(3):
        # 1. Image
        ax = axes[0, i]
        ax.imshow(slices_img[i], cmap='gray')
        ax.set_title(f"{titles[i]} - Image", fontsize=12)
        ax.axis('off')
        
        # 2. GT Overlay
        ax = axes[1, i]
        ax.imshow(slices_img[i], cmap='gray', alpha=0.6)
        masked_gt = np.ma.masked_where(slices_gt[i] == 0, slices_gt[i])
        ax.imshow(masked_gt, cmap='tab20', alpha=0.7, vmin=0, vmax=15)
        ax.set_title(f"{titles[i]} - Ground Truth", fontsize=12)
        ax.axis('off')

        # 3. Prediction Overlay
        if pred_np is not None:
            ax = axes[2, i]
            ax.imshow(slices_img[i], cmap='gray', alpha=0.6)
            masked_pred = np.ma.masked_where(slices_pred[i] == 0, slices_pred[i])
            ax.imshow(masked_pred, cmap='tab20', alpha=0.7, vmin=0, vmax=15)
            ax.set_title(f"{titles[i]} - Prediction", fontsize=12)
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_name)
    print(f"Saved visualization to {save_name}")
    plt.close()


def post_process_gap_closing(pred_numpy, kernel_size=3):
    """
    针对“断裂”问题的形态学修复
    """
    # 结果容器
    result = np.zeros_like(pred_numpy)
    
    # 🟢 修复核心：自动获取输入数据的维度 (ndim)
    # 如果输入是 [D, H, W]，ndim=3；如果多了个 batch 维度 [1, D, H, W]，ndim=4
    ndim = pred_numpy.ndim 
    
    # 🟢 动态生成结构元素，保证维度永远匹配
    struct = ndimage.generate_binary_structure(ndim, 1) 
    
    # 获取图中出现的所有类别 (跳过背景 0)
    classes = np.unique(pred_numpy)
    classes = classes[classes != 0]
    
    if len(classes) == 0:
        return pred_numpy
        
    for c in classes:
        # 1. 提取当前器官的二值 Mask
        binary_mask = (pred_numpy == c)
        
        # 2. 执行闭运算
        try:
            closed_mask = ndimage.binary_closing(binary_mask, structure=struct, iterations=1)
            result[closed_mask] = c
        except RuntimeError as e:
            print(f"⚠️ Morphology error for class {c}: {e}")
            result[binary_mask] = c # 出错就保持原样
            
    return result

def fast_roi_inference(inputs, model, input_size=(160, 160, 96)):
    """
    输入: [B, C, D, H, W] 原始大图
    输出: [B, C, D, H, W] 原始尺寸的 Mask
    """
    # 1. 记录原始尺寸
    original_size = inputs.shape[2:] 
    
    # 2. 极速降采样 (Downsample)
    # mode='area' 或 'trilinear' 速度很快
    # input_size 越小越快，建议尝试 (64, 128, 128) 甚至 (48, 96, 96)
    inputs_small = F.interpolate(inputs, size=input_size, mode='area')
    
    # 3. 开启半精度 (FP16) 加速推理
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            outputs_small = model(inputs_small)
            
    # 4. 极速上采样 (Upsample) 回原尺寸
    # 此时不需要 softmax，直接插值 logits 甚至更快
    outputs_large = F.interpolate(outputs_small, size=original_size, mode='trilinear', align_corners=False)
    
    # 5. 二值化 (在 GPU 上完成，不要转 CPU!)
    pred_mask = torch.argmax(outputs_large, dim=1, keepdim=True)
    
    return pred_mask

# ================= 主逻辑 =================
def main():
    # 定义后处理：保留最大连通域 (解决"多预测几个小的"问题)
    # 注意：post_process 需要作用在 One-Hot 格式或者是 [C, D, H, W] 格式上
    post_process = KeepLargestConnectedComponent(applied_labels=None, is_onehot=True)

    val_ds = AMOS_Seg_Dataset(Config.img_root, Config.label_root, split='val')
    # 建议加上 target_id 过滤，固定看某一个病人，方便对比
    # val_ds.data_list = [item for item in val_ds.data_list if "0507" in item['image']] 
    
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False) # Debug 时建议 shuffle=False
    
    print(">>> Loading one sample from validation set...")
    batch = next(iter(val_loader))
    image = batch['image'] 
    label = batch['label'] 
    
    # ---------------- 推理核心修改区域 ----------------
    pred_np = None # 初始化为 None
    
    if Config.checkpoint_path and os.path.exists(Config.checkpoint_path):
        print(f">>> Loading model from {Config.checkpoint_path}...")
        model = create_mednext_v1(
            num_input_channels=1,
            num_classes=Config.num_classes,
            model_id='S',
            kernel_size=3,
            deep_supervision=False 
        ).cuda()
        
        # 加载权重
        state_dict = torch.load(Config.checkpoint_path)
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        
        print(">>> Running Sliding Window Inference & Post-Processing...")
        with torch.no_grad():
            img_cuda = image.cuda()
            
            # # --- 1. 滑动窗口推理 ---
            # outputs = sliding_window_inference(
            #     img_cuda, 
            #     roi_size=Config.crop_size, 
            #     sw_batch_size=4, 
            #     predictor=model, 
            #     overlap=0.5, 
            #     mode='gaussian' 
            # )
            
            # if isinstance(outputs, (list, tuple)): outputs = outputs[0]

            # # --- 2. 准备后处理数据 ---
            # # 转 One-Hot: [B, num_classes, D, H, W]
            # pred_idx = torch.argmax(outputs, dim=1, keepdim=True)
            # pred_onehot = torch.nn.functional.one_hot(pred_idx.squeeze(1), Config.num_classes).permute(0, 4, 1, 2, 3)
            
            # # --- 3. 最大连通域后处理 (MONAI) ---
            # # 解决假阳性小块
            # pred_post_list = decollate_batch(pred_onehot)
            # pred_post_processed = post_process(pred_post_list[0])
            
            # # --- 4. 转回索引并转为 Numpy ---
            # # [C, D, H, W] -> argmax -> [D, H, W] (Tensor)
            # pred_final_tensor = torch.argmax(pred_post_processed, dim=0)
            
            # # 🟢 关键修正步骤：先转 CPU Numpy，赋值给 pred_np
            # pred_np = pred_final_tensor.cpu().numpy()
            
            # # --- 5. 形态学闭运算 (Custom) ---
            # # 解决"断裂"问题。此时 pred_np 已经是 Numpy 数组了，不会报错
            # pred_np = post_process_gap_closing(pred_np, kernel_size=3)

            # 注意：fast_roi_inference 返回的就是 [B, 1, D, H, W] 的 mask 索引
            pred_mask_tensor = fast_roi_inference(img_cuda, model, input_size=(160, 160, 96))
            
            # 因为函数内部已经做过 argmax 了，所以这里直接转 numpy 即可
            # [B, 1, D, H, W] -> [D, H, W]
            pred_np = pred_mask_tensor[0, 0].cpu().numpy()

    # ---------------- 可视化 ----------------
    # 只有当 pred_np 成功生成时才画图，避免 None 报错
    if pred_np is not None:
        img_np = image[0, 0].numpy()
        gt_np = label[0, 0].numpy()
        show_slices(img_np, gt_np, pred_np, save_name='debug_vis_optimized.png')
    else:
        print("❌ Model inference failed or checkpoint not found.")

if __name__ == '__main__':
    main()