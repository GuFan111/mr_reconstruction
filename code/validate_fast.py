import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
from monai.metrics import DiceMetric
from monai.utils import set_determinism
from dataset_seg import AMOS_Seg_Dataset
from models.mednext.create_mednext_v1 import create_mednext_v1

# ================= 配置 =================
class Config:
    img_root = '/root/autodl-tmp/Proj/data/amos_mri_npy'
    label_root = '/root/autodl-tmp/Proj/data/amos_mri_label_npy'
    checkpoint_path = './logs/mednext_seg_v1_finetune/best_finetuned_model.pth'
    
    num_classes = 16
    # 🔴 核心：推理尺寸 (必须与训练 Patch Size 一致: X, Y, Z)
    infer_size = (160, 160, 96) 

# ================= 极速推理管道 (复用 measure_speed 的逻辑) =================
class FastROIPipeline:
    def __init__(self, model, input_size):
        self.model = model
        self.input_size = input_size # (160, 160, 96)
        
    def gpu_morphology_closing(self, mask_tensor, kernel_size=5):
        """ GPU 形态学闭运算 """
        pad = kernel_size // 2
        # 1. 膨胀
        dilated = F.max_pool3d(mask_tensor, kernel_size=kernel_size, stride=1, padding=pad)
        # 2. 腐蚀
        closed = -F.max_pool3d(-dilated, kernel_size=kernel_size, stride=1, padding=pad)
        return closed

    def predict(self, inputs):
        """
        输入: [B, 1, D, H, W] 原始图像
        输出: [B, 1, D, H, W] 预测 Mask (Index格式 0-15)
        """
        original_size = inputs.shape[2:]
        
        # 1. 降采样
        inputs_small = F.interpolate(inputs, size=self.input_size, mode='area')
        
        # 2. 推理
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                logits_small = self.model(inputs_small)
        
        # 3. 上采样 (logits 插值更平滑)
        logits_large = F.interpolate(logits_small, size=original_size, mode='trilinear', align_corners=False)
        
        # 4. 生成 Mask
        pred_mask = torch.argmax(logits_large, dim=1, keepdim=True).float()
        
        # 5. 后处理
        final_mask = self.gpu_morphology_closing(pred_mask, kernel_size=5)
        
        return final_mask

# ================= 主函数 =================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 数据加载
    val_ds = AMOS_Seg_Dataset(Config.img_root, Config.label_root, split='val')
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)
    
    print(f"Dataset Size: {len(val_ds)}")
    
    # 2. 模型加载
    print(f"Loading model from {Config.checkpoint_path}...")
    model = create_mednext_v1(
        num_input_channels=1,
        num_classes=Config.num_classes,
        model_id='S',
        kernel_size=3,
        deep_supervision=False
    ).to(device)
    
    checkpoint = torch.load(Config.checkpoint_path, map_location=device)
    state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # 3. 初始化管道
    pipeline = FastROIPipeline(model, input_size=Config.infer_size)
    
    # 4. 评估指标
    # include_background=False: 不计算背景的 Dice (通常背景 Dice 很高，会虚高分数)
    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
    
    print(">>> Starting Fast Validation...")
    
    with torch.no_grad():
        for batch in tqdm(val_loader):
            inputs, labels = batch["image"].to(device), batch["label"].to(device)
            
            # A. 极速推理
            pred_mask = pipeline.predict(inputs) # [B, 1, D, H, W]
            
            # B. 转换 One-Hot (用于计算 Dice)
            # labels: [B, 1, D, H, W] -> [B, C, D, H, W]
            labels_oh = torch.nn.functional.one_hot(labels.long().squeeze(1), Config.num_classes).permute(0, 4, 1, 2, 3)
            
            # preds: [B, 1, D, H, W] -> [B, C, D, H, W]
            preds_oh = torch.nn.functional.one_hot(pred_mask.long().squeeze(1), Config.num_classes).permute(0, 4, 1, 2, 3)
            
            # C. 计算 Batch Dice
            dice_metric(y_pred=preds_oh, y=labels_oh)
            
    # 5. 汇总结果
    # aggregate 返回的是 [num_classes] 的 tensor
    metric_per_class = dice_metric.aggregate()
    mean_dice = torch.mean(metric_per_class).item()
    
    dice_metric.reset()
    
    print("\n" + "="*40)
    print(f"🚀 Fast Inference Dice Score (Mean): {mean_dice:.4f}")
    print("="*40)
    
    # 打印详细器官分数
    organ_names = [
        "Spleen", "R.Kidney", "L.Kidney", "Gallbladder", "Esophagus", 
        "Liver", "Stomach", "Aorta", "IVC", "Pancreas", 
        "R.Adrenal", "L.Adrenal", "Duodenum", "Bladder", "Prostate/Uterus"
    ]
    
    print(f"{'Organ Name':<15} | {'Dice':<8}")
    print("-" * 26)
    for i, name in enumerate(organ_names):
        # metric_per_class 0对应背景(如果我们设了include_background=True)，但这里是False
        # 如果 include_background=False，metric_per_class[0] 就是第一个器官(Spleen)
        score = metric_per_class[i].item()
        print(f"{name:<15} | {score:.4f}")
    print("="*40)

if __name__ == '__main__':
    main()