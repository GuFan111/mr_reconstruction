import os
os.environ['OMP_NUM_THREADS'] = '1'
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from monai.losses import GeneralizedDiceLoss, DiceCELoss
from monai.metrics import DiceMetric
from monai.utils import set_determinism

# 导入你新写的 Dataset
from dataset_seg import AMOS_Seg_Dataset
# 导入 MedNeXt 工厂函数
from models.mednext.create_mednext_v1 import create_mednext_v1

# ================= 配置 =================
class Config:
    img_root = '/root/autodl-tmp/Proj/data/amos_mri_npy'
    label_root = '/root/autodl-tmp/Proj/data/amos_mri_label_npy'
    save_dir = './logs/mednext_seg_v1_finetune' # 🟢 改个名字，别覆盖了之前的 Log
    
    # 🟢 指向你之前 Dice 0.76 的最佳模型
    pretrained_path = './logs/mednext_seg_v1/best_metric_model.pth'
    
    # 🟢 微调参数
    batch_size = 2        # 因为全图 Resize 变小了，显存够用，可以加大 BS
    lr = 1e-4             # 🟢 降低学习率 (从 1e-3 降到 1e-4)
    epochs = 50           # 🟢 只需要跑 30-50 轮
    
    crop_size = (160, 160, 96) # 这是 Resize 的目标尺寸 (X, Y, Z)
    num_classes = 16 

# ================= 极速推理函数 (嵌入在这里) =================
def fast_roi_inference(inputs, model, input_size=(160, 160, 96)):
    """
    Validation 时使用的快速推理
    """
    original_size = inputs.shape[2:] 
    
    # 1. 降采样
    inputs_small = F.interpolate(inputs, size=input_size, mode='area')
    
    # 2. 推理
    with torch.cuda.amp.autocast():
        # model 输出可能是 list (深监督)，取第一个
        outputs = model(inputs_small)
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]
            
    # 3. 上采样 logits (比上采样 mask 更平滑)
    outputs_large = F.interpolate(outputs, size=original_size, mode='trilinear', align_corners=False)
    
    # 4. 生成 Mask (Index)
    pred_mask = torch.argmax(outputs_large, dim=1, keepdim=True)
    
    return pred_mask

# ================= 验证函数 (使用极速模式) =================
def validate(model, loader):
    model.eval()
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating (Fast Mode)"):
            inputs, labels = batch["image"].cuda(), batch["label"].cuda()
            
            # 使用极速推理 (模拟部署时的环境)
            pred_mask = fast_roi_inference(inputs, model, input_size=Config.crop_size)
            
            # 转 One-Hot 计算 Dice
            outputs_onehot = torch.nn.functional.one_hot(
                pred_mask.squeeze(1).long(), 
                num_classes=Config.num_classes
            ).permute(0, 4, 1, 2, 3)
            
            labels_oh = torch.nn.functional.one_hot(
                labels.squeeze(1).long(), 
                num_classes=Config.num_classes
            ).permute(0, 4, 1, 2, 3)

            dice_metric(y_pred=outputs_onehot, y=labels_oh)
            
    return dice_metric.aggregate().item()

# ================= 主训练逻辑 =================
def train():
    os.makedirs(Config.save_dir, exist_ok=True)
    set_determinism(seed=0)
    
    # 1. 数据加载
    # train_ds 会自动做 Resize，val_ds 返回原图
    train_ds = AMOS_Seg_Dataset(Config.img_root, Config.label_root, split='train', crop_size=Config.crop_size)
    val_ds = AMOS_Seg_Dataset(Config.img_root, Config.label_root, split='val')
    
    train_loader = DataLoader(train_ds, batch_size=Config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)
    
    print(f"Training on resized images: {Config.crop_size}")
    
    # 2. 模型初始化
    model = create_mednext_v1(
        num_input_channels=1,
        num_classes=Config.num_classes,
        model_id='S',
        kernel_size=3,
        deep_supervision=True
    ).cuda()

    # 3. 加载预训练权重 (关键)
    if Config.pretrained_path and os.path.exists(Config.pretrained_path):
        print(f"🔄 Loading pretrained weights from {Config.pretrained_path}...")
        checkpoint = torch.load(Config.pretrained_path)
        new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        model.load_state_dict(new_state_dict, strict=False)
        print("✅ Weights loaded! Starting Low-Res Fine-tuning.")
    else:
        print("❌ ERROR: No pretrained weights found! Fine-tuning requires a base model.")
        return
    
    # 4. 优化器与 Loss
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.epochs, eta_min=1e-6)
    
    # 深监督权重
    ds_weights = [1.0, 0.5, 0.25, 0.125, 0.0625]
    best_dice = 0.0
    
    # 5. 训练循环
    for epoch in range(Config.epochs):
        model.train()
        epoch_loss = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.epochs}") as pbar:
            for batch in pbar:
                inputs, labels = batch["image"].cuda(), batch["label"].cuda()
                
                optimizer.zero_grad()
                
                # 开启混合精度训练 (加速 + 省显存)
                with torch.cuda.amp.autocast():
                    outputs = model(inputs) # outputs list
                    
                    loss = 0
                    for i, output in enumerate(outputs):
                        # 🟢 修复核心：如果 output 尺寸变小了，就把 label 也变小
                        if output.shape[2:] != labels.shape[2:]:
                            # 使用 nearest 插值保持标签为整数
                            labels_ds = torch.nn.functional.interpolate(
                                labels.float(), 
                                size=output.shape[2:], 
                                mode='nearest'
                            ).long()
                            loss += ds_weights[i] * loss_function(output, labels_ds)
                        else:
                            # 尺寸一致直接计算
                            loss += ds_weights[i] * loss_function(output, labels)
                
                # 这里的 scaler 需要在外部定义，简单起见直接 backward
                # 如果显存不够，建议加上 GradScaler
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})
    
        scheduler.step()
        
        # 每 2 个 Epoch 验证一次 (因为微调很快)
        if (epoch + 1) % 2 == 0:
            # 这里的 validate 用的是 fast_roi_inference
            current_dice = validate(model, val_loader)
            print(f"Epoch {epoch+1} Fast Val Dice: {current_dice:.4f}")
            
            if current_dice > best_dice:
                best_dice = current_dice
                torch.save(model.state_dict(), os.path.join(Config.save_dir, "best_finetuned_model.pth"))
                print("💾 Best model saved!")

if __name__ == "__main__":
    train()