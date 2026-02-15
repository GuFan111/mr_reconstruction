import os
os.environ['OMP_NUM_THREADS'] = '1'
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from monai.losses import GeneralizedDiceLoss, DiceCELoss
from monai.metrics import DiceMetric
from monai.utils import set_determinism
from monai.inferers import sliding_window_inference
from monai.transforms import KeepLargestConnectedComponent
from monai.data import decollate_batch

# 导入你新写的 Dataset
from dataset_seg import AMOS_Seg_Dataset
# 导入 MedNeXt 工厂函数
from models.mednext.create_mednext_v1 import create_mednext_v1

# 配置
class Config:
    img_root = '/root/autodl-tmp/Proj/data/amos_mri_npy'
    label_root = '/root/autodl-tmp/Proj/data/amos_mri_label_npy'
    save_dir = './logs/mednext_seg_v1'
    # pretrained_path = './logs/mednext_seg_v1/best_metric_model.pth'
    pretrained_path = None
    batch_size = 1 # 根据显存调整，MedNeXt-S 比较轻量，2-4 应该没问题
    lr = 1e-3
    epochs = 100
    crop_size = (160, 160, 96) # 训练时的 Patch 大小
    num_classes = 16 # AMOS 也是 16 类 (含背景)

def train():
    os.makedirs(Config.save_dir, exist_ok=True)
    set_determinism(seed=0)
    
    # 1. 数据加载
    train_ds = AMOS_Seg_Dataset(Config.img_root, Config.label_root, split='train', crop_size=Config.crop_size)
    val_ds = AMOS_Seg_Dataset(Config.img_root, Config.label_root, split='val')
    
    train_loader = DataLoader(train_ds, batch_size=Config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    # 验证集 Batch Size 设为 1，因为全图尺寸较大
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)
    
    # 2. 模型初始化 (MedNeXt Small)
    model = create_mednext_v1(
        num_input_channels=1,
        num_classes=Config.num_classes,
        model_id='S',             # S: Small, M: Medium
        kernel_size=3,            # 3x3x3 卷积
        deep_supervision=True     # 开启深监督
    ).cuda()

    if Config.pretrained_path and os.path.exists(Config.pretrained_path):
        print(f"Loading pretrained weights from {Config.pretrained_path}...")
        checkpoint = torch.load(Config.pretrained_path)
        
        # 处理可能的 'module.' 前缀 (DataParallel 遗留问题)
        new_state_dict = {}
        for k, v in checkpoint.items():
            name = k.replace('module.', '') # 去掉 module.
            new_state_dict[name] = v
            
        # strict=False 允许忽略一些不匹配的层 (虽然这里应该完全匹配)
        model.load_state_dict(new_state_dict, strict=False)
        print("✅ Weights loaded successfully! Starting Fine-tuning.")
    else:
        print("⚠️ No pretrained path found, training from scratch (NOT RECOMMENDED).")
    
    # 3. 损失函数与优化器
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.epochs, eta_min=1e-6)
    
    # 深监督权重 (MedNeXt 默认输出 5 个尺度的结果)
    ds_weights = [1.0, 0.5, 0.25, 0.125, 0.0625]
    
    best_dice = 0.0
    
    # 4. 训练循环
    for epoch in range(Config.epochs):
        model.train()
        epoch_loss = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.epochs}") as pbar:
            for batch in pbar:
                inputs, labels = batch["image"].cuda(), batch["label"].cuda()
                
                optimizer.zero_grad()
                outputs = model(inputs) # outputs 是一个列表 (Deep Supervision)
                
                # 计算深监督 Loss
                loss = 0
                for i, output in enumerate(outputs):
                    # 如果 output 尺寸和 label 不一致，需要对 label 进行下采样
                    if output.shape[2:] != labels.shape[2:]:
                        # 简单的最近邻下采样 label
                        labels_ds = torch.nn.functional.interpolate(labels.float(), size=output.shape[2:], mode='nearest').long()
                        loss += ds_weights[i] * loss_function(output, labels_ds)
                    else:
                        loss += ds_weights[i] * loss_function(output, labels)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})
    
        scheduler.step()
        if (epoch + 1) % 10 == 0:
             print(f"Epoch {epoch+1} LR: {scheduler.get_last_lr()[0]:.6f}")       

    
        # 5. 简单验证 (保存最佳模型)
        # 这里为了速度，仅每 5 个 Epoch 跑一次验证
        if (epoch + 1) % 5 == 0:
            current_dice = validate(model, val_loader)
            print(f"Epoch {epoch+1} Val Dice: {current_dice:.4f}")
            
            if current_dice > best_dice:
                best_dice = current_dice
                torch.save(model.state_dict(), os.path.join(Config.save_dir, "best_metric_model.pth"))
        
        # 定期保存
        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), os.path.join(Config.save_dir, f"epoch_{epoch+1}.pth"))



# 定义后处理：只保留最大的连通域
# applied_labels: 这里填你想要进行连通域处理的类别 ID 列表
# 比如 1(右肾), 2(左肾), 3(肝脏)... AMOS 中除了血管等散状结构，大部分器官都适用
# 这里为了演示，假设所有前景类别都只保留最大连通域
post_process = KeepLargestConnectedComponent(applied_labels=None, is_onehot=True)

# def validate(model, loader):
#     model.eval()
#     torch.cuda.empty_cache()
#     dice_metric = DiceMetric(include_background=False, reduction="mean")
    
#     with torch.no_grad():
#         for batch in tqdm(loader, desc="Validating"):
#             inputs, labels = batch["image"].cuda(), batch["label"].cuda()
            
#             # -----------------------------------------------------------
#             # 修改点 1: 使用滑动窗口推理 (Sliding Window Inference)
#             # 解决边缘烂的问题
#             # -----------------------------------------------------------
#             # roi_size: 必须与训练时的 crop_size 一致 (128, 128, 64)
#             # overlap: 0.5 表示窗口重叠一半，高斯融合效果最好
#             outputs = sliding_window_inference(
#                 inputs, 
#                 roi_size=(160, 160, 96), 
#                 sw_batch_size=4, 
#                 predictor=model, 
#                 overlap=0.5, 
#                 mode='gaussian'  # 关键！使用高斯加权平滑边缘
#             )
            
#             # MedNeXt 深监督模式下可能返回列表，处理一下
#             if isinstance(outputs, (list, tuple)):
#                 outputs = outputs[0] # 取最高分辨率输出
            
#             # [B, C, D, H, W] -> Argmax 转为类别索引 -> One-Hot
#             # 注意：后处理通常在 One-Hot 格式上进行
#             outputs_onehot = torch.nn.functional.one_hot(
#                 torch.argmax(outputs, dim=1), 
#                 num_classes=Config.num_classes
#             ).permute(0, 4, 1, 2, 3) # [B, C, D, H, W]
            
#             labels_oh = torch.nn.functional.one_hot(
#                 labels.squeeze(1).long(), 
#                 num_classes=Config.num_classes
#             ).permute(0, 4, 1, 2, 3)

#             # -----------------------------------------------------------
#             # 修改点 2: 最大连通域后处理 (Post Processing)
#             # 解决“多预测出几个小的”问题
#             # -----------------------------------------------------------
#             # decollate_batch 将 Batch 拆开，因为后处理是对单张图做的
#             outputs_list = decollate_batch(outputs_onehot)
#             outputs_post = []
            
#             for pred in outputs_list:
#                 # 对该样本的所有通道应用“保留最大连通域”
#                 # 注意：这步计算量稍大，如果太慢可以只在测试时用
#                 try:
#                     pred_pp = post_process(pred)
#                     outputs_post.append(pred_pp)
#                 except Exception as e:
#                     # 万一报错（比如全黑），就退回原预测
#                     outputs_post.append(pred)
            
#             # 重新堆叠回 Batch
#             outputs_final = torch.stack(outputs_post)
            
#             # 计算 Dice
#             dice_metric(y_pred=outputs_final, y=labels_oh)
            
#     return dice_metric.aggregate().item()

def validate(model, loader):
    model.eval()
    dice_metric = DiceMetric(...)
    
    with torch.no_grad():
        for batch in tqdm(loader):
            inputs, labels = batch["image"].cuda(), batch["label"].cuda()
            
            # 🔴 替换为快速推理
            # 注意：fast_roi_inference 返回的是 index (0,1,2...)
            # 而 DiceMetric 需要 One-Hot
            pred_mask = fast_roi_inference(inputs, model, input_size=(96, 160, 160))
            
            # Index -> One-Hot
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


if __name__ == "__main__":
    train()