# import os
# import glob
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# from monai.transforms import (
#     Compose,
#     RandCropByPosNegLabeld,
#     RandRotate90d,
#     RandFlipd,
#     RandShiftIntensityd,
#     EnsureTyped,
#     ToTensord
# )

# class AMOS_Seg_Dataset(Dataset):
#     def __init__(self, img_root, label_root, split='train', crop_size=(128, 128, 64), cache=False):
#         self.img_root = img_root
#         self.label_root = label_root
#         self.split = split
#         self.crop_size = crop_size
#         self.cache = cache
        
#         # 获取共有 ID
#         self.img_files = sorted(glob.glob(os.path.join(img_root, '*.npy')))
#         self.data_list = []
        
#         for img_path in self.img_files:
#             name = os.path.basename(img_path)
#             idx = name.split('_')[1].split('.')[0] # amos_0500.npy -> 0500
#             label_name = f"amos_{idx}_label.npy"
#             label_path = os.path.join(label_root, label_name)
            
#             if os.path.exists(label_path):
#                 self.data_list.append({"image": img_path, "label": label_path})
        
#         # 简单划分 (80% Train, 20% Val)
#         split_idx = int(0.8 * len(self.data_list))
#         if split == 'train':
#             self.data_list = self.data_list[:split_idx]
#         else:
#             self.data_list = self.data_list[split_idx:]
            
#         # 定义训练时的数据增强流水线
#         # 注意：这里输入已经是 numpy array，不需要 LoadImage
#         self.train_transforms = Compose([
#             EnsureTyped(keys=["image", "label"]),
#             # 1. 随机裁剪：保证裁剪块中包含前景（器官）
#             RandCropByPosNegLabeld(
#                 keys=["image", "label"],
#                 label_key="label",
#                 spatial_size=crop_size,
#                 pos=2, neg=1, # 2:1 的比例采样前景和背景
#                 num_samples=1,
#                 image_key="image",
#                 image_threshold=0,
#             ),
#             # 2. 空间增强
#             RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 1)),
#             RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
#             RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
#             RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
#             # 3. 强度增强 (仅对 Image)
#             RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
#         ])

#     def __len__(self):
#         return len(self.data_list)

#     def __getitem__(self, index):
#         item = self.data_list[index]
        
#         # 加载数据
#         img = np.load(item["image"])   # [256, 256, 128]
#         lbl = np.load(item["label"])   # [256, 256, 128]
        
#         # 增加 Channel 维度: [C, D, H, W]
#         img = img[None, ...] 
#         lbl = lbl[None, ...]
        
#         data_dict = {"image": img, "label": lbl}
        
#         if self.split == 'train':
#             # 应用增强，Crop 出小块进行训练 (节省显存)
#             data_dict = self.train_transforms(data_dict)[0] # MONAI Crop 返回列表，取第0个
#         else:
#             # 验证集直接转 Tensor，不裁剪 (或使用滑动窗口推理)
#             data_dict["image"] = torch.from_numpy(data_dict["image"]).float()
#             data_dict["label"] = torch.from_numpy(data_dict["label"]).long()
            
#         return data_dict


import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from monai.transforms import (
    Compose,
    Resized,                  # 🟢 改为 Resized (带 d)
    RandRotate90d,
    RandFlipd,
    RandShiftIntensityd,
    RandBiasFieldd,          
    NormalizeIntensityd,     
    EnsureTyped,
    ToTensord
)

class AMOS_Seg_Dataset(Dataset):
    def __init__(self, img_root, label_root, split='train', crop_size=(160, 160, 96), cache=False):
        """
        crop_size: 在微调阶段，这实际上是 target_size (缩放目标尺寸)
        """
        self.img_root = img_root
        self.label_root = label_root
        self.split = split
        self.crop_size = crop_size # e.g. (160, 160, 96)
        
        # 获取共有 ID
        self.img_files = sorted(glob.glob(os.path.join(img_root, '*.npy')))
        self.data_list = []
        
        for img_path in self.img_files:
            name = os.path.basename(img_path)
            idx = name.split('_')[1].split('.')[0] 
            label_name = f"amos_{idx}_label.npy"
            label_path = os.path.join(label_root, label_name)
            
            if os.path.exists(label_path):
                self.data_list.append({"image": img_path, "label": label_path})
        
        # 简单划分 (80% Train, 20% Val)
        split_idx = int(0.8 * len(self.data_list))
        if split == 'train':
            self.data_list = self.data_list[:split_idx]
        else:
            self.data_list = self.data_list[split_idx:]
            
        # ====================================================
        # 🟢 微调阶段的核心修改：全图 Resize，不再 Crop
        # ====================================================
        self.train_transforms = Compose([
            EnsureTyped(keys=["image", "label"]),
            
            # 1. 强制缩放到推理尺寸 (160, 160, 96)
            # image 使用 trilinear (平滑插值)
            # label 使用 nearest (最近邻，保证标签是整数)
            Resized(keys=["image"], spatial_size=crop_size, mode="trilinear"),
            Resized(keys=["label"], spatial_size=crop_size, mode="nearest"),
            
            # 2. 强度归一化 (非常重要，防止过拟合)
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            
            # 3. 空间增强 (在 Resize 之后做)
            RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 1)),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),

            ToTensord(keys=["image", "label"]),
        ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        item = self.data_list[index]
        
        # 加载数据
        img = np.load(item["image"])   # [H, W, D]
        lbl = np.load(item["label"])   # [H, W, D]
        
        # 增加 Channel 维度: [C, H, W, D] -> 对应 [C, X, Y, Z]
        img = img[None, ...] 
        lbl = lbl[None, ...]
        
        data_dict = {"image": img, "label": lbl}
        
        if self.split == 'train':
            # 应用 Resize 和增强
            data_dict = self.train_transforms(data_dict)
            # 此时返回的是 (160, 160, 96) 的数据
        else:
            # 验证集返回原图，交给 validate 函数里的 fast_inference 去缩放
            # 这样算出来的 Dice 才是真实的
            data_dict["image"] = torch.from_numpy(data_dict["image"]).float()
            data_dict["label"] = torch.from_numpy(data_dict["label"]).long()
            
        return data_dict