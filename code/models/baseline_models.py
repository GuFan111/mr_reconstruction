import torch
import torch.nn as nn
from monai.networks.nets import UNet, SwinUNETR

class SparseVolumeBuilder(nn.Module):
    """
    公平性防线：三通道稀疏 3D 矩阵构建器
    作用：将输入的 3D 图像抹除 97% 的信息，只保留 (cx, cy, cz) 三个相交平面的像素。
    同时注入先验切片，确保 Baseline 与 DIF-Net 获得绝对同等的信息量。
    """
    def __init__(self):
        super().__init__()

    def forward(self, full_target_img, full_prior_img, prior_mask, coords):
        """
        full_target_img: [B, 1, 128, 128, 128] 完整的第二天目标灰度图
        full_prior_img:  [B, 1, 128, 128, 128] 完整的第一天先验灰度图
        prior_mask:      [B, 1, 128, 128, 128] 完美对齐的刚性先验掩码
        coords:          [B, 3] 记录了 (cx, cy, cz) 的切面坐标索引
        """
        B, C, W, H, D = full_target_img.shape

        # 1. 构造全零的稀疏容器
        sparse_target = torch.zeros_like(full_target_img)
        sparse_prior = torch.zeros_like(full_prior_img)

        # 2. 遍历 Batch，提取十字交叉的 3 帧切片
        for b in range(B):
            cx, cy, cz = coords[b].long()

            # 填入今天的切片 (Target Slices)
            sparse_target[b, :, cx, :, :] = full_target_img[b, :, cx, :, :]
            sparse_target[b, :, :, cy, :] = full_target_img[b, :, :, cy, :]
            sparse_target[b, :, :, :, cz] = full_target_img[b, :, :, :, cz]

            # 填入昨天的切片 (Prior Slices)
            sparse_prior[b, :, cx, :, :] = full_prior_img[b, :, cx, :, :]
            sparse_prior[b, :, :, cy, :] = full_prior_img[b, :, :, cy, :]
            sparse_prior[b, :, :, :, cz] = full_prior_img[b, :, :, :, cz]

        # 3. 在通道维度拼接：通道0(稀疏今日), 通道1(稀疏昨日), 通道2(密集先验掩码)
        # 输出尺寸: [B, 3, 128, 128, 128]
        network_input = torch.cat([sparse_target, sparse_prior, prior_mask], dim=1)
        return network_input


class Baseline_3DUNet(nn.Module):
    """经典 3D 卷积基线：用于证明特征稀释效应"""
    def __init__(self):
        super().__init__()
        self.builder = SparseVolumeBuilder()
        self.net = UNet(
            spatial_dims=3,
            in_channels=3,   # 🔴 关键修改：接收 3 通道输入
            out_channels=1,  # 输出单一概率图
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm="instance"
        )

    def forward(self, full_target_img, full_prior_img, prior_mask, coords):
        x = self.builder(full_target_img, full_prior_img, prior_mask, coords)
        logits = self.net(x)
        return logits


class Baseline_SwinUNETR(nn.Module):
    """适配最新/特殊版本 MONAI 的 SwinUNETR"""
    def __init__(self, img_size=(128, 128, 128)):
        super().__init__()
        self.builder = SparseVolumeBuilder()

        # 🟢 根据你的 inspect 结果，直接移除所有 size 相关的参数
        from monai.networks.nets import SwinUNETR as MONAI_SwinUNETR
        self.net = MONAI_SwinUNETR(
            in_channels=3,
            out_channels=1,
            feature_size=24,
            use_checkpoint=True,
            spatial_dims=3  # 确保显式指定为 3D 模式
        )

    def forward(self, full_target_img, full_prior_img, prior_mask, coords):
        x = self.builder(full_target_img, full_prior_img, prior_mask, coords)
        logits = self.net(x)
        return logits