# dataset.py

import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from copy import deepcopy
from tqdm import tqdm
import scipy.ndimage as ndimage

def apply_biological_smoothing(mask_np, sigma=1.2):
    """
    利用 3D 高斯滤波与等值面截断，消除 Z 轴层厚导致的阶梯离散化伪影，
    还原器官真实的平滑生物力学边界。
    sigma=1.2 对应约 1~2 毫米的物理平滑半径，符合前列腺包膜的实际物理厚度。
    """
    # 1. 对离散的 0/1 阶梯进行低通滤波，转换为连续概率场
    smoothed_prob = ndimage.gaussian_filter(mask_np.astype(np.float32), sigma=sigma)

    # 2. 重新提取 0.5 的等值面作为全新的平滑生物学边界
    smooth_mask = (smoothed_prob > 0.5).astype(np.float32)
    return smooth_mask


class OrthogonalGeometry(object):
    def project(self, points, view_idx):
        p = deepcopy(points)
        # 对应：0-Axial(XY), 1-Coronal(XZ), 2-Sagittal(YZ)
        if view_idx == 0:   uv = p[:, [0, 1]]
        elif view_idx == 1: uv = p[:, [0, 2]]
        elif view_idx == 2: uv = p[:, [1, 2]]
        return uv

class ProstateDeformer:
    """
    引入盆腔生物力学约束与亚体素抗混叠重采样的增强版 CPU 形变器
    """
    def __init__(self, max_translation=15.0, max_scale=0.05, grid_size=7, max_displacement=10.0):
        self.max_t = max_translation
        self.max_s = max_scale
        self.grid_size = grid_size
        self.max_d = max_displacement / 64.0

    def __call__(self, image, mask):
        img_t = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
        mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
        D, H, W = image.shape

        # 1. 生成粗糙网格上的随机位移场 (Elastic Deformation)
        noise_grid = (torch.rand(1, 3, self.grid_size, self.grid_size, self.grid_size) * 2.0 - 1.0) * self.max_d

        # 3. 缓和生物力学约束，不要把 X 轴彻底锁死，给一点弹性空间防止撕裂
        # X(左右) 0.3, Y(前后) 1.2, Z(上下) 1.0
        biomechanical_weights = torch.tensor([0.3, 1.2, 1.0]).view(1, 3, 1, 1, 1)
        noise_grid = noise_grid * biomechanical_weights

        # 利用 PyTorch 极速三线性插值放大到全尺寸
        disp_field = F.interpolate(noise_grid, size=(D, H, W), mode='trilinear', align_corners=True)
        disp_field = disp_field.permute(0, 2, 3, 4, 1)

        # 2. 计算全局仿射分量
        tx = np.random.uniform(-self.max_t, self.max_t) / (D / 2)
        ty = np.random.uniform(-self.max_t, self.max_t) / (H / 2)
        tz = np.random.uniform(-self.max_t, self.max_t) / (W / 2)

        sx = np.random.uniform(1.0 - self.max_s, 1.0 + self.max_s)
        sy = np.random.uniform(1.0 - self.max_s, 1.0 + self.max_s)
        sz = np.random.uniform(1.0 - self.max_s, 1.0 + self.max_s)

        theta = torch.tensor([[[1.0/sx, 0, 0, tx],
                               [0, 1.0/sy, 0, ty],
                               [0, 0, 1.0/sz, tz]]], dtype=torch.float32)

        # 3. 融合仿射网格与弹性位移场
        base_grid = F.affine_grid(theta, img_t.size(), align_corners=True)
        final_grid = base_grid + disp_field

        # 4. 执行极速空间重采样
        deformed_img = F.grid_sample(img_t, final_grid, mode='bilinear', padding_mode='zeros', align_corners=True)

        # ==========================================
        # 🟢 升级二：亚体素抗混叠掩码重采样
        # ==========================================
        # 废弃 mode='nearest'，改用 'bilinear' 对 Mask 进行连续空间拉扯，防止离散化锯齿
        soft_deformed_mask = F.grid_sample(mask_t, final_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        # 重新提取 0.5 的等值面，获得完美平滑的生物力学边界
        deformed_mask = (soft_deformed_mask > 0.5).float()

        return deformed_img.squeeze().numpy(), deformed_mask.squeeze().numpy()

class AMOS_Dataset(Dataset):
    def __init__(self, data_root, label_root, split='train', npoint=5000, out_res=(128, 128, 128), preload=False):
        super().__init__()
        self.data_root = data_root
        self.label_root = label_root
        self.split = split
        self.npoint = npoint
        self.out_res = out_res
        self.geo = OrthogonalGeometry()
        self.preload = preload
        self.data_cache = []

        # 实例化前列腺形变器
        self.deformer = ProstateDeformer(max_translation=15.0, max_scale=0.05)

        # ==========================================
        # 🟢 核心修复：由于数据已经在物理文件夹层面划分好了
        # 直接读取当前目录下的所有 npy 文件，不进行二次分割
        # ==========================================
        self.file_list = sorted(glob.glob(os.path.join(self.data_root, '*.npy')))

        # 增加安全阻断机制，如果没读到文件，立刻抛出详细错误
        if len(self.file_list) == 0:
            raise ValueError(f"[CRITICAL ERROR] 在目录 {self.data_root} 中没有找到任何 .npy 文件！请检查预处理路径与 train.py 配置。")

        print(f"[Dataset] {self.split.capitalize()} Split: {len(self.file_list)} samples loaded from {self.data_root}")

        if self.preload:
            for path in tqdm(self.file_list, desc=f"[{self.split}] Pre-loading"):
                self.data_cache.append(np.load(path))

    def __len__(self):
        return len(self.file_list)

    def eval_points_as_indices(self):
        res_x, res_y, res_z = self.out_res
        grid = np.mgrid[:res_x, :res_y, :res_z]
        return grid.reshape(3, -1).transpose(1, 0) # [N, 3]

    def __getitem__(self, index):
        # 1. 加载 Prior 数据
        if self.preload:
            prior_image = self.data_cache[index]
            name = os.path.basename(self.file_list[index]).split('.')[0]
        else:
            path = self.file_list[index]
            name = os.path.basename(path).split('.')[0]
            prior_image = np.load(path)

        res_x, res_y, res_z = self.out_res

        label_path = os.path.join(self.label_root, f"{name}_label.npy")
        if os.path.exists(label_path):
            raw_prior_mask = np.load(label_path)
            # ==========================================
            # 🟢 核心手术：在生成形变 Target 之前，彻底修复 Prior 的梯田伪影
            # ==========================================
            prior_mask = apply_biological_smoothing(raw_prior_mask, sigma=1.2)
        else:
            prior_mask = np.zeros_like(prior_image)

            # 2. 执行物理形变，生成当天的 Target 数据
        # 因为 prior_mask 已经被平滑，形变器生成的 target_mask 自然也是完美平滑的锥形体
        if self.split == 'train':
            target_image, target_mask = self.deformer(prior_image, prior_mask)
        else:
            seed_val = 2026 + index
            np.random.seed(seed_val)
            torch.manual_seed(seed_val)

            target_image, target_mask = self.deformer(prior_image, prior_mask)

            # 释放种子，防止污染后续训练的随机性
            np.random.seed()
            torch.seed()

        # 3. 提取 3 张稀疏正交切片 (作为网络输入)
        # 获取 Target Mask 的质心，模拟医生在正交视图上的定位
        nz_target = np.argwhere(target_mask > 0)
        if len(nz_target) > 0:
            cx, cy, cz = nz_target.mean(axis=0).astype(int)
        else:
            cx, cy, cz = res_x//2, res_y//2, res_z//2

        # [3, 1, H, W] 的稀疏视图 (Target 当天切片)
        res_max = max(res_x, res_y, res_z)
        projs = np.zeros((3, 1, res_max, res_max), dtype=np.float32)
        # 0: Axial(XY), 1: Coronal(XZ), 2: Sagittal(YZ)
        projs[0, 0, :res_x, :res_y] = target_image[:, :, cz]
        projs[1, 0, :res_x, :res_z] = target_image[:, cy, :]
        projs[2, 0, :res_y, :res_z] = target_image[cx, :, :]

        # ==========================================
        # 🟢 核心补丁：同步提取 Prior (昨天) 的对应切片
        # ==========================================
        prior_projs = np.zeros((3, 1, res_max, res_max), dtype=np.float32)
        prior_projs[0, 0, :res_x, :res_y] = prior_image[:, :, cz]
        prior_projs[1, 0, :res_x, :res_z] = prior_image[:, cy, :]
        prior_projs[2, 0, :res_y, :res_z] = prior_image[cx, :, :]

        # 4. 实时动态靶区计算 (80% ROI + 20% Global)
        if self.split == 'train':
            if len(nz_target) > 0:
                mins = nz_target.min(axis=0)
                maxs = nz_target.max(axis=0)

                margin = 20 # 前列腺较小，裕度可稍微收紧
                min_0 = max(0, mins[0] - margin)
                max_0 = min(res_x, maxs[0] + margin)
                min_1 = max(0, mins[1] - margin)
                max_1 = min(res_y, maxs[1] + margin)
                min_2 = max(0, mins[2] - margin)
                max_2 = min(res_z, maxs[2] + margin)

                n_fg = int(self.npoint * 0.8)
                n_bg = self.npoint - n_fg

                coords_fg = np.stack([
                    np.random.randint(min_0, max_0, n_fg),
                    np.random.randint(min_1, max_1, n_fg),
                    np.random.randint(min_2, max_2, n_fg)
                ], axis=1)

                coords_bg = np.stack([
                    np.random.randint(0, res_x, n_bg),
                    np.random.randint(0, res_y, n_bg),
                    np.random.randint(0, res_z, n_bg)
                ], axis=1)

                coords = np.concatenate([coords_fg, coords_bg], axis=0)
                is_fg = np.concatenate([np.ones(n_fg), np.zeros(n_bg)]).astype(np.float32)

                shuffle_idx = np.random.permutation(self.npoint)
                coords = coords[shuffle_idx]
                is_fg = is_fg[shuffle_idx]

            else:
                coords = np.stack([
                    np.random.randint(0, res_x, self.npoint),
                    np.random.randint(0, res_y, self.npoint),
                    np.random.randint(0, res_z, self.npoint)
                ], axis=1)
                is_fg = np.zeros(self.npoint, dtype=np.float32)
        else:
            coords = self.eval_points_as_indices()
            is_fg = np.zeros(coords.shape[0], dtype=np.float32)

            # 5. 物理空间归一化 [-1, 1]
        values = target_mask[coords[:, 0], coords[:, 1], coords[:, 2]]

        res_array = np.array([res_x, res_y, res_z], dtype=np.float32)
        points_norm = ((coords.astype(np.float32) / (res_array - 1)) - 0.5) * 2

        proj_points = np.stack([
            self.geo.project(points_norm, 0),
            self.geo.project(points_norm, 1),
            self.geo.project(points_norm, 2)
        ], axis=0)

        return {
            'name': name,
            'projs': projs,
            'prior_projs': prior_projs,
            'points': points_norm.astype(np.float32),
            'proj_points': proj_points.astype(np.float32),
            'p_gt': values[None, :].astype(np.float32),
            'prior_mask': prior_mask[None, ...].astype(np.float32),
            # 🟢 新增下面这一行：把形变后的当天 3D 灰度底图传出去，专供可视化做背景
            'target_image': target_image[None, ...].astype(np.float32),
            'point_is_fg': is_fg
        }