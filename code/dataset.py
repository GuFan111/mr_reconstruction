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

# ==========================================
# 🟢 物理学预处理模块
# ==========================================

def apply_biological_smoothing(mask_np, sigma=1.2):
    """
    生物形态学平滑 (Biomorphological Smoothing)
    物理意义：真实的临床 MRI 往往存在 Z 轴层厚过大 (例如 3mm) 的问题，导致 Ground Truth
    呈现非生理性的“喀斯特阶梯伪影”。本函数通过 3D 高斯滤波与等值面重采样，重建出符合
    真实前列腺包膜表面张力的连续物理边界。
    学术贡献：消除了切片各向异性导致的离散化噪声，防止网络过拟合人工标注瑕疵。
    """
    # 1. 对离散的 0/1 阶梯进行低通滤波，转换为连续概率场
    smoothed_prob = ndimage.gaussian_filter(mask_np.astype(np.float32), sigma=sigma)

    # 2. 重新提取 0.5 的等值面作为全新的平滑生物学边界
    smooth_mask = (smoothed_prob > 0.5).astype(np.float32)
    return smooth_mask

class OrthogonalGeometry(object):
    """正交投影器：将 3D 物理空间坐标精确映射到 3 张 2D 临床视图上"""
    def project(self, points, view_idx):
        p = deepcopy(points)
        # 对应：0-Axial(XY), 1-Coronal(XZ), 2-Sagittal(YZ)
        if view_idx == 0:   uv = p[:, [0, 1]]
        elif view_idx == 1: uv = p[:, [0, 2]]
        elif view_idx == 2: uv = p[:, [1, 2]]
        return uv

# ==========================================
# 🟢 核心组件 1：盆腔生物力学形变模拟器
# (Biomechanical-constrained Deformer)
# ==========================================

class ProstateDeformer:
    """
    引入盆腔生物力学约束的增强版形变引擎
    学术创新：传统的弹性形变 (Elastic Deformation) 是各向同性的，容易产生解剖学上不可能的撕裂。
    本形变器根据前列腺的物理环境（左右受骨盆限制，前后受直肠挤压），施加了非对称的力学权重。
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

        # 1. 生成粗糙网格上的随机位移场
        noise_grid = (torch.rand(1, 3, self.grid_size, self.grid_size, self.grid_size) * 2.0 - 1.0) * self.max_d

        # 2. 施加非各向同性的生物力学权重约束 (Biomechanic Weights)
        # 限制 X 轴(左右)形变，允许 Y(前后) 和 Z(上下) 存在较大的生理性位移
        biomechanical_weights = torch.tensor([0.3, 1.2, 1.0]).view(1, 3, 1, 1, 1)
        noise_grid = noise_grid * biomechanical_weights

        # 利用 PyTorch 极速三线性插值放大位移场
        disp_field = F.interpolate(noise_grid, size=(D, H, W), mode='trilinear', align_corners=True)
        disp_field = disp_field.permute(0, 2, 3, 4, 1)

        # 3. 计算全局仿射分量 (模拟大尺度的器官漂移)
        tx = np.random.uniform(-self.max_t, self.max_t) / (D / 2)
        ty = np.random.uniform(-self.max_t, self.max_t) / (H / 2)
        tz = np.random.uniform(-self.max_t, self.max_t) / (W / 2)

        sx = np.random.uniform(1.0 - self.max_s, 1.0 + self.max_s)
        sy = np.random.uniform(1.0 - self.max_s, 1.0 + self.max_s)
        sz = np.random.uniform(1.0 - self.max_s, 1.0 + self.max_s)

        theta = torch.tensor([[[1.0/sx, 0, 0, tx],
                               [0, 1.0/sy, 0, ty],
                               [0, 0, 1.0/sz, tz]]], dtype=torch.float32)

        base_grid = F.affine_grid(theta, img_t.size(), align_corners=True)
        final_grid = base_grid + disp_field

        # 4. 执行极速连续空间重采样
        deformed_img = F.grid_sample(img_t, final_grid, mode='bilinear', padding_mode='zeros', align_corners=True)

        # 亚体素抗混叠重采样：对 Mask 采用 bilinear 防止产生锯齿，而后重新提取 0.5 等值面
        soft_deformed_mask = F.grid_sample(mask_t, final_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        deformed_mask = (soft_deformed_mask > 0.5).float()

        return deformed_img.squeeze().numpy(), deformed_mask.squeeze().numpy()

# ==========================================
# 🟢 核心数据管线 (Data Pipeline)
# 承载了粗到细架构 (Coarse-to-fine) 的第一步
# ==========================================

class Prostate_Dataset(Dataset):
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

        self.deformer = ProstateDeformer(max_translation=15.0, max_scale=0.05)

        self.file_list = sorted(glob.glob(os.path.join(self.data_root, '*.npy')))

        if len(self.file_list) == 0:
            raise ValueError(f"[CRITICAL ERROR] 在目录 {self.data_root} 中没有找到任何 .npy 文件！")

        print(f"[Dataset] {self.split.capitalize()} Split: {len(self.file_list)} samples loaded from {self.data_root}")

        if self.preload:
            for path in tqdm(self.file_list, desc=f"[{self.split}] Pre-loading"):
                self.data_cache.append(np.load(path))

    def __len__(self):
        return len(self.file_list)

    def eval_points_as_indices(self):
        """生成用于评估的密集全空间网格系"""
        res_x, res_y, res_z = self.out_res
        grid = np.mgrid[:res_x, :res_y, :res_z]
        return grid.reshape(3, -1).transpose(1, 0) # [N, 3]

    def __getitem__(self, index):
        # 1. 加载 先验 (Prior) 数据
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
            prior_mask = apply_biological_smoothing(raw_prior_mask, sigma=1.2)
        else:
            prior_mask = np.zeros_like(prior_image)

            # 2. 形变引擎介入，合成当天的 靶区 (Target) 数据
        if self.split == 'train':
            target_image, target_mask = self.deformer(prior_image, prior_mask)
        else:
            # 评估模式使用固定随机种子，保证每次评测的数据集是一致的
            seed_val = 2026 + index
            np.random.seed(seed_val)
            torch.manual_seed(seed_val)
            target_image, target_mask = self.deformer(prior_image, prior_mask)
            np.random.seed()
            torch.seed()

        # ==========================================
        # 🟢 粗对齐：基于 2D 切片投影的质心估算
        # ==========================================
        # 1. 获取 Prior 的质心 (即 MR-Linac 机器当天的扫描等中心 Isocenter)
        nz_target = np.argwhere(target_mask > 0)
        nz_prior = np.argwhere(prior_mask > 0)
        if len(nz_prior) > 0:
            px, py, pz = nz_prior.mean(axis=0).astype(int)
        else:
            px, py, pz = res_x//2, res_y//2, res_z//2

        # 2. 机器在 (px, py, pz) 处切下当天的三张 2D Target 切片
        slice_ax = target_mask[:, :, pz]  # Axial 切片包含 X, Y 坐标
        slice_cor = target_mask[:, py, :] # Coronal 切片包含 X, Z 坐标
        slice_sag = target_mask[px, :, :] # Sagittal 切片包含 Y, Z 坐标

        # 3. 从 2D 切片中分别计算对应轴的 1D 质心投影
        def get_2d_com(mask_2d):
            pts = np.argwhere(mask_2d > 0)
            return pts.mean(axis=0) if len(pts) > 0 else None

        com_ax = get_2d_com(slice_ax)
        com_cor = get_2d_com(slice_cor)
        com_sag = get_2d_com(slice_sag)

        # 4. 融合 2D 投影，计算当天的估算 3D 质心 (Estimated Target CoM)
        cx_list, cy_list, cz_list = [], [], []
        if com_ax is not None: cx_list.append(com_ax[0]); cy_list.append(com_ax[1])
        if com_cor is not None: cx_list.append(com_cor[0]); cz_list.append(com_cor[1])
        if com_sag is not None: cy_list.append(com_sag[0]); cz_list.append(com_sag[1])

        # 如果某一切片完全没切到器官(漂移过大)，则默认使用 Prior 的对应轴坐标保底
        cx = int(np.mean(cx_list)) if cx_list else px
        cy = int(np.mean(cy_list)) if cy_list else py
        cz = int(np.mean(cz_list)) if cz_list else pz

        # 5. 计算合法的空间平移向量并执行预对齐
        shift_vec = (cx - px, cy - py, cz - pz)

        # 将 Prior 强行推到基于 2D 切片估算出的物理质心位置
        aligned_prior_mask = ndimage.shift(prior_mask, shift_vec, order=0)
        aligned_prior_image = ndimage.shift(prior_image, shift_vec, order=1)

        # ==========================================
        # 提取用于网络输入的 3 张正交视图切片
        # 注意：因为机器追踪到了新的中心 (cx, cy, cz)，我们需要提取新中心处的特征
        # ==========================================
        res_max = max(res_x, res_y, res_z)
        projs = np.zeros((3, 1, res_max, res_max), dtype=np.float32)
        projs[0, 0, :res_x, :res_y] = target_image[:, :, cz]
        projs[1, 0, :res_x, :res_z] = target_image[:, cy, :]
        projs[2, 0, :res_y, :res_z] = target_image[cx, :, :]

        prior_projs = np.zeros((3, 1, res_max, res_max), dtype=np.float32)
        prior_projs[0, 0, :res_x, :res_y] = aligned_prior_image[:, :, cz]
        prior_projs[1, 0, :res_x, :res_z] = aligned_prior_image[:, cy, :]
        prior_projs[2, 0, :res_y, :res_z] = aligned_prior_image[cx, :, :]

        # ==========================================
        # 🟢 核心创新：双轨狄利克雷物理约束采样 (Dirichlet Boundary-Constrained Sampling)
        # 学术意义：通过分离“强锁定锚点”与“游离推演点”，解决极稀疏视图重建的核心拓扑歧义。
        # ==========================================
        if self.split == 'train':
            # 分配一半的算力给切片锚点，一半给空间盲区
            n_lock = self.npoint // 2
            n_free = self.npoint - n_lock

            # --- A. 强锁定区 (Lock Points)：死死钉在三个正交平面上 ---
            n_plane = n_lock // 3
            lock_x = np.stack([np.full(n_plane, cx), np.random.randint(0, res_y, n_plane), np.random.randint(0, res_z, n_plane)], axis=1)
            lock_y = np.stack([np.random.randint(0, res_x, n_plane), np.full(n_plane, cy), np.random.randint(0, res_z, n_plane)], axis=1)
            n_remain = n_lock - 2 * n_plane
            lock_z = np.stack([np.random.randint(0, res_x, n_remain), np.random.randint(0, res_y, n_remain), np.full(n_remain, cz)], axis=1)

            coords_lock = np.concatenate([lock_x, lock_y, lock_z], axis=0)

            # --- B. 自由推演区 (Free Points)：在目标边界框及其周围游走 ---
            if len(nz_target) > 0:
                mins, maxs = nz_target.min(axis=0), nz_target.max(axis=0)
                margin = 20
                min_0, max_0 = max(0, mins[0]-margin), min(res_x, maxs[0]+margin)
                min_1, max_1 = max(0, mins[1]-margin), min(res_y, maxs[1]+margin)
                min_2, max_2 = max(0, mins[2]-margin), min(res_z, maxs[2]+margin)

                n_free_fg = int(n_free * 0.8)
                n_free_bg = n_free - n_free_fg

                free_fg = np.stack([np.random.randint(min_0, max_0, n_free_fg), np.random.randint(min_1, max_1, n_free_fg), np.random.randint(min_2, max_2, n_free_fg)], axis=1)
                free_bg = np.stack([np.random.randint(0, res_x, n_free_bg), np.random.randint(0, res_y, n_free_bg), np.random.randint(0, res_z, n_free_bg)], axis=1)
                coords_free = np.concatenate([free_fg, free_bg], axis=0)
            else:
                coords_free = np.stack([np.random.randint(0, res_x, n_free), np.random.randint(0, res_y, n_free), np.random.randint(0, res_z, n_free)], axis=1)

            # 聚合坐标体系并打乱，输出 0/1 标记以供 loss 函数进行空间加权劫持
            coords = np.concatenate([coords_lock, coords_free], axis=0)
            is_lock = np.concatenate([np.ones(n_lock), np.zeros(n_free)]).astype(np.float32)

            shuffle_idx = np.random.permutation(self.npoint)
            coords = coords[shuffle_idx]
            is_lock = is_lock[shuffle_idx]
        else:
            # 评估阶段采用全空间无差别密集采样
            coords = self.eval_points_as_indices()
            is_lock = np.zeros(coords.shape[0], dtype=np.float32)

            # 4. 提取 Ground Truth (占据概率基准)
        values = target_mask[coords[:, 0], coords[:, 1], coords[:, 2]]

        # 5. 坐标系向隐式空间归一化 [-1, 1]
        res_array = np.array([res_x, res_y, res_z], dtype=np.float32)
        points_norm = ((coords.astype(np.float32) / (res_array - 1)) - 0.5) * 2

        # 执行坐标轴的正交投影
        proj_points = np.stack([
            self.geo.project(points_norm, 0),
            self.geo.project(points_norm, 1),
            self.geo.project(points_norm, 2)
        ], axis=0)

        # 6. 数据字典封装并出栈
        return {
            'name': name,
            'projs': projs,
            'prior_projs': prior_projs,
            'points': points_norm.astype(np.float32),
            'proj_points': proj_points.astype(np.float32),
            'p_gt': values[None, :].astype(np.float32),
            'prior_mask': aligned_prior_mask[None, ...].astype(np.float32), # 网络看到的先验已完全去除了大尺度位移
            'target_image': target_image[None, ...].astype(np.float32),
            'is_lock': is_lock[None, :].astype(np.float32) # 输出锁定信标，供 Loss 计算
        }