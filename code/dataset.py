# dataset.py

import os
import json
import glob
import numpy as np
from torch.utils.data import Dataset
from copy import deepcopy
from tqdm import tqdm


class OrthogonalGeometry(object):
    def project(self, points, view_idx):
        p = deepcopy(points)
        # 对应：0-Axial(XY), 1-Coronal(XZ), 2-Sagittal(YZ)
        if view_idx == 0:   uv = p[:, [0, 1]]
        elif view_idx == 1: uv = p[:, [0, 2]]
        elif view_idx == 2: uv = p[:, [1, 2]]
        return uv

class AMOS_Dataset(Dataset):
    def __init__(self, data_root, label_root, split='train', npoint=5000, out_res=(256, 256, 128), preload=False):
        super().__init__()
        self.data_root = data_root
        self.label_root = label_root # 你需要这个目录来存放器官坐标
        self.split = split
        self.npoint = npoint
        self.out_res = out_res
        self.geo = OrthogonalGeometry()
        self.preload = preload
        self.data_cache = []

        # --- 保留你原来的文件划分逻辑 ---
        all_files = sorted(glob.glob(os.path.join(data_root, '*.npy')))
        n = len(all_files)
        # (此处省略你原来的 80/10/10 划分代码，保持不变即可)
        self.file_list = all_files # 假设已划分

        if self.preload:
            for path in tqdm(self.file_list, desc=f"[{split}] Pre-loading"):
                self.data_cache.append(np.load(path))

    def __len__(self):
        """返回数据集样本的总数"""
        return len(self.file_list)

    def eval_points_as_indices(self):
        """生成用于评估的全图索引网格"""
        res_x, res_y, res_z = self.out_res
        # 注意：这里我们只生成索引，真正的归一化和 projection 在下面统一做
        grid = np.mgrid[:res_x, :res_y, :res_z]
        return grid.reshape(3, -1).transpose(1, 0) # [N, 3]

    def __getitem__(self, index):
        # 1. 加载图像数据
        if self.preload:
            vol_clean = self.data_cache[index]
            name = os.path.basename(self.file_list[index]).split('.')[0]
        else:
            path = self.file_list[index]
            name = os.path.basename(path).split('.')[0]
            vol_clean = np.load(path)

        res_x, res_y, res_z = self.out_res

        # 2. 提前加载 Mask 标签
        label_path = os.path.join(self.label_root, f"{name}_label.npy")
        if os.path.exists(label_path):
            mask_np = np.load(label_path)
        else:
            print(f"\n[CRITICAL FATAL] 找不到标签文件: {label_path}")
            mask_np = np.zeros_like(vol_clean) # 兜底

        res_max = max(res_x, res_y, res_z)
        projs = np.zeros((3, 1, res_max, res_max), dtype=np.float32)

        # 4. 实时动态靶区计算 (ROI 暴力聚焦)
        if self.split == 'train':
            nz = np.argwhere(mask_np > 0)
            if len(nz) > 0:
                mins = nz.min(axis=0)
                maxs = nz.max(axis=0)

                margin = 35 # 脂肪缓冲带

                min_0 = max(0, mins[0] - margin)
                max_0 = min(vol_clean.shape[0], maxs[0] + margin)
                min_1 = max(0, mins[1] - margin)
                max_1 = min(vol_clean.shape[1], maxs[1] + margin)
                min_2 = max(0, mins[2] - margin)
                max_2 = min(vol_clean.shape[2], maxs[2] + margin)

                # 100% 算力死死锁在膨胀靶区内
                coords = np.stack([
                    np.random.randint(min_0, max_0, self.npoint),
                    np.random.randint(min_1, max_1, self.npoint),
                    np.random.randint(min_2, max_2, self.npoint)
                ], axis=1)
                # else:
                coords = np.stack([
                    np.random.randint(0, res_x, self.npoint),
                    np.random.randint(0, res_y, self.npoint),
                    np.random.randint(0, res_z, self.npoint)
                ], axis=1)
        else:
            # 推理模式：全图网格采样
            coords = self.eval_points_as_indices()

            # 5. 物理空间归一化 [-1, 1]
        values = vol_clean[coords[:, 0], coords[:, 1], coords[:, 2]]
        res_array = np.array([res_x, res_y, res_z], dtype=np.float32)
        points_norm = ((coords.astype(np.float32) / (res_array - 1)) - 0.5) * 2

        # 投影坐标映射
        proj_points = np.stack([
            self.geo.project(points_norm, 0),
            self.geo.project(points_norm, 1),
            self.geo.project(points_norm, 2)
        ], axis=0)

        return {
            'name': name,
            'projs': projs,  # 传递空壳占位符
            'points': points_norm.astype(np.float32),
            'proj_points': proj_points.astype(np.float32),
            'p_gt': values[None, :].astype(np.float32),
            'image': vol_clean[None, ...],
            'mask': mask_np[None, ...]
        }