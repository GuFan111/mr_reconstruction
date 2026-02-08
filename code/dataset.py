# dataset.py

import os
import json
import glob
import numpy as np
import SimpleITK as sitk
import scipy.ndimage
from torch.utils.data import Dataset
from copy import deepcopy
from tqdm import tqdm



class OrthogonalGeometry(object):
    def __init__(self):
        pass

    def project(self, points, view_idx):
        p = deepcopy(points)
        if view_idx == 0:   # Axial (xy)
            uv = p[:, [0, 1]]
        elif view_idx == 1: # Coronal (xz)
            uv = p[:, [0, 2]]
        elif view_idx == 2: # Sagittal (yz)
            uv = p[:, [1, 2]]
        else:
            raise ValueError("View index must be 0, 1, or 2")
        return uv

class BraTS_Dataset(Dataset):
    def __init__(self, data_root, split='train', npoint=5000, out_res=(256,256,128), preload=False):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.npoint = npoint
        self.out_res = out_res
        self.geo = OrthogonalGeometry()
        self.preload = preload
        self.data_cache = []

        all_files = sorted(glob.glob(os.path.join(data_root, '*.npy')))

        if len(all_files) == 0:
            raise RuntimeError(f"No .npy files found in {data_root}. Please run preprocess script first.")

        n = len(all_files)
        if n < 3:
            # 如果数据极少，则不进行严格划分，全部用于训练和验证
            self.file_list = all_files
            print(f"[{split}] Warning: Dataset too small ({n} files). Using all files for {split}.")
        else:
            train_end = int(0.8 * n)
            eval_end = int(0.9 * n)

            # 确保 eval 至少有一个文件
            if train_end == eval_end and n >= 2:
                eval_end = train_end + 1

            if split == 'train':
                self.file_list = all_files[:train_end]
            elif split == 'eval':
                self.file_list = all_files[train_end:eval_end]
            else:
                self.file_list = all_files[eval_end:]

        # 如果划分后依然为空（例如只有1个文件且请求eval），强制给它分配一个
        if len(self.file_list) == 0:
            self.file_list = [all_files[-1]]

        print(f"[{split}] Dataset loaded. Found {len(self.file_list)} pre-processed files.")

        if self.preload:
            print(f"[{split}] Pre-loading to RAM...")
            for path in tqdm(self.file_list, desc="Loading .npy"):
                # 读取并转置回 (x, y, z)
                vol = np.load(path)
                self.data_cache.append(vol)

        # 预生成评估网格
        res_x, res_y, res_z = out_res
        grid = np.mgrid[:res_x, :res_y, :res_z]
        # 归一化时需分别除以各自的分辨率
        grid_norm = np.stack([
            grid[0] / (res_x - 1),
            grid[1] / (res_y - 1),
            grid[2] / (res_z - 1)
        ], axis=0)
        grid_norm = grid_norm.reshape(3, -1).transpose(1, 0)
        self.eval_points = grid_norm

    def __len__(self):
        return len(self.file_list)


    def __getitem__(self, index):
        # 1. 获取干净的 GT 数据 (Planning Scan)
        if self.preload:
            vol_clean = self.data_cache[index]
            name = os.path.basename(self.file_list[index]).split('.')[0]
        else:
            path = self.file_list[index]
            name = os.path.basename(path).split('.')[0]
            vol_clean = np.load(path)

        # 解包非立方体分辨率
        res_x, res_y, res_z = self.out_res
        res_max = max(res_x, res_y, res_z)

        projs = np.zeros((3, 1, res_max, res_max), dtype=np.float32)

        # 制作标签 (GT)
        if self.split == 'train':
            # 1. 找到所有非零区域
            foreground_indices = np.argwhere(vol_clean > 1e-4)

            if len(foreground_indices) > 0:
                # 2. 采样策略
                n_fg = int(self.npoint * 0.7)
                n_rnd = self.npoint - n_fg

                # A. 前景采样
                choice_fg = np.random.choice(len(foreground_indices), n_fg, replace=True)
                coords_fg = foreground_indices[choice_fg]

                # B. 背景/全局采样
                # --- 修改点 2: 针对不同轴设置随机上限 ---
                coords_rnd_x = np.random.randint(0, res_x, n_rnd)
                coords_rnd_y = np.random.randint(0, res_y, n_rnd)
                coords_rnd_z = np.random.randint(0, res_z, n_rnd)
                coords_rnd = np.stack([coords_rnd_x, coords_rnd_y, coords_rnd_z], axis=1)

                # C. 合并
                coords = np.concatenate([coords_fg, coords_rnd], axis=0)
            else:
                # 异常保护
                coords_x = np.random.randint(0, res_x, self.npoint)
                coords_y = np.random.randint(0, res_y, self.npoint)
                coords_z = np.random.randint(0, res_z, self.npoint)
                coords = np.stack([coords_x, coords_y, coords_z], axis=1)

            # 获取 GT 值
            values = vol_clean[coords[:, 0], coords[:, 1], coords[:, 2]]

            # 坐标归一化
            res_array = np.array([res_x, res_y, res_z], dtype=np.float32)
            points = coords.astype(np.float32) / (res_array - 1)
        else:
            # 全图评估
            points = self.eval_points
            values = np.zeros(len(points))

            # 5. 投影坐标
        points_norm = (points - 0.5) * 2
        proj_points = []
        for i in range(3):
            # 投影逻辑由 OrthogonalGeometry 处理
            proj_points.append(self.geo.project(points_norm, i))
        proj_points = np.stack(proj_points, axis=0)

        return {
            'name': name,
            'dst_name': 'BraTS',
            'projs': projs.astype(np.float32),
            'points': points_norm.astype(np.float32),
            'proj_points': proj_points.astype(np.float32),
            'p_gt': values[None, :].astype(np.float32),
            'image': vol_clean[None, ...]
        }