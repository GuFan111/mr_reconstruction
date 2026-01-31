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
    def __init__(self, data_root, split='train', npoint=5000, out_res=128, preload=False):
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

        # 8:1:1 划分
        n = len(all_files)
        if split == 'train':
            self.file_list = all_files[:int(0.8 * n)]
        elif split == 'eval':
            self.file_list = all_files[int(0.8 * n):int(0.9 * n)]
        else:
            self.file_list = all_files[int(0.9 * n):]

        # print("单张图片过拟合测试 !!!")
        # self.file_list = all_files[:1]

        print(f"[{split}] Dataset loaded. Found {len(self.file_list)} pre-processed files.")

        if self.preload:
            print(f"[{split}] Pre-loading to RAM...")
            for path in tqdm(self.file_list, desc="Loading .npy"):
                # 读取并转置回 (x, y, z)
                vol = np.load(path)
                self.data_cache.append(vol)

        # 预生成评估网格
        grid = np.mgrid[:out_res, :out_res, :out_res]
        grid = grid.astype(float) / (out_res - 1) 
        grid = grid.reshape(3, -1).transpose(1, 0)
        self.eval_points = grid

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

        res = self.out_res
        projs = np.zeros((3, 1, res, res), dtype=np.float32)

        #  制作标签 (GT)
        if self.split == 'train':
            # 1. 找到所有非零区域（脑子）
            foreground_indices = np.argwhere(vol_clean > 1e-4)
            
            if len(foreground_indices) > 0:
                # 2. 采样策略：70% 脑子，30% 背景
                n_fg = int(self.npoint * 0.7)
                n_rnd = self.npoint - n_fg
                
                # A. 前景采样 (从 foreground_indices 里随机挑)
                choice_fg = np.random.choice(len(foreground_indices), n_fg, replace=True)
                coords_fg = foreground_indices[choice_fg]
                
                # B. 背景/全局采样 (随机坐标)
                coords_rnd = np.random.randint(0, self.out_res, (n_rnd, 3))
                
                # C. 合并
                coords = np.concatenate([coords_fg, coords_rnd], axis=0)
            else:
                # 异常保护：如果是全黑图
                coords = np.random.randint(0, self.out_res, (self.npoint, 3))

            # 获取 GT 值
            values = vol_clean[coords[:, 0], coords[:, 1], coords[:, 2]]
            
            # 归一化坐标到 [0, 1] 供投影使用
            points = coords.astype(np.float32) / (self.out_res - 1)
        else:
            # 全图评估
            points = self.eval_points 
            values = np.zeros(len(points)) 

        # 5. 投影坐标
        points_norm = (points - 0.5) * 2 
        proj_points = []
        for i in range(3):
            proj_points.append(self.geo.project(points_norm, i))
        proj_points = np.stack(proj_points, axis=0) 

        return {
            'name': name,
            'dst_name': 'BraTS',
            'projs': projs.astype(np.float32),          # Input (Noisy)
            'points': points_norm.astype(np.float32),
            'proj_points': proj_points.astype(np.float32),
            'p_gt': values[None, :].astype(np.float32), # GT (Clean)
            'image': vol_clean[None, ...]               # Vis (Clean)
        }