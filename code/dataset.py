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
    def __init__(self, data_root, json_root, split='train', npoint=5000, out_res=(256, 256, 128), preload=False):
        super().__init__()
        self.data_root = data_root
        self.json_root = json_root # 你需要这个目录来存放器官坐标
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
        # 1. 加载数据
        if self.preload:
            vol_clean = self.data_cache[index]
            name = os.path.basename(self.file_list[index]).split('.')[0]
        else:
            path = self.file_list[index]
            name = os.path.basename(path).split('.')[0]
            vol_clean = np.load(path)

        res_x, res_y, res_z = self.out_res

        # --- 修改 1: 模拟 MR-Linac 的三帧投影 (MIP) ---
        # 即使是训练，也需要给模型“看”投影图，否则注意力无法对焦
        res_max = max(res_x, res_y, res_z)
        projs = np.zeros((3, 1, res_max, res_max), dtype=np.float32)
        projs[0, 0, :res_x, :res_y] = np.max(vol_clean, axis=2) # Axial
        projs[1, 0, :res_x, :res_z] = np.max(vol_clean, axis=1) # Coronal
        projs[2, 0, :res_y, :res_z] = np.max(vol_clean, axis=0) # Sagittal

        # --- 修改 2: 引入 ROI 引导采样 ---
        if self.split == 'train':
            json_path = os.path.join(self.json_root, f"{name}.json")
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    roi = json.load(f)
                # 70% 的点落在器官框内，解决“糊”的关键
                n_roi = int(self.npoint * 0.7)
                coords_roi = np.stack([
                    np.random.randint(roi['mins'][0], roi['maxs'][0], n_roi),
                    np.random.randint(roi['mins'][1], roi['maxs'][1], n_roi),
                    np.random.randint(roi['mins'][2], roi['maxs'][2], n_roi)
                ], axis=1)
                # 30% 全局随机
                coords_rnd = np.stack([
                    np.random.randint(0, res_x, self.npoint - n_roi),
                    np.random.randint(0, res_y, self.npoint - n_roi),
                    np.random.randint(0, res_z, self.npoint - n_roi)
                ], axis=1)
                coords = np.concatenate([coords_roi, coords_rnd], axis=0)
            else:
                # 备份：如果没 JSON，用你原来的 > 1e-4 逻辑
                fg_idx = np.argwhere(vol_clean > 0.1)
                coords = fg_idx[np.random.choice(len(fg_idx), self.npoint)]
        else:
            # 推理模式：全图网格采样 (维持你原来的 eval_points 逻辑)
            coords = self.eval_points_as_indices() # 假设你生成的索引

        # --- 修改 3: 物理空间归一化 ---
        values = vol_clean[coords[:, 0], coords[:, 1], coords[:, 2]]
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
            'points': points_norm.astype(np.float32),
            'proj_points': proj_points.astype(np.float32),
            'p_gt': values[None, :].astype(np.float32),
            'image': vol_clean[None, ...]
        }