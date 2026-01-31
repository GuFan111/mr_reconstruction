import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.unet import UNet
from models.point_classifier import SurfaceClassifier
from models.attention import SVC_Block

def index_2d(feat, uv):
    # feat: [B, C, H, W]
    # uv: [B, N, 2]
    uv = uv.unsqueeze(2) # [B, N, 1, 2]
    feat = feat.transpose(2, 3) # [W, H] -> 适配 grid_sample 的 (x, y) 坐标系
    samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True) # [B, C, N, 1]
    return samples[:, :, :, 0] # [B, C, N]

# --- 新增: 3D 残差块 (用于处理 MS-3DV 中的体素特征) ---
class ResBlock3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(channels)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(channels)

    def forward(self, x):
        residual = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = self.act(x)
        return x

# --- 新增: MS-3DV 模块 ---
class MS3DV(nn.Module):
    def __init__(self, in_ch, out_ch, grid_res_base=16, scales=[1, 2, 4]):
        """
        :param in_ch: 输入 2D 特征通道数 (e.g. 128)
        :param out_ch: 输出 3D 特征通道数 (e.g. 128)
        :param grid_res_base: 基础网格分辨率 (Scale 1), 论文中为 16
        :param scales: 下采样倍率列表
        """
        super().__init__()
        self.scales = scales
        self.grid_res_base = grid_res_base
        
        # 多尺度 3D 卷积处理层
        self.convs_3d = nn.ModuleList()
        for _ in scales:
            # 论文: 3-layer 3D residual convolution
            # 这里简化为一个 Conv3d + ResBlock
            self.convs_3d.append(nn.Sequential(
                nn.Conv3d(in_ch, in_ch, kernel_size=1), # Channel mapping if needed, here identity
                ResBlock3D(in_ch),
                ResBlock3D(in_ch)
            ))
            
        # 融合 MLP: Concatenate [S1, S2, S3] -> out_ch
        self.fusion_mlp = nn.Sequential(
            nn.Linear(in_ch * len(scales), in_ch * 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch * 2, out_ch)
        )

    def make_grid(self, B, res, device):
        # 生成 3D 网格坐标 [-1, 1]
        # shape: [1, R, R, R, 3]
        d = torch.linspace(-1, 1, res, device=device)
        mesh = torch.stack(torch.meshgrid(d, d, d, indexing='ij'), dim=-1)
        # 转为 (x, y, z) 顺序
        # meshgrid 'ij' -> (d1, d2, d3). 
        # 我们假设对应 (x, y, z) 顺序跟 dataset.py 里的 transpose 对应
        return mesh.unsqueeze(0).expand(B, -1, -1, -1, -1)

    def back_project(self, feat_2d_list, grid_3d):
        """
        Orthogonal Back-Projection (Simplified for BraTS Orthogonal views)
        feat_2d_list: [B, Views, C, H, W]
        grid_3d: [B, R, R, R, 3] (x, y, z)
        """
        B, Views, C, H, W = feat_2d_list.shape
        B, R, _, _, _ = grid_3d.shape
        
        # Flatten grid for sampling: [B, R^3, 3]
        pts = grid_3d.reshape(B, -1, 3) 
        
        # 投影到 2D 平面 (模拟 dataset.py 中的 OrthogonalGeometry)
        # View 0 (Axial): x, y
        # View 1 (Coronal): x, z
        # View 2 (Sagittal): y, z
        
        vol_feats = []
        for v_idx in range(Views):
            if v_idx == 0:   uv = pts[..., [0, 1]] # x, y
            elif v_idx == 1: uv = pts[..., [0, 2]] # x, z
            elif v_idx == 2: uv = pts[..., [1, 2]] # y, z
            else: continue # 只支持3视图
            
            # Sample: [B, C, N]
            sampled = index_2d(feat_2d_list[:, v_idx], uv)
            # Reshape back to volume: [B, C, R, R, R]
            vol_feats.append(sampled.reshape(B, C, R, R, R))
            
        # Aggregation: Max-Pooling across views (Eqn 4 in paper)
        vol_stack = torch.stack(vol_feats, dim=1) # [B, V, C, R, R, R]
        vol_agg, _ = torch.max(vol_stack, dim=1) # [B, C, R, R, R]
        return vol_agg

    def forward(self, proj_feats, query_points):
        """
        proj_feats: [B, Views, C, H, W]
        query_points: [B, N, 3]
        """
        B, V, C, H, W = proj_feats.shape
        device = proj_feats.device
        
        feat_samples = []
        
        # 对每个尺度进行处理
        for i, s in enumerate(self.scales):
            # 1. 2D 下采样 (得到 F^s)
            if s > 1:
                # view合并到batch做downsample
                pf_s = F.avg_pool2d(proj_feats.view(B*V, C, H, W), kernel_size=s, stride=s)
                pf_s = pf_s.view(B, V, C, H//s, W//s)
            else:
                pf_s = proj_feats
            
            # 2. 构建 3D 网格 & 反投影
            res = self.grid_res_base // s # e.g., 16 -> 8 -> 4 (Resolution decreases as per paper logic?)
            # 论文中: r^1=16, r^s=0.5*r^{s-1}. 
            # 通常 deeper scale 特征分辨率更低，这里我们假设 grid_res 随 scale 减小
            # 如果 grid_res 变得太小 (e.g. < 4)，可能效果不好，这里根据 Config 调整
            # 修正: 论文 Table 4 使用 r^1=16. 
            
            # 限制最小分辨率为 4
            res = max(res, 4)
            
            grid_3d = self.make_grid(B, res, device) # [B, R, R, R, 3]
            
            # 得到粗糙体素特征 [B, C, R, R, R]
            vol_feat = self.back_project(pf_s, grid_3d)
            
            # 3. 3D 卷积细化 (Eqn 5) -> \hat{F}^s
            vol_feat = self.convs_3d[i](vol_feat)
            
            # 4. 采样 Query Points (Eqn 7 part 1)
            # grid_sample expects [B, C, D, H, W] input and [B, D_out, H_out, W_out, 3] grid
            # query_points: [B, N, 3] -> [B, 1, 1, N, 3]
            q_grid = query_points.view(B, 1, 1, -1, 3)
            
            # grid_sample 需要 coordinate 在 [-1, 1]，input 也是这个范围
            # 注意: grid_sample 默认 align_corners=False (in newer pytorch), paper usually uses True/False consistent
            # index_2d 用了 True，这里也用 True
            sampled = F.grid_sample(vol_feat, q_grid, align_corners=True) # [B, C, 1, 1, N]
            sampled = sampled.view(B, C, -1).transpose(1, 2) # [B, N, C]
            
            feat_samples.append(sampled)
            
        # 5. 拼接与融合 (Eqn 7)
        # Cat: [B, N, C * 3]
        cat_feat = torch.cat(feat_samples, dim=-1)
        out_feat = self.fusion_mlp(cat_feat) # [B, N, C]
        
        return out_feat


class MLP(nn.Module):
    def __init__(self, mlp_list, use_bn=False):
        super().__init__()
        layers = []
        for i in range(len(mlp_list) - 1):
            layers += [nn.Conv2d(mlp_list[i], mlp_list[i + 1], kernel_size=1)]
            if use_bn:
                layers += [nn.BatchNorm2d(mlp_list[i + 1])]
            layers += [nn.LeakyReLU(inplace=True),]
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class DIF_Net(nn.Module):
    def __init__(self, num_views, combine, mid_ch=128):
        super().__init__()
        self.combine = combine

        # 2D Encoder (U-Net)
        self.image_encoder = UNet(1, mid_ch)

        # [新增] MS-3DV 模块
        # 论文配置: S=3, r1=16. Scales 可以是 [1, 2, 4] 对应 encoder output (128) 下采样到 128, 64, 32
        # 注意：这里的 scales 是相对于 proj_feats (128x128) 的。
        # grid_res 是体素网格分辨率，论文建议 16 -> 8 -> 4
        if self.combine == 'attention':
            self.ms3dv = MS3DV(in_ch=mid_ch, out_ch=mid_ch, grid_res_base=16, scales=[1, 2, 4])
            self.fusion_block = SVC_Block(dim=mid_ch, num_heads=4, mlp_ratio=2.)
            
        elif self.combine == 'mlp':
            self.view_mixer = MLP([num_views, num_views // 2, 1])
        
        self.point_classifier = SurfaceClassifier(
            [mid_ch, 256, 64, 16, 1],
            no_residual=False
        )
        print(f'DIF_Net with MS-3DV, mid_ch: {mid_ch}, combine: {self.combine}')

    def forward(self, data, is_eval=False, eval_npoint=100000):
        # projection encoding
        projs = data['projs'] # B, M, C, W, H
        b, m, c, w, h = projs.shape
        projs = projs.reshape(b * m, c, w, h) # B', C, W, H
        
        # 提取特征
        proj_feats = self.image_encoder(projs)
        # proj_feats: [B*M, C, W, H]
        
        proj_feats = list(proj_feats) if type(proj_feats) is tuple else [proj_feats]
        for i in range(len(proj_feats)):
            _, c_, w_, h_ = proj_feats[i].shape
            proj_feats[i] = proj_feats[i].reshape(b, m, c_, w_, h_) # B, M, C, W, H

        # point-wise forward
        if not is_eval:
            p_pred = self.forward_points(proj_feats, data)
            p_gt = data['p_gt']
            return p_pred, p_gt
        
        else:
            total_npoint = data['proj_points'].shape[2]
            n_batch = int(np.ceil(total_npoint / eval_npoint))

            pred_list = []
            for i in range(n_batch):
                left = i * eval_npoint
                right = min((i + 1) * eval_npoint, total_npoint)
                # 构造 eval batch 数据
                batch_data = {
                    'proj_points': data['proj_points'][..., left:right, :],
                    'points': data['points'][..., left:right, :], # 需要 points 坐标用于 MS-3DV
                }
                p_pred = self.forward_points(proj_feats, batch_data) # B, C, N
                pred_list.append(p_pred)

            pred = torch.cat(pred_list, dim=2)
            return pred
    
    def forward_points(self, proj_feats, data):
        # proj_feats: List[[B, M, C, W, H]] (Usually length 1)
        feat_map = proj_feats[0] # [B, M, C, W, H]
        n_view = feat_map.shape[1]

        # 1. Pixel-aligned features (原始 DIF-Net 逻辑)
        # Query view-specific features from 2D maps
        p_list = []
        for i in range(n_view):
            # View i
            feat = feat_map[:, i, ...] # B, C, W, H
            p = data['proj_points'][:, i, ...] # B, N, 2 (UV coords)
            p_feats = index_2d(feat, p) # B, C, N
            p_list.append(p_feats)
        
        # [B, C, N, M] -> [B, M, N, C]
        p_feats = torch.stack(p_list, dim=-1).permute(0, 3, 2, 1) 

        if self.combine == 'attention':
            # 2. [新增] Voxel-aligned features (MS-3DV)
            # 使用 3D 坐标直接从体素特征中采样，作为 Query
            # data['points']: [B, N, 3] (3D coords in -1~1)
            ms3dv_feat = self.ms3dv(feat_map, data['points']) # [B, N, C]
            
            # 3. Scale-View Cross-Attention
            # Query: 3D feature (MS-3DV), Key/Value: 2D view features
            # output: [B, N, C]
            p_fused = self.fusion_block(x_query=ms3dv_feat, x_slices=p_feats)
            
            # [B, N, C] -> [B, C, N] for classifier
            p_fused = p_fused.transpose(1, 2)
            
        elif self.combine == 'max':
            # Fallback (Old logic)
            # [B, M, N, C] -> [B, C, N, M]
            p_feats = p_feats.permute(0, 3, 2, 1)
            p_fused = F.max_pool2d(p_feats, (1, n_view)).squeeze(-1)
            
        elif self.combine == 'mlp':
            # Fallback (Old logic)
            # [B, M, N, C] -> [B, C, N, M] -> [B, M, C, N] ?
            # Logic in old code: permute(0, 3, 1, 2) -> [B, M, C, N]
            # view_mixer expects [B, M, C, N] (Conv2d on M)
            p_feats = p_feats.permute(0, 1, 3, 2) 
            p_fused = self.view_mixer(p_feats).squeeze(1)
            
        else:
            raise NotImplementedError

        # 4. Point-wise classification
        p_pred = self.point_classifier(p_fused)
        return p_pred

def print_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    trainable = sum([param.nelement() for param in model.parameters() if param.requires_grad])
    print(f"  + Number of params: {total / 1e6:.2f} M")
    print(f"  + Trainable params: {trainable / 1e6:.2f} M")

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DIF_Net(num_views=3, combine='attention', mid_ch=128).to(device)
    print_model_parm_nums(model)
    
    # Fake data test
    data = {
        'projs': torch.randn(2, 3, 1, 128, 128).to(device),
        'proj_points': torch.randn(2, 3, 100, 2).to(device),
        'points': torch.rand(2, 100, 3).to(device) * 2 - 1, # [-1, 1]
        'p_gt': torch.randn(2, 100).to(device)
    }
    out, _ = model(data)
    print("Output shape:", out.shape)