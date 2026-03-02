# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.unet import UNet
from models.point_classifier import SurfaceClassifier
from models.attention import SVC_Block


def positional_encoding(p, L=10):
    pi = 3.1415926
    out = [p]
    for i in range(L):
        out.append(torch.sin(2 ** i * pi * p))
        out.append(torch.cos(2 ** i * pi * p))
    return torch.cat(out, dim=-1)

def index_2d(feat, uv):
    uv = uv.unsqueeze(2)
    feat = feat.transpose(2, 3)
    samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True)
    return samples[:, :, :, 0]

def index_3d(feat, uv):
    uv_sampling = uv[..., [2, 1, 0]]
    uv_sampling = uv_sampling.reshape(uv.shape[0], 1, 1, uv.shape[1], 3)
    samples = torch.nn.functional.grid_sample(feat, uv_sampling, align_corners=True)
    return samples[:, :, 0, 0, :]

def index_3d_deform_local(feat, uv):
    """
    feat: [B, C, D, H, W] (采样源，如 prior volume)
    uv: [B, N, 3] (3D 采样坐标)
    """
    # 将坐标顺序从 (z, y, x) 转为 grid_sample 要求的 (x, y, z)
    # 并调整形状为 [B, 1, 1, N, 3] 以适配 5D grid_sample
    uv_sampling = uv[..., [2, 1, 0]]
    uv_sampling = uv_sampling.reshape(uv.shape[0], 1, 1, uv.shape[1], 3)

    # 执行采样，使用 border 模式处理越界坐标
    samples = F.grid_sample(feat, uv_sampling, align_corners=True, padding_mode='border')

    # 转换回 [B, C, N] 格式
    return samples.reshape(feat.shape[0], feat.shape[1], -1)


class ConvDownsampler(nn.Module):
    def __init__(self, in_ch, scale):
        super().__init__()
        self.scale = scale
        if scale == 1:
            self.net = nn.Identity()
        else:
            layers = []
            current_scale = 1
            while current_scale < scale:
                layers.append(nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=2, padding=1))
                layers.append(nn.InstanceNorm2d(in_ch))
                layers.append(nn.LeakyReLU(inplace=True))
                current_scale *= 2
            self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class ResBlock3D(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.norm1 = nn.InstanceNorm3d(channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.norm2 = nn.InstanceNorm3d(channels)

    def forward(self, x):
        residual = x
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act(out + residual)

class PriorEncoder(nn.Module):
    def __init__(self, in_ch=1, out_ch=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, 16, 3, padding=1),
            nn.InstanceNorm3d(16),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down_conv = nn.Sequential(
            nn.Conv3d(16, 32, 3, stride=2, padding=1),
            nn.InstanceNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.bottleneck = nn.Sequential(
            ResBlock3D(32, dilation=1),
            ResBlock3D(32, dilation=2),
            ResBlock3D(32, dilation=4),
            ResBlock3D(32, dilation=1)
        )
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(32 + 16, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.down_conv(c1)
        c2 = self.bottleneck(c2)
        c2_up = self.up(c2)
        out = self.fusion_conv(torch.cat([c2_up, c1], dim=1))
        return out


class MS3DV(nn.Module):
    def __init__(self, in_ch, out_ch, grid_res_base=32, scales=[1, 2, 4]):
        super().__init__()
        self.scales = scales
        self.grid_res_base = grid_res_base

        self.downsamplers = nn.ModuleList([
            ConvDownsampler(in_ch, s) for s in scales
        ])

        self.convs_3d = nn.ModuleList()
        for _ in scales:
            self.convs_3d.append(nn.Sequential(
                nn.Conv3d(in_ch, in_ch, kernel_size=1),
                ResBlock3D(in_ch),
                ResBlock3D(in_ch)
            ))

        self.fusion_mlp = nn.Sequential(
            nn.Linear(in_ch * len(scales), in_ch * 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch * 2, out_ch)
        )

    def make_grid(self, B, res, device):
        d = torch.linspace(-1, 1, res, device=device)
        mesh = torch.stack(torch.meshgrid(d, d, d, indexing='ij'), dim=-1)
        return mesh.unsqueeze(0).expand(B, -1, -1, -1, -1)

    def back_project(self, feat_2d_list, grid_3d):
        B, Views, C, H, W = feat_2d_list.shape
        B, R, _, _, _ = grid_3d.shape
        pts = grid_3d.reshape(B, -1, 3)

        vol_feats = []
        for v_idx in range(Views):
            if v_idx == 0:   uv = pts[..., [0, 1]]
            elif v_idx == 1: uv = pts[..., [0, 2]]
            elif v_idx == 2: uv = pts[..., [1, 2]]
            else: continue

            sampled = index_2d(feat_2d_list[:, v_idx], uv)
            vol_feats.append(sampled.reshape(B, C, R, R, R))

        vol_stack = torch.stack(vol_feats, dim=1)
        vol_agg, _ = torch.max(vol_stack, dim=1)
        return vol_agg

    def build_volumes(self, proj_feats):
        B, V, C, H, W = proj_feats.shape
        device = proj_feats.device
        volumes = []

        for i, s in enumerate(self.scales):
            pf_s = proj_feats.view(B*V, C, H, W)
            pf_s = self.downsamplers[i](pf_s)
            _, _, H_s, W_s = pf_s.shape
            pf_s = pf_s.view(B, V, C, H_s, W_s)

            res = max(self.grid_res_base // s, 4)
            grid_3d = self.make_grid(B, res, device)
            vol_feat = self.back_project(pf_s, grid_3d)
            vol_feat = self.convs_3d[i](vol_feat)
            volumes.append(vol_feat)

        return volumes

    def sample_features(self, volumes, query_points):
        B, N, _ = query_points.shape
        C = volumes[0].shape[1]
        feat_samples = []

        q_grid = query_points.view(B, 1, 1, -1, 3)

        for vol in volumes:
            sampled = F.grid_sample(vol, q_grid, align_corners=True)
            sampled = sampled.view(B, C, -1).transpose(1, 2)
            feat_samples.append(sampled)

        cat_feat = torch.cat(feat_samples, dim=-1)
        out_feat = self.fusion_mlp(cat_feat)
        return out_feat

    def forward(self, proj_feats, query_points):
        volumes = self.build_volumes(proj_feats)
        return self.sample_features(volumes, query_points)

class TriPlaneViewAttention(nn.Module):
    def __init__(self, view_ch, prior_ch, hidden_ch=64):
        """
        view_ch: 2D 投影特征的通道数 (mid_ch=128)
        prior_ch: 3D 先验特征的通道数 (prior_ch=64)
        """
        super().__init__()
        # 修复：打分器的输入是 3D 坐标 + 先验特征 (3 + prior_ch = 67)
        self.view_scorer = nn.Sequential(
            nn.Linear(3 + prior_ch, hidden_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_ch, hidden_ch // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_ch // 2, 3),
            nn.Softmax(dim=-1)
        )

        # 修复：融合映射是对 2D 视角特征进行的 (view_ch = 128)
        self.out_proj = nn.Sequential(
            nn.Linear(view_ch, view_ch),
            nn.LayerNorm(view_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, pts_3d, prior_feats, view_feats_stack):
        B, C, N, V = view_feats_stack.shape

        # 1. 准备 Query: 拼接坐标与先验特征 [B, N, 3 + prior_ch]
        prior_feats_t = prior_feats.transpose(1, 2)
        attn_query = torch.cat([pts_3d, prior_feats_t], dim=-1)

        # 2. 计算视角注意力权重
        view_weights = self.view_scorer(attn_query)

        # 3. 动态加权融合
        view_weights_exp = view_weights.unsqueeze(1)
        fused_view_feats = (view_feats_stack * view_weights_exp).sum(dim=-1)

        # 4. 最终映射
        fused_view_feats = fused_view_feats.transpose(1, 2)
        out_feats = self.out_proj(fused_view_feats)

        return out_feats.transpose(1, 2), view_weights


class MLP(nn.Module):
    def __init__(self, mlp_list, use_bn=False):
        super().__init__()
        layers = []
        for i in range(len(mlp_list) - 1):
            layers += [nn.Conv2d(mlp_list[i], mlp_list[i + 1], kernel_size=1)]
            if use_bn:
                layers += [nn.InstanceNorm2d(mlp_list[i + 1])]
            layers += [nn.LeakyReLU(inplace=True),]
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class DIF_Net(nn.Module):
    def __init__(self, num_views, combine, mid_ch=128):
        super().__init__()
        self.combine = combine
        self.image_encoder = UNet(2, mid_ch)

        prior_ch = mid_ch // 2
        self.prior_encoder = PriorEncoder(out_ch=prior_ch)

        self.pe_L = 6
        pe_dim = 3 + 3 * 2 * self.pe_L

        if self.combine == 'attention':
            self.triplane_attn = TriPlaneViewAttention(view_ch=mid_ch, prior_ch=prior_ch)

            in_dim = mid_ch + prior_ch + pe_dim
            self.point_classifier = SurfaceClassifier(
                [in_dim, 256, 256, 256, 256, 256, 256, 256, 3], no_residual=True)

        elif self.combine == 'mlp':
            self.view_mixer = MLP([num_views, num_views // 2, 1])
            in_dim = mid_ch + prior_ch + pe_dim
            self.point_classifier = SurfaceClassifier(
                [in_dim, 256, 64, 16, 3], no_residual=True)
        else:
            raise NotImplementedError

        print(f'DIF_Net Optimized, mid: {mid_ch}, prior: {prior_ch}')

        # 用于存储 3D 先验特征的缓存变量
        self.cached_prior_feats = None
        print(f'DIF_Net Optimized with Feature Caching, mid: {mid_ch}')

    def clear_cache(self):
        self.cached_prior_feats = None

    def forward(self, data, is_eval=False, eval_npoint=100000, use_cache=False):
        # 1. 获取 3D 先验特征 (保持不变)
        if is_eval and use_cache and self.cached_prior_feats is not None:
            prior_feats_vol = self.cached_prior_feats
        else:
            prior_vol = data['prior']
            prior_feats_vol = self.prior_encoder(prior_vol)
            if is_eval and use_cache:
                self.cached_prior_feats = prior_feats_vol

        # ==========================================================
        # 架构升级 2：截取并拼接 2D 切片
        # ==========================================================
        # 获取形变后的 Target 切片和原始的 Prior 切片
        target_projs = data['projs']             # [B, M, 1, W, H]
        prior_projs = data['prior_projs']        # [B, M, 1, W, H]

        # 在通道维度 (dim=2) 拼接，生成 2 通道的输入张量
        combined_projs = torch.cat([target_projs, prior_projs], dim=2) # [B, M, 2, W, H]

        b, m, c, w, h = combined_projs.shape     # 此时的 c 变成了 2

        # 折叠 Batch 和 View 维度，送入 UNet
        combined_projs = combined_projs.reshape(b * m, c, w, h)

        # 此时 UNet 直接“看”到了两张图的差异，输出的特征将携带极其强烈的位移梯度
        proj_feats = self.image_encoder(combined_projs)

        proj_feats = list(proj_feats) if type(proj_feats) is tuple else [proj_feats]
        for i in range(len(proj_feats)):
            _, c_, w_, h_ = proj_feats[i].shape
            proj_feats[i] = proj_feats[i].reshape(b, m, c_, w_, h_)

        if not is_eval:
            # 训练模式：直接调用，移除 ms3dv_volumes 参数
            p_pred, delta_coords = self.forward_points(
                proj_feats, prior_feats_vol, data)
            return p_pred, delta_coords
        else:
            # 推理模式：分块处理
            total_npoint = data['proj_points'].shape[2]
            n_batch = int(np.ceil(total_npoint / eval_npoint))
            pred_list = []
            delta_list = []

            for i in range(n_batch):
                left = i * eval_npoint
                right = min((i + 1) * eval_npoint, total_npoint)

                batch_data = {
                    'proj_points': data['proj_points'][..., left:right, :],
                    'points': data['points'][..., left:right, :],
                    'prior': data['prior']
                }

                # 同理，在推理循环中移除 ms3dv_volumes 参数
                p_pred_batch, delta_batch = self.forward_points(
                    proj_feats, prior_feats_vol, batch_data)

                pred_list.append(p_pred_batch)
                delta_list.append(delta_batch)

            # 统一在 dim=2 拼接
            pred = torch.cat(pred_list, dim=2)  # [B, 1, total_N]
            delta = torch.cat(delta_list, dim=2) # [B, 3, total_N]
            return pred, delta

    def forward_points(self, proj_feats, prior_feats_vol, data, return_delta_only=False):
        feat_map = proj_feats[0]
        n_view = feat_map.shape[1]

        # ==========================================================
        # 回滚 1：恢复平滑的 2D 特征投影 (移除距离门控 cutoff)
        # ==========================================================
        p_list = []
        for i in range(n_view):
            feat = feat_map[:, i, ...]
            p = data['proj_points'][:, i, ...]
            p_feats = index_2d(feat, p)
            p_list.append(p_feats)
        p_stack = torch.stack(p_list, dim=-1)

        # 2. 先验特征采样
        p_prior = index_3d(prior_feats_vol, data['points'])

        if self.combine == 'attention':
            fused_view_feats, view_weights = self.triplane_attn(
                pts_3d=data['points'],
                prior_feats=p_prior,
                view_feats_stack=p_stack
            )
            # 拼接: 融合后的2D视角特征 + 3D先验特征
            p_fused = torch.cat([fused_view_feats.transpose(1, 2), p_prior.transpose(1, 2)], dim=-1)

        elif self.combine == 'mlp':
            p_feats = p_stack.permute(0, 3, 1, 2)
            p_fused = self.view_mixer(p_feats).squeeze(1)
            p_fused = p_fused.permute(0, 2, 1)
            p_fused = torch.cat([p_fused, p_prior.transpose(1, 2)], dim=-1)
        else:
            raise NotImplementedError

        # 位置编码
        pos_enc = positional_encoding(data['points'], L=self.pe_L)
        p_in = torch.cat([p_fused, pos_enc], dim=-1)

        # 3. 坐标变形预测
        p_in = p_in.transpose(1, 2)
        out = self.point_classifier(p_in)

        # ==========================================================
        # 回滚 2：放宽物理截断边界
        # 相比于 [0.02, 0.06, 0.20]，给予更宽容的缓冲空间防止撞墙产生色块
        # ==========================================================
        amp = torch.tensor([0.05, 0.10, 0.25], device=out.device).view(1, 3, 1)
        delta_coords = torch.tanh(out) * amp

        corrected_coords = data['points'] + delta_coords.transpose(1, 2)
        sampled_val = index_3d_deform_local(data['prior'], corrected_coords)

        return sampled_val, delta_coords