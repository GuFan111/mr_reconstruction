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


# ==========================================
# 🟢 依然保留：利用正交切片构建 3D 可视外壳
# 极其有用的全局上下文脚手架
# ==========================================
def construct_visual_hull(projs, size=(128, 128, 128)):
    ax = projs[:, 0]
    co = projs[:, 1]
    sa = projs[:, 2]

    ax_3d = ax.unsqueeze(-1).expand(-1, -1, -1, -1, size[2])
    co_3d = co.unsqueeze(3).expand(-1, -1, -1, size[1], -1)
    sa_3d = sa.unsqueeze(2).expand(-1, -1, size[0], -1, -1)

    visual_hull = (ax_3d + co_3d + sa_3d) / 3.0
    return visual_hull


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
    def __init__(self, in_ch=2, out_ch=32):
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


class TriPlaneViewAttention(nn.Module):
    def __init__(self, view_ch, prior_ch, hidden_ch=64):
        super().__init__()
        self.view_scorer = nn.Sequential(
            nn.Linear(3 + prior_ch, hidden_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_ch, hidden_ch // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_ch // 2, 3),
            nn.Softmax(dim=-1)
        )

        self.out_proj = nn.Sequential(
            nn.Linear(view_ch, view_ch),
            nn.LayerNorm(view_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, pts_3d, prior_feats, view_feats_stack):
        B, C, N, V = view_feats_stack.shape
        prior_feats_t = prior_feats.transpose(1, 2)
        attn_query = torch.cat([pts_3d, prior_feats_t], dim=-1)
        view_weights = self.view_scorer(attn_query)
        view_weights_exp = view_weights.unsqueeze(1)
        fused_view_feats = (view_feats_stack * view_weights_exp).sum(dim=-1)
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

        self.pe_L = 2
        pe_dim = 3 + 3 * 2 * self.pe_L

        if self.combine == 'attention':
            self.triplane_attn = TriPlaneViewAttention(view_ch=mid_ch, prior_ch=prior_ch)
            in_dim = mid_ch + prior_ch + pe_dim
            self.point_classifier = SurfaceClassifier(
                [in_dim, 256, 256, 256, 256, 256, 256, 256, 1], no_residual=True) # 3 -> 1
        elif self.combine == 'mlp':
            self.view_mixer = MLP([num_views, num_views // 2, 1])
            in_dim = mid_ch + prior_ch + pe_dim
            self.point_classifier = SurfaceClassifier(
                [in_dim, 256, 64, 16, 1], no_residual=True) # 3 -> 1
        else:
            raise NotImplementedError

        self.cached_prior_feats = None

    def clear_cache(self):
        self.cached_prior_feats = None

    def forward(self, data, is_eval=False, eval_npoint=100000, use_cache=False):
        # 1. 获取 3D 先验特征
        if is_eval and use_cache and self.cached_prior_feats is not None:
            prior_feats_vol = self.cached_prior_feats
        else:
            prior_vol = data['prior_mask']
            target_projs = data['projs']

            visual_hull = construct_visual_hull(target_projs, size=prior_vol.shape[2:])
            prior_in = torch.cat([prior_vol, visual_hull], dim=1)

            prior_feats_vol = self.prior_encoder(prior_in)

            if is_eval and use_cache:
                self.cached_prior_feats = prior_feats_vol

        # 2. 2D 视角特征提取
        target_projs = data['projs']
        prior_projs = data['prior_projs']
        combined_projs = torch.cat([target_projs, prior_projs], dim=2)
        b, m, c, w, h = combined_projs.shape

        combined_projs = combined_projs.reshape(b * m, c, w, h)
        proj_feats = self.image_encoder(combined_projs)

        proj_feats = list(proj_feats) if type(proj_feats) is tuple else [proj_feats]
        for i in range(len(proj_feats)):
            _, c_, w_, h_ = proj_feats[i].shape
            proj_feats[i] = proj_feats[i].reshape(b, m, c_, w_, h_)

        # 3. 点位查询
        if not is_eval:
            # 训练模式：直接返回预测概率 (Logits)
            p_pred = self.forward_points(proj_feats, prior_feats_vol, data)
            return p_pred
        else:
            # 推理模式：分块查询防止 OOM
            total_npoint = data['proj_points'].shape[2]
            n_batch = int(np.ceil(total_npoint / eval_npoint))
            pred_list = []

            for i in range(n_batch):
                left = i * eval_npoint
                right = min((i + 1) * eval_npoint, total_npoint)

                batch_data = {
                    'proj_points': data['proj_points'][..., left:right, :],
                    'points': data['points'][..., left:right, :]
                }

                p_pred_batch = self.forward_points(proj_feats, prior_feats_vol, batch_data)
                pred_list.append(p_pred_batch)

                # 沿着 N 轴拼接所有的预测概率
            pred = torch.cat(pred_list, dim=2)
            return pred

    def forward_points(self, proj_feats, prior_feats_vol, data):
        feat_map = proj_feats[0]
        n_view = feat_map.shape[1]

        p_list = []
        for i in range(n_view):
            feat = feat_map[:, i, ...]
            p = data['proj_points'][:, i, ...]
            p_feats = index_2d(feat, p)
            p_list.append(p_feats)
        p_stack = torch.stack(p_list, dim=-1)

        p_prior = index_3d(prior_feats_vol, data['points'])

        if self.combine == 'attention':
            fused_view_feats, view_weights = self.triplane_attn(
                pts_3d=data['points'],
                prior_feats=p_prior,
                view_feats_stack=p_stack
            )
            p_fused = torch.cat([fused_view_feats.transpose(1, 2), p_prior.transpose(1, 2)], dim=-1)
        elif self.combine == 'mlp':
            p_feats = p_stack.permute(0, 3, 1, 2)
            p_fused = self.view_mixer(p_feats).squeeze(1)
            p_fused = p_fused.permute(0, 2, 1)
            p_fused = torch.cat([p_fused, p_prior.transpose(1, 2)], dim=-1)

        pos_enc = positional_encoding(data['points'], L=self.pe_L)
        p_in = torch.cat([p_fused, pos_enc], dim=-1)

        p_in = p_in.transpose(1, 2)

        out = self.point_classifier(p_in)

        return out