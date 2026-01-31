# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np

# from models.unet import UNet
# from models.point_classifier import SurfaceClassifier
# from models.attention import SVC_Block

# def index_2d(feat, uv):
#     # feat: [B, C, H, W]
#     # uv: [B, N, 2]
#     uv = uv.unsqueeze(2) # [B, N, 1, 2]
#     feat = feat.transpose(2, 3) # [W, H]
#     samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True) # [B, C, N, 1]
#     return samples[:, :, :, 0] # [B, C, N]

# # --- 改进: 3D 残差块 ---
# class ResBlock3D(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm3d(channels)
#         self.act = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm3d(channels)

#     def forward(self, x):
#         residual = x
#         x = self.act(self.bn1(self.conv1(x)))
#         x = self.bn2(self.conv2(x))
#         x += residual
#         x = self.act(x)
#         return x

# # --- 改进: 可学习的下采样块 (替代 AvgPool) ---
# class ConvDownsampler(nn.Module):
#     def __init__(self, in_ch, scale):
#         super().__init__()
#         self.scale = scale
#         if scale == 1:
#             self.net = nn.Identity()
#         else:
#             # 使用 strided convolution 进行下采样，保留更多特征
#             # scale=2 -> stride=2, scale=4 -> stride=4 (或者两层 stride=2)
#             layers = []
#             current_scale = 1
#             while current_scale < scale:
#                 layers.append(nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=2, padding=1))
#                 layers.append(nn.BatchNorm2d(in_ch))
#                 layers.append(nn.LeakyReLU(inplace=True))
#                 current_scale *= 2
#             self.net = nn.Sequential(*layers)
            
#     def forward(self, x):
#         return self.net(x)

# # --- 改进: MS-3DV 模块 ---
# class MS3DV(nn.Module):
#     def __init__(self, in_ch, out_ch, grid_res_base=32, scales=[1, 2, 4]):
#         """
#         grid_res_base: 建议提升到 32 或 64 以保留几何细节 (论文为 16)
#         """
#         super().__init__()
#         self.scales = scales
#         self.grid_res_base = grid_res_base
        
#         # 1. 多尺度下采样 (Learnable)
#         self.downsamplers = nn.ModuleList([
#             ConvDownsampler(in_ch, s) for s in scales
#         ])
        
#         # 2. 多尺度 3D 处理
#         self.convs_3d = nn.ModuleList()
#         for _ in scales:
#             self.convs_3d.append(nn.Sequential(
#                 nn.Conv3d(in_ch, in_ch, kernel_size=1), 
#                 ResBlock3D(in_ch),
#                 ResBlock3D(in_ch)
#             ))
            
#         # 3. 融合 MLP
#         self.fusion_mlp = nn.Sequential(
#             nn.Linear(in_ch * len(scales), in_ch * 2),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_ch * 2, out_ch)
#         )

#     def make_grid(self, B, res, device):
#         d = torch.linspace(-1, 1, res, device=device)
#         mesh = torch.stack(torch.meshgrid(d, d, d, indexing='ij'), dim=-1)
#         return mesh.unsqueeze(0).expand(B, -1, -1, -1, -1)

#     def back_project(self, feat_2d_list, grid_3d):
#         B, Views, C, H, W = feat_2d_list.shape
#         B, R, _, _, _ = grid_3d.shape
#         pts = grid_3d.reshape(B, -1, 3) 
        
#         vol_feats = []

#         for v_idx in range(Views):
#             if v_idx == 0:   uv = pts[..., [0, 1]] 
#             elif v_idx == 1: uv = pts[..., [0, 2]] 
#             elif v_idx == 2: uv = pts[..., [1, 2]] 
#             else: continue 

#         # for v_idx in range(Views):
#         #     # --- 【修正开始】 ---
#         #     # 这里的 pts 是 (D, H, W) 顺序 (即 z, y, x)
#         #     # grid_sample 需要 (x, y) 顺序
            
#         #     # View 0 (Axial, H-W平面): 
#         #     # 对应 pts 的后两维 (H, W)。即 indices [1, 2]
#         #     if v_idx == 0:   
#         #         uv = pts[..., [2, 1]] # 取 (x=W, y=H) 适配 grid_sample
            
#         #     # View 1 (Coronal, H-D平面): 
#         #     # 对应 pts 的 (D, H)。即 indices [0, 1]
#         #     # 注意：slice_coronal 切的是 W，剩下 (D, H)。grid_sample 需要 (H, D) 或 (D, H) 取决于 transpose
#         #     # 你的 index_2d 做了 transpose(2,3)，所以这里取 (x=H, y=D)
#         #     elif v_idx == 1: 
#         #         uv = pts[..., [1, 0]] 
            
#         #     # View 2 (Sagittal, W-D平面): 
#         #     # 对应 pts 的 (D, W)。即 indices [0, 2]
#         #     # index_2d transpose 后需要 (x=W, y=D)
#         #     elif v_idx == 2: 
#         #         uv = pts[..., [2, 0]] 
#         #     else: continue

            
#             sampled = index_2d(feat_2d_list[:, v_idx], uv)
#             vol_feats.append(sampled.reshape(B, C, R, R, R))
            
#         vol_stack = torch.stack(vol_feats, dim=1) 
#         vol_agg, _ = torch.max(vol_stack, dim=1) 
#         return vol_agg

#     def forward(self, proj_feats, query_points):
#         B, V, C, H, W = proj_feats.shape
#         device = proj_feats.device
#         feat_samples = []
        
#         for i, s in enumerate(self.scales):
#             # 1. 下采样
#             # Reshape [B, V, ...] -> [B*V, ...] specifically for Conv layers if needed
#             # But ConvDownsampler expects [N, C, H, W]
#             pf_s = proj_feats.view(B*V, C, H, W)
#             pf_s = self.downsamplers[i](pf_s)
#             _, _, H_s, W_s = pf_s.shape
#             pf_s = pf_s.view(B, V, C, H_s, W_s)
            
#             # 2. 反投影
#             res = max(self.grid_res_base // s, 4)
#             grid_3d = self.make_grid(B, res, device)
#             vol_feat = self.back_project(pf_s, grid_3d)
            
#             # 3. 3D 卷积
#             vol_feat = self.convs_3d[i](vol_feat)
            
#             # 4. 采样
#             q_grid = query_points.view(B, 1, 1, -1, 3)
#             sampled = F.grid_sample(vol_feat, q_grid, align_corners=True) 
#             sampled = sampled.view(B, C, -1).transpose(1, 2) 
#             feat_samples.append(sampled)
            
#         cat_feat = torch.cat(feat_samples, dim=-1)
#         out_feat = self.fusion_mlp(cat_feat)
#         return out_feat

# class MLP(nn.Module):
#     def __init__(self, mlp_list, use_bn=False):
#         super().__init__()
#         layers = []
#         for i in range(len(mlp_list) - 1):
#             layers += [nn.Conv2d(mlp_list[i], mlp_list[i + 1], kernel_size=1)]
#             if use_bn:
#                 layers += [nn.BatchNorm2d(mlp_list[i + 1])]
#             layers += [nn.LeakyReLU(inplace=True),]
#         self.layer = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.layer(x)

# class DIF_Net(nn.Module):
#     def __init__(self, num_views, combine, mid_ch=128):
#         super().__init__()
#         self.combine = combine
#         self.image_encoder = UNet(1, mid_ch)

#         if self.combine == 'attention':
#             # [关键修改] grid_res_base 提升至 32 (原16)，增加细节
#             self.ms3dv = MS3DV(in_ch=mid_ch, out_ch=mid_ch, grid_res_base=32, scales=[1, 2, 4])
#             self.fusion_block = SVC_Block(dim=mid_ch, num_heads=4, mlp_ratio=2.)
            
#             # [关键修改] 融合层输入维度翻倍 (mid_ch * 2)
#             # 因为我们要拼接: [SVC_Output, Max_Pixel_Feat]
#             self.point_classifier = SurfaceClassifier(
#                 [mid_ch * 2, 256, 64, 16, 1], 
#                 no_residual=False
#             )
#         elif self.combine == 'mlp':
#             self.view_mixer = MLP([num_views, num_views // 2, 1])
#             self.point_classifier = SurfaceClassifier(
#                 [mid_ch, 256, 64, 16, 1],
#                 no_residual=False
#             )
        
#         # Fallback for old combines
#         if self.combine != 'attention':
#              self.point_classifier = SurfaceClassifier(
#                 [mid_ch, 256, 64, 16, 1],
#                 no_residual=False
#             )

#         print(f'DIF_Net Enhanced, mid_ch: {mid_ch}, combine: {self.combine}')

#     def forward(self, data, is_eval=False, eval_npoint=100000):
#         projs = data['projs'] 
#         b, m, c, w, h = projs.shape
#         projs = projs.reshape(b * m, c, w, h) 
#         proj_feats = self.image_encoder(projs)
        
#         proj_feats = list(proj_feats) if type(proj_feats) is tuple else [proj_feats]
#         for i in range(len(proj_feats)):
#             _, c_, w_, h_ = proj_feats[i].shape
#             proj_feats[i] = proj_feats[i].reshape(b, m, c_, w_, h_) 

#         if not is_eval:
#             p_pred = self.forward_points(proj_feats, data)
#             p_gt = data['p_gt']
#             return p_pred, p_gt
#         else:
#             total_npoint = data['proj_points'].shape[2]
#             n_batch = int(np.ceil(total_npoint / eval_npoint))
#             pred_list = []
#             for i in range(n_batch):
#                 left = i * eval_npoint
#                 right = min((i + 1) * eval_npoint, total_npoint)
#                 batch_data = {
#                     'proj_points': data['proj_points'][..., left:right, :],
#                     'points': data['points'][..., left:right, :], 
#                 }
#                 p_pred = self.forward_points(proj_feats, batch_data) 
#                 pred_list.append(p_pred)
#             pred = torch.cat(pred_list, dim=2)
#             return pred
    
#     def forward_points(self, proj_feats, data):
#         feat_map = proj_feats[0] # [B, M, C, W, H]
#         n_view = feat_map.shape[1]

#         # 1. Pixel-aligned features
#         p_list = []
#         for i in range(n_view):
#             feat = feat_map[:, i, ...] 
#             p = data['proj_points'][:, i, ...] 
#             p_feats = index_2d(feat, p) 
#             p_list.append(p_feats)
        
#         # [B, C, N, M]
#         p_stack = torch.stack(p_list, dim=-1) 

#         if self.combine == 'attention':
#             # 准备 Attention 输入
#             # Key/Value: [B, M, N, C]
#             x_slices = p_stack.permute(0, 3, 2, 1) 
            
#             # Query: [B, N, C] (来自 MS-3DV)
#             ms3dv_feat = self.ms3dv(feat_map, data['points']) 
            
#             # Cross-Attention
#             # Output: [B, N, C]
#             svc_out = self.fusion_block(x_query=ms3dv_feat, x_slices=x_slices)
            
#             # [关键修改] Shortcut Connection (直连通路)
#             # 我们从原始 Pixel Features 中提取一个强特征 (e.g., Max Pooling)
#             # 这样保证即使 3D 模块糊了，原始的高清纹理还在
#             # p_stack: [B, C, N, M] -> max -> [B, C, N] -> permute -> [B, N, C]
#             pixel_max, _ = torch.max(p_stack, dim=-1)
#             pixel_max = pixel_max.permute(0, 2, 1)
            
#             # Concatenate: [B, N, C*2]
#             p_final = torch.cat([svc_out, pixel_max], dim=-1)
            
#             # [B, N, C*2] -> [B, C*2, N] for classifier
#             p_fused = p_final.transpose(1, 2)
            
#         elif self.combine == 'max':
#             p_feats = p_stack.permute(0, 3, 2, 1)
#             p_fused = F.max_pool2d(p_feats, (1, n_view)).squeeze(-1)
#             p_fused = p_fused.transpose(1, 2) # fix dim for consistency if needed, but old code was squeezed
            
#         elif self.combine == 'mlp':
#             p_feats = p_stack.permute(0, 3, 1, 2) 
#             p_fused = self.view_mixer(p_feats).squeeze(1)
            
#         else:
#             raise NotImplementedError

#         # 3. Point-wise classification
#         p_pred = self.point_classifier(p_fused)
#         return p_pred

# def print_model_parm_nums(model):
#     total = sum([param.nelement() for param in model.parameters()])
#     trainable = sum([param.nelement() for param in model.parameters() if param.requires_grad])
#     print(f"  + Number of params: {total / 1e6:.2f} M")
#     print(f"  + Trainable params: {trainable / 1e6:.2f} M")






import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.unet import UNet
from models.point_classifier import SurfaceClassifier
from models.attention import SVC_Block

# --- [新增] 位置编码函数 ---
def positional_encoding(p, L=10):
    """
    p: [B, N, 3] 归一化到 [-1, 1] 或 [0, 1] 的坐标
    L: 频率数量
    Return: [B, N, 3 + 3*2*L]
    """
    pi = 3.1415926
    out = [p]
    for i in range(L):
        out.append(torch.sin(2 ** i * pi * p))
        out.append(torch.cos(2 ** i * pi * p))
    return torch.cat(out, dim=-1)

def index_2d(feat, uv):
    # feat: [B, C, H, W]
    # uv: [B, N, 2]
    uv = uv.unsqueeze(2) # [B, N, 1, 2]
    feat = feat.transpose(2, 3) # [W, H]
    samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True) # [B, C, N, 1]
    return samples[:, :, :, 0] # [B, C, N]

# --- 改进: 3D 残差块 ---
class ResBlock3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        # self.bn1 = nn.BatchNorm3d(channels)
        self.bn1 = nn.InstanceNorm3d(channels)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm3d(channels)
        self.bn2 = nn.InstanceNorm3d(channels)

    def forward(self, x):
        residual = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = self.act(x)
        return x

# --- 改进: 可学习的下采样块 (替代 AvgPool) ---
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
                # layers.append(nn.BatchNorm2d(in_ch))
                layers.append(nn.InstanceNorm2d(in_ch))
                layers.append(nn.LeakyReLU(inplace=True))
                current_scale *= 2
            self.net = nn.Sequential(*layers)
            
    def forward(self, x):
        return self.net(x)

# --- 改进: MS-3DV 模块 ---
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

    def forward(self, proj_feats, query_points):
        B, V, C, H, W = proj_feats.shape
        device = proj_feats.device
        feat_samples = []
        
        for i, s in enumerate(self.scales):
            pf_s = proj_feats.view(B*V, C, H, W)
            pf_s = self.downsamplers[i](pf_s)
            _, _, H_s, W_s = pf_s.shape
            pf_s = pf_s.view(B, V, C, H_s, W_s)
            
            res = max(self.grid_res_base // s, 4)
            grid_3d = self.make_grid(B, res, device)
            vol_feat = self.back_project(pf_s, grid_3d)
            
            vol_feat = self.convs_3d[i](vol_feat)
            
            q_grid = query_points.view(B, 1, 1, -1, 3)
            sampled = F.grid_sample(vol_feat, q_grid, align_corners=True) 
            sampled = sampled.view(B, C, -1).transpose(1, 2) 
            feat_samples.append(sampled)
            
        cat_feat = torch.cat(feat_samples, dim=-1)
        out_feat = self.fusion_mlp(cat_feat)
        return out_feat

class MLP(nn.Module):
    def __init__(self, mlp_list, use_bn=False):
        super().__init__()
        layers = []
        for i in range(len(mlp_list) - 1):
            layers += [nn.Conv2d(mlp_list[i], mlp_list[i + 1], kernel_size=1)]
            if use_bn:
                # layers += [nn.BatchNorm2d(mlp_list[i + 1])]
                layers += [nn.InstanceNorm2d(mlp_list[i + 1])]
            layers += [nn.LeakyReLU(inplace=True),]
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class DIF_Net(nn.Module):
    def __init__(self, num_views, combine, mid_ch=128):
        super().__init__()
        self.combine = combine
        self.image_encoder = UNet(1, mid_ch)
        
        # [新增] 位置编码参数
        self.pe_L = 10
        pe_dim = 3 + 3 * 2 * self.pe_L # 3 + 36 = 39

        if self.combine == 'attention':
            self.ms3dv = MS3DV(in_ch=mid_ch, out_ch=mid_ch, grid_res_base=32, scales=[1, 2, 4])
            self.fusion_block = SVC_Block(dim=mid_ch, num_heads=4, mlp_ratio=2.)
            
            # [关键修改] 输入维度 = Feature Dim + PosEncoding Dim
            # Feature Dim = mid_ch * 2 (Attention + MaxPool Shortcut)
            in_dim = mid_ch * 2 + pe_dim
            
            self.point_classifier = SurfaceClassifier(
                # 输入 -> 6层 hidden -> 输出
                [in_dim, 256, 256, 256, 256, 256, 128, 1], 
                no_residual=False, # 你的代码里实现了 Dense/Res 连接，深网络更需要这个
                last_op=nn.Sigmoid()
            )
        elif self.combine == 'mlp':
            self.view_mixer = MLP([num_views, num_views // 2, 1])
            
            # Feature Dim = mid_ch (MLP output)
            in_dim = mid_ch + pe_dim
            
            self.point_classifier = SurfaceClassifier(
                [in_dim, 256, 64, 16, 1],
                no_residual=False,
                last_op=nn.Sigmoid()
            )
        
        # Fallback
        if self.combine != 'attention' and self.combine != 'mlp':
             # Feature Dim = mid_ch (Assuming MaxPool)
             in_dim = mid_ch + pe_dim
             self.point_classifier = SurfaceClassifier(
                [in_dim, 256, 64, 16, 1],
                no_residual=False,
                last_op=nn.Sigmoid()
            )

        print(f'DIF_Net Enhanced with PosEnc, mid_ch: {mid_ch}, combine: {self.combine}, PE_dim: {pe_dim}')

    def forward(self, data, is_eval=False, eval_npoint=100000):
        projs = data['projs'] 
        b, m, c, w, h = projs.shape
        projs = projs.reshape(b * m, c, w, h) 
        proj_feats = self.image_encoder(projs)
        
        proj_feats = list(proj_feats) if type(proj_feats) is tuple else [proj_feats]
        for i in range(len(proj_feats)):
            _, c_, w_, h_ = proj_feats[i].shape
            proj_feats[i] = proj_feats[i].reshape(b, m, c_, w_, h_) 

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
                batch_data = {
                    'proj_points': data['proj_points'][..., left:right, :],
                    'points': data['points'][..., left:right, :], 
                }
                p_pred = self.forward_points(proj_feats, batch_data) 
                pred_list.append(p_pred)
            pred = torch.cat(pred_list, dim=2)
            return pred
    
    def forward_points(self, proj_feats, data):
        feat_map = proj_feats[0] # [B, M, C, W, H]
        n_view = feat_map.shape[1]

        # 1. Pixel-aligned features
        p_list = []
        for i in range(n_view):
            feat = feat_map[:, i, ...] 
            p = data['proj_points'][:, i, ...] 
            p_feats = index_2d(feat, p) 
            p_list.append(p_feats)
        
        p_stack = torch.stack(p_list, dim=-1) # [B, C, N, M]

        if self.combine == 'attention':
            x_slices = p_stack.permute(0, 3, 2, 1) 
            ms3dv_feat = self.ms3dv(feat_map, data['points']) 
            svc_out = self.fusion_block(x_query=ms3dv_feat, x_slices=x_slices)
            
            pixel_max, _ = torch.max(p_stack, dim=-1)
            pixel_max = pixel_max.permute(0, 2, 1)
            
            # [修改] 拼接特征
            p_fused = torch.cat([svc_out, pixel_max], dim=-1)
            
        elif self.combine == 'max':
            p_feats = p_stack.permute(0, 3, 2, 1)
            p_fused = F.max_pool2d(p_feats, (1, n_view)).squeeze(-1)
            p_fused = p_fused.permute(0, 2, 1) # [B, N, C]
            
        elif self.combine == 'mlp':
            p_feats = p_stack.permute(0, 3, 1, 2) 
            p_fused = self.view_mixer(p_feats).squeeze(1)
            p_fused = p_fused.permute(0, 2, 1) # [B, N, C]
            
        else:
            raise NotImplementedError

        # [新增] 位置编码注入
        # data['points']: [B, N, 3] -> PosEnc -> [B, N, 39]
        pos_enc = positional_encoding(data['points'], L=self.pe_L)
        
        # 拼接: [B, N, FeatDim] + [B, N, PEDim] -> [B, N, TotalDim]
        p_in = torch.cat([p_fused, pos_enc], dim=-1)
        
        # 调整为 Conv1d 输入格式: [B, TotalDim, N]
        p_in = p_in.transpose(1, 2)

        # 3. Point-wise classification
        p_pred = self.point_classifier(p_in)
        return p_pred

def print_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    trainable = sum([param.nelement() for param in model.parameters() if param.requires_grad])
    print(f"  + Number of params: {total / 1e6:.2f} M")
    print(f"  + Trainable params: {trainable / 1e6:.2f} M")