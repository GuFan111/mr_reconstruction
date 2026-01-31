# attention.py

import torch
import torch.nn as nn
from einops import rearrange



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class OrthogonalCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Query 映射 (来自 3D 坐标/特征)
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        
        # Key/Value 映射 (来自 3个切片)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_query, x_slices):
        B, N, C = x_query.shape
        
        # 1. Prepare Q, K, V
        # Q: [B, Heads, N, C/Heads]
        q = self.q_proj(x_query).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        # K, V 来自切片
        # x_slices shape: [B, 3, N, C] -> flatten views -> [B, N, 3, C]
        x_slices = x_slices.permute(0, 2, 1, 3) # [B, N, 3, C]
        
        # K: [B, Heads, N, 3, C/Heads]
        k = self.k_proj(x_slices).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(0, 3, 1, 2, 4)
        
        # V: [B, Heads, N, 3, C/Heads]
        v = self.v_proj(x_slices).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(0, 3, 1, 2, 4)

        # 2. Attention Score
        # 在 3 个视图间做 Attention
        # Q: [BH, N, 1, D], K: [BH, N, 3, D] -> Attn: [BH, N, 1, 3]
        
        # (Query * Key^T) * scale
        # Einsum 解释: bh: Batch*Head, n: Points, d: Dim, v: Views(3)
        # q (bh, n, 1, d) @ k.T (bh, n, d, v) -> (bh, n, 1, v)
        q = q.unsqueeze(-2) # [BH, N, 1, D]
        attn = (q @ k.transpose(-2, -1)) * self.scale # [BH, N, 1, 3]
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 3. Aggregation
        # (Attn * V) -> [BH, N, 1, D]
        x = (attn @ v).squeeze(2) # [BH, N, D]
        
        # Restore shape
        x = x.transpose(1, 2).reshape(B, N, C) # [B, N, C]
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class SVC_Block(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.cross_attn = OrthogonalCrossAttention(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)

    def forward(self, x_query, x_slices):
        # 1. Cross Attention: Query 关注 Slices
        # Residual connection
        x = x_query + self.cross_attn(self.norm1(x_query), x_slices)
        
        # 2. FFN
        x = x + self.mlp(self.norm2(x))
        return x