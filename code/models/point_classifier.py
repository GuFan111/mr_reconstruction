# point_classifier.py

import torch
import torch.nn as nn
import torch.nn.functional as F



# class SurfaceClassifier(nn.Module):
#     def __init__(self, filter_channels, no_residual=True, last_op=None):
#         super().__init__()

#         self.filters = []
#         self.no_residual = no_residual
#         self.last_op = last_op

#         if self.no_residual:
#             for l in range(len(filter_channels) - 1):
#                 self.filters.append(nn.Conv1d(
#                     filter_channels[l],
#                     filter_channels[l + 1],
#                     1))
#                 self.add_module("conv%d" % l, self.filters[l])
#         else:
#             for l in range(len(filter_channels) - 1):
#                 if 0 != l:
#                     self.filters.append(
#                         nn.Conv1d(
#                             filter_channels[l] + filter_channels[0],
#                             filter_channels[l + 1],
#                             1))
#                 else:
#                     self.filters.append(nn.Conv1d(
#                         filter_channels[l],
#                         filter_channels[l + 1],
#                         1))

#                 self.add_module("conv%d" % l, self.filters[l])

#         # if last_op is not None:
#         #     if self.filters[-1].bias is not None:
#         #         nn.init.constant_(self.filters[-1].bias, 0)

#     def forward(self, feature):
#         '''
#         :param feature: [B, C_in, N]
#         :return: [B, C_out, N]
#         '''
#         y = feature
#         tmpy = feature
#         for i in range(len(self.filters)):
#             if self.no_residual:
#                 y = self._modules['conv' + str(i)](y)
#             else:
#                 y = self._modules['conv' + str(i)](
#                     y if i == 0
#                     else torch.cat([y, tmpy], 1)
#                 )
            
#             if i != len(self.filters) - 1:
#                 y = F.leaky_relu(y)

#         if self.last_op:
#             y = self.last_op(y)

#         return y


class SurfaceClassifier(nn.Module):
    def __init__(self, filter_channels, no_residual=True, last_op=None):
        super().__init__()

        self.filters = nn.ModuleList()
        self.no_residual = no_residual
        
        # 网络层数的一半处拼接
        self.skip_layer = (len(filter_channels) - 1) // 2
        
        # 记录原始输入维度，用于计算拼接后的通道数
        self.input_dim = filter_channels[0]

        for l in range(len(filter_channels) - 1):
            in_ch = filter_channels[l]
            out_ch = filter_channels[l + 1]

            # 如果当前层是跳跃层，输入通道数 = 上一层输出 + 原始输入
            if l == self.skip_layer:
                in_ch += self.input_dim

            self.filters.append(nn.Conv1d(in_ch, out_ch, 1))

        # 2. 执行防撕裂初始化
        self._init_weights()

    def _init_weights(self):
        # A. 中间层：Kaiming 初始化
        for i, m in enumerate(self.filters[:-1]):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        # B. 最后一层：微小噪声初始化
        last_layer = self.filters[-1]
        nn.init.normal_(last_layer.weight, mean=0.0, std=1e-5)
        nn.init.constant_(last_layer.bias, 0)
        
    def forward(self, feature):
        y = feature
        original_input = feature

        for i, layer in enumerate(self.filters):
            if i == self.skip_layer:
                y = torch.cat([y, original_input], dim=1)
            
            y = layer(y)

            if i != len(self.filters) - 1:
                y = F.leaky_relu(y, 0.2, inplace=True)

        return y

