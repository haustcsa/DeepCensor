import torch
import torch.nn as nn
import torch.nn.functional as F

# 下采样卷积模块
class DownsampleConv(nn.Module):
    def __init__(self, in_channels):
        super(DownsampleConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2)

    def forward(self, X_s, X_f):
        X_s = self.conv(X_s)  # 对 X_s 进行下采样
        X_f = self.conv(X_f)  # 对 X_f 进行下采样
        out = X_s + X_f  # 相加得到 out
        return out

# 空间金字塔池化（SPP）模块
class SpatialPyramidPooling(nn.Module):
    def __init__(self):
        super(SpatialPyramidPooling, self).__init__()
        # 定义池化操作
        self.pool1x1 = nn.AdaptiveAvgPool2d((1, 1))  # 1x1 池化
        self.pool4x4 = nn.MaxPool2d(kernel_size=4, stride=4)  # 4x4 池化
        self.pool2x2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2 池化

    def forward(self, x):
        # 输入特征图大小为 1×512×8×8
        b, c, h, w = x.size()

        # 1×1 池化
        out_1x1 = self.pool1x1(x)  # 输出 1×512×1×1
        out_1x1 = out_1x1.view(b, -1)  # 转化为 1×512 向量
        #print('bbbbbbbbbb', out_1x1.size())
        # 4×4 池化
        out_4x4 = self.pool4x4(x)  # 输出 1×512×2×2
        #out_4x4 = nn.AdaptiveAvgPool2d((1, 1))(out_4x4)
        out_4x4 = out_4x4.view(b, -1)  # 转化为 1×2048 向量
        #print('bbbbbbbbbb2', out_4x4.size())
        # # 2×2 池化
        # out_2x2 = self.pool2x2(x)  # 输出 1×512×4×4
        # out_2x2 = out_2x2.view(b, -1)  # 转化为 1×8192 向量

        # 将所有池化结果拼接
        spp_out = torch.cat([out_1x1, out_4x4], dim=1)  # 最终输出 1×10752 的向量
        return spp_out