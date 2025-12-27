import time

import numpy as np
# -----frequency modules----- #
import torch
import torch.nn as nn
import torch.nn.functional as F

def StdFeatureMap(x, unbiased=False):
    """
    高效实现：通过卷积预计算差值，避免展开和掩码操作（修正版）
    """
    b, c, h, w = x.shape
    device = x.device

    # Step 1: Padding
    x_pad = F.pad(x, (1, 1, 1, 1))  # [b, c, h+2, w+2]

    # Step 2: 定义8个卷积核，对应3x3窗口中除中心外的8个位置
    kernels = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1)
    ]

    # 构造卷积核权重：形状为 [8*c, 1, 3, 3]
    weight = torch.zeros(8 * c, 1, 3, 3, device=device)
    for i in range(c):
        for j, (dx, dy) in enumerate(kernels):
            weight[i * 8 + j, 0, 1 + dx, 1 + dy] = 1.0
            weight[i * 8 + j, 0, 1, 1] = -1.0

    # Step 3: 使用分组卷积（groups=c），输出形状 [b, 8*c, h, w]
    diff = F.conv2d(
        x_pad,
        weight,
        groups=c,  # 关键修改：分组数等于通道数
        padding=0
    )

    # Step 4: 调整形状为 [b, c, h, w, 8]
    diff = diff.view(b, c, 8, h, w).permute(0, 1, 3, 4, 2)

    # Step 5: 计算标准差
    return diff.std(dim=-1, unbiased=unbiased)

# -----basic functions of FSMConv----- #
class FirstFSMConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        #print('FirstFSMConv的init开始了')
        super(FirstFSMConv, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        #print('执行池化操作')
        self.AvgPool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.MaxPool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        #print('池化执行完成')

        self.s2s = torch.nn.Conv2d(in_channels, out_channels,
                                   kernel_size, 1, padding, dilation, groups, bias)
        # 第一个分支：7x7和5x5
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, 3, kernel_size=7, stride=1, padding=3, groups=in_channels, bias=bias),  # 深度卷积
            nn.Conv2d(3, 3, kernel_size=5, stride=2, padding=2, groups=in_channels, bias=bias),  # 深度卷积
            nn.Conv2d(3, 32, kernel_size=1, bias=bias)  # 逐点卷积
        )

        # 第二个分支：5x5和3x3
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, 3, kernel_size=5, stride=1, padding=2, groups=in_channels, bias=bias),  # 深度卷积
            nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1, groups=in_channels, bias=bias),  # 深度卷积
            nn.Conv2d(3, 32, kernel_size=1, bias=bias)  # 逐点卷积
        )

        # 第三个分支：3x3和1x1
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, 3, kernel_size=3, stride=2, padding=1, groups=in_channels, bias=bias),  # 深度卷积
            nn.Conv2d(3, 32, kernel_size=1, bias=bias)  # 逐点卷积
        )
        #print('空间特征')
        self.f2f = torch.nn.Conv2d(3, 32, 3, 2, 1, dilation, groups, bias)
        # frequency modules
        #self.MCSConv1 = MCSConv1(32)
        #self.MCSConv2 = MCSConv2(32)
        #self.MCSConv0 = MCSConv3(3)
        #print(self.MCSConv0.const_weight)
        #self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
        #                       bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        # 实例化 StdFeatureMapCalculator
        #self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    def forward(self, x):
        X_s1 = self.branch1(x)
        X_s2 = self.branch2(x)
        X_s3 = self.branch3(x)
        #print('3333',X_s3.size())
        X_s = X_s1 + X_s2 + X_s3
        # 分别通过频域模块 MCSConv 处理
        #X_f = self.MCSConv0(x)
        #start_time = time.time()
        X_f = StdFeatureMap(x)
        #end_time = time.time()
        #print(f"运行时间: {(end_time - start_time) * 1000:.3f} ms")
        #print('ffffffff00', X_f.size())
        #X_f = self.MaxPool(X_f)
          # Multichannel Constrained Separable Conv

        X_f = self.f2f(X_f)
        #print('ffffffff00', X_f.size())
        return X_s, X_f

class FSMConvBR0(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, norm_layer=nn.BatchNorm2d):
        #print('FSMConv的init开始了',in_channels,out_channels)
        super(FSMConvBR0, self).__init__()
        kernel_size = kernel_size[0]
        self.conv = FSMConv0(in_channels, out_channels, kernel_size, )
        self.bn_s = norm_layer(out_channels)
        self.bn_f = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        #print('FSMConv的forward开始了')
        X_s2s, X_f2f = self.conv(x)
        X_s2s = self.bn_s(X_s2s)
        X_s2s = self.relu(X_s2s)
        X_f2f = self.bn_f(X_f2f)
        X_f2f = self.relu(X_f2f)
        return X_s2s, X_f2f

class FSMConvB0(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, norm_layer=nn.BatchNorm2d):
        # print('FSMConv的init开始了',in_channels,out_channels)
        super(FSMConvB0, self).__init__()
        kernel_size = kernel_size[0]
        self.conv = FSMConv0(in_channels, out_channels, kernel_size)
        self.bn_s = norm_layer(out_channels)
        self.bn_f = norm_layer(out_channels)

    def forward(self, x):
        # print('FSMConv的forward开始了')
        X_s2s, X_f2f = self.conv(x)
        X_s2s = self.bn_s(X_s2s)
        X_f2f = self.bn_f(X_f2f)
        return X_s2s, X_f2f
class FSMConv0(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, alpha= 0.5,
                 groups=1, bias=False,norm_layer=nn.BatchNorm2d):
        #print('FSMConv的init开始了',in_channels,out_channels)
        super(FSMConv0, self).__init__()
        self.is_dw = groups == in_channels
        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride
        assert 0 <= alpha <= 1, "Alpha should be in the interval from 0 to 1."
        self.alpha = alpha
        self.f2f = None if alpha == 0 else \
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias)
        self.s2s = None if alpha == 1 else \
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias)

    def forward(self, x):
        #print('FSMConv的forward开始了')
        X_s, X_f = x
        X_s2s = self.s2s(X_s)
        X_f2f = self.f2f(X_f)
        return X_s2s, X_f2f


class AdaptiveFusionModule(nn.Module):
    def __init__(self, channels):
        super(AdaptiveFusionModule, self).__init__()

        # 特征交互层
        self.interaction = GatedInteraction(channels)

        # 生成融合权重的网络
        # 生成 4 个权重
        self.fusion_weights = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels * 4, 1),  # 生成 4 个权重
            nn.Tanh()  # 让权重值稳定在 (-1,1)
        )


class GatedInteraction(nn.Module):
    def __init__(self, channels):
        super(GatedInteraction, self).__init__()
        self.conv1 = nn.Conv2d(channels * 2, channels * 2, 3, stride=1, padding=1, groups=channels * 2)
        self.conv2 = nn.Conv2d(channels * 2, channels, 1)
        self.norm = nn.GroupNorm(8, channels)  # GroupNorm 替代 BatchNorm
        self.relu = nn.ReLU(inplace=True)

        # 门控机制：决定哪些特征保留，哪些特征需要抑制
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, 1),  # 生成门控权重
            nn.Sigmoid()  # 将权重值限制在 (0,1) 之间
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.norm(x)
        x = self.relu(x)

        gate = self.gate(x)  # 计算门控权重
        return x * gate  # 应用门控机制
class FSMConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(FSMConv, self).__init__()
        kernel_size = kernel_size[0]
        self.AvgPool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.MaxPool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.stride = stride
        self.is_dw = groups == in_channels
        self.alpha = alpha

        # 主要特征提取分支
        self.f2f = None if alpha == 0 else \
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, 1, bias)
        self.s2s = None if alpha == 1 else \
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, 1, bias)

        # 自适应特征融合模块
        self.fusion_module = AdaptiveFusionModule(out_channels)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

        self.w_s1 = nn.Parameter(torch.tensor(1.0))
        self.w_s2 = nn.Parameter(torch.tensor(1.0))
        self.w_f1 = nn.Parameter(torch.tensor(1.0))
        self.w_f2 = nn.Parameter(torch.tensor(1.0))
    def forward(self, x):
        X_s, X_f = x

        # if self.stride == 2:
        #     X_s, X_f = self.MaxPool(X_s), self.MaxPool(X_f)

        # 获取主要特征
        X_s2s = self.s2s(X_s)
        X_f2f = self.f2f(X_f)

        # 1. 特征交互融合
        concat_features = torch.cat([X_s2s, X_f2f], dim=1)
        fusion_feat = self.fusion_module.interaction(concat_features)

        # 2. 生成自适应融合权重
        # fusion_weights = self.fusion_module.fusion_weights(fusion_feat)  # (B, C*4, 1, 1)
        # w_s1, w_s2, w_f1, w_f2 = torch.chunk(fusion_weights, 4, dim=1)  # 分成 4 组
        #
        # # 3. 特征增强
        # X_s_final = w_s1 * X_s2s + w_s2 * fusion_feat + self.dropout(w_s2 * fusion_feat)
        # X_f_final = w_f1 * X_f2f + w_f2 * fusion_feat + self.dropout(w_f2 * fusion_feat)
        weights_s = F.softmax(torch.stack([self.w_s1, self.w_s2]), dim=0)
        weights_f = F.softmax(torch.stack([self.w_f1, self.w_f2]), dim=0)
        X_s_final = weights_s[0] * X_s2s + weights_s[1] * fusion_feat
        X_f_final = weights_f[0] * X_f2f + weights_f[1] * fusion_feat
        return X_s_final, X_f_final


class LastFSMConv0(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(LastFSMConv0, self).__init__()
        self.stride = stride
        kernel_size = kernel_size[0]
        self.AvgPool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)

        # self.s2s = torch.nn.Conv2d(in_channels, out_channels,
        #                            kernel_size, 1, padding, dilation, out_channels, bias)
        # self.f2f = torch.nn.Conv2d(in_channels, out_channels,
        #                            kernel_size, 1, padding, dilation, out_channels, bias)
        self.bn_s = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        X_s, X_f = x

        if self.stride == 2:
            X_s, X_f = self.AvgPool(X_s), self.AvgPool(X_f)

        # Perform convolution on X_s and X_f
        #X_s2s = self.s2s(X_s)
        #X_f2f = self.f2f(X_f)

        # Concatenate X_s2s and X_f2f directly
        X_out = torch.cat((X_s, X_f ), dim=1)  # Concatenate along channel dimension

        return X_out



class LastFSMConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d):
        #print('LastFSMConv的init开始了')
        super(LastFSMConv, self).__init__()
        self.stride = stride
        kernel_size = kernel_size[0]
        self.AvgPool = nn.AvgPool2d(kernel_size=(2,2), stride=2)

        self.s2s = torch.nn.Conv2d(in_channels, out_channels,
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.f2f = torch.nn.Conv2d(in_channels, out_channels,
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.bn_s = norm_layer(out_channels)
        self.bn_f = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        #print('LastFSMConv的forward开始了')
        X_s, X_f = x
        if self.stride ==2:
            X_s, X_f = self.AvgPool(X_s), self.AvgPool(X_f)
        X_s2s = self.s2s(X_s)
        X_f2f = self.f2f(X_f)
        x_s = self.bn_s(X_s2s)
        x_f = self.bn_f(X_f2f)
        return x_s, x_f

# -----FSMConv functions used in backbone----- #
class FSMConvBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d):

        #print('FSMConvBR的init开始了')
        super(FSMConvBR, self).__init__()
        self.conv = FSMConv(in_channels,out_channels,kernel_size, alpha, stride, padding, dilation, groups, bias)
        self.bn_s = norm_layer(out_channels)
        self.bn_f = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        #print('FSMConvBR的forward开始了')
        x_s, x_f = self.conv(x)
        x_s = self.relu(self.bn_s(x_s))
        x_f = self.relu(self.bn_f(x_f))
        return x_s, x_f


class FSMConvB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d):
        #print('FSMConvB的init开始了')
        super(FSMConvB, self).__init__()
        self.conv = FSMConv(in_channels, out_channels, kernel_size, alpha, stride, padding, dilation, groups, bias)
        self.bn_s = norm_layer(out_channels)
        self.bn_f = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        #print('FSMConvB的forward开始了')
        x_s, x_f = self.conv(x)
        x_s = self.bn_s(x_s)
        x_f = self.bn_f(x_f)
        return x_s, x_f


class FirstFSMConvBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False,norm_layer=nn.BatchNorm2d):
        #print('FirstFSMConvBR的init开始了')
        super(FirstFSMConvBR, self).__init__()
        self.conv = FirstFSMConv(in_channels,out_channels,kernel_size, alpha,stride,padding,dilation,groups,bias)
        self.bn_s = norm_layer(int(out_channels * (1 - alpha)))
        self.bn_f = norm_layer(int(out_channels * alpha))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        #print('FirstFSMConvBR的forward开始了')
        x_s, x_f = self.conv(x)
        x_s = self.relu(self.bn_s(x_s))
        x_f = self.relu(self.bn_f(x_f))
        return x_s, x_f


class LastFSMConvBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d):
        #print('LastFSMConvBR的init开始了')
        super(LastFSMConvBR, self).__init__()
        self.conv = LastFSMConv(in_channels, out_channels, kernel_size, alpha, stride, padding, dilation, groups, bias)
        self.bn_s = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        #print('LastFSMConvBR的forward开始了')
        x_s = self.conv(x)
        x_s = self.relu(self.bn_s(x_s))
        return x_s


class FirstFSMConvB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d):
        #print('FirstFSMConvB的init开始了')
        super(FirstFSMConvB, self).__init__()
        self.conv = FirstFSMConv(in_channels, out_channels, kernel_size, alpha, stride, padding, dilation, groups, bias)
        self.bn_s = norm_layer(int(out_channels * (1 - alpha)))
        self.bn_f = norm_layer(int(out_channels * alpha))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        #print('FirstFSMConvB的forward开始了')
        x_s, x_f = self.conv(x)
        x_s = self.bn_s(x_s)
        x_f = self.bn_f(x_f)
        return x_s, x_f


class LastFSMConvB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d):
        #print('LastFSMConvB的init开始了')
        super(LastFSMConvB, self).__init__()
        self.conv = LastFSMConv(in_channels, out_channels, kernel_size, alpha, stride, padding, dilation, groups, bias)
        self.bn_s = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        #print('LastFSMConvB的forward开始了')
        x_s = self.conv(x)
        x_s = self.bn_s(x_s)
        return x_s


if __name__ == '__main__':
    # ----------------------------------------------------------------------------------------------------- #
    # test FirstMYconv
    i = torch.Tensor(1, 3, 256, 256)
    FirstMYconv = FirstFSMConv(kernel_size=(3, 3), in_channels=3, out_channels=64, alpha=0.5)
    x_out, y_out = FirstMYconv(i)
    print("FirstMYconv: ", x_out.size(), y_out.size())

    # test MYconv
    MYconv = FSMConvBR0(kernel_size=(1,1), in_channels=64, out_channels=64, bias=False, alpha=0.5)
    i = x_out, y_out
    print('iiiiiiii',x_out.size())
    x_out, y_out = MYconv(i)
    print("MYconv: ", x_out.size(), y_out.size())
    #
    # MYconv_b = FSMConvB(in_channels=64, out_channels=64, alpha=0.5).cuda()
    # i = x_out, y_out
    # x_out, y_out = MYconv_b(i)
    # print("MYconv_b:",x_out.size(),y_out.size())
    #
    # # test LastMYconv
    # LastMYconv = LastFSMConv(kernel_size=(3, 3), in_channels=64, out_channels=256, alpha=0.5).cuda()
    # i = x_out, y_out
    # out = LastMYconv(i)
    # print("LastMYconv: ", out.size())
    # ----------------------------------------------------------------------------------------------------- #

