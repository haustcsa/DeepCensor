import os

import torch.nn as nn
from PIL import Image
from torchvision import transforms
from FSMC_ResNet.network.FSMConv import *

__all__ = ['FSMresnet26','FSMresnet50','FSMresnet101','FSMresnet152']

from FSMC_ResNet.network.SPP import SpatialPyramidPooling


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 conv with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(3,3), stride=stride,
                     padding=1, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 conv"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1,1), stride=stride, bias=False, padding=0)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

import torch
import torch.nn as nn
import torch.nn.functional as F

class ImprovedChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ImprovedChannelAttentionModule, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 独立的MLP用于平均池化和最大池化
        self.mlp_avg = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels)
        )

        self.mlp_max = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels)
        )

        # 可学习的融合权重，初始化为随机值
        self.fusion_weights = nn.Parameter(torch.randn(5, 1))  # 5个区域的权重
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape

        # 全局特征处理
        avg_global = self.avg_pool(x).view(b, c)  # 平均池化 [b, c]
        max_global = self.max_pool(x).view(b, c)  # 最大池化 [b, c]

        # 分别对平均池化和最大池化进行MLP处理
        avg_global_feat = self.mlp_avg(avg_global)  # [b, c]
        max_global_feat = self.mlp_max(max_global)  # [b, c]

        # 将两种池化特征相加
        global_feat = avg_global_feat + max_global_feat  # [b, c]

        # 动态划分局部区域
        h_split, w_split = h // 2, w // 2
        x1 = x[:, :, :h_split, :w_split]  # 左上
        x2 = x[:, :, :h_split, w_split:]  # 右上
        x3 = x[:, :, h_split:, :w_split]  # 左下
        x4 = x[:, :, h_split:, w_split:]  # 右下

        # 处理每个局部区域
        def process_local(x_local):
            avg_local = self.avg_pool(x_local).view(b, c)  # 平均池化 [b, c]
            max_local = self.max_pool(x_local).view(b, c)  # 最大池化 [b, c]

            # 分别对平均池化和最大池化进行MLP处理
            avg_local_feat = self.mlp_avg(avg_local)  # [b, c]
            max_local_feat = self.mlp_max(max_local)  # [b, c]

            # 将两种池化特征相加
            return avg_local_feat + max_local_feat  # [b, c]

        local_feat1 = process_local(x1)  # [b, c]
        local_feat2 = process_local(x2)  # [b, c]
        local_feat3 = process_local(x3)  # [b, c]
        local_feat4 = process_local(x4)  # [b, c]

        # 将所有特征堆叠起来
        all_feats = torch.stack([
            global_feat,  # 全局特征
            local_feat1,  # 局部特征1
            local_feat2,  # 局部特征2
            local_feat3,  # 局部特征3
            local_feat4   # 局部特征4
        ], dim=2)  # [b, c, 5]

        # 应用注意力权重
        fusion_weights = F.softmax(self.fusion_weights, dim=0)  # 确保权重和为1
        fusion_weights = fusion_weights.unsqueeze(0).unsqueeze(0)  # 形状从 [5, 1] 变为 [1, 1, 5, 1]
        fused_feat = torch.sum(all_feats.unsqueeze(-1) * fusion_weights, dim=2).squeeze(-1)  # [b, c]

        # 生成全局注意力权重
        global_attention = self.sigmoid(fused_feat).view(b, c, 1, 1)  # [b, c, 1, 1]

        # 生成局部区域的独立注意力权重
        attention1 = self.sigmoid(local_feat1).view(b, c, 1, 1)  # [b, c, 1, 1]
        attention2 = self.sigmoid(local_feat2).view(b, c, 1, 1)  # [b, c, 1, 1]
        attention3 = self.sigmoid(local_feat3).view(b, c, 1, 1)  # [b, c, 1, 1]
        attention4 = self.sigmoid(local_feat4).view(b, c, 1, 1)  # [b, c, 1, 1]

        # 全局特征保留：将全局注意力权重与局部注意力权重结合
        x1 = x1 * (attention1 + global_attention)  # 左上区域加权
        x2 = x2 * (attention2 + global_attention)  # 右上区域加权
        x3 = x3 * (attention3 + global_attention)  # 左下区域加权
        x4 = x4 * (attention4 + global_attention)  # 右下区域加权

        # 重新组合四个区域
        top = torch.cat([x1, x2], dim=3)  # 上部分
        bottom = torch.cat([x3, x4], dim=3)  # 下部分
        out = torch.cat([top, bottom], dim=2)  # 完整特征图

        return out
class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None,First=False, is_first_block=False):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        #width = int(planes * (base_width / 64.)) * groups
        width = int(planes)
        #print('inplane========',inplanes)
        #width = planes
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.first = First
        self.is_first_block = is_first_block
        self.FSMC1 = FSMConvBR0(inplanes, width, kernel_size=(1, 1), norm_layer=norm_layer)
        self.FSMC2 = FSMConvB(width,  width, kernel_size=(3, 3), stride=stride, groups=groups, norm_layer=norm_layer)
        #self.FSMC22 = FSMConvB(width, width, kernel_size=(3, 3), stride=1, groups=groups, norm_layer=norm_layer)
        #self.FSMC3 = FSMConvB0(width, planes, kernel_size=(1, 1), norm_layer=norm_layer)
        #print('FSMiiiiii', planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        # 通道注意力机制
        #self.channel_attention1 = ChannelAttentionModule(width)  # 在此应用注意力机制
        self.channel_attention2 = ImprovedChannelAttentionModule(planes)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        x_s_res, x_f_res = x
        #print('SF0000', x_s_res.size())
        x_s1, x_f1 = self.FSMC1((x_s_res,x_f_res))
        #print('FSM11111', x_s1.size())
        x_s, x_f = self.FSMC2((x_s1, x_f1))
        #print('FSM2222222', x_s.size(),self.stride)
        #x_s, x_f = self.FSMC22((x_s, x_f))
        #print('FSM111112222222', x_s.size())
        #x_s, x_f = self.FSMC3((x_s, x_f))
        #print('FSM333333', x_s.size())
        # 应用通道注意力机制
        x_s = self.channel_attention2(x_s)  # 对x_s应用注意力权重
        x_f = self.channel_attention2(x_f)  # 对x_f应用注意力权重


        #print('FSM注意力机制====', x_s.size())
        if self.downsample is not None:
            #print('开始下采样==',)
            x_s_res, x_f_res= self.downsample(x)
        x_s += x_s_res
        x_f += x_f_res

        x_s = self.relu(x_s)
        x_f = self.relu(x_f)

        return x_s, x_f


class BottleneckLast(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None):
        super(BottleneckLast, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        #width = int(planes * (base_width / 64.)) * groups
        #width = int(planes * 2)
        width = planes
        #print('wwwwwwwwwwww', width)
        #print('inplanesinplanes', width)
        # Last means the end of two branch
        #self.FSMC1 = FSMConvBR0(inplanes, width,kernel_size=(1, 1), norm_layer=norm_layer)
        #self.FSMC2 = FSMConvBR(width, width, kernel_size=(3, 3), stride=1, groups=groups, norm_layer=norm_layer)
        #self.FSMC22 = FSMConvBR(width, width, kernel_size=(3, 3), stride=1, groups=groups, norm_layer=norm_layer)
        self.FSMC3 = LastFSMConv0(width, planes, kernel_size=(1, 1), norm_layer=norm_layer, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self,x):

        x_s_res, x_f_res = x
        #print('FSM00000mmm',x_s_res.size())
        #x_s, x_f = self.FSMC1((x_s_res, x_f_res))
        #print('FSM111111mmm', x_s.size())
        #x_s, x_f = self.FSMC2((x_s_res, x_f_res))
        #x_s, x_f = self.FSMC22((x_s, x_f))
        #print('FSM22222mmm', x_s.size())
        x_s = self.FSMC3((x_s_res, x_f_res))
        #print('FSM033333mmm', x_s.size())
        x_s = self.relu(x_s)
      #  print('lasttttttttttt', x_s.size())

        return x_s
class BottleneckOrigin(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None):
        super(BottleneckOrigin, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        #width = int(planes * (base_width / 64.)) * groups
        #width = int(planes * 2)
        width = int(planes*2)
       # print('wwwwwwwwww', width,inplanes,planes)
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        #self.conv1 = conv1x1(inplanes, width)
        #self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups)
        #print(self.conv2)
        self.bn2 = norm_layer(width)
        #self.conv3 = conv1x1(width, planes*2)
        self.bn3 = norm_layer(planes*2)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        #print('convoooooooo', x.size())
        #out = self.conv1(x)
        #print('conv11111111', out.size())
        #out = self.bn1(out)
        #out = self.relu(out)

        out = self.conv2(x)
        #print('conv222222', out.size())
        out = self.bn2(out)
        out = self.relu(out)

        return out



class FSMConvResNet(nn.Module):

    def __init__(self, block, layers, num_classes=5, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):
        super(FSMConvResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.inplanes = 32
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.conv2 = FirstFSMConv(in_channels=3, out_channels=self.inplanes, kernel_size=3, stride=1, padding=0)
        self.bn_s = norm_layer(self.inplanes)
        self.bn_f = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=2, norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_last_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer)
        #self.conv3 = LastFSMConv1(in_channels=1024, out_channels=2048, kernel_size=(3, 3), norm_layer=norm_layer, padding=1,stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(5120, num_classes)
        self.spp = SpatialPyramidPooling()
        self.dropout = nn.Dropout(0.3)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None, First=False):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                LastFSMConv(in_channels=self.inplanes, out_channels=planes, kernel_size=(3, 3), stride=2,
                             padding=1)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, norm_layer, First, is_first_block=True))
        self.inplanes = planes
        #print('inininin==', self.inplanes)
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _make_last_layer(self, block, planes, blocks, stride=1, norm_layer=None, First=False):

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                LastFSMConv(in_channels=self.inplanes, out_channels=planes, kernel_size=(3, 3), stride=2,
                            padding=1)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, norm_layer, First, is_first_block=True))
        self.inplanes = planes

        for _ in range(1, blocks):
            layers.append(BottleneckLast(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)
        x_s, x_f = self.conv2(x)
        x_s = self.bn_s(x_s)
        x_s = self.relu(x_s)
        x_f = self.bn_f(x_f)
        x_f = self.relu(x_f)
        #print('conv22222222', x_f.size(), x_s.size())
        x_s, x_f = self.layer1((x_s, x_f))
        #print('laryer111111111', x_s.size())
        x_s, x_f = self.layer2((x_s,x_f))
        #print('laryer222222', x_s.size())
        x_s, x_f = self.layer3((x_s,x_f))
        #print('laryer33333333', x_s.size())
        x_s = self.layer4((x_s,x_f))
        #print('laryer4444444', x_s)
        x_s = self.dropout(x_s)
        #print('laryer4444444',x_s.size())
        #x_s = self.conv3((x_s, x_f))
        #print('laryer5555555555', x_s.size())
        #x = self.avgpool(x_s)
        x = self.spp(x_s)  # 经过SPP

        #print('xxxxxxxxxxxxx',x.size())
        x = x.view(x.size(0), -1)
        #print('xxxxxxxxxxxxxxx========',x.shape)
        x = self.fc(x)

        return x

def FSMresnet26(pretrained=False, **kwargs):
    """Constructs a ResNet-26 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FSMConvResNet(Bottleneck, [1, 3, 3, 2], **kwargs)
    return model

def FSMresnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FSMConvResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

def FSMresnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FSMConvResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model

def FSMresnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FSMConvResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


# 加载和处理图像的函数
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 如果需要调整大小
    transforms.ToTensor(),          # 转换为 Tensor
])

def load_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        if img is None:
            print(f"Failed to load image from {image_path}")
        img = transform(img)  # 将图像转换为 Tensor
        return img
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
if __name__ == '__main__':
    model = FSMresnet26(num_classes=2)
    # print(model)
    # 2. 指定图片路径（修改为你想加载的图片路径）
    image_path = "E:/dataset/NeuralTextures/001_870_6.jpg"  # 图片路径
    img = load_image(image_path)

    if img is not None:
        img = img.unsqueeze(0)  # 为批量处理添加一个批次维度
        output = model(img)
        print('mmmm', output.size())
    else:
        print("图片加载失败，无法传递给模型。")

