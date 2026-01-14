import torch
import torch.nn as nn
from network.encoder import spatial_features_extractor
from network.encoder import Bottleneck
from network.ResBlock import ResBlock
from network.Attention import ResBlock_CBAM, ChannelAttentionModule, SpatialAttentionModule
from network.simAM import Simam_module
from config import training_config as cfg
from network.ConvBlock import ConvBlock
import torch.nn.functional as F
import torch.nn.init as init

class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out
class DifferencePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.net1 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=5, padding=2),
            nn.ReLU(),

        )
        self.net2 = nn.Sequential(
            nn.Conv2d(8, 1, kernel_size=1),
            nn.ReLU(),

        )
        self.cbam1=CBAM(32)
        self.cbam2 = CBAM(8)

    def forward(self, x):
        a1=self.net(x)
        a2=self.cbam1(a1)
        a3=self.net1(a2)
        a4=self.cbam2(a3)
        a5=self.net2(a4)
        return a5

class DifferenceLearner(nn.Module):
    def __init__(self):
        super().__init__()
        self.predictor = DifferencePredictor()

    def forward(self, attacked, original=None, use_pred=False):
        pred_diff = self.predictor(attacked)
        diff_loss = torch.tensor(0.0).to(attacked.device)
        if original is not None:
            true_diff = attacked - original
            diff_loss = nn.MSELoss()(pred_diff, true_diff)
            if use_pred:
                used_diff = pred_diff
            else:
                used_diff = true_diff
        else:
            used_diff = pred_diff
        return used_diff, diff_loss
class Bottleneck1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Bottleneck1, self).__init__()
        self.bn1 = nn.InstanceNorm2d(in_channels)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, 4 * out_channels, kernel_size=1)
        self.bn2 = nn.InstanceNorm2d(4 * out_channels)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(4 * out_channels, out_channels, kernel_size=5, padding=2)

    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.relu2(self.bn2(out)))
        out = torch.cat([out, x], 1)
        return out
# -------------------
# 改进后的分类头
# -------------------
class PerturbationClassifier(nn.Module):
    def __init__(self, in_channels=1, num_classes=4):
        super(PerturbationClassifier, self).__init__()

        # 特征重提取卷积块
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # 全局平均池化 + 最大池化
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

        # 分类器
        self.fc = nn.Sequential(
            nn.Linear(256 * 2, 256),  # 拼接 GAP 和 GMP 输出
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # 输入: [B, 256, 32, 32]
        feat = self.refine(x)  # [B, 64, 32, 32]
        # print('rererere',feat.size())
        avg_pool = self.gap(feat).view(feat.size(0), -1)  # [B, 64]
        max_pool = self.gmp(feat).view(feat.size(0), -1)  # [B, 64]
        feat_cat = torch.cat([avg_pool, max_pool], dim=1)  # [B, 128]
        out = self.fc(feat_cat)  # [B, num_classes]
        return out

class Decoder(nn.Module):
    def __init__(self, type, num_classes=4):
        super(Decoder, self).__init__()
        self.type = type

        if self.type == "tracer":
            self.in_channels = 6
            self.classifier = None  # tracer 模式不需要分类器
        elif self.type == "detector":
            self.in_channels = 3
            self.classifier = PerturbationClassifier(in_channels=3, num_classes=num_classes)
        else:
            raise ValueError(f"Unknown decoder type: {self.type}")

        self.feature_extractor = spatial_features_extractor(in_c=self.in_channels + 16)
        self.simam = Simam_module()

        self.b1 = Bottleneck(64, 64)
        self.b2 = Bottleneck1(128, 64)
        self.b3 = Bottleneck(192, 64)

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 输入通道为1（U分量）
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.down = nn.Sequential(
            nn.InstanceNorm2d(256),
            nn.Tanh(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),

            nn.InstanceNorm2d(256),
            nn.Tanh(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),

            nn.InstanceNorm2d(128),
            nn.Tanh(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),

            nn.InstanceNorm2d(64),
            nn.Tanh(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),

            nn.InstanceNorm2d(64),
            nn.Tanh(),
            nn.Conv2d(64, cfg.wm_channels, kernel_size=3, padding=1),
        )

        self.conv_wm = ConvBlock(cfg.wm_channels, 1, blocks=2)
        self.fc = nn.Linear(cfg.message_length ** 2, cfg.message_length)

    def forward(self, x, difference_features=None, aa=None):

        # 如果有差异特征，扩展并拼接
        if difference_features is not None:
            difference_features = self.cnn(difference_features)
            x = torch.cat([x, difference_features], dim=1)
        fm = self.feature_extractor(x)

        o = self.b1(fm)
        o = o + self.simam(o)
        o = self.b2(o)
        o = o + self.simam(o)
        o = self.b3(o)
        o = o + self.simam(o)

        message = self.down(o)
        message = self.conv_wm(message)
        message = F.interpolate(message, size=(cfg.message_length, cfg.message_length), mode="nearest")
        message = message.squeeze(1).view(message.size(0), -1)
        message = self.fc(message)

        # 只有 detector 才输出分类
        if self.type == "detector":
            perturb_pred = self.classifier(aa)
        else:
            perturb_pred = None

        return message, perturb_pred
