import torch
import torch.nn as nn
from network.encoder import spatial_features_extractor
from network.encoder import Bottleneck
from network.simAM import Simam_module
from config import training_config as cfg
from network.ConvBlock import ConvBlock
import torch.nn.functional as F

class DifferencePredictor(nn.Module):
    # MODIFIED: Added a simple CNN predictor to learn to predict the difference from attacked U only
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # 简化通道数
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 稍深一层
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.net(x)

class DifferenceLearner(nn.Module):
    def __init__(self):
        super().__init__()
        # MODIFIED: Added predictor to learn difference without original U
        self.predictor = DifferencePredictor()

    # MODIFIED: Updated forward to support learning predictor in first 10 epochs using true diff,
    # then using predicted diff in later epochs; always compute diff_loss if original provided
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

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
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
        self.b2 = Bottleneck(128, 64)
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

        if self.type == "detector":
            perturb_pred = self.classifier(aa)
        else:
            perturb_pred = None

        return message, perturb_pred
