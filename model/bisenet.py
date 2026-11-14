"""Minimal BiSeNet implementation compatible with zllrunning face-parsing weights."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = _conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = _conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = self.conv1(x)
        residual = F.relu(self.bn1(residual))
        residual = self.bn2(self.conv2(residual))
        shortcut = x if self.downsample is None else self.downsample(x)
        return self.relu(shortcut + residual)


def _basic_layer(in_channels, out_channels, count, stride=1):
    layers = [BasicBlock(in_channels, out_channels, stride)]
    layers.extend(BasicBlock(out_channels, out_channels) for _ in range(count - 1))
    return nn.Sequential(*layers)


class Resnet18(nn.Module):
    """The checkpoint's ResNet-18 context backbone (without classification head)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = _basic_layer(64, 64, count=2)
        self.layer2 = _basic_layer(64, 128, count=2, stride=2)
        self.layer3 = _basic_layer(128, 256, count=2, stride=2)
        self.layer4 = _basic_layer(256, 512, count=2, stride=2)

    def forward(self, x):
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        feature8 = self.layer2(x)
        feature16 = self.layer3(feature8)
        feature32 = self.layer4(feature16)
        return feature8, feature16, feature32


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride,
            padding=padding, bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class BiSeNetOutput(nn.Module):
    def __init__(self, in_channels, mid_channels, n_classes):
        super().__init__()
        self.conv = ConvBNReLU(in_channels, mid_channels)
        self.conv_out = nn.Conv2d(mid_channels, n_classes, kernel_size=1, bias=False)

    def forward(self, x):
        return self.conv_out(self.conv(x))


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBNReLU(in_channels, out_channels)
        self.conv_atten = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_channels)
        self.sigmoid_atten = nn.Sigmoid()

    def forward(self, x):
        feature = self.conv(x)
        attention = F.avg_pool2d(feature, feature.shape[2:])
        attention = self.sigmoid_atten(self.bn_atten(self.conv_atten(attention)))
        return feature * attention


class ContextPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = Resnet18()
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128)
        self.conv_head16 = ConvBNReLU(128, 128)
        self.conv_avg = ConvBNReLU(512, 128, kernel_size=1, padding=0)

    def forward(self, x):
        feature8, feature16, feature32 = self.resnet(x)
        height8, width8 = feature8.shape[2:]
        height16, width16 = feature16.shape[2:]
        average = F.avg_pool2d(feature32, feature32.shape[2:])
        average = F.interpolate(self.conv_avg(average), feature32.shape[2:], mode="nearest")
        feature32 = self.arm32(feature32) + average
        feature32 = self.conv_head32(
            F.interpolate(feature32, (height16, width16), mode="nearest")
        )
        feature16 = self.arm16(feature16) + feature32
        feature16 = self.conv_head16(
            F.interpolate(feature16, (height8, width8), mode="nearest")
        )
        return feature8, feature16, feature32


class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convblk = ConvBNReLU(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv1 = nn.Conv2d(out_channels, out_channels // 4, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels // 4, out_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, spatial, context):
        feature = self.convblk(torch.cat([spatial, context], dim=1))
        attention = F.avg_pool2d(feature, feature.shape[2:])
        attention = self.sigmoid(self.conv2(self.relu(self.conv1(attention))))
        return feature * attention + feature


class BiSeNet(nn.Module):
    """19-class face parser used by the published ``79999_iter.pth`` checkpoint."""

    def __init__(self, n_classes=19):
        super().__init__()
        self.cp = ContextPath()
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, n_classes)
        self.conv_out16 = BiSeNetOutput(128, 64, n_classes)
        self.conv_out32 = BiSeNetOutput(128, 64, n_classes)

    def forward(self, x):
        height, width = x.shape[2:]
        feature_res8, feature_cp8, feature_cp16 = self.cp(x)
        feature_fused = self.ffm(feature_res8, feature_cp8)
        output = F.interpolate(
            self.conv_out(feature_fused), (height, width), mode="bilinear", align_corners=True
        )
        output16 = F.interpolate(
            self.conv_out16(feature_cp8), (height, width), mode="bilinear", align_corners=True
        )
        output32 = F.interpolate(
            self.conv_out32(feature_cp16), (height, width), mode="bilinear", align_corners=True
        )
        return output, output16, output32
