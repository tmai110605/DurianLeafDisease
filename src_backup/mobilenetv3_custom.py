import torch
import torch.nn as nn
import torch.nn.functional as F
from MAFC import *

class HSwish(nn.Module):
    def forward(self, x):
        return x * F.relu6(x + 3, inplace=True) / 6


class HSigmoid(nn.Module):
    def forward(self, x):
        return F.relu6(x + 3, inplace=True) / 6


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, squeeze_channels):
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.fc2 = nn.Conv2d(squeeze_channels, in_channels, kernel_size=1)
        self.activation = nn.ReLU(inplace=True)
        self.scale_activation = HSigmoid()

    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        scale = self.scale_activation(scale)
        return x * scale


def conv_bn_act(in_channels, out_channels, kernel_size, stride, groups=1, activation="relu"):
    padding = (kernel_size - 1) // 2

    if activation == "hswish":
        act_layer = HSwish()
    elif activation == "relu":
        act_layer = nn.ReLU(inplace=True)
    else:
        act_layer = nn.Identity()

    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False
        ),
        nn.BatchNorm2d(out_channels),
        act_layer
    )


class InvertedResidual(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        expand_channels,
        use_se,
        activation
    ):
        super().__init__()

        self.use_residual = stride == 1 and in_channels == out_channels

        layers = []

        if expand_channels != in_channels:
            layers.append(
                conv_bn_act(
                    in_channels,
                    expand_channels,
                    kernel_size=1,
                    stride=1,
                    activation=activation
                )
            )

        layers.append(
            conv_bn_act(
                expand_channels,
                expand_channels,
                kernel_size=kernel_size,
                stride=stride,
                groups=expand_channels,
                activation=activation
            )
        )

        if use_se:
            squeeze_channels = max(1, expand_channels // 4)
            layers.append(SqueezeExcitation(expand_channels, squeeze_channels))

        layers.append(
            nn.Sequential(
                nn.Conv2d(expand_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        )

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)

        if self.use_residual:
            out = out + x

        return out


class MobileNetV3SmallMultiTask(nn.Module):
    def __init__(self, num_disease_classes=5, num_severity_classes=4, dropout=0.2):
        super().__init__()

        self.features = nn.Sequential(
            conv_bn_act(3, 16, kernel_size=3, stride=2, activation="hswish"),

            InvertedResidual(16, 16, 3, 2, 16, True, "relu"),
            InvertedResidual(16, 24, 3, 2, 72, False, "relu"),
            InvertedResidual(24, 24, 3, 1, 88, False, "relu"),

            InvertedResidual(24, 40, 5, 2, 96, True, "hswish"),
            InvertedResidual(40, 40, 5, 1, 240, True, "hswish"),
            InvertedResidual(40, 40, 5, 1, 240, True, "hswish"),

            InvertedResidual(40, 48, 5, 1, 120, True, "hswish"),
            InvertedResidual(48, 48, 5, 1, 144, True, "hswish"),

            InvertedResidual(48, 96, 5, 2, 288, True, "hswish"),
            InvertedResidual(96, 96, 5, 1, 576, True, "hswish"),
            InvertedResidual(96, 96, 5, 1, 576, True, "hswish"),

            conv_bn_act(96, 576, kernel_size=1, stride=1, activation="hswish")
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.shared_classifier = nn.Sequential(
            nn.Linear(576, 1024),
            HSwish(),
            nn.Dropout(dropout)
        )

        self.disease_head = nn.Linear(1024, num_disease_classes)
        self.severity_head = nn.Linear(1024, num_severity_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)

        x = self.shared_classifier(x)

        disease_output = self.disease_head(x)
        severity_output = self.severity_head(x)

        return disease_output, severity_output
    
class MobileNetV3LargeMultiTask(nn.Module):
    def __init__(self, num_disease_classes=5, num_severity_classes=4, dropout=0.2):
        super().__init__()

        self.features = nn.Sequential(
            conv_bn_act(3, 16, kernel_size=3, stride=2, activation="hswish"),

            InvertedResidual(16, 16, 3, 1, 16, False, "relu"),
            InvertedResidual(16, 24, 3, 2, 64, False, "relu"),
            InvertedResidual(24, 24, 3, 1, 72, False, "relu"),

            InvertedResidual(24, 40, 5, 2, 72, True, "relu"),
            InvertedResidual(40, 40, 5, 1, 120, True, "relu"),
            InvertedResidual(40, 40, 5, 1, 120, True, "relu"),

            InvertedResidual(40, 80, 3, 2, 240, False, "hswish"),
            InvertedResidual(80, 80, 3, 1, 200, False, "hswish"),
            InvertedResidual(80, 80, 3, 1, 184, False, "hswish"),
            InvertedResidual(80, 80, 3, 1, 184, False, "hswish"),

            InvertedResidual(80, 112, 3, 1, 480, True, "hswish"),
            InvertedResidual(112, 112, 3, 1, 672, True, "hswish"),

            InvertedResidual(112, 160, 5, 2, 672, True, "hswish"),
            InvertedResidual(160, 160, 5, 1, 960, True, "hswish"),
            InvertedResidual(160, 160, 5, 1, 960, True, "hswish"),

            conv_bn_act(160, 960, kernel_size=1, stride=1, activation="hswish")
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.shared_classifier = nn.Sequential(
            nn.Linear(960, 1280),
            HSwish(),
            nn.Dropout(dropout)
        )

        self.disease_head = nn.Linear(1280, num_disease_classes)
        self.severity_head = nn.Linear(1280, num_severity_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)

        x = self.shared_classifier(x)

        disease_output = self.disease_head(x)
        severity_output = self.severity_head(x)

        return disease_output, severity_output


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2

        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )


class InvertedResidualV2(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()

        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_residual = stride == 1 and in_channels == out_channels

        layers = []

        if expand_ratio != 1:
            layers.append(
                ConvBNReLU(
                    in_channels,
                    hidden_dim,
                    kernel_size=1,
                    stride=1
                )
            )

        layers.extend([
            ConvBNReLU(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=stride,
                groups=hidden_dim
            ),
            nn.Conv2d(
                hidden_dim,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(out_channels)
        ])

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)

        if self.use_residual:
            out = x + out

        return out


class MobileNetV2MultiTask(nn.Module):
    def __init__(self, num_disease_classes=5, num_severity_classes=4, dropout=0.2, width_mult=1.0):
        super().__init__()

        def make_divisible(v, divisor=8):
            new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        input_channel = make_divisible(32 * width_mult)
        last_channel = make_divisible(1280 * max(1.0, width_mult))

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = [
            ConvBNReLU(3, input_channel, stride=2)
        ]

        for t, c, n, s in inverted_residual_setting:
            output_channel = make_divisible(c * width_mult)

            for i in range(n):
                stride = s if i == 0 else 1

                features.append(
                    InvertedResidualV2(
                        input_channel,
                        output_channel,
                        stride,
                        expand_ratio=t
                    )
                )

                input_channel = output_channel

        features.append(
            ConvBNReLU(
                input_channel,
                last_channel,
                kernel_size=1,
                stride=1
            )
        )

        self.features = nn.Sequential(*features)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.shared_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(last_channel, 512),
            nn.ReLU6(inplace=True),
            nn.Dropout(dropout)
        )

        self.disease_head = nn.Linear(512, num_disease_classes)
        self.severity_head = nn.Linear(512, num_severity_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)

        x = self.shared_classifier(x)

        disease_output = self.disease_head(x)
        severity_output = self.severity_head(x)

        return disease_output, severity_output