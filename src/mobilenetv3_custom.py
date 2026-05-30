import torch
import torch.nn as nn
import torch.nn.functional as F


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