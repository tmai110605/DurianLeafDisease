import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(name="relu"):
    if name is None:
        return nn.Identity()

    name = name.lower()

    if name == "relu":
        return nn.ReLU(inplace=True)

    if name == "relu6":
        return nn.ReLU6(inplace=True)

    if name == "hswish":
        return nn.Hardswish(inplace=True)

    if name == "sigmoid":
        return nn.Sigmoid()

    raise ValueError(f"Unsupported activation: {name}")


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=None,
        groups=1,
        use_bn=True,
        activation="relu",
        bias=False
    ):
        super().__init__()

        if padding is None:
            padding = (kernel_size - 1) // 2

        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=bias
            )
        ]

        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))

        if activation is not None:
            layers.append(get_activation(activation))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


def conv1x1_block(
    in_channels,
    out_channels,
    stride=1,
    groups=1,
    use_bn=True,
    activation="relu"
):
    return ConvBNAct(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        padding=0,
        groups=groups,
        use_bn=use_bn,
        activation=activation
    )


def conv3x3_block(
    in_channels,
    out_channels,
    stride=1,
    activation="relu"
):
    return ConvBNAct(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        groups=1,
        use_bn=True,
        activation=activation
    )


def conv3x3_dw_block_all(
    channels,
    stride=1,
    activation="relu"
):
    return ConvBNAct(
        in_channels=channels,
        out_channels=channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        groups=channels,
        use_bn=True,
        activation=activation
    )


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        hidden_channels = max(1, channels // reduction)

        self.fc1 = nn.Conv2d(
            channels,
            hidden_channels,
            kernel_size=1
        )
        self.fc2 = nn.Conv2d(
            hidden_channels,
            channels,
            kernel_size=1
        )

    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = F.relu(self.fc1(scale), inplace=True)
        scale = torch.sigmoid(self.fc2(scale))

        return x * scale


class FRPDPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.pw1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=in_channels,
            use_bn=False,
            activation=None
        )

        self.dw = conv3x3_dw_block_all(
            channels=in_channels,
            stride=stride,
            activation="relu"
        )

        self.pw2 = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            groups=1,
            use_bn=True,
            activation="relu"
        )

        self.se = SEBlock(out_channels, reduction=16)

        self.proj = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            use_bn=True,
            activation=None
        )

    def forward(self, x):
        residual = x

        out = self.pw1(x)
        out = self.dw(out)
        out = self.pw2(out)
        out = self.se(out)

        if self.stride == 1 and self.in_channels == self.out_channels:
            out = out + residual
        else:
            residual = self.proj(residual)
            out = out + residual

        return out


class TickNetSingleTask(nn.Module):
    def __init__(
        self,
        init_conv_channels,
        init_conv_stride,
        channels,
        strides,
        num_classes=5,
        in_channels=3,
        use_data_batchnorm=True,
        dropout=0.3
    ):
        super().__init__()

        self.use_data_batchnorm = use_data_batchnorm

        backbone = []

        if use_data_batchnorm:
            backbone.append(nn.BatchNorm2d(num_features=in_channels))

        backbone.append(
            conv3x3_block(
                in_channels=in_channels,
                out_channels=init_conv_channels,
                stride=init_conv_stride,
                activation="relu"
            )
        )

        current_channels = init_conv_channels

        for stage_id, stage_channels in enumerate(channels):
            stage = []

            for unit_id, unit_channels in enumerate(stage_channels):
                stride = strides[stage_id] if unit_id == 0 else 1

                stage.append(
                    FRPDPBlock(
                        in_channels=current_channels,
                        out_channels=unit_channels,
                        stride=stride
                    )
                )

                current_channels = unit_channels

            backbone.append(nn.Sequential(*stage))

        self.final_conv_channels = 1024

        backbone.append(
            conv1x1_block(
                in_channels=current_channels,
                out_channels=self.final_conv_channels,
                activation="relu"
            )
        )

        self.backbone = nn.Sequential(*backbone)
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)

        self.classifier = nn.Sequential(
            nn.Linear(self.final_conv_channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.head = nn.Linear(512, num_classes)

        self.init_params()

    def init_params(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)

                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.backbone(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)
        output = self.head(x)

        return output


def TickNetLargeSingleTask(
    num_classes=5,
    dropout=0.3
):
    init_conv_channels = 32
    init_conv_stride = 2

    channels = [
        [128],
        [64, 128],
        [256, 512, 128, 64, 128, 256],
        [512, 128, 64, 128, 256],
        [512]
    ]

    strides = [2, 1, 2, 2, 2]

    return TickNetSingleTask(
        init_conv_channels=init_conv_channels,
        init_conv_stride=init_conv_stride,
        channels=channels,
        strides=strides,
        num_classes=num_classes,
        dropout=dropout
    )
