import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(name="relu"):
    name = name.lower()

    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "relu6":
        return nn.ReLU6(inplace=True)
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "hswish":
        return nn.Hardswish(inplace=True)
    if name == "identity" or name is None:
        return nn.Identity()

    raise ValueError(f"Unsupported activation: {name}")


class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
        activation="relu"
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )

        self.bn = nn.BatchNorm2d(
            out_planes,
            eps=1e-5,
            momentum=0.01,
            affine=True
        ) if bn else None

        self.relu = get_activation(activation) if relu else None

    def forward(self, x):
        x = self.conv(x)

        if self.bn is not None:
            x = self.bn(x)

        if self.relu is not None:
            x = self.relu(x)

        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(
        self,
        gate_channels,
        reduction_ratio=16,
        pool_types=None,
        activation="relu",
        excite_activation="sigmoid"
    ):
        super().__init__()

        if pool_types is None:
            pool_types = ["avg", "std"]

        self.pool_types = pool_types

        hidden_channels = gate_channels // reduction_ratio
        if hidden_channels == 0:
            hidden_channels = 1

        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(len(pool_types) * gate_channels, hidden_channels),
            get_activation(activation),
            nn.Linear(hidden_channels, gate_channels)
        )

        self.activation2 = get_activation(excite_activation)

    def forward(self, x):
        squeeze_all = None

        for pool_type in self.pool_types:
            if pool_type == "avg":
                squeeze = F.adaptive_avg_pool2d(x, 1)

            elif pool_type == "max":
                squeeze = F.adaptive_max_pool2d(x, 1)

            elif pool_type == "std":
                std = torch.std(x, dim=(2, 3), unbiased=False)
                squeeze = std.view(std.size(0), std.size(1), 1, 1)

            else:
                raise ValueError(f"Unsupported pool type: {pool_type}")

            if squeeze_all is None:
                squeeze_all = squeeze
            else:
                squeeze_all = torch.cat((squeeze_all, squeeze), dim=1)

        channel_att = self.mlp(squeeze_all)
        scale = self.activation2(channel_att).unsqueeze(2).unsqueeze(3)

        return x * scale


class ChannelPool(nn.Module):
    def __init__(self, pool_types=None):
        super().__init__()

        if pool_types is None:
            pool_types = ["avg", "std"]

        self.pool_types = pool_types

    def forward(self, x):
        spatial_all = None

        for pool_type in self.pool_types:
            if pool_type == "avg":
                spatial = torch.mean(x, dim=1, keepdim=True)

            elif pool_type == "max":
                spatial = torch.max(x, dim=1, keepdim=True)[0]

            elif pool_type == "std":
                spatial = torch.std(x, dim=1, keepdim=True, unbiased=False)

            else:
                raise ValueError(f"Unsupported pool type: {pool_type}")

            if spatial_all is None:
                spatial_all = spatial
            else:
                spatial_all = torch.cat((spatial_all, spatial), dim=1)

        return spatial_all


class SpatialGate(nn.Module):
    def __init__(self, pool_types=None):
        super().__init__()

        if pool_types is None:
            pool_types = ["avg", "std"]

        kernel_size = 7
        self.compress = ChannelPool(pool_types)

        self.spatial = BasicConv(
            in_planes=len(pool_types),
            out_planes=1,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            relu=False,
            activation="relu"
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)

        return x * scale


class MAFC(nn.Module):
    def __init__(
        self,
        gate_channels,
        reduction_ratio=16,
        pool_types=None,
        no_spatial=False,
        activation="relu",
        excite_activation="sigmoid"
    ):
        super().__init__()

        if pool_types is None:
            pool_types = ["avg", "std"]

        self.channel_gate = ChannelGate(
            gate_channels=gate_channels,
            reduction_ratio=reduction_ratio,
            pool_types=pool_types,
            activation=activation,
            excite_activation=excite_activation
        )

        self.no_spatial = no_spatial

        if not no_spatial:
            self.spatial_gate = SpatialGate(pool_types=pool_types)

    def forward(self, x):
        x_out = self.channel_gate(x)

        if not self.no_spatial:
            x_out = self.spatial_gate(x_out)

        return x_out