import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size=4, drop_rate=0.0):
        super().__init__()

        inter_channels = bn_size * growth_rate

        self.norm1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_channels,
            inter_channels,
            kernel_size=1,
            stride=1,
            bias=False
        )

        self.norm2 = nn.BatchNorm2d(inter_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            inter_channels,
            growth_rate,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = self.conv1(self.relu1(self.norm1(x)))
        new_features = self.conv2(self.relu2(self.norm2(new_features)))

        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features,
                p=self.drop_rate,
                training=self.training
            )

        return torch.cat([x, new_features], dim=1)


class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, bn_size=4, drop_rate=0.0):
        super().__init__()

        layers = []
        channels = in_channels

        for _ in range(num_layers):
            layer = DenseLayer(
                in_channels=channels,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate
            )
            layers.append(layer)
            channels += growth_rate

        self.block = nn.Sequential(*layers)
        self.out_channels = channels

    def forward(self, x):
        return self.block(x)


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            bias=False
        )
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(self.relu(self.norm(x)))
        x = self.pool(x)
        return x


class DenseNetMultiTask(nn.Module):
    def __init__(
        self,
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        num_init_features=64,
        bn_size=4,
        drop_rate=0.0,
        num_disease_classes=5,
        num_severity_classes=4,
        dropout=0.3
    ):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                3,
                num_init_features,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            ),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        num_features = num_init_features

        dense_blocks = []

        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers=num_layers,
                in_channels=num_features,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate
            )
            dense_blocks.append(block)
            num_features = block.out_channels

            if i != len(block_config) - 1:
                out_features = num_features // 2
                trans = Transition(
                    in_channels=num_features,
                    out_channels=out_features
                )
                dense_blocks.append(trans)
                num_features = out_features

        self.dense_features = nn.Sequential(*dense_blocks)
        self.final_norm = nn.BatchNorm2d(num_features)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.shared_classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.disease_head = nn.Linear(512, num_disease_classes)
        self.severity_head = nn.Linear(512, num_severity_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="relu"
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.dense_features(x)
        x = self.final_norm(x)
        x = F.relu(x, inplace=True)

        x = self.pool(x)
        x = torch.flatten(x, 1)

        x = self.shared_classifier(x)

        disease_output = self.disease_head(x)
        severity_output = self.severity_head(x)

        return disease_output, severity_output


def DenseNet121MultiTask(
    num_disease_classes=5,
    num_severity_classes=4,
    dropout=0.3
):
    return DenseNetMultiTask(
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        num_init_features=64,
        bn_size=4,
        drop_rate=0.0,
        num_disease_classes=num_disease_classes,
        num_severity_classes=num_severity_classes,
        dropout=dropout
    )