import torch
import torch.nn as nn

from terratorch.registry import BACKBONE_REGISTRY

"""
Defines the two model classes for oil slick detection: 
- BaselineCNN: Small CNN trained from scratch.
- TerraMindClassifier: Uses the pre-trained TerraMind backbone (frozen) with a custom classification head.
"""


def init_kaiming_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.Linear):
        if m.out_features > 1:
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        else:
            nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class BaselineCNN(nn.Module):
    def __init__(self, num_classes=2, in_channels=2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=16, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, num_classes - 1),
        )
        self.apply(init_kaiming_weights)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class TerraMindClassifier(nn.Module):
    def __init__(self, num_classes=2, freeze_backbone=True):
        super().__init__()
        self.backbone = BACKBONE_REGISTRY.build(
            "terramind_v1_small",
            pretrained=True,
            modalities=["S1GRD"],
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        embedding_dim = self.backbone.out_channels[-1]

        self.head = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, num_classes - 1),
        )

        self.head.apply(init_kaiming_weights)

    def forward(self, x):
        feats = self.backbone(x)

        if isinstance(feats, (dict, list, tuple)):
            feats = list(feats.values())[-1] if isinstance(feats, dict) else feats[-1]

        if feats.dim() == 3:
            feats = torch.mean(feats, dim=1)

        return self.head(feats)
