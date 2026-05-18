import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from terratorch.registry import BACKBONE_REGISTRY

"""
Defines the two model classes for oil slick detection: 
- ResNet50Classifier: CNN baseline with modified input and output layers.
- TerraMindClassifier: Uses the TerraMind backbone with a custom classification head.
"""


class ResNet50Classifier(nn.Module):
    def __init__(
        self, num_classes=2, in_channels=2, pretrained=True, freeze_backbone=False
    ):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = resnet50(weights=weights)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        orig_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels,
            orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=False,
        )

        if pretrained:
            with torch.no_grad():
                self.backbone.conv1.weight[:, :in_channels] = orig_conv.weight[
                    :, :in_channels
                ]

        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


class TerraMindClassifier(nn.Module):
    def __init__(self, num_classes=2, freeze_backbone=False):
        super().__init__()
        self.backbone = BACKBONE_REGISTRY.build(
            "terramind_v1_small", pretrained=True, modalities=["S1GRD"]
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.head = nn.LazyLinear(num_classes)

    def forward(self, x):
        feats = self.backbone(x)

        if isinstance(feats, (dict, list, tuple)):
            feats = list(feats.values())[-1] if isinstance(feats, dict) else feats[-1]

        if feats.dim() == 4:
            feats = feats.mean(dim=[2, 3])
        elif feats.dim() == 3:
            feats = feats.mean(dim=1)

        return self.head(feats)
