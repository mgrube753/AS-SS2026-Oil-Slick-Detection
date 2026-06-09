import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from terratorch.registry import BACKBONE_REGISTRY

"""
Defines the two model classes for oil slick detection: 
- ResNet50Classifier: CNN baseline (pre-trained, unfrozen) with modified input and output layers.
- TerraMindClassifier: Uses the pre-trained TerraMind backbone (frozen) with a custom classification head.
"""


class ResNet50Classifier(nn.Module):
    # todo: use small baseline only instead of resnet50, from SCRATCH (input layer 2-channeled, conv1 16 channels, conv2 32 channels, 32 or 64 neurons hidden layer, 1 output neuron + sigmoid as below)
    def __init__(self, num_classes=2, in_channels=2, pretrained=True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = resnet50(weights=weights)

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
                mean = orig_conv.weight.mean(dim=1, keepdim=True)
                self.backbone.conv1.weight[:, 0:1, :, :] = mean
                self.backbone.conv1.weight[:, 1:2, :, :] = mean

        num_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes - 1),
        )

    def forward(self, x):
        return self.backbone(x)


class TerraMindClassifier(nn.Module):
    def __init__(self, num_classes=2, freeze_backbone=False):
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
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes - 1),
        )

    def forward(self, x):
        feats = self.backbone(x)

        if isinstance(feats, (dict, list, tuple)):
            feats = list(feats.values())[-1] if isinstance(feats, dict) else feats[-1]

        if feats.dim() == 3:
            feats = torch.mean(feats, dim=1)
        elif feats.dim() == 4:
            feats = torch.mean(feats, dim=[2, 3])

        return self.head(feats)
