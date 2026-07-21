"""
Pre-activation ResNet-20 for Christlein-style surrogate patch classification.

Methodological target:
- Input: grayscale 32x32 handwriting patches.
- Architecture: CIFAR-style pre-activation ResNet-20.
- Penultimate descriptor: 64-dimensional pooled activation.
- Classifier: surrogate cluster labels, e.g. 5000 classes.

Layer counting:
- Initial 3x3 convolution: 1 conv layer.
- 3 stages × 3 residual blocks × 2 conv layers = 18 conv layers.
- Final linear classifier: 1 layer.
- Total = 20 layers, following the usual CIFAR ResNet-20 convention.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PreActBasicBlock(nn.Module):
    """Pre-activation residual block with two 3x3 convolutions."""

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Conv2d(
                in_planes,
                planes,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(x), inplace=True)

        # In pre-activation ResNet, the projection shortcut is applied to the
        # pre-activated input when dimensions change.
        shortcut = self.shortcut(out) if not isinstance(self.shortcut, nn.Identity) else x

        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out), inplace=True))

        return out + shortcut


class PreActResNet20(nn.Module):
    """CIFAR-style pre-activation ResNet-20 for 32x32 grayscale patches."""

    def __init__(
        self,
        num_classes: int = 5000,
        in_channels: int = 1,
        base_channels: int = 16,
        embedding_dim: int = 64,
    ) -> None:
        super().__init__()

        if embedding_dim != 64:
            raise ValueError(
                "For faithful Christlein-compatible ResNet-20, embedding_dim must be 64."
            )

        self.num_classes = int(num_classes)
        self.in_channels = int(in_channels)
        self.base_channels = int(base_channels)
        self.embedding_dim = int(embedding_dim)

        self.in_planes = base_channels

        self.conv1 = nn.Conv2d(
            in_channels,
            base_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.layer1 = self._make_layer(base_channels, num_blocks=3, stride=1)
        self.layer2 = self._make_layer(base_channels * 2, num_blocks=3, stride=2)
        self.layer3 = self._make_layer(base_channels * 4, num_blocks=3, stride=2)

        self.bn_final = nn.BatchNorm2d(base_channels * 4)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(base_channels * 4, num_classes)

        if base_channels * 4 != embedding_dim:
            raise ValueError(
                f"base_channels * 4 must equal embedding_dim=64. "
                f"Got base_channels={base_channels}, base_channels*4={base_channels * 4}."
            )

        self._initialize_weights()

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []

        for s in strides:
            blocks.append(PreActBasicBlock(self.in_planes, planes, stride=s))
            self.in_planes = planes * PreActBasicBlock.expansion

        return nn.Sequential(*blocks)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="linear",
                )
                nn.init.zeros_(module.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = F.relu(self.bn_final(out), inplace=True)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)

        return out

    def extract_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        embedding = self.forward_features(x)
        logits = self.classifier(embedding)

        if return_embedding:
            return embedding, logits

        return logits


def build_resnet20_64d(num_classes: int = 5000, in_channels: int = 1) -> PreActResNet20:
    return PreActResNet20(
        num_classes=num_classes,
        in_channels=in_channels,
        base_channels=16,
        embedding_dim=64,
    )


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable