"""Shared ResNet-50 encoder + projection head.

The backbone outputs 2048-d features (final FC stripped); a 2-layer MLP
projection head maps those to a lower-dim space for the contrastive terms.
``forward`` returns both representations: heads consume ``features``,
contrastive/SupCon/NT-Xent consume ``projection``, CORAL consumes ``features``.
"""

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class ResNetEncoder(nn.Module):
    """ResNet-50 backbone (FC stripped) with a SimCLR-style projection head.

    The whole stack is trainable — the backbone is deliberately not frozen.
    """

    def __init__(
        self,
        encoder_dim: int = 2048,
        projection_dim: int = 512,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)
        # Strip the final classifier so the backbone emits encoder_dim features.
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.encoder_dim = encoder_dim
        self.projection_dim = projection_dim

        self.projection = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.BatchNorm1d(encoder_dim),
            nn.ReLU(inplace=True),
            nn.Linear(encoder_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        projection = self.projection(features)
        return features, projection
