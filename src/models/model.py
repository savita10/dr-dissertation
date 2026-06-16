"""DRModel — shared encoder + three output heads, single forward pass."""

import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.models.encoder import ResNetEncoder
from src.models.heads import BiomarkerHead, DRHead, RegressionHead


class DRModel(nn.Module):
    """Wraps the shared encoder and the DR / biomarker / regression heads.

    ``forward`` returns a dict with the encoder ``features`` and ``projection``
    plus the three head outputs, so a single pass feeds every loss term.
    """

    def __init__(
        self,
        encoder_dim: int = 2048,
        projection_dim: int = 512,
        num_dr_classes: int = 5,
        num_biomarkers: int = 9,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = ResNetEncoder(
            encoder_dim=encoder_dim,
            projection_dim=projection_dim,
            pretrained=pretrained,
        )
        self.dr_head = DRHead(encoder_dim=encoder_dim, num_dr_classes=num_dr_classes)
        self.biomarker_head = BiomarkerHead(
            encoder_dim=encoder_dim, num_biomarkers=num_biomarkers
        )
        self.regression_head = RegressionHead(encoder_dim=encoder_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features, projection = self.encoder(x)
        return {
            "features": features,
            "projection": projection,
            "dr_logits": self.dr_head(features),
            "biomarker_logits": self.biomarker_head(features),
            "regression": self.regression_head(features),
        }

    @classmethod
    def from_config(cls, cfg: DictConfig, pretrained: bool = True) -> "DRModel":
        """Build a DRModel from a loaded ``configs/model.yaml`` config.

        ``pretrained`` is exposed separately so the smoke test can build the
        model offline with random weights.
        """
        return cls(
            encoder_dim=int(cfg.encoder_dim),
            projection_dim=int(cfg.projection_dim),
            num_dr_classes=int(cfg.num_dr_classes),
            num_biomarkers=int(cfg.num_biomarkers),
            pretrained=pretrained,
        )
