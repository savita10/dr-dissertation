"""Output heads consuming the 2048-d encoder features.

Each head is a single linear layer; dimensions are config-driven.
"""

import torch
import torch.nn as nn


class DRHead(nn.Module):
    """5-class DR severity classifier (EyePACS supervision)."""

    def __init__(self, encoder_dim: int = 2048, num_dr_classes: int = 5) -> None:
        super().__init__()
        self.fc = nn.Linear(encoder_dim, num_dr_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)


class BiomarkerHead(nn.Module):
    """Multi-label biomarker classifier (OLIVES tier-2 supervision)."""

    def __init__(self, encoder_dim: int = 2048, num_biomarkers: int = 9) -> None:
        super().__init__()
        self.fc = nn.Linear(encoder_dim, num_biomarkers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)


class RegressionHead(nn.Module):
    """BCVA/CST regressor (OLIVES tier-1), outputs in standardised space."""

    def __init__(self, encoder_dim: int = 2048) -> None:
        super().__init__()
        self.fc = nn.Linear(encoder_dim, 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)
