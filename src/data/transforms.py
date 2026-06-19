"""Tensor-based augmentation for already-ImageNet-normalised fundus images.

The saved tensors are ImageNet-normalised (values centred near 0, some
negative). Photometric ops are only meaningful in raw [0,1] space, so every
augmented path is wrapped: de-normalise -> augment in [0,1] -> clamp ->
re-normalise. All transforms operate on tensors via torchvision.transforms.v2.
"""

from typing import Callable

import torch
from omegaconf import DictConfig
from torchvision.transforms import v2


def _stat_tensor(values: object) -> torch.Tensor:
    """Turn a 3-element mean/std sequence into a broadcastable [3, 1, 1]."""
    return torch.as_tensor(list(values), dtype=torch.float32).view(-1, 1, 1)


def denormalize(
    x: torch.Tensor, mean: object, std: object
) -> torch.Tensor:
    """Undo ImageNet normalisation and clamp to the valid [0, 1] image range."""
    m = _stat_tensor(mean)
    s = _stat_tensor(std)
    return (x * s + m).clamp(0.0, 1.0)


def normalize(x: torch.Tensor, mean: object, std: object) -> torch.Tensor:
    """Apply ImageNet normalisation: (x - mean) / std."""
    m = _stat_tensor(mean)
    s = _stat_tensor(std)
    return (x - m) / s


class _DeNormAugReNorm:
    """Wraps a [0, 1]-space augmentation between de-norm and re-norm."""

    def __init__(
        self, aug: Callable[[torch.Tensor], torch.Tensor], mean: object, std: object
    ) -> None:
        self.aug = aug
        self.mean = _stat_tensor(mean)
        self.std = _stat_tensor(std)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = (x * self.std + self.mean).clamp(0.0, 1.0)  # de-normalise to [0, 1]
        x = self.aug(x)
        x = x.clamp(0.0, 1.0)  # keep photometric output in a valid image range
        return (x - self.mean) / self.std  # re-normalise


class _ValTransform:
    """Deterministic, label-preserving near-identity (images already 224²)."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x


class TwoViewTransform:
    """Returns two independent draws of ``transform`` for NT-Xent SSL."""

    def __init__(self, transform: Callable[[torch.Tensor], torch.Tensor]) -> None:
        self.transform = transform

    def __call__(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.transform(x), self.transform(x)


def build_train_transform(cfg: DictConfig) -> _DeNormAugReNorm:
    """Field-stop-targeting train augmentation (applied identically to both
    datasets). ``cfg`` is the full data config (reads ``augmentation`` and the
    ImageNet stats)."""
    aug_cfg = cfg.augmentation
    aug = v2.Compose(
        [
            v2.RandomResizedCrop(
                size=224,
                scale=tuple(float(s) for s in aug_cfg.rrc_scale),
                ratio=tuple(float(r) for r in aug_cfg.rrc_ratio),
                antialias=True,
            ),
            v2.ColorJitter(
                brightness=float(aug_cfg.jitter_brightness),
                contrast=float(aug_cfg.jitter_contrast),
            ),
            v2.RandomRotation(degrees=float(aug_cfg.rotation_degrees)),
            v2.RandomHorizontalFlip(p=float(aug_cfg.hflip_prob)),
        ]
    )
    return _DeNormAugReNorm(aug, cfg.imagenet_mean, cfg.imagenet_std)


def build_val_transform(cfg: DictConfig) -> _ValTransform:
    """Deterministic validation transform (no photometric ops)."""
    return _ValTransform()
