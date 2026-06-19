"""Statistics computation, batch collation, and DataLoader assembly.

``build_dataloaders`` is the single entry point used by the trainer (Phase 2c):
it performs the splits, computes the persisted/derived statistics, builds the
datasets and the joint train sampler, and returns everything needed to train.
"""

import json
from pathlib import Path
from typing import Sequence

import torch
from omegaconf import DictConfig
from torch.utils.data import ConcatDataset, DataLoader

from src.data.datasets import (
    USABLE_BIOMARKER_INDICES,
    EyePACSDataset,
    OLIVESDataset,
)
from src.data.samplers import JointBatchSampler, patient_aware_split, random_split
from src.data.transforms import (
    TwoViewTransform,
    build_train_transform,
    build_val_transform,
)

NUM_DR_CLASSES = 5


def compute_regression_stats(
    olives: dict, train_olives_indices: Sequence[int], stats_path: str | Path
) -> dict:
    """Mean/std of BCVA and CST over the train-split tier-1 valid samples.

    Idempotent: if ``stats_path`` exists it is loaded and reused. NaN-bearing
    samples are excluded (``has_clinical & ~isnan``).
    """
    stats_path = Path(stats_path)
    if stats_path.exists():
        with open(stats_path) as fh:
            stats = json.load(fh)
        print(f"[stats] regression stats loaded from {stats_path}")
        return stats

    idx = torch.tensor(list(train_olives_indices), dtype=torch.long)
    bcva = olives["bcva"][idx].float()
    cst = olives["cst"][idx].float()
    valid = olives["has_clinical"][idx].bool() & ~torch.isnan(bcva) & ~torch.isnan(cst)
    bcva_v = bcva[valid]
    cst_v = cst[valid]
    stats = {
        "bcva_mean": float(bcva_v.mean()),
        "bcva_std": float(bcva_v.std(unbiased=False)),
        "cst_mean": float(cst_v.mean()),
        "cst_std": float(cst_v.std(unbiased=False)),
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"[stats] regression stats computed over {int(valid.sum())} tier-1 samples")
    print(f"[stats] written to {stats_path}")
    return stats


def compute_dr_class_weights(
    eyepacs_labels: torch.Tensor, train_eyepacs_indices: Sequence[int]
) -> torch.Tensor:
    """Inverse-frequency class weights (length 5) from the EyePACS train split."""
    idx = torch.tensor(list(train_eyepacs_indices), dtype=torch.long)
    counts = torch.bincount(
        eyepacs_labels[idx].long(), minlength=NUM_DR_CLASSES
    ).float()
    weights = counts.sum() / (NUM_DR_CLASSES * counts.clamp(min=1.0))
    return weights


def compute_biomarker_pos_weight(
    olives: dict, train_olives_indices: Sequence[int]
) -> torch.Tensor:
    """Per-column neg/pos pos_weight (length 9) over train-split tier-2 samples."""
    idx = torch.tensor(list(train_olives_indices), dtype=torch.long)
    has_bio = olives["has_biomarkers"][idx].bool()
    rows = idx[has_bio]
    bio = (olives["biomarkers"][rows][:, USABLE_BIOMARKER_INDICES] > 0.5).float()
    pos = bio.sum(dim=0)
    neg = bio.shape[0] - pos
    return torch.where(pos > 0, neg / pos.clamp(min=1.0), torch.ones_like(pos))


def collate_fn(samples: list[dict]) -> dict:
    """Assemble per-sample dicts into the batch contract dict."""
    first_image = samples[0]["images"]
    two_view = isinstance(first_image, tuple)

    batch: dict[str, torch.Tensor] = {}
    if two_view:
        batch["images"] = torch.stack([s["images"][0] for s in samples])
        batch["images_v2"] = torch.stack([s["images"][1] for s in samples])
    else:
        batch["images"] = torch.stack([s["images"] for s in samples])

    batch["dataset"] = torch.tensor([s["dataset"] for s in samples], dtype=torch.long)
    batch["dr_labels"] = torch.tensor(
        [s["dr_labels"] for s in samples], dtype=torch.long
    )
    batch["biomarkers"] = torch.stack([s["biomarkers"] for s in samples]).float()
    batch["has_biomarkers"] = torch.tensor(
        [s["has_biomarkers"] for s in samples], dtype=torch.bool
    )
    batch["bcva"] = torch.tensor(
        [float(s["bcva"]) for s in samples], dtype=torch.float32
    )
    batch["cst"] = torch.tensor(
        [float(s["cst"]) for s in samples], dtype=torch.float32
    )
    batch["has_clinical"] = torch.tensor(
        [s["has_clinical"] for s in samples], dtype=torch.bool
    )
    batch["patient_ids"] = torch.tensor(
        [s["patient_ids"] for s in samples], dtype=torch.long
    )
    return batch


def build_dataloaders(cfg: DictConfig) -> dict:
    """Build train/val/test loaders plus loss-weighting stats from ``cfg``.

    Returns a dict with: ``train_loader``, ``val_loader``, ``test_loader``,
    ``class_weights``, ``pos_weight``, ``regression_stats``, and the per-dataset
    split index lists.
    """
    paths = cfg.paths
    ratios = (float(cfg.split.train), float(cfg.split.val), float(cfg.split.test))
    seed = int(cfg.split.seed)

    train_tf = build_train_transform(cfg)
    if bool(cfg.augmentation.two_view):
        train_tf = TwoViewTransform(train_tf)
    val_tf = build_val_transform(cfg)

    # [1/6] OLIVES into RAM once; reuse for splits, stats, weights, dataset.
    print("[1/6] loading OLIVES into RAM")
    olives = torch.load(
        str(paths.olives_file), map_location="cpu", weights_only=False
    )

    # [2/6] splits
    print("[2/6] computing splits")
    ol_train, ol_val, ol_test = patient_aware_split(
        olives["patient_ids"], olives["has_clinical"], olives["has_biomarkers"], ratios, seed
    )
    # A throwaway full dataset triggers the cache copy and gives the total count.
    ep_full = EyePACSDataset(
        str(paths.eyepacs_dir), str(paths.eyepacs_cache), allowed_indices=None
    )
    ep_train, ep_val, ep_test = random_split(ep_full.total, ratios, seed)
    eyepacs_labels = ep_full.labels

    # [3/6] statistics
    print("[3/6] computing statistics")
    regression_stats = compute_regression_stats(
        olives, ol_train, paths.regression_stats
    )
    class_weights = compute_dr_class_weights(eyepacs_labels, ep_train)
    pos_weight = compute_biomarker_pos_weight(olives, ol_train)

    # [4/6] datasets
    print("[4/6] building datasets")

    def make_eyepacs(indices: Sequence[int], transform: object) -> EyePACSDataset:
        return EyePACSDataset(
            str(paths.eyepacs_dir),
            str(paths.eyepacs_cache),
            allowed_indices=indices,
            transform=transform,
        )

    def make_olives(indices: Sequence[int], transform: object) -> OLIVESDataset:
        return OLIVESDataset(olives, indices, transform, regression_stats)

    ep_train_ds = make_eyepacs(ep_train, train_tf)
    ol_train_ds = make_olives(ol_train, train_tf)
    val_ds = ConcatDataset(
        [make_eyepacs(ep_val, val_tf), make_olives(ol_val, val_tf)]
    )
    test_ds = ConcatDataset(
        [make_eyepacs(ep_test, val_tf), make_olives(ol_test, val_tf)]
    )

    # [5/6] train sampler (guarantees both datasets per batch for CORAL)
    print("[5/6] building joint train sampler")
    train_concat = ConcatDataset([ep_train_ds, ol_train_ds])
    sampler = JointBatchSampler(
        n_eyepacs=len(ep_train_ds),
        n_olives=len(ol_train_ds),
        eyepacs_per_batch=int(cfg.batch.eyepacs_per_batch),
        olives_per_batch=int(cfg.batch.olives_per_batch),
        seed=seed,
    )

    # [6/6] loaders
    print("[6/6] assembling DataLoaders")
    eval_bs = int(cfg.batch.eval_batch_size)
    train_loader = DataLoader(
        train_concat, batch_sampler=sampler, collate_fn=collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=eval_bs, shuffle=False, collate_fn=collate_fn, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=eval_bs, shuffle=False, collate_fn=collate_fn, num_workers=0
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "class_weights": class_weights,
        "pos_weight": pos_weight,
        "regression_stats": regression_stats,
        "splits": {
            "eyepacs": {"train": ep_train, "val": ep_val, "test": ep_test},
            "olives": {"train": ol_train, "val": ol_val, "test": ol_test},
        },
    }
