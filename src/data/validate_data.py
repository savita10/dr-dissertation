"""CLI validation for the Phase 2b data layer (requires Drive mounted).

Runs against the real preprocessed files and prints [k/7] checks, exiting
non-zero on any failure. Memory-light: EyePACS is accessed lazily via the
dataset; OLIVES field-level checks use a memory-mapped load of the small fields.

Run (in Colab, with Drive mounted):
    python -m src.data.validate_data --config configs/data.yaml
"""

import argparse
import sys
import traceback

import torch
from omegaconf import DictConfig

from src.data.dataloaders import build_dataloaders
from src.data.datasets import USABLE_BIOMARKER_INDICES
from src.data.transforms import build_train_transform, build_val_transform
from src.utils.config import load_config

EXPECTED_BIOMARKER_COUNTS = [68, 67, 110, 134, 48, 164, 114, 174, 29]

# (key, dtype, trailing dims after batch) for the batch contract.
CONTRACT = [
    ("images", torch.float32, (3, 224, 224)),
    ("dataset", torch.int64, ()),
    ("dr_labels", torch.int64, ()),
    ("biomarkers", torch.float32, (9,)),
    ("has_biomarkers", torch.bool, ()),
    ("bcva", torch.float32, ()),
    ("cst", torch.float32, ()),
    ("has_clinical", torch.bool, ()),
    ("patient_ids", torch.int64, ()),
]


def _stat_tensor(values: object) -> torch.Tensor:
    return torch.as_tensor(list(values), dtype=torch.float32).view(-1, 1, 1)


def check_contract(batch: dict, expected_b: int, two_view: bool) -> None:
    print("[1/7] batch contract — keys, shapes, dtypes:")
    for key, dtype, tail in CONTRACT:
        tensor = batch[key]
        shape = tuple(tensor.shape)
        print(f"  {key:<15} {str(shape):<22} {tensor.dtype}")
        assert tensor.dtype == dtype, f"{key} dtype {tensor.dtype} != {dtype}"
        assert shape == (expected_b, *tail), f"{key} shape {shape} unexpected"
    if two_view:
        assert "images_v2" in batch, "images_v2 missing while two_view=true"
    else:
        assert "images_v2" not in batch, "images_v2 present while two_view=false"
    print("  [1/7] PASS")


def check_routing(batch: dict) -> None:
    ds = batch["dataset"]
    ep = ds == 0
    ol = ds == 1
    assert bool((~batch["has_clinical"][ep]).all()), "EyePACS rows must have has_clinical=False"
    assert bool((~batch["has_biomarkers"][ep]).all()), "EyePACS rows must have has_biomarkers=False"
    assert bool((batch["patient_ids"][ep] == -1).all()), "EyePACS patient_ids must be -1"
    assert bool((batch["dr_labels"][ol] == -1).all()), "OLIVES dr_labels must be -1"
    # Tier consistency: every biomarker-bearing OLIVES row is a valid OLIVES row.
    assert bool(ol[batch["has_biomarkers"]].all()), "biomarker rows must be OLIVES"
    print("[2/7] tier + dataset routing consistent — PASS")


def check_biomarker_columns(olives: dict) -> None:
    bio = olives["biomarkers"]
    has_bio = olives["has_biomarkers"].bool()
    counts = (
        (bio[has_bio][:, USABLE_BIOMARKER_INDICES] > 0.5).sum(dim=0).long().tolist()
    )
    if counts != EXPECTED_BIOMARKER_COUNTS:
        print(f"  actual counts:   {counts}")
        print(f"  expected counts: {EXPECTED_BIOMARKER_COUNTS}")
        raise AssertionError(
            "biomarker column counts mismatch — tensor column order differs "
            "from the documented order; correct USABLE_BIOMARKER_INDICES."
        )
    print(f"[3/7] biomarker columns reproduce {EXPECTED_BIOMARKER_COUNTS} — PASS")


def check_augmentation(cfg: DictConfig, olives: dict) -> None:
    mean = _stat_tensor(cfg.imagenet_mean)
    std = _stat_tensor(cfg.imagenet_std)
    sample = olives["images"][0].clone()

    train_tf = build_train_transform(cfg)
    val_tf = build_val_transform(cfg)

    aug = train_tf(sample)
    # De-normalise the augmented view WITHOUT clamping; must land within [0,1].
    denorm = aug * std + mean
    eps = 1e-3
    assert float(denorm.min()) >= -eps, f"augmented view below 0 ({float(denorm.min())})"
    assert float(denorm.max()) <= 1.0 + eps, f"augmented view above 1 ({float(denorm.max())})"

    val = val_tf(sample)
    assert torch.allclose(val, sample, atol=1e-4), "val transform must preserve the image"
    print("[4/7] augmentation round-trip within [0,1]; val transform identity — PASS")


def check_split_integrity(splits: dict, olives: dict) -> None:
    pid = olives["patient_ids"].long()

    def patient_set(indices: list[int]) -> set:
        return {int(pid[i]) for i in indices if int(pid[i]) != -1}

    train = patient_set(splits["train"])
    val = patient_set(splits["val"])
    test = patient_set(splits["test"])
    assert not (train & val), "patient overlap between train and val"
    assert not (train & test), "patient overlap between train and test"
    assert not (val & test), "patient overlap between val and test"
    print("[5/7] patient-aware split — no patient crosses train/val/test — PASS")


def check_regression_stats(
    stats: dict, splits: dict, olives: dict
) -> None:
    idx = torch.tensor(splits["train"], dtype=torch.long)
    bcva = olives["bcva"][idx].float()
    cst = olives["cst"][idx].float()
    valid = olives["has_clinical"][idx].bool() & ~torch.isnan(bcva) & ~torch.isnan(cst)
    bcva_std = (bcva[valid] - stats["bcva_mean"]) / stats["bcva_std"]
    cst_std = (cst[valid] - stats["cst_mean"]) / stats["cst_std"]
    for name, z in (("bcva", bcva_std), ("cst", cst_std)):
        assert abs(float(z.mean())) < 1e-3, f"{name} standardised mean not ~0"
        assert abs(float(z.std(unbiased=False)) - 1.0) < 1e-2, f"{name} standardised std not ~1"
    print("[6/7] regression stats — train tier-1 standardised to mean~0, std~1 — PASS")


def check_joint_quota(batch: dict, cfg: DictConfig) -> None:
    ds = batch["dataset"]
    n_ep = int((ds == 0).sum())
    n_ol = int((ds == 1).sum())
    assert n_ep == int(cfg.batch.eyepacs_per_batch), f"EyePACS quota {n_ep} unexpected"
    assert n_ol == int(cfg.batch.olives_per_batch), f"OLIVES quota {n_ol} unexpected"
    print(f"[7/7] joint-batch quota: {n_ep} EyePACS + {n_ol} OLIVES per batch — PASS")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate the Phase 2b data layer against the real files."
    )
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
        two_view = bool(cfg.augmentation.two_view)
        expected_b = int(cfg.batch.eyepacs_per_batch) + int(cfg.batch.olives_per_batch)

        print("[validate] building dataloaders")
        bundle = build_dataloaders(cfg)
        batch = next(iter(bundle["train_loader"]))

        # Memory-light access to OLIVES fields for the column/split/stats checks.
        print("[validate] memory-mapping OLIVES fields")
        olives = torch.load(
            str(cfg.paths.olives_file),
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )

        check_contract(batch, expected_b, two_view)
        check_routing(batch)
        check_biomarker_columns(olives)
        check_augmentation(cfg, olives)
        check_split_integrity(bundle["splits"]["olives"], olives)
        check_regression_stats(
            bundle["regression_stats"], bundle["splits"]["olives"], olives
        )
        check_joint_quota(batch, cfg)
    except Exception as exc:
        print(f"\n[FAIL] data validation failed: {exc}")
        traceback.print_exc()
        return 1

    print("\n[validate] all 7 checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
