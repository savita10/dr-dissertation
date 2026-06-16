"""CPU-only, offline smoke test for the Phase 2a model + losses.

Builds the model from config with random weights (``pretrained=False``),
runs a forward pass and the combined loss over several mixed/single-dataset
batches, and asserts shapes, finiteness, empty-mask zeros, and backprop.

Run:
    python -m src.models.smoke_test --config configs/model.yaml
"""

import argparse
import copy
import math
import sys
import traceback

import torch
from omegaconf import DictConfig

from src.models.losses import combined_loss
from src.models.model import DRModel
from src.utils.config import load_config

IMAGE_SHAPE = (3, 224, 224)
N_CHECKS = 6
SEED = 0


def _images(n: int) -> torch.Tensor:
    return torch.randn(n, *IMAGE_SHAPE)


def _empty_batch(n: int) -> dict[str, torch.Tensor]:
    """A batch with every mask off and all optional fields NaN/zero."""
    return {
        "dataset": torch.ones(n, dtype=torch.long),  # all OLIVES by default
        "dr_labels": torch.zeros(n, dtype=torch.long),
        "biomarkers": torch.full((n, 9), float("nan")),
        "has_biomarkers": torch.zeros(n, dtype=torch.bool),
        "bcva": torch.full((n,), float("nan")),
        "cst": torch.full((n,), float("nan")),
        "has_clinical": torch.zeros(n, dtype=torch.bool),
    }


def _grads_have_nan(model: torch.nn.Module) -> bool:
    for param in model.parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            return True
    return False


def _check_forward(model: DRModel, cfg: DictConfig) -> None:
    out = model(_images(8))
    expected = {
        "features": (8, int(cfg.encoder_dim)),
        "projection": (8, int(cfg.projection_dim)),
        "dr_logits": (8, int(cfg.num_dr_classes)),
        "biomarker_logits": (8, int(cfg.num_biomarkers)),
        "regression": (8, 2),
    }
    for key, shape in expected.items():
        assert tuple(out[key].shape) == shape, (
            f"{key} shape {tuple(out[key].shape)} != {shape}"
        )
    print("[1/6] forward: all five output shapes correct")


def _check_mixed_batch(model: DRModel, cfg: DictConfig) -> None:
    # 12 samples: EyePACS / tier-0 / tier-1 (one NaN BCVA) / tier-2.
    batch = _empty_batch(12)
    # EyePACS (0-3): dataset 0, has_dr; a repeated label gives a positive pair.
    batch["dataset"][0:4] = 0
    batch["dr_labels"][0:4] = torch.tensor([0, 0, 1, 2])
    # tier-0 OLIVES (4-5): all masks already off.
    # tier-1 OLIVES (6-8): clinical valid, with one NaN BCVA at index 8.
    batch["has_clinical"][6:9] = True
    batch["bcva"][6:9] = torch.tensor([0.3, -0.5, float("nan")])
    batch["cst"][6:9] = torch.tensor([0.1, 0.2, -0.4])
    # tier-2 OLIVES (9-11): biomarkers valid.
    batch["has_biomarkers"][9:12] = True
    batch["biomarkers"][9:12] = (torch.rand(3, 9) > 0.5).float()

    out = model(_images(12))
    total, comps = combined_loss(out, batch, cfg)

    assert math.isfinite(comps["total"]), "total not finite"
    for name, value in comps.items():
        assert math.isfinite(value), f"component {name} not finite ({value})"

    model.zero_grad()
    total.backward()
    assert not _grads_have_nan(model), "NaN found in gradients"
    print("[2/6] mixed-tier batch: total + components finite, backward grads clean")


def _check_all_tier0(model: DRModel, cfg: DictConfig) -> None:
    # OLIVES-only, every mask off.
    batch = _empty_batch(8)
    out = model(_images(8))
    total, comps = combined_loss(out, batch, cfg)

    for name in ("dr", "biomarker", "regression", "supcon", "coral"):
        assert comps[name] == 0.0, f"{name} should be 0.0, got {comps[name]}"
    assert math.isfinite(comps["total"]), "total not finite"
    print("[3/6] all-tier-0 OLIVES batch: dr/biomarker/regression/supcon/coral == 0")


def _check_eyepacs_only(model: DRModel, cfg: DictConfig) -> None:
    batch = _empty_batch(8)
    batch["dataset"][:] = 0  # all EyePACS
    # Valid label mix with positive pairs in multiple classes.
    batch["dr_labels"][:] = torch.tensor([0, 0, 1, 1, 2, 2, 3, 4])

    out = model(_images(8))
    _, comps = combined_loss(out, batch, cfg)

    assert comps["coral"] == 0.0, "coral should be 0.0 (single dataset)"
    assert comps["biomarker"] == 0.0, "biomarker should be 0.0"
    assert comps["regression"] == 0.0, "regression should be 0.0"
    assert comps["dr"] > 0.0, f"dr should be > 0, got {comps['dr']}"
    assert comps["supcon"] > 0.0, f"supcon should be > 0, got {comps['supcon']}"
    print("[4/6] EyePACS-only batch: coral/biomarker/regression == 0; dr, supcon > 0")


def _check_olives_only(model: DRModel, cfg: DictConfig) -> None:
    batch = _empty_batch(8)  # all OLIVES
    # tier-1 (0-3) clinical valid, tier-2 (4-7) biomarkers valid.
    batch["has_clinical"][0:4] = True
    batch["bcva"][0:4] = torch.tensor([0.2, -0.1, 0.5, -0.3])
    batch["cst"][0:4] = torch.tensor([0.1, -0.2, 0.4, 0.0])
    batch["has_biomarkers"][4:8] = True
    batch["biomarkers"][4:8] = (torch.rand(4, 9) > 0.5).float()

    out = model(_images(8))
    _, comps = combined_loss(out, batch, cfg)

    assert comps["coral"] == 0.0, "coral should be 0.0 (single dataset)"
    assert comps["dr"] == 0.0, "dr should be 0.0 (no EyePACS)"
    assert comps["supcon"] == 0.0, "supcon should be 0.0 (no EyePACS)"
    assert comps["biomarker"] > 0.0, "biomarker should be active (> 0)"
    assert comps["regression"] > 0.0, "regression should be active (> 0)"
    print("[5/6] OLIVES-only batch: coral/dr/supcon == 0; biomarker, regression > 0")


def _check_ntxent(model: DRModel, cfg: DictConfig) -> None:
    batch = _empty_batch(8)
    batch["dataset"][0:4] = 0
    batch["dr_labels"][0:4] = torch.tensor([0, 0, 1, 2])

    out = model(_images(8))

    # Default path: w_ntxent = 0, no second view -> term skipped cleanly.
    _, comps_default = combined_loss(out, batch, cfg, projection_v2=None)
    assert "ntxent" not in comps_default, "ntxent should be skipped by default"

    # Enabled path: w_ntxent > 0 with a supplied second view -> finite.
    cfg_on = copy.deepcopy(cfg)
    cfg_on.loss_weights.w_ntxent = 0.1
    out_v2 = model(_images(8))
    total, comps_on = combined_loss(
        out, batch, cfg_on, projection_v2=out_v2["projection"]
    )
    assert "ntxent" in comps_on, "ntxent should be active when enabled"
    assert math.isfinite(comps_on["ntxent"]), "ntxent not finite"
    assert math.isfinite(comps_on["total"]), "total not finite with ntxent"
    print("[6/6] NT-Xent: skipped by default; finite when enabled with two views")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="CPU-only smoke test for the Phase 2a model + losses."
    )
    parser.add_argument("--config", default="configs/model.yaml")
    args = parser.parse_args(argv)

    torch.manual_seed(SEED)

    try:
        cfg = load_config(args.config)
        print(f"[smoke] building DRModel from {args.config} (pretrained=False)")
        model = DRModel.from_config(cfg, pretrained=False)
        model.train()

        _check_forward(model, cfg)
        _check_mixed_batch(model, cfg)
        _check_all_tier0(model, cfg)
        _check_eyepacs_only(model, cfg)
        _check_olives_only(model, cfg)
        _check_ntxent(model, cfg)
    except Exception as exc:
        print(f"\n[FAIL] smoke test failed: {exc}")
        traceback.print_exc()
        return 1

    print(f"\n[smoke] all {N_CHECKS} checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
