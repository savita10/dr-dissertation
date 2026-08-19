"""Training entry point for the Design C1 harmonised run (greyscale experiment).

Reuses the EXACT Phase 2 training logic from the unmodified pipeline —
``build_merged_config`` + ``build_dataloaders`` + ``DRModel`` + ``Trainer`` (same
architecture, heads, losses, optimizer, schedule, seed 42, splits) — and inserts
the ONE difference: EyePACS-only green + histogram-match harmonisation, applied
between dataloader construction and training via
``harmonise_eyepacs_in_bundle``. OLIVES is left raw (Design C1).

Nothing in ``src/`` is modified; this is a NEW wrapper on the
``greyscale-experiment``. The best checkpoint (by validation DR QWK) is
written by ``Trainer`` to ``cfg.checkpoint.dir / phase2_{run_name}_best.pt`` —
i.e. ``greyscale_experiment/phase2_greenhistmatch_best.pt`` — never the original
``phase2_coral_on_best.pt``.

Run:
    python -m scripts.train_greyscale --config configs/greyscale_train.yaml [--smoke]
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import torch
from omegaconf import DictConfig

from src.data.dataloaders import build_dataloaders
from src.models.model import DRModel
from src.preprocessing.harmonise import harmonise_eyepacs_in_bundle
from src.training.train import (
    N_SMOKE_CHECKS,
    _make_smoke_train_loader,
    _make_smoke_val_loader,
    build_merged_config,
    run_smoke_assertions,
    set_seeds,
)
from src.training.trainer import Trainer

# Phase 2 CORAL-on in-domain validation QWK, printed alongside the harmonised
# result so a large drop (harmonisation hurting in-domain grading) is visible.
PHASE2_REFERENCE_QWK = 0.615


def _apply_harmonisation(bundle: dict, cfg: DictConfig) -> None:
    """Insert the Design C1 EyePACS-only harmonisation (the only diff vs Phase 2)."""
    if not bool(cfg.get("harmonise", False)):
        print("[greyscale] harmonise flag off — plain colour EyePACS path (no C1 transform)")
        return
    ref = str(cfg.reference_hist)
    if not Path(ref).exists():
        raise FileNotFoundError(
            f"reference histogram not found: {ref} — run "
            "`python -m scripts.compute_olives_histogram` first."
        )
    n = harmonise_eyepacs_in_bundle(
        bundle,
        ref,
        imagenet_mean=cfg.imagenet_mean,
        imagenet_std=cfg.imagenet_std,
        source_channel=int(cfg.get("reference_channel", 1)),
    )
    print(f"[greyscale] Design C1: harmonised {n} EyePACS dataset(s); OLIVES left raw")
    print(f"[greyscale] reference histogram: {ref}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train the Design C1 harmonised DR model (greyscale experiment)."
    )
    parser.add_argument("--config", default="configs/greyscale_train.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--resume", default=None, help="override checkpoint.resume: auto | never | <path>"
    )
    args = parser.parse_args(argv)

    try:
        cfg = build_merged_config(args.config)
        if args.resume is not None:
            cfg.checkpoint.resume = args.resume

        set_seeds(int(cfg.seed))

        if float(cfg.loss_weights.w_ntxent) > 0.0:
            assert bool(cfg.augmentation.two_view), (
                "w_ntxent > 0 requires data.augmentation.two_view: true"
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[greyscale] run_name={cfg.run_name} device={device} smoke={args.smoke}")

        # Save-safety (pre-flight): Google Drive does not auto-create parent
        # folders. Ensure the checkpoint dir exists before any save.
        ckpt_dir = Path(str(cfg.checkpoint.dir))
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(f"[greyscale] checkpoint dir ready: {ckpt_dir} (exists={ckpt_dir.exists()})")

        print("[greyscale] building dataloaders (Phase 2 pipeline, unmodified)")
        bundle = build_dataloaders(cfg)

        # The ONLY difference vs Phase 2: EyePACS-only harmonisation.
        _apply_harmonisation(bundle, cfg)

        train_loader = bundle["train_loader"]
        val_loader = bundle["val_loader"]

        if args.smoke:
            # Isolate the smoke gate: own run_name, always-fresh, capped subsets.
            # Smoke loaders are rebuilt from the already-harmonised datasets.
            cfg.run_name = f"{cfg.run_name}_smoke"
            cfg.checkpoint.resume = "never"
            cfg.train.max_epochs = int(cfg.smoke.epochs)
            train_loader = _make_smoke_train_loader(cfg, bundle)
            val_loader = _make_smoke_val_loader(cfg, bundle)

        print("[greyscale] building model (pretrained=True)")
        model = DRModel.from_config(cfg, pretrained=True)

        trainer = Trainer(
            cfg=cfg,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=bundle["test_loader"],
            class_weights=bundle["class_weights"],
            pos_weight=bundle["pos_weight"],
            regression_stats=bundle["regression_stats"],
            device=device,
        )

        trainer.fit()

        if args.smoke:
            print("\n[smoke] running training gate assertions")
            run_smoke_assertions(trainer, expected_epochs=int(cfg.smoke.epochs))
            print(f"\n[smoke] all {N_SMOKE_CHECKS} checks passed.")
        else:
            best = float(trainer.best_metric)
            print(f"\n[greyscale] best validation dr_qwk = {best:.4f}")
            print(
                f"[greyscale] vs Phase 2 in-domain QWK ~{PHASE2_REFERENCE_QWK} "
                f"(delta {best - PHASE2_REFERENCE_QWK:+.4f})"
            )
            print(f"[greyscale] best checkpoint: {trainer._ckpt_path('best')}")
    except Exception as exc:
        print(f"\n[FAIL] greyscale training failed: {exc}")
        traceback.print_exc()
        return 1

    print(f"\n[greyscale] done (run_name={cfg.run_name}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
