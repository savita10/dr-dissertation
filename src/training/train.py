"""Training entry point for the cross-dataset DR model.

Merges the model + data + train configs into one OmegaConf object, applies the
loss-weight overrides (the CORAL ablation switch), builds the loaders and model,
and runs the Trainer. With ``--smoke`` it caps the train/val subsets, runs a
short 2-epoch fit, and asserts the [1/6]…[6/6] training gate.

Run:
    python -m src.training.train --config configs/train.yaml [--smoke]
"""

import argparse
import math
import random
import sys
import traceback

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset, DataLoader, Subset

from src.data.dataloaders import build_dataloaders, collate_fn
from src.data.samplers import JointBatchSampler
from src.models.model import DRModel
from src.training.trainer import LOSS_TERM_KEYS, Trainer
from src.utils.config import load_config

N_SMOKE_CHECKS = 6


def build_merged_config(train_config_path: str) -> DictConfig:
    """Merge model + data + train configs and apply loss-weight overrides."""
    train_cfg = load_config(train_config_path)
    model_cfg = load_config(str(train_cfg.paths.model_config))
    data_cfg = load_config(str(train_cfg.paths.data_config))
    cfg = OmegaConf.merge(model_cfg, data_cfg, train_cfg)

    overrides = cfg.get("loss_weight_overrides", {})
    if overrides:
        cfg.loss_weights = OmegaConf.merge(cfg.loss_weights, overrides)
        print(f"[config] applied loss_weight_overrides: {OmegaConf.to_container(overrides)}")
    return cfg


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_smoke_train_loader(cfg: DictConfig, bundle: dict) -> DataLoader:
    """Cap the train datasets to the smoke subsets, rebuilding the joint sampler."""
    ep_ds, ol_ds = bundle["train_loader"].dataset.datasets
    n_ep = min(int(cfg.smoke.eyepacs_subset), len(ep_ds))
    n_ol = min(int(cfg.smoke.olives_subset), len(ol_ds))
    ep_sub = Subset(ep_ds, list(range(n_ep)))
    ol_sub = Subset(ol_ds, list(range(n_ol)))
    concat = ConcatDataset([ep_sub, ol_sub])
    sampler = JointBatchSampler(
        n_eyepacs=n_ep,
        n_olives=n_ol,
        eyepacs_per_batch=int(cfg.batch.eyepacs_per_batch),
        olives_per_batch=int(cfg.batch.olives_per_batch),
        seed=int(cfg.split.seed),
    )
    print(f"[smoke] train capped to {n_ep} EyePACS + {n_ol} OLIVES")
    return DataLoader(concat, batch_sampler=sampler, collate_fn=collate_fn, num_workers=0)


def _make_smoke_val_loader(cfg: DictConfig, bundle: dict) -> DataLoader:
    """Cap the val set too, so the smoke gate is fast on CPU."""
    ep_val, ol_val = bundle["val_loader"].dataset.datasets
    n_ep = min(int(cfg.smoke.eyepacs_subset), len(ep_val))
    n_ol = min(int(cfg.smoke.olives_subset), len(ol_val))
    concat = ConcatDataset(
        [Subset(ep_val, list(range(n_ep))), Subset(ol_val, list(range(n_ol)))]
    )
    print(f"[smoke] val capped to {n_ep} EyePACS + {n_ol} OLIVES")
    return DataLoader(
        concat,
        batch_size=int(cfg.batch.eval_batch_size),
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )


def run_smoke_assertions(trainer: Trainer, expected_epochs: int) -> None:
    history = trainer.history

    n = 0
    n += 1
    assert len(history) == expected_epochs, (
        f"expected {expected_epochs} epochs, got {len(history)}"
    )
    print(f"[1/{N_SMOKE_CHECKS}] {expected_epochs} epochs completed end-to-end")

    n += 1
    for record in history:
        for phase in ("train", "val"):
            for key in LOSS_TERM_KEYS:
                value = record[phase]["loss"].get(key)
                if value is not None:
                    assert math.isfinite(value), f"{phase} loss '{key}' not finite: {value}"
    print(f"[2/{N_SMOKE_CHECKS}] every logged loss term finite across all epochs")

    n += 1
    assert any(r["train"]["coral_active"] for r in history), (
        "CORAL term never > 0 — both domains not co-located in a batch"
    )
    print(f"[3/{N_SMOKE_CHECKS}] CORAL term > 0 on at least one joint batch")

    n += 1
    for record in history:
        assert math.isfinite(float(record["val"]["dr_qwk"])), "dr_qwk not finite"
    print(f"[4/{N_SMOKE_CHECKS}] validation dr_qwk finite for every epoch")

    n += 1
    best_path = trainer._ckpt_path("best")
    last_path = trainer._ckpt_path("last")
    assert best_path.exists(), f"missing {best_path}"
    assert last_path.exists(), f"missing {last_path}"
    best_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    last_ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
    assert "model" in best_ckpt and "model" in last_ckpt
    print(f"[5/{N_SMOKE_CHECKS}] best.pt and last.pt exist on Drive and reload")

    n += 1
    for key in ("epoch", "optimizer", "scheduler", "scaler", "rng"):
        assert key in last_ckpt, f"resume state missing '{key}'"
    assert int(last_ckpt["epoch"]) == trainer.last_completed_epoch, (
        f"checkpoint epoch {last_ckpt['epoch']} != last completed "
        f"{trainer.last_completed_epoch}"
    )
    print(f"[6/{N_SMOKE_CHECKS}] resume state intact (epoch counter matches)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the cross-dataset DR model.")
    parser.add_argument("--config", default="configs/train.yaml")
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
        print(f"[train] run_name={cfg.run_name} device={device} smoke={args.smoke}")

        print("[train] building dataloaders")
        bundle = build_dataloaders(cfg)
        train_loader = bundle["train_loader"]
        val_loader = bundle["val_loader"]

        if args.smoke:
            # Isolate the smoke gate: its own run_name (so it never clobbers a
            # real run's checkpoints) and always-fresh so it is re-runnable.
            cfg.run_name = f"{cfg.run_name}_smoke"
            cfg.checkpoint.resume = "never"
            cfg.train.max_epochs = int(cfg.smoke.epochs)
            train_loader = _make_smoke_train_loader(cfg, bundle)
            val_loader = _make_smoke_val_loader(cfg, bundle)

        print("[train] building model (pretrained=True)")
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
    except Exception as exc:
        print(f"\n[FAIL] training failed: {exc}")
        traceback.print_exc()
        return 1

    print(f"\n[train] done (run_name={cfg.run_name}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
