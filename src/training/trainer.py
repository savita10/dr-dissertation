"""Training loop for the cross-dataset DR model.

The Trainer owns the optimizer/scheduler/AMP machinery, the train and
validation passes, checkpointing with full resume, and model selection on
validation DR quadratic-weighted kappa (QWK). It consumes the public Phase 2a
model + loss and Phase 2b loaders without modifying them.
"""

import json
import math
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.models.losses import combined_loss

# Component keys that must stay finite (loss terms); QWK/AUC are graded separately.
LOSS_TERM_KEYS = ("dr", "biomarker", "regression", "supcon", "coral", "ntxent", "total")


def _json_safe(value: object) -> object:
    """Recursively convert tensors/non-finite floats so json.dump emits valid JSON."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if torch.is_tensor(value):
        return _json_safe(value.item() if value.ndim == 0 else value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return _json_safe(float(value))
    return value


def safe_qwk(y_true: list[int], y_pred: list[int]) -> float:
    """Quadratic-weighted Cohen kappa; returns 0.0 when undefined (single class)."""
    if len(y_true) == 0:
        return 0.0
    try:
        value = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    except Exception:
        return 0.0
    return float(value) if value is not None and math.isfinite(value) else 0.0


class Trainer:
    def __init__(
        self,
        cfg: DictConfig,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        class_weights: torch.Tensor,
        pos_weight: torch.Tensor,
        regression_stats: dict,
        device: str,
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.class_weights = class_weights.to(device) if class_weights is not None else None
        self.pos_weight = pos_weight.to(device) if pos_weight is not None else None
        self.regression_stats = regression_stats

        self.w_ntxent = float(cfg.loss_weights.w_ntxent)
        self.two_view = self.w_ntxent > 0.0
        self.grad_clip = float(cfg.train.grad_clip_norm)
        self.max_epochs = int(cfg.train.max_epochs)
        self.patience = int(cfg.train.early_stopping_patience)

        # AMP only on CUDA; on CPU it is a no-op so the smoke run still works.
        self.use_amp = bool(cfg.train.amp) and device == "cuda"
        self.amp_device = "cuda" if device == "cuda" else "cpu"
        self.scaler = GradScaler(self.amp_device, enabled=self.use_amp)

        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler(len(train_loader))

        self.run_name = str(cfg.run_name)
        self.ckpt_dir = Path(str(cfg.checkpoint.dir))
        self.best_metric = -math.inf
        self.epochs_no_improve = 0
        self.start_epoch = 0
        self.history: list[dict] = []
        self.last_completed_epoch = -1

    # ------------------------------------------------------------------ build
    def build_optimizer(self) -> torch.optim.Optimizer:
        backbone_params = list(self.model.encoder.backbone.parameters())
        backbone_ids = {id(p) for p in backbone_params}
        other_params = [p for p in self.model.parameters() if id(p) not in backbone_ids]

        n_all = sum(1 for _ in self.model.parameters())
        assert len(backbone_params) + len(other_params) == n_all, (
            "optimizer parameter groups do not cover all parameters exactly"
        )

        wd = float(self.cfg.optimizer.weight_decay)
        groups = [
            {"params": backbone_params, "lr": float(self.cfg.optimizer.lr_backbone)},
            {"params": other_params, "lr": float(self.cfg.optimizer.lr_heads)},
        ]
        print(
            f"[optim] AdamW: backbone {len(backbone_params)} params @ "
            f"{self.cfg.optimizer.lr_backbone}, heads {len(other_params)} params @ "
            f"{self.cfg.optimizer.lr_heads}, wd={wd}"
        )
        return torch.optim.AdamW(groups, weight_decay=wd)

    def build_scheduler(self, steps_per_epoch: int) -> SequentialLR:
        total_steps = max(1, self.max_epochs * steps_per_epoch)
        # Clamp warmup so cosine always has at least one step (short smoke runs).
        warmup = min(int(self.cfg.schedule.warmup_steps), max(1, total_steps // 2))
        cosine_steps = max(1, total_steps - warmup)
        print(
            f"[sched] warmup={warmup} steps, cosine={cosine_steps} steps "
            f"(total {total_steps})"
        )
        warmup_sched = LinearLR(
            self.optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup
        )
        cosine_sched = CosineAnnealingLR(self.optimizer, T_max=cosine_steps, eta_min=0.0)
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup],
        )

    # ------------------------------------------------------------------ helpers
    def _to_device(self, batch: dict) -> dict:
        return {
            k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()
        }

    @staticmethod
    def _outputs_to_fp32(out: dict) -> dict:
        return {
            k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
            for k, v in out.items()
        }

    # ------------------------------------------------------------------ train
    def train_one_epoch(self, epoch: int) -> tuple[dict, bool]:
        self.model.train()
        sums: dict[str, float] = {}
        n_batches = 0
        coral_active = False

        desc = f"train e{epoch + 1}/{self.max_epochs}"
        for batch in tqdm(self.train_loader, desc=desc, unit="batch"):
            batch = self._to_device(batch)
            self.optimizer.zero_grad(set_to_none=True)

            with autocast(self.amp_device, enabled=self.use_amp):
                out = self.model(batch["images"])
                projection_v2 = None
                if self.two_view:
                    out_v2 = self.model(batch["images_v2"])
                    projection_v2 = out_v2["projection"]

            # Loss in fp32 so CORAL covariance / SupCon logits never use fp16.
            out = self._outputs_to_fp32(out)
            if projection_v2 is not None:
                projection_v2 = projection_v2.float()

            total, comps = combined_loss(
                out,
                batch,
                self.cfg,
                class_weights=self.class_weights,
                pos_weight=self.pos_weight,
                projection_v2=projection_v2,
            )

            self.scaler.scale(total).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            for key, value in comps.items():
                sums[key] = sums.get(key, 0.0) + float(value)
            if comps.get("coral", 0.0) > 0.0:
                coral_active = True
            n_batches += 1

        means = {k: v / max(1, n_batches) for k, v in sums.items()}
        return means, coral_active

    # ------------------------------------------------------------------ validate
    @torch.no_grad()
    def validate(self) -> dict:
        self.model.eval()
        sums: dict[str, float] = {}
        n_batches = 0
        dr_true: list[int] = []
        dr_pred: list[int] = []
        bio_logits: list[torch.Tensor] = []
        bio_true: list[torch.Tensor] = []
        reg_pred: list[torch.Tensor] = []
        reg_true: list[torch.Tensor] = []

        for batch in tqdm(self.val_loader, desc="val", unit="batch"):
            batch = self._to_device(batch)
            with autocast(self.amp_device, enabled=self.use_amp):
                out = self.model(batch["images"])
            out = self._outputs_to_fp32(out)

            _, comps = combined_loss(
                out,
                batch,
                self.cfg,
                class_weights=self.class_weights,
                pos_weight=self.pos_weight,
            )
            for key, value in comps.items():
                sums[key] = sums.get(key, 0.0) + float(value)
            n_batches += 1

            ds = batch["dataset"]
            ep = ds == 0
            if bool(ep.any()):
                preds = out["dr_logits"][ep].argmax(dim=1)
                dr_pred.extend(preds.cpu().tolist())
                dr_true.extend(batch["dr_labels"][ep].cpu().tolist())

            hb = batch["has_biomarkers"]
            if bool(hb.any()):
                bio_logits.append(out["biomarker_logits"][hb].cpu())
                bio_true.append(batch["biomarkers"][hb].cpu())

            valid = (
                batch["has_clinical"]
                & ~torch.isnan(batch["bcva"])
                & ~torch.isnan(batch["cst"])
            )
            if bool(valid.any()):
                reg_pred.append(out["regression"][valid].cpu())
                target = torch.stack([batch["bcva"], batch["cst"]], dim=1)[valid]
                reg_true.append(target.cpu())

        metrics: dict = {"loss": {k: v / max(1, n_batches) for k, v in sums.items()}}

        # DR metrics over EyePACS-val rows.
        metrics["dr_qwk"] = safe_qwk(dr_true, dr_pred)
        if dr_true:
            correct = sum(int(a == b) for a, b in zip(dr_true, dr_pred))
            metrics["dr_acc"] = correct / len(dr_true)
        else:
            metrics["dr_acc"] = float("nan")

        # Per-biomarker ROC-AUC (NaN where a column is single-class in val).
        metrics["biomarker_auc"] = self._biomarker_auc(bio_logits, bio_true)

        # Regression MSE in standardised space, plus inverted to real units.
        metrics.update(self._regression_mse(reg_pred, reg_true))
        return metrics

    @staticmethod
    def _biomarker_auc(
        logits: list[torch.Tensor], targets: list[torch.Tensor]
    ) -> list[float]:
        if not logits:
            return [float("nan")] * 9
        scores = torch.sigmoid(torch.cat(logits)).numpy()
        labels = torch.cat(targets).numpy()
        aucs: list[float] = []
        for col in range(labels.shape[1]):
            try:
                aucs.append(float(roc_auc_score(labels[:, col], scores[:, col])))
            except Exception:
                aucs.append(float("nan"))
        return aucs

    def _regression_mse(
        self, preds: list[torch.Tensor], targets: list[torch.Tensor]
    ) -> dict:
        if not preds:
            return {
                "bcva_mse_std": float("nan"),
                "cst_mse_std": float("nan"),
                "bcva_mse_real": float("nan"),
                "cst_mse_real": float("nan"),
            }
        pred = torch.cat(preds)
        true = torch.cat(targets)
        se = (pred - true) ** 2
        bcva_std = float(se[:, 0].mean())
        cst_std = float(se[:, 1].mean())
        bcva_scale = float(self.regression_stats["bcva_std"]) ** 2
        cst_scale = float(self.regression_stats["cst_std"]) ** 2
        return {
            "bcva_mse_std": bcva_std,
            "cst_mse_std": cst_std,
            "bcva_mse_real": bcva_std * bcva_scale,
            "cst_mse_real": cst_std * cst_scale,
        }

    # ------------------------------------------------------------------ checkpoint
    def _ckpt_path(self, kind: str) -> Path:
        return self.ckpt_dir / f"phase2_{self.run_name}_{kind}.pt"

    def _history_path(self) -> Path:
        return self.ckpt_dir / f"phase2_{self.run_name}_history.json"

    def save_checkpoint(self, epoch: int, is_best: bool) -> None:
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "best_metric": self.best_metric,
            "epochs_no_improve": self.epochs_no_improve,
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None,
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
            "config": OmegaConf.to_container(self.cfg, resolve=True),
        }
        last_path = self._ckpt_path("last")
        tmp = last_path.with_suffix(".pt.tmp")
        torch.save(state, tmp)
        tmp.replace(last_path)
        if is_best:
            shutil.copyfile(last_path, self._ckpt_path("best"))

    def maybe_resume(self) -> None:
        resume = str(self.cfg.checkpoint.resume)
        if resume == "never":
            print("[resume] resume=never — starting fresh")
            return

        if resume == "auto":
            ckpt_path = self._ckpt_path("last")
            if not ckpt_path.exists():
                print(f"[resume] no checkpoint at {ckpt_path} — starting fresh")
                return
        else:
            ckpt_path = Path(resume)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"resume checkpoint not found: {ckpt_path}")

        print(f"[resume] restoring from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.best_metric = ckpt["best_metric"]
        self.epochs_no_improve = ckpt["epochs_no_improve"]
        self.start_epoch = int(ckpt["epoch"]) + 1

        rng = ckpt["rng"]
        torch.set_rng_state(rng["torch"])
        if rng["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
        print(
            f"[resume] resumed at epoch {self.start_epoch + 1} "
            f"(best {self.cfg.train.select_metric}={self.best_metric:.4f})"
        )

    def _append_history(self, record: dict) -> None:
        self.history.append(record)
        path = self._history_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        on_disk = []
        if path.exists():
            with open(path) as fh:
                on_disk = json.load(fh)
        on_disk.append(_json_safe(record))
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w") as fh:
            json.dump(on_disk, fh, indent=2)
        tmp.replace(path)

    # ------------------------------------------------------------------ fit
    def fit(self) -> list[dict]:
        self.maybe_resume()
        # Fresh start (no resume) begins a clean history; resume keeps the old.
        if self.start_epoch == 0:
            hist_path = self._history_path()
            if hist_path.exists():
                hist_path.unlink()
        for epoch in range(self.start_epoch, self.max_epochs):
            train_means, coral_active = self.train_one_epoch(epoch)
            val_metrics = self.validate()

            qwk = float(val_metrics["dr_qwk"])
            improved = qwk > self.best_metric
            if improved:
                self.best_metric = qwk
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1

            record = {
                "epoch": epoch + 1,
                "lr_backbone": self.optimizer.param_groups[0]["lr"],
                "lr_heads": self.optimizer.param_groups[1]["lr"],
                "train": {"loss": train_means, "coral_active": coral_active},
                "val": val_metrics,
                "best_metric": self.best_metric,
                "epochs_no_improve": self.epochs_no_improve,
            }
            self._append_history(record)
            self.save_checkpoint(epoch, is_best=improved)
            self.last_completed_epoch = epoch

            marker = "  <== best" if improved else ""
            print(
                f"[epoch {epoch + 1}/{self.max_epochs}] "
                f"train_loss={train_means.get('total', float('nan')):.4f} "
                f"val_loss={val_metrics['loss'].get('total', float('nan')):.4f} "
                f"dr_qwk={qwk:.4f} dr_acc={val_metrics['dr_acc']:.4f} "
                f"bcva_mse={val_metrics['bcva_mse_std']:.4f} "
                f"cst_mse={val_metrics['cst_mse_std']:.4f} "
                f"coral_active={coral_active}{marker}"
            )

            if self.epochs_no_improve >= self.patience:
                print(
                    f"[early-stop] no {self.cfg.train.select_metric} improvement in "
                    f"{self.patience} epochs — stopping at epoch {epoch + 1}"
                )
                break
        return self.history
