"""Test-time-augmentation (TTA) uncertainty engine for the frozen DR model.

Phase 3a obtains an uncertainty-aware DR grade ("grade 2, 85% confident") from
the *frozen* Phase 2 checkpoint with **no retraining and no architecture
change**. The model has no dropout layers, so MC Dropout is not applicable;
instead we perturb the input with the same augmentation used in training and
measure how stable the grade is across many augmented forward passes.

This module is deliberately **dataset-agnostic** — it never touches EyePACS
labels or calibration logic — so Phase 4a can reuse it unchanged on OLIVES. All
EyePACS-specific / calibration code lives in ``calibration_eyepacs.py``.

Public API
----------
* ``load_full_model(checkpoint, device)`` — rebuild the Phase 2 ``DRModel``
  (encoder + projection + all heads) and strict-load the checkpoint weights.
* ``read_merged_config(checkpoint)`` — recover the merged model+data+train
  config stored inside the checkpoint (paths / split / augmentation).
* ``tta_predict(...)`` — per-batch TTA: mean softmax, point grade, and a family
  of category-based and ordinal-aware confidence measures.
* ``run_tta_over_loader(...)`` — apply ``tta_predict`` across a loader, returning
  per-image arrays (labels optional); idempotent cache to ``features_dir``.
"""

from __future__ import annotations

import math
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.models.model import DRModel
from src.utils.config import load_config

NUM_DR_CLASSES = 5
AugmentFn = Callable[[torch.Tensor], torch.Tensor]


# --------------------------------------------------------------------- loading
def read_merged_config(checkpoint_path: str | Path) -> DictConfig:
    """Return the merged model+data+train config stored in the checkpoint.

    Phase 2 saves ``OmegaConf.to_container(cfg, resolve=True)`` under the
    ``config`` key, so this recovers ``paths`` / ``split`` / ``augmentation`` /
    ``batch`` exactly as trained — used to rebuild the matching test loader.
    """
    checkpoint_path = Path(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "config" not in ckpt:
        raise RuntimeError(
            f"No 'config' key in {checkpoint_path} — cannot recover the "
            "train-time paths/split/augmentation."
        )
    return OmegaConf.create(ckpt["config"])


def load_full_model(
    checkpoint_path: str | Path,
    device: str,
    model_config: str | DictConfig | None = None,
) -> nn.Module:
    """Build the Phase 2 ``DRModel`` and strict-load the checkpoint weights.

    The architecture is taken from the checkpoint's own merged ``config`` (or an
    explicit ``model_config`` fallback), built with ``pretrained=False`` so no
    ImageNet download happens, then loaded with ``strict=True``. A clean load is
    asserted (no missing/unexpected keys, tensor counts match) so a silent
    architecture drift cannot pass through. Returns ``model.eval()`` on
    ``device``. Encoder + DR head are what grading needs; the biomarker /
    regression heads load too but stay unused.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path} — expected the frozen "
            "Phase 2 checkpoint (phase2_coral_on_best.pt)."
        )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model" not in ckpt:
        raise RuntimeError(f"No 'model' state_dict in {checkpoint_path}.")
    state = ckpt["model"]

    if model_config is not None:
        arch_cfg = (
            load_config(model_config) if isinstance(model_config, str) else model_config
        )
    elif "config" in ckpt:
        arch_cfg = OmegaConf.create(ckpt["config"])
    else:
        raise RuntimeError(
            f"{checkpoint_path} has no 'config' and no model_config was given — "
            "cannot determine the architecture."
        )

    model = DRModel.from_config(arch_cfg, pretrained=False)
    result = model.load_state_dict(state, strict=True)
    assert not result.missing_keys and not result.unexpected_keys, (
        f"strict load reported missing={result.missing_keys} "
        f"unexpected={result.unexpected_keys}"
    )
    n_loaded = len(state)
    n_expected = len(model.state_dict())
    assert n_loaded == n_expected, (
        f"loaded {n_loaded} tensors but model expects {n_expected}"
    )
    n_dr = sum(1 for k in state if k.startswith("dr_head."))
    n_enc = sum(1 for k in state if k.startswith("encoder."))
    print(
        f"[model] strict load OK: {n_loaded}/{n_expected} tensors "
        f"(encoder={n_enc}, dr_head={n_dr}); eval on {device}"
    )
    model.eval()
    model.to(device)
    return model


# ------------------------------------------------------------------ TTA engine
@torch.no_grad()
def tta_predict(
    model: nn.Module,
    images: torch.Tensor,
    n_passes: int,
    augment_fn: AugmentFn,
    device: str,
    include_clean: bool = True,
) -> dict[str, torch.Tensor]:
    """Run TTA on one batch and return per-image predictions + confidences.

    For each image, ``n_passes`` augmented forward passes (plus optionally one
    clean, un-augmented pass) are collected as softmax vectors over the 5 grades.
    Augmentation is applied on CPU through the training wrapper (de-normalise ->
    augment -> re-normalise); each view is then moved to ``device`` for a
    ``no_grad`` + autocast forward. Returns CPU tensors of length ``B``:

    Point prediction
        ``probs_mean`` [B,5], ``pred_grade`` [B].
    Category-based confidence
        ``confidence_agreement`` (fraction of passes whose argmax == pred_grade),
        ``confidence_maxprob`` (probs_mean at pred_grade),
        ``entropy`` and ``entropy_norm`` (predictive entropy of probs_mean, and
        that divided by ln 5).
    Ordinal-aware measures
        ``pred_grade_mean`` (Σ k·p_k, continuous severity),
        ``grade_std`` (std of per-pass argmax grades on the 0–4 scale),
        ``within_one_frac`` (fraction of passes within ±1 grade of pred_grade).
    When ``include_clean`` also ``pred_grade_clean`` and ``confidence_clean``.
    """
    model.eval()
    images_cpu = images.detach().to("cpu")
    use_amp = isinstance(device, str) and device.startswith("cuda")

    def _forward(x: torch.Tensor) -> torch.Tensor:
        x = x.to(device, non_blocking=True)
        ctx = autocast("cuda") if use_amp else nullcontext()
        with ctx:
            logits = model(x)["dr_logits"].float()
        return torch.softmax(logits, dim=1).to("cpu")

    prob_list: list[torch.Tensor] = []
    clean_probs: Optional[torch.Tensor] = None
    if include_clean:
        clean_probs = _forward(images_cpu)  # un-augmented reference view
        prob_list.append(clean_probs)
    for _ in range(n_passes):
        prob_list.append(_forward(augment_fn(images_cpu)))

    probs = torch.stack(prob_list, dim=0)  # [P, B, 5]
    probs_mean = probs.mean(dim=0)  # [B, 5]
    pred_grade = probs_mean.argmax(dim=1)  # [B]
    per_pass_grade = probs.argmax(dim=2)  # [P, B]

    grades = torch.arange(NUM_DR_CLASSES, dtype=probs_mean.dtype)  # [5]
    agreement = (per_pass_grade == pred_grade.unsqueeze(0)).float().mean(dim=0)
    maxprob = probs_mean.gather(1, pred_grade.unsqueeze(1)).squeeze(1)
    entropy = -(probs_mean.clamp_min(1e-12).log() * probs_mean).sum(dim=1)
    entropy_norm = entropy / math.log(NUM_DR_CLASSES)
    pred_grade_mean = (probs_mean * grades.unsqueeze(0)).sum(dim=1)
    grade_std = per_pass_grade.float().std(dim=0, unbiased=False)
    within_one = (
        (per_pass_grade - pred_grade.unsqueeze(0)).abs() <= 1
    ).float().mean(dim=0)

    out: dict[str, torch.Tensor] = {
        "probs_mean": probs_mean,
        "pred_grade": pred_grade.long(),
        "confidence_agreement": agreement,
        "confidence_maxprob": maxprob,
        "entropy": entropy,
        "entropy_norm": entropy_norm,
        "pred_grade_mean": pred_grade_mean,
        "grade_std": grade_std,
        "within_one_frac": within_one,
    }
    if clean_probs is not None:
        out["pred_grade_clean"] = clean_probs.argmax(dim=1).long()
        out["confidence_clean"] = clean_probs.max(dim=1).values
    return out


def run_tta_over_loader(
    model: nn.Module,
    loader: DataLoader,
    cfg: DictConfig,
    device: str,
    augment_fn: AugmentFn,
    tag: str,
) -> dict[str, Any]:
    """Apply ``tta_predict`` across a loader; return per-image arrays + labels.

    Reads ``cfg.tta.n_passes`` / ``cfg.tta.include_clean_pass`` and caches the
    result to ``cfg.features_dir/tta_{tag}.pt`` (idempotent — a present cache is
    loaded and the passes are skipped). ``augment_fn`` is injected (the shared
    training-strength transform) so this stays dataset-agnostic. Labels are
    collected only if the loader provides ``dr_labels`` (so it also runs on
    unlabelled data, where they may be absent or sentinel ``-1``). Returns a dict
    of NumPy arrays plus a ``meta`` block.
    """
    features_dir = Path(str(cfg.features_dir))
    cache_path = features_dir / f"tta_{tag}.pt"
    if cache_path.exists():
        print(f"[tta {tag}] cache present — loading {cache_path} (skipping passes)")
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    n_passes = int(cfg.tta.n_passes)
    include_clean = bool(cfg.tta.include_clean_pass)
    n_batches = len(loader)
    print(
        f"[tta {tag}] {n_batches} batches x ({n_passes} augmented"
        f"{' + 1 clean' if include_clean else ''}) passes on {device}"
    )

    accum: dict[str, list[torch.Tensor]] = defaultdict(list)
    log_every = max(1, n_batches // 10)
    for i, batch in enumerate(tqdm(loader, desc=f"TTA {tag}", unit="batch"), start=1):
        res = tta_predict(
            model, batch["images"], n_passes, augment_fn, device, include_clean
        )
        for key, value in res.items():
            accum[key].append(value)
        if "dr_labels" in batch:
            accum["labels"].append(batch["dr_labels"].long())
        if i % log_every == 0 or i == n_batches:
            print(f"[tta {tag}] batch {i}/{n_batches}")

    out: dict[str, Any] = {
        key: torch.cat(parts, dim=0).numpy() for key, parts in accum.items()
    }
    n_images = int(out["pred_grade"].shape[0])
    out["meta"] = {
        "tag": tag,
        "n_passes": n_passes,
        "include_clean": include_clean,
        "n_images": n_images,
        "has_labels": "labels" in out,
    }

    features_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".pt.tmp")
    torch.save(out, tmp)
    tmp.replace(cache_path)
    print(f"[tta {tag}] saved {n_images} predictions -> {cache_path}")
    return out
