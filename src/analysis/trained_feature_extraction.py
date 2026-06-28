"""Extract features from a trained Phase 2 encoder, mirroring Phase 1 exactly.

Phase 1 (``feature_extraction.py``) ran a *pretrained* ResNet-50 over the data
to characterise the cross-dataset gap (FID 240.65). Phase 2d re-runs the same
extraction with the *trained* encoder backbone lifted out of a checkpoint, so
the resulting features can be fed to the unchanged Phase 1 diagnostics for an
apples-to-apples before/after comparison.

Design notes
------------
* Iteration reuses Phase 1's ``_iter_inputs`` / ``_extract_chunk`` so sampling
  order matches the published baseline.
* Per-image acquisition stats (the black-border proxy the PC1 correlation
  needs) are computed with Phase 1's ``_image_stats_chunked`` on exactly the
  images pulled for each split — so the stats stay row-aligned with the
  features for both ``full`` and the held-out ``test`` subset.
* The ``test`` split is reproduced with the same ``src.data.samplers`` routines
  and the same ratios/seed the checkpoint was trained under, so it matches the
  Phase 2b held-out set.
"""

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torchvision.models import resnet50

from src.analysis.distribution_diagnostics import (
    EYEPACS_SHARD_GLOB,
    _image_stats_chunked,
    _load_eyepacs_labels,
)
from src.analysis.feature_extraction import (
    FEATURE_DIM,
    MODEL_NAME,
    _extract_chunk,
    _iter_inputs,
)
from src.data.samplers import patient_aware_split, random_split

# Trained checkpoints store the full DRModel; the ResNet backbone lives under
# this prefix (model.encoder = ResNetEncoder, ResNetEncoder.backbone = resnet50).
ENCODER_BACKBONE_PREFIX = "encoder.backbone."
DEFAULT_BATCH_SIZE = 64
STAT_KEYS = ("brightness", "black_pixel", "contrast")


def load_trained_backbone(
    checkpoint_path: Path | str, device: str
) -> tuple[nn.Module, dict[str, Any]]:
    """Lift the trained ResNet-50 backbone out of a Phase 2 checkpoint.

    Builds a weightless ``resnet50`` with ``fc = Identity`` (matching the Phase 1
    extractor), selects the ``encoder.backbone.*`` keys from ``ckpt["model"]``,
    strips the prefix, and loads them with ``strict=True``. A clean strict load
    is asserted so a silent prefix mismatch cannot pass through as a randomly
    initialised backbone.

    Returns the eval-mode model on ``device`` plus a provenance dict carrying the
    checkpoint ``config``, ``epoch``, ``best_metric``, ``run_name``, the resolved
    ``loss_weights``, and the ``split`` config used for training.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path} — expected a trained "
            "Phase 2 checkpoint (e.g. phase2_coral_on_best.pt)."
        )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state = ckpt["model"]
    backbone_state = {
        key[len(ENCODER_BACKBONE_PREFIX) :]: value
        for key, value in model_state.items()
        if key.startswith(ENCODER_BACKBONE_PREFIX)
    }
    if not backbone_state:
        raise RuntimeError(
            f"No keys under prefix {ENCODER_BACKBONE_PREFIX!r} in "
            f"{checkpoint_path} — checkpoint layout changed?"
        )

    model = resnet50(weights=None)
    model.fc = nn.Identity()
    try:
        model.load_state_dict(backbone_state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Backbone state_dict did not map cleanly onto resnet50 after "
            f"stripping {ENCODER_BACKBONE_PREFIX!r} — prefix mismatch or "
            f"architecture drift. Original error: {exc}"
        ) from exc

    n_loaded = len(backbone_state)
    n_expected = len(model.state_dict())
    assert n_loaded == n_expected, (
        f"loaded {n_loaded} tensors but resnet50(fc=Identity) expects "
        f"{n_expected} — strict load should have caught this"
    )
    print(
        f"[backbone] loaded {n_loaded}/{n_expected} tensors from "
        f"'{ENCODER_BACKBONE_PREFIX}*' (clean strict load)"
    )

    model.eval()
    model.to(device)

    config = dict(ckpt.get("config", {}))
    provenance: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "run_name": config.get("run_name"),
        "epoch": int(ckpt.get("epoch", -1)),
        "best_metric": float(ckpt.get("best_metric", float("nan"))),
        "loss_weights": dict(config.get("loss_weights", {})),
        "split": dict(config.get("split", {})),
    }
    return model, provenance


def _arm_from_checkpoint_path(checkpoint_path: Path | str) -> str:
    """Recover the arm name from ``phase2_<arm>_<best|last>.pt``."""
    name = Path(checkpoint_path).stem
    if name.startswith("phase2_"):
        name = name[len("phase2_") :]
    for suffix in ("_best", "_last"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


def feature_paths(
    features_dir: Path | str, arm: str, split: str
) -> tuple[Path, Path]:
    """Drive paths for the cached EyePACS/OLIVES feature files of one arm+split."""
    features_dir = Path(features_dir)
    return (
        features_dir / f"phase2_{arm}_{split}_eyepacs_features.pt",
        features_dir / f"phase2_{arm}_{split}_olives_features.pt",
    )


def load_saved_features(path: Path | str) -> dict[str, Any]:
    """Load a feature blob saved by this module (features/labels/stats/metadata)."""
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def _split_ratios_seed(
    cfg: DictConfig, provenance: dict[str, Any]
) -> tuple[tuple[float, float, float], int]:
    """Use the checkpoint's own split config so the test set matches training."""
    split = provenance.get("split", {}) or {}
    ratios = (
        float(split.get("train", 0.70)),
        float(split.get("val", 0.15)),
        float(split.get("test", 0.15)),
    )
    seed = int(split.get("seed", cfg.get("seed", 42)))
    return ratios, seed


def _concat_stats(parts: Sequence[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        key: np.concatenate([p[key] for p in parts]) for key in STAT_KEYS
    }


def _extract_eyepacs_full(
    model: nn.Module,
    eyepacs_dir: Path,
    batch_size: int,
    device: str,
    use_amp: bool,
    desc_prefix: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, np.ndarray]]:
    """All EyePACS shards in Phase 1 order: features, DR labels, image stats."""
    feats: list[torch.Tensor] = []
    stats: list[dict[str, np.ndarray]] = []
    for name, images in _iter_inputs(eyepacs_dir, "eyepacs"):
        feats.append(
            _extract_chunk(
                model, images, batch_size, device, use_amp, desc=f"{desc_prefix} {name}"
            )
        )
        stats.append(_image_stats_chunked(images))
        del images
    features = torch.cat(feats, dim=0).float()
    labels = _load_eyepacs_labels(eyepacs_dir)
    labels_t = torch.from_numpy(labels).long()
    if labels_t.shape[0] != features.shape[0]:
        raise RuntimeError(
            f"EyePACS labels ({labels_t.shape[0]}) and features "
            f"({features.shape[0]}) disagree — shard order mismatch."
        )
    return features, labels_t, _concat_stats(stats)


def _extract_eyepacs_subset(
    model: nn.Module,
    eyepacs_dir: Path,
    indices_sorted: np.ndarray,
    batch_size: int,
    device: str,
    use_amp: bool,
    desc_prefix: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, np.ndarray]]:
    """Stream only the given global indices across sorted shards (test split).

    Mirrors the streaming selection in Phase 1's ``_eyepacs_image_stats_at_indices``
    so features, labels, and stats all come out in sorted-global-index order.
    """
    shards = sorted(Path(eyepacs_dir).glob(EYEPACS_SHARD_GLOB))
    if not shards:
        raise FileNotFoundError(f"No EyePACS shards in {eyepacs_dir}.")

    n = int(indices_sorted.shape[0])
    feats: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    stats: list[dict[str, np.ndarray]] = []

    global_offset = 0
    cursor = 0
    for shard_path in shards:
        if cursor >= n:
            break
        data = torch.load(shard_path, map_location="cpu", weights_only=False)
        images = data["images"]
        shard_labels = data["labels"]
        shard_size = int(images.shape[0])
        end_cursor = cursor
        while (
            end_cursor < n
            and indices_sorted[end_cursor] < global_offset + shard_size
        ):
            end_cursor += 1
        if cursor < end_cursor:
            local = torch.from_numpy(
                (indices_sorted[cursor:end_cursor] - global_offset).astype(np.int64)
            )
            sub_images = images[local]
            feats.append(
                _extract_chunk(
                    model,
                    sub_images,
                    batch_size,
                    device,
                    use_amp,
                    desc=f"{desc_prefix} {shard_path.name}",
                )
            )
            labels.append(shard_labels[local].long())
            stats.append(_image_stats_chunked(sub_images))
            cursor = end_cursor
        global_offset += shard_size
        del data, images

    if cursor != n:
        raise RuntimeError(
            f"Only {cursor}/{n} EyePACS test indices were loaded — "
            "shards may be missing samples."
        )
    return (
        torch.cat(feats, dim=0).float(),
        torch.cat(labels, dim=0).long(),
        _concat_stats(stats),
    )


def _extract_olives(
    model: nn.Module,
    images: torch.Tensor,
    batch_size: int,
    device: str,
    use_amp: bool,
    desc: str,
) -> tuple[torch.Tensor, dict[str, np.ndarray]]:
    features = _extract_chunk(
        model, images, batch_size, device, use_amp, desc=desc
    ).float()
    stats = _image_stats_chunked(images)
    return features, stats


def _save_features(
    path: Path,
    features: torch.Tensor,
    labels: torch.Tensor | None,
    stats: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> None:
    if features.shape[1] != FEATURE_DIM:
        raise RuntimeError(
            f"Expected feature_dim={FEATURE_DIM}, got {features.shape[1]} — "
            "backbone fc-strip likely failed."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "features": features,
        "labels": labels,
        "image_stats": stats,
        "metadata": metadata,
    }
    tmp_path = path.with_suffix(".pt.tmp")
    torch.save(blob, tmp_path)
    tmp_path.replace(path)


def extract_features_for_checkpoint(
    checkpoint_path: Path | str,
    cfg: DictConfig,
    split: str,
    device: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """Extract (or load cached) trained-encoder features for one arm and split.

    ``split="full"`` mirrors Phase 1 (all data, no subsampling). ``split="test"``
    restricts to the Phase 2b held-out indices, reproduced with the same samplers
    and the checkpoint's own ratios/seed. Idempotent: if both feature files exist
    they are loaded and returned, and the extraction is skipped.

    Returns a dict with ``eyepacs_path``, ``olives_path``, ``arm``, ``split``,
    and ``provenance`` (read back from the cached metadata on a skip).
    """
    if split not in ("full", "test"):
        raise ValueError(f"split must be 'full' or 'test', got {split!r}")

    arm = _arm_from_checkpoint_path(checkpoint_path)
    features_dir = Path(str(cfg.features_dir))
    eye_path, oli_path = feature_paths(features_dir, arm, split)

    if eye_path.exists() and oli_path.exists():
        print(
            f"[extract] {arm}/{split}: cached features present — skipping "
            f"({eye_path.name}, {oli_path.name})"
        )
        eye_blob = load_saved_features(eye_path)
        return {
            "eyepacs_path": eye_path,
            "olives_path": oli_path,
            "arm": arm,
            "split": split,
            "provenance": dict(eye_blob["metadata"].get("provenance", {})),
        }

    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[extract] cuda requested but unavailable — falling back to cpu")
        device = "cpu"
    use_amp = device.startswith("cuda")

    print(f"[extract] {arm}/{split}: loading trained backbone (device={device})")
    model, provenance = load_trained_backbone(checkpoint_path, device)
    print(
        f"[extract] {arm}/{split}: run_name={provenance['run_name']} "
        f"epoch={provenance['epoch']} best_metric={provenance['best_metric']:.4f} "
        f"w_coral={provenance['loss_weights'].get('w_coral')}"
    )

    eyepacs_dir = Path(str(cfg.eyepacs_dir))
    olives_file = Path(str(cfg.olives_file))
    t0 = time.time()

    if split == "full":
        print(f"[extract] {arm}/full: EyePACS (all shards)")
        eye_feats, eye_labels, eye_stats = _extract_eyepacs_full(
            model, eyepacs_dir, batch_size, device, use_amp, f"{arm}/full eyepacs"
        )
        print(f"[extract] {arm}/full: OLIVES (full set)")
        for name, images in _iter_inputs(olives_file, "olives"):
            oli_feats, oli_stats = _extract_olives(
                model, images, batch_size, device, use_amp, f"{arm}/full {name}"
            )
            del images
    else:
        ratios, seed = _split_ratios_seed(cfg, provenance)
        print(
            f"[extract] {arm}/test: reproducing held-out split "
            f"(ratios={ratios}, seed={seed})"
        )
        olives = torch.load(olives_file, map_location="cpu", weights_only=False)
        _, _, ol_test = patient_aware_split(
            olives["patient_ids"],
            olives["has_clinical"],
            olives["has_biomarkers"],
            ratios,
            seed,
        )
        ol_idx = torch.tensor(ol_test, dtype=torch.long)
        oli_feats, oli_stats = _extract_olives(
            model,
            olives["images"][ol_idx],
            batch_size,
            device,
            use_amp,
            f"{arm}/test olives",
        )
        del olives

        total_eyepacs = int(_load_eyepacs_labels(eyepacs_dir).shape[0])
        _, _, ep_test = random_split(total_eyepacs, ratios, seed)
        ep_idx_sorted = np.sort(np.asarray(ep_test, dtype=np.int64))
        print(
            f"[extract] {arm}/test: EyePACS held-out {ep_idx_sorted.shape[0]} / "
            f"{total_eyepacs} samples"
        )
        eye_feats, eye_labels, eye_stats = _extract_eyepacs_subset(
            model,
            eyepacs_dir,
            ep_idx_sorted,
            batch_size,
            device,
            use_amp,
            f"{arm}/test eyepacs",
        )

    extraction_date = datetime.now(timezone.utc).isoformat()
    base_meta: dict[str, Any] = {
        "arm": arm,
        "split": split,
        "feature_dim": FEATURE_DIM,
        "model_name": MODEL_NAME,
        "extraction_date": extraction_date,
        "provenance": provenance,
    }

    _save_features(
        eye_path,
        eye_feats,
        eye_labels,
        eye_stats,
        {**base_meta, "dataset": "eyepacs", "n_samples": int(eye_feats.shape[0])},
    )
    _save_features(
        oli_path,
        oli_feats,
        None,
        oli_stats,
        {**base_meta, "dataset": "olives", "n_samples": int(oli_feats.shape[0])},
    )

    elapsed = time.time() - t0
    print(
        f"[extract] {arm}/{split}: saved EyePACS {tuple(eye_feats.shape)} + "
        f"OLIVES {tuple(oli_feats.shape)} in {elapsed:.1f}s"
    )
    return {
        "eyepacs_path": eye_path,
        "olives_path": oli_path,
        "arm": arm,
        "split": split,
        "provenance": provenance,
    }
