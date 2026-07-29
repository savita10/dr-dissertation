"""One-off: compute the OLIVES near-IR reference intensity histogram / CDF.

Loads every OLIVES fundus image via the existing on-disk tensor
(``olives_fundus.pt``, ImageNet-normalised), de-normalises to raw [0, 1] space,
takes the reference channel (green by default; OLIVES near-IR is effectively
single-channel so any channel ~ intensity), aggregates a global intensity
histogram over all pixels, and saves the histogram + CDF as the reference the
Design C1 ``GreenHistMatch`` transform matches EyePACS toward.

Reuses the existing loader path and the existing ``olives_fundus.pt`` — it never
rebuilds or moves OLIVES data. Idempotent: an existing output is loaded and
summarised, not overwritten (pass ``--force`` to recompute). CPU is fine.

All artefacts land under ``greyscale_experiment/`` on Drive (new subfolder); no
existing report/figure/checkpoint is touched.

Run (from the repo root, in Colab after mounting Drive):
    python -m scripts.compute_olives_histogram --config configs/greyscale_train.yaml
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from tqdm.auto import tqdm

from src.utils.config import load_config


def _denormalise_channel(
    images: torch.Tensor, channel: int, mean: list[float], std: list[float]
) -> torch.Tensor:
    """De-normalise one channel of an ImageNet-normalised batch back to [0, 1]."""
    m = float(mean[channel])
    s = float(std[channel])
    chan = images[:, channel, :, :].float() * s + m
    return chan.clamp(0.0, 1.0)


def compute_reference_histogram(
    olives_file: str | Path,
    channel: int,
    n_bins: int,
    imagenet_mean: list[float],
    imagenet_std: list[float],
    batch: int = 256,
) -> dict:
    """Aggregate the global [0, 1] intensity histogram + CDF over all OLIVES pixels."""
    olives = torch.load(str(olives_file), map_location="cpu", weights_only=False)
    images = olives["images"]
    if images.dim() != 4 or images.shape[1] != 3:
        raise ValueError(
            f"Expected OLIVES images [N, 3, H, W], got {tuple(images.shape)}."
        )
    n = int(images.shape[0])
    print(f"[hist] OLIVES images: {n} (all tiers) from {olives_file}")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float64)
    hist = np.zeros(n_bins, dtype=np.float64)

    for start in tqdm(range(0, n, batch), desc="OLIVES histogram", unit="batch"):
        end = min(start + batch, n)
        chan = _denormalise_channel(
            images[start:end], channel, imagenet_mean, imagenet_std
        )
        counts, _ = np.histogram(chan.numpy().reshape(-1), bins=bin_edges)
        hist += counts

    total = float(hist.sum())
    if total <= 0:
        raise RuntimeError("Empty OLIVES intensity histogram — no pixels counted.")

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    cdf = np.cumsum(hist) / total  # nondecreasing in [0, 1]; inverse-CDF target

    return {
        "hist": hist,
        "bin_edges": bin_edges,
        "bin_centers": bin_centers,
        "cdf": cdf,
        "value_range": np.array([0.0, 1.0], dtype=np.float64),
        "channel": int(channel),
        "n_images": int(n),
        "n_bins": int(n_bins),
        "imagenet_mean": np.asarray(imagenet_mean, dtype=np.float64),
        "imagenet_std": np.asarray(imagenet_std, dtype=np.float64),
        "source": str(olives_file),
        "description": (
            "OLIVES near-IR reference intensity CDF in de-normalised [0,1] space; "
            "target for Design C1 GreenHistMatch harmonisation of EyePACS."
        ),
    }


def _save(reference: dict, out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npy")
    np.save(tmp, reference, allow_pickle=True)
    tmp.replace(out)
    print(f"[hist] wrote reference histogram -> {out}")


def _summarise(reference: dict) -> None:
    centers = np.asarray(reference["bin_centers"], dtype=np.float64)
    cdf = np.asarray(reference["cdf"], dtype=np.float64)
    mean_intensity = float((centers * np.asarray(reference["hist"])).sum()
                           / max(1.0, float(np.asarray(reference["hist"]).sum())))
    # Quantile intensities via the inverse CDF.
    q = {p: float(np.interp(p, cdf, centers)) for p in (0.05, 0.5, 0.95)}
    print(
        f"[hist] n_images={reference.get('n_images')} "
        f"channel={reference.get('channel')} n_bins={reference.get('n_bins')}"
    )
    print(
        f"[hist] intensity mean={mean_intensity:.4f} "
        f"p05={q[0.05]:.4f} p50={q[0.5]:.4f} p95={q[0.95]:.4f} (in [0,1])"
    )


def run(cfg: DictConfig, force: bool = False) -> None:
    data_cfg = load_config(str(cfg.paths.data_config))
    olives_file = str(data_cfg.paths.olives_file)
    imagenet_mean = [float(v) for v in data_cfg.imagenet_mean]
    imagenet_std = [float(v) for v in data_cfg.imagenet_std]

    out_path = Path(str(cfg.reference_hist))
    channel = int(cfg.get("reference_channel", 1))
    n_bins = int(cfg.get("reference_hist_bins", 256))

    if out_path.exists() and not force:
        print(f"[hist] output {out_path} already exists — loading + summarising (idempotent).")
        _summarise(np.load(out_path, allow_pickle=True).item())
        return

    reference = compute_reference_histogram(
        olives_file, channel, n_bins, imagenet_mean, imagenet_std
    )
    _save(reference, out_path)
    _summarise(reference)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute the OLIVES near-IR reference intensity histogram (Design C1)."
    )
    parser.add_argument("--config", default="configs/greyscale_train.yaml")
    parser.add_argument(
        "--force", action="store_true", help="recompute even if the output exists"
    )
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
        run(cfg, force=args.force)
    except Exception as exc:
        print(f"\n[ERROR] OLIVES histogram computation failed: {exc}")
        traceback.print_exc()
        return 1

    print("\n[DONE] OLIVES reference histogram ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
