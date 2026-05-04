"""Extract ResNet-50 features from preprocessed fundus image tensors."""

import argparse
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.amp import autocast
from torchvision.models import ResNet50_Weights, resnet50
from tqdm.auto import tqdm

from src.utils.config import load_config

FEATURE_DIM = 2048
MODEL_NAME = "resnet50_imagenet1k_v2"
EYEPACS_FEATURES_FILENAME = "eyepacs_resnet50_features.pt"
OLIVES_FEATURES_FILENAME = "olives_resnet50_features.pt"
OLIVES_INPUT_FILENAME = "olives_fundus.pt"
EYEPACS_SHARD_GLOB = "eyepacs_shard_*.pt"


def _build_encoder(device: str) -> nn.Module:
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Identity()
    model.eval()
    model.to(device)
    return model


def _iter_inputs(
    input_path: Path, dataset_name: str
) -> Iterator[tuple[str, torch.Tensor]]:
    if dataset_name == "eyepacs":
        shards = sorted(input_path.glob(EYEPACS_SHARD_GLOB))
        if not shards:
            raise FileNotFoundError(
                f"No EyePACS shards matching {EYEPACS_SHARD_GLOB} under "
                f"{input_path} — run preprocess --dataset eyepacs first."
            )
        for shard_path in shards:
            data = torch.load(
                shard_path, map_location="cpu", weights_only=False
            )
            yield shard_path.name, data["images"]
    else:
        if not input_path.exists():
            raise FileNotFoundError(
                f"OLIVES input not found at {input_path} — run "
                "preprocess --dataset olives first."
            )
        data = torch.load(input_path, map_location="cpu", weights_only=False)
        yield input_path.name, data["images"]


def _extract_chunk(
    encoder: nn.Module,
    images: torch.Tensor,
    batch_size: int,
    device: str,
    use_amp: bool,
    desc: str,
) -> torch.Tensor:
    n = images.shape[0]
    out: list[torch.Tensor] = []
    with tqdm(total=n, desc=desc, unit="img") as pbar:
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = images[start:end].to(device, non_blocking=True)
            t0 = time.time()
            with torch.no_grad():
                if use_amp:
                    with autocast("cuda"):
                        feats = encoder(batch)
                else:
                    feats = encoder(batch)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed = time.time() - t0
            ips = (end - start) / elapsed if elapsed > 0 else 0.0
            out.append(feats.float().cpu())
            pbar.update(end - start)
            pbar.set_postfix(ips=f"{ips:.1f}")
    return torch.cat(out, dim=0)


def extract_features(
    input_path: Path | str,
    output_path: Path | str,
    dataset_name: str,
    batch_size: int = 64,
    device: str = "cuda",
) -> None:
    """Run ResNet-50 over preprocessed image tensors and save features + meta."""
    if dataset_name not in ("eyepacs", "olives"):
        raise ValueError(
            f"dataset_name must be 'eyepacs' or 'olives', got {dataset_name!r}"
        )

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        print(f"[features] {output_path} already exists — skipping")
        return

    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[features] cuda requested but unavailable — falling back to cpu")
        device = "cpu"
    use_amp = device.startswith("cuda")

    print(
        f"[features] dataset={dataset_name} device={device} "
        f"batch_size={batch_size}"
    )
    print(f"[features] loading encoder: {MODEL_NAME}")
    t0 = time.time()
    encoder = _build_encoder(device)
    print(f"  encoder ready in {time.time() - t0:.1f}s")

    print(f"[features] reading inputs from {input_path}")
    t_total = time.time()
    chunks: list[torch.Tensor] = []
    n_total = 0
    for name, images in _iter_inputs(input_path, dataset_name):
        feats = _extract_chunk(
            encoder, images, batch_size, device, use_amp, desc=name
        )
        chunks.append(feats)
        n_total += feats.shape[0]
        del images

    features = torch.cat(chunks, dim=0).float()
    if features.shape[1] != FEATURE_DIM:
        raise RuntimeError(
            f"Expected feature_dim={FEATURE_DIM}, got {features.shape[1]} — "
            "encoder fc-strip likely failed."
        )

    metadata: dict[str, object] = {
        "dataset_name": dataset_name,
        "n_samples": n_total,
        "feature_dim": FEATURE_DIM,
        "extraction_date": datetime.now(timezone.utc).isoformat(),
        "model_name": MODEL_NAME,
    }

    tmp_path = output_path.with_suffix(".pt.tmp")
    torch.save({"features": features, "metadata": metadata}, tmp_path)
    tmp_path.replace(output_path)

    elapsed = time.time() - t_total
    ips_overall = n_total / elapsed if elapsed > 0 else 0.0
    print(
        f"[features] saved {tuple(features.shape)} ({features.dtype}) to "
        f"{output_path} in {elapsed:.1f}s ({ips_overall:.1f} img/s overall)"
    )


def _resolve_paths(cfg: DictConfig, dataset: str) -> tuple[Path, Path]:
    features_dir = Path(str(cfg.features_dir))
    features_dir.mkdir(parents=True, exist_ok=True)
    if dataset == "eyepacs":
        return (
            Path(str(cfg.eyepacs_output)),
            features_dir / EYEPACS_FEATURES_FILENAME,
        )
    return (
        Path(str(cfg.olives_output)) / OLIVES_INPUT_FILENAME,
        features_dir / OLIVES_FEATURES_FILENAME,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract ResNet-50 features from preprocessed tensors."
    )
    parser.add_argument(
        "--dataset", required=True, choices=["eyepacs", "olives"]
    )
    parser.add_argument("--config", default="configs/preprocess.yaml")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        cfg = load_config(args.config)
    except Exception as exc:
        print(f"[ERROR] failed to load config from {args.config}: {exc}")
        traceback.print_exc()
        return 2

    try:
        input_path, output_path = _resolve_paths(cfg, args.dataset)
        extract_features(
            input_path=input_path,
            output_path=output_path,
            dataset_name=args.dataset,
            batch_size=args.batch_size,
            device=args.device,
        )
    except Exception as exc:
        print(f"\n[ERROR] {args.dataset} feature extraction failed: {exc}")
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
