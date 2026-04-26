import argparse
import json
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, load_from_disk
from omegaconf import DictConfig
from PIL import Image
from tqdm.auto import tqdm

from src.utils.config import load_config

EYEPACS_SHARD_PREFIX = "eyepacs_shard"
OLIVES_OUTPUT_FILENAME = "olives_fundus.pt"
METADATA_FILENAME = "metadata.json"
DR_LABEL_COLUMN = "DRIL"
FUNDUS_TOKEN = "fundus"
PATH_COLUMN = "Path (Trial/Arm/Folder/Visit/Eye/Image Name)"
BIOMARKER_COLUMNS = (
    "Atrophy / thinning of retinal layers",
    "Disruption of EZ",
    "DRIL",
    "IR hemorrhages",
    "IR HRF",
    "Partially attached vitreous face",
    "Fully attached vitreous face",
    "Preretinal tissue/hemorrhage",
    "Vitreous debris",
    "VMT",
    "DRT/ME",
    "Fluid (IRF)",
    "Fluid (SRF)",
    "Disruption of RPE",
    "PED (serous)",
    "SHRM",
)
SECTION_BAR = "=" * 72


def _section(title: str) -> None:
    print(f"\n{SECTION_BAR}\n{title}\n{SECTION_BAR}", flush=True)


def _stage(label: str) -> None:
    print(f"\n[{label}]", flush=True)


def _build_transform(
    image_size: int, mean: list[float], std: list[float]
) -> Callable[[Image.Image], torch.Tensor]:
    mean_t = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def transform(img: Image.Image) -> torch.Tensor:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = img.resize((image_size, image_size), Image.LANCZOS)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return (tensor - mean_t) / std_t

    return transform


def _write_metadata(
    output_dir: Path,
    total_images: int,
    num_shards: int,
    image_size: int,
    mean: list[float],
    std: list[float],
    class_distribution: dict[str, int],
) -> None:
    metadata = {
        "total_images": total_images,
        "num_shards": num_shards,
        "image_size": [image_size, image_size],
        "normalisation": {"mean": mean, "std": std},
        "class_distribution": class_distribution,
        "preprocessing_date": datetime.now(timezone.utc).isoformat(),
    }
    with open(output_dir / METADATA_FILENAME, "w") as f:
        json.dump(metadata, f, indent=2)


def _normalise_csv_path(value: Any) -> str:
    return str(value).lstrip("/").replace("\\", "/").strip()


def _resolve_eyepacs_split(dataset: Dataset | DatasetDict) -> Dataset:
    if isinstance(dataset, DatasetDict):
        if "train" in dataset:
            return dataset["train"]
        return dataset[next(iter(dataset.keys()))]
    return dataset


def preprocess_eyepacs(cfg: DictConfig) -> None:
    _section("EyePACS preprocessing")

    output_dir = Path(str(cfg.eyepacs_output))
    output_dir.mkdir(parents=True, exist_ok=True)

    image_size = int(cfg.image_size)
    mean = list(cfg.normalise_mean)
    std = list(cfg.normalise_std)
    shard_size = int(cfg.eyepacs_shard_size)
    transform = _build_transform(image_size, mean, std)

    _stage(f"1/4 Loading dataset from {cfg.eyepacs_source}")
    t0 = time.time()
    dataset = _resolve_eyepacs_split(load_from_disk(str(cfg.eyepacs_source)))
    n = len(dataset)
    print(f"loaded {n} rows in {time.time() - t0:.1f}s")

    num_shards = (n + shard_size - 1) // shard_size
    _stage(f"2/4 Plan: {num_shards} shards of up to {shard_size} images each")

    _stage("3/4 Writing shards")
    label_counter: Counter = Counter()
    t_all = time.time()

    for shard_idx in range(num_shards):
        shard_path = output_dir / f"{EYEPACS_SHARD_PREFIX}_{shard_idx:03d}.pt"
        start = shard_idx * shard_size
        end = min(start + shard_size, n)

        if shard_path.exists():
            existing = torch.load(shard_path, map_location="cpu", weights_only=False)
            label_counter.update(existing["labels"].tolist())
            print(
                f"  skip existing {shard_path.name} "
                f"({len(existing['labels'])} images)"
            )
            continue

        shard_n = end - start
        images_buf = torch.empty(
            (shard_n, 3, image_size, image_size), dtype=torch.float32
        )
        labels_buf = torch.empty((shard_n,), dtype=torch.long)
        filenames: list[str] = []

        t_shard = time.time()
        shard_view = dataset.select(range(start, end))
        for i, row in enumerate(
            tqdm(
                shard_view,
                desc=f"shard {shard_idx + 1}/{num_shards}",
                total=shard_n,
            )
        ):
            images_buf[i] = transform(row["image"])
            label = int(row["label_code"])
            labels_buf[i] = label
            label_counter[label] += 1
            filenames.append(f"eyepacs_{start + i:06d}")

        tmp_path = shard_path.with_suffix(".pt.tmp")
        torch.save(
            {"images": images_buf, "labels": labels_buf, "filenames": filenames},
            tmp_path,
        )
        tmp_path.replace(shard_path)
        print(
            f"  wrote {shard_path.name} "
            f"({shard_n} images in {time.time() - t_shard:.1f}s)"
        )

    print(f"all shards done in {time.time() - t_all:.1f}s")

    _stage("4/4 Writing metadata.json")
    _write_metadata(
        output_dir,
        total_images=n,
        num_shards=num_shards,
        image_size=image_size,
        mean=mean,
        std=std,
        class_distribution={str(k): int(v) for k, v in sorted(label_counter.items())},
    )
    print(f"output: {output_dir}")


def _extract_fundus_to_scratch(
    olives_zip_path: str, scratch_dir: Path
) -> None:
    """
    Extract fundus images from triple-nested zip structure:
    olive.zip → OLIVES.zip → TREX_DME.zip and Prime_FULL.zip

    Only files with 'fundus' in the filename are extracted.
    Preserves the internal folder structure so paths can be
    matched against the CSV Path column.

    Structure inside olive.zip:
      OLIVES.zip
        └── OLIVES/
              ├── TREX_DME.zip
              └── Prime_FULL.zip

    Each inner zip contains fundus images named:
      TREX DME: TREX DME/{site}/{patient}/V{visit}/{eye}/
                fundus_{eye}_V{visit}.tif
      Prime:    Prime_FULL/{patient}/W{week}/{eye}/
                fundus_{eye}_W{week}.tif
    """
    import zipfile

    inner_zips = ["OLIVES/TREX_DME.zip", "OLIVES/Prime_FULL.zip"]

    print(f"  Opening outer zip: {olives_zip_path}")
    with zipfile.ZipFile(olives_zip_path, "r") as outer:

        # Open OLIVES.zip as a stream (second level)
        print("  Opening OLIVES.zip as stream...")
        with outer.open("OLIVES.zip") as olives_stream:
            with zipfile.ZipFile(olives_stream, "r") as olives_zip:

                for inner_zip_name in inner_zips:
                    trial = (
                        "TREX_DME"
                        if "TREX" in inner_zip_name
                        else "Prime_FULL"
                    )
                    print(f"  Processing {inner_zip_name}...")

                    with olives_zip.open(inner_zip_name) as inner_stream:
                        with zipfile.ZipFile(inner_stream, "r") as inner_zip:

                            # Find all fundus files
                            fundus_files = [
                                f
                                for f in inner_zip.namelist()
                                if FUNDUS_TOKEN in f.lower()
                                and not f.endswith("/")
                            ]
                            print(
                                f"    Found {len(fundus_files)} "
                                f"fundus files in {trial}"
                            )

                            # Extract each fundus file preserving
                            # folder structure
                            for file_path in tqdm(
                                fundus_files, desc=f"Extracting {trial}"
                            ):
                                target = scratch_dir / file_path
                                target.parent.mkdir(
                                    parents=True, exist_ok=True
                                )
                                if target.exists():
                                    continue  # resumable
                                with inner_zip.open(file_path) as src:
                                    target.write_bytes(src.read())


def _parse_olives_metadata(rel_path: str) -> dict[str, str]:
    parts = Path(rel_path).parts
    if not parts:
        return {"trial": "", "patient_id": "", "visit": "", "eye": ""}
    if "TREX" in parts[0]:
        return {
            "trial": "TREX_DME",
            "patient_id": parts[2] if len(parts) > 2 else "",
            "visit": parts[3] if len(parts) > 3 else "",
            "eye": parts[4] if len(parts) > 4 else "",
        }
    return {
        "trial": "Prime_FULL",
        "patient_id": parts[1] if len(parts) > 1 else "",
        "visit": parts[2] if len(parts) > 2 else "",
        "eye": parts[3] if len(parts) > 3 else "",
    }


def _safe_int(value: Any) -> int:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def preprocess_olives(cfg: DictConfig) -> None:
    _section("OLIVES preprocessing")

    output_dir = Path(str(cfg.olives_output))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OLIVES_OUTPUT_FILENAME

    if output_path.exists():
        print(f"output {output_path} already exists — skipping")
        return

    image_size = int(cfg.image_size)
    mean = list(cfg.normalise_mean)
    std = list(cfg.normalise_std)
    transform = _build_transform(image_size, mean, std)

    scratch_dir = Path(str(cfg.scratch_dir))
    scratch_dir.mkdir(parents=True, exist_ok=True)

    _stage(f"1/5 Extracting fundus images to {scratch_dir}")
    t0 = time.time()
    _extract_fundus_to_scratch(str(cfg.olives_zip), scratch_dir)
    print(f"extraction complete in {time.time() - t0:.1f}s")

    _stage(f"2/5 Loading labels from {cfg.olives_labels_csv}")
    t0 = time.time()
    labels_df = pd.read_csv(str(cfg.olives_labels_csv))
    labels_df["_normalised_path"] = labels_df[PATH_COLUMN].map(_normalise_csv_path)
    csv_lookup = {
        path: idx
        for idx, path in enumerate(labels_df["_normalised_path"])
        if FUNDUS_TOKEN in path.lower()
    }
    print(
        f"loaded {len(labels_df)} rows ({len(csv_lookup)} fundus rows) "
        f"in {time.time() - t0:.1f}s"
    )

    _stage("3/5 Walking scratch dir for fundus files")
    fundus_files = [
        p
        for p in scratch_dir.rglob("*")
        if p.is_file() and FUNDUS_TOKEN in p.name.lower()
    ]
    print(f"found {len(fundus_files)} fundus files on disk")

    _stage("4/5 Preprocessing matched images")
    images_list: list[torch.Tensor] = []
    labels_list: list[int] = []
    metadata_list: list[dict[str, str]] = []
    biomarkers_list: list[list[float]] = []
    bcva_list: list[float] = []
    cst_list: list[float] = []
    label_counter: Counter = Counter()
    unmatched = 0
    failed = 0

    t0 = time.time()
    for path in tqdm(fundus_files, desc="OLIVES preprocess"):
        rel_path = path.relative_to(scratch_dir).as_posix()
        normalised = _normalise_csv_path(rel_path)
        row_idx = csv_lookup.get(normalised)
        if row_idx is None:
            unmatched += 1
            continue
        try:
            with Image.open(path) as img:
                tensor = transform(img)
        except Exception as exc:
            failed += 1
            print(f"  failed to load {path}: {exc}")
            continue

        row = labels_df.iloc[row_idx]
        meta = _parse_olives_metadata(rel_path)
        label = _safe_int(row.get(DR_LABEL_COLUMN, 0))
        biomarkers = [_safe_float(row.get(c, float("nan"))) for c in BIOMARKER_COLUMNS]

        images_list.append(tensor)
        labels_list.append(label)
        metadata_list.append(meta)
        biomarkers_list.append(biomarkers)
        bcva_list.append(_safe_float(row.get("BCVA", float("nan"))))
        cst_list.append(_safe_float(row.get("CST", float("nan"))))
        label_counter[label] += 1

    n_processed = len(images_list)
    print(
        f"processed={n_processed}, unmatched={unmatched}, failed={failed} "
        f"in {time.time() - t0:.1f}s"
    )
    if n_processed == 0:
        raise RuntimeError(
            "No OLIVES fundus images were processed — check that "
            f"olives_labels_csv Path column matches files under {scratch_dir}"
        )

    _stage("5/5 Saving tensor file and metadata.json")
    t0 = time.time()
    images = torch.stack(images_list)
    labels = torch.tensor(labels_list, dtype=torch.long)
    biomarkers_t = torch.tensor(biomarkers_list, dtype=torch.float32)
    bcva_t = torch.tensor(bcva_list, dtype=torch.float32)
    cst_t = torch.tensor(cst_list, dtype=torch.float32)

    tmp_path = output_path.with_suffix(".pt.tmp")
    torch.save(
        {
            "images": images,
            "labels": labels,
            "metadata": metadata_list,
            "biomarkers": biomarkers_t,
            "bcva": bcva_t,
            "cst": cst_t,
        },
        tmp_path,
    )
    tmp_path.replace(output_path)
    _write_metadata(
        output_dir,
        total_images=n_processed,
        num_shards=1,
        image_size=image_size,
        mean=mean,
        std=std,
        class_distribution={str(k): int(v) for k, v in sorted(label_counter.items())},
    )
    print(f"saved in {time.time() - t0:.1f}s — output: {output_path}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-time preprocessing for EyePACS or OLIVES."
    )
    parser.add_argument(
        "--dataset", required=True, choices=["eyepacs", "olives"]
    )
    parser.add_argument(
        "--config", default="configs/preprocess.yaml"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        cfg = load_config(args.config)
    except Exception as exc:
        print(f"[ERROR] Failed to load config from {args.config}: {exc}")
        traceback.print_exc()
        return 2

    try:
        if args.dataset == "eyepacs":
            preprocess_eyepacs(cfg)
        else:
            preprocess_olives(cfg)
    except Exception as exc:
        print(f"\n[ERROR] {args.dataset} preprocessing failed: {exc}")
        traceback.print_exc()
        return 1

    print(f"\n[DONE] {args.dataset} preprocessing finished successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
