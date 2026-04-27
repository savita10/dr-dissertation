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
# Clinical_Data_Images.xlsx uses a different path-column header than the
# biomarker CSV. The path-format itself is identical (leading slash, same
# Trial/.../patient/visit/eye/file.tif structure) — first row example:
#   /Prime_FULL/02-010/W0/OD/15.tif
# so _csv_path_to_key works unchanged on both files.
CLINICAL_PATH_COLUMN = "File_Path"
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
DRIL_INDEX = BIOMARKER_COLUMNS.index(DR_LABEL_COLUMN)
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


def _csv_path_to_key(
    csv_path: str,
) -> tuple[str, str, str, str] | None:
    """
    Extract (trial, patient, visit, eye) from a CSV Path entry.

    CSV format has a leading slash:
      /Prime_FULL/patient/visit/eye/filename.tif → parts[1, 2, 3, 4]
      /TREX DME/arm/patient/visit/eye/filename.tif → parts[1, 3, 4, 5]
    """
    parts = str(csv_path).split("/")
    if len(parts) < 2:
        return None
    trial = parts[1]
    if trial == "Prime_FULL" and len(parts) >= 5:
        return (trial, parts[2], parts[3], parts[4])
    if trial == "TREX DME" and len(parts) >= 6:
        return (trial, parts[3], parts[4], parts[5])
    return None


def _disk_path_to_key(
    rel_path: str,
) -> tuple[str, str, str, str] | None:
    """
    Extract (trial, patient, visit, eye) from a scratch-relative fundus path.

    Disk format has no leading slash:
      Prime_FULL/patient/visit/eye/fundus_*.tif → parts[0, 1, 2, 3]
      TREX DME/arm/patient/visit/eye/fundus_*.tif → parts[0, 2, 3, 4]
    """
    parts = rel_path.split("/")
    if not parts:
        return None
    trial = parts[0]
    if trial == "Prime_FULL" and len(parts) >= 4:
        return (trial, parts[1], parts[2], parts[3])
    if trial == "TREX DME" and len(parts) >= 5:
        return (trial, parts[2], parts[3], parts[4])
    return None


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


def _to_long_or_missing(value: Any) -> int:
    """Cast value to int; return -1 for None/NaN/non-numeric (e.g. string IDs)."""
    if value is None:
        return -1
    if isinstance(value, float) and np.isnan(value):
        return -1
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _build_clinical_lookup(
    xlsx_path: str,
) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    """Tier-1 lookup: BCVA/CST/Patient_ID/Eye_ID per (trial, patient, visit, eye)."""
    df = pd.read_excel(xlsx_path)
    if CLINICAL_PATH_COLUMN not in df.columns:
        raise KeyError(
            f"Expected column {CLINICAL_PATH_COLUMN!r} in {xlsx_path}. "
            f"Found: {list(df.columns)}"
        )
    df = df.assign(
        _key=df[CLINICAL_PATH_COLUMN].map(lambda p: _csv_path_to_key(str(p)))
    )
    df = df[df["_key"].notna()].drop_duplicates(subset="_key")
    lookup: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for _, row in df.iterrows():
        lookup[row["_key"]] = {
            "BCVA": row.get("BCVA"),
            "CST": row.get("CST"),
            "Patient_ID": row.get("Patient_ID"),
            "Eye_ID": row.get("Eye_ID"),
        }
    return lookup


def _build_biomarker_lookup(
    csv_path: str,
) -> dict[tuple[str, str, str, str], np.ndarray]:
    """Tier-2 lookup: 16-element float32 biomarker vector per visit-eye key."""
    df = pd.read_csv(csv_path)
    if PATH_COLUMN not in df.columns:
        raise KeyError(
            f"Expected column {PATH_COLUMN!r} in {csv_path}. "
            f"Found: {list(df.columns)}"
        )
    df = df.assign(_key=df[PATH_COLUMN].map(lambda p: _csv_path_to_key(str(p))))
    df = df[df["_key"].notna()].drop_duplicates(subset="_key")
    lookup: dict[tuple[str, str, str, str], np.ndarray] = {}
    for _, row in df.iterrows():
        vec = np.array(
            [_safe_float(row.get(c, float("nan"))) for c in BIOMARKER_COLUMNS],
            dtype=np.float32,
        )
        lookup[row["_key"]] = vec
    return lookup


def preprocess_olives(cfg: DictConfig) -> None:
    """
    Three-tier label assignment for OLIVES fundus images.

    Tier 0 — fundus image only (no clinical, no biomarker labels).
    Tier 1 — fundus + clinical (BCVA, CST, Patient_ID, Eye_ID).
    Tier 2 — fundus + clinical + 16 biomarker labels.

    All fundus files on disk are retained regardless of label availability;
    has_clinical and has_biomarkers masks indicate which labels are present
    per sample. NaN values in tensors mark missing labels and must be masked
    by downstream losses.
    """
    _section("OLIVES preprocessing")

    output_dir = Path(str(cfg.olives_output))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OLIVES_OUTPUT_FILENAME

    if output_path.exists():
        print(f"output {output_path} already exists — skipping")
        return

    xlsx_path = cfg.get("olives_clinical_xlsx", None)
    if not xlsx_path:
        raise ValueError(
            "preprocess.yaml is missing required key 'olives_clinical_xlsx' — "
            "this should point to Clinical_Data_Images.xlsx (tier-1 BCVA/CST "
            "labels). See configs/preprocess.yaml for the expected format."
        )

    image_size = int(cfg.image_size)
    mean = list(cfg.normalise_mean)
    std = list(cfg.normalise_std)
    transform = _build_transform(image_size, mean, std)

    scratch_dir = Path(str(cfg.scratch_dir))
    scratch_dir.mkdir(parents=True, exist_ok=True)

    _stage(f"1/6 Extracting fundus images to {scratch_dir}")
    t0 = time.time()
    _extract_fundus_to_scratch(str(cfg.olives_zip), scratch_dir)
    print(f"extraction complete in {time.time() - t0:.1f}s")

    _stage(f"2/6 Loading biomarker labels from {cfg.olives_labels_csv}")
    t0 = time.time()
    biomarker_lookup = _build_biomarker_lookup(str(cfg.olives_labels_csv))
    print(
        f"Biomarker lookup: {len(biomarker_lookup)} keys "
        f"(loaded in {time.time() - t0:.1f}s)"
    )

    _stage(f"3/6 Loading clinical labels from {xlsx_path}")
    t0 = time.time()
    clinical_lookup = _build_clinical_lookup(str(xlsx_path))
    print(
        f"Clinical lookup: {len(clinical_lookup)} keys "
        f"(loaded in {time.time() - t0:.1f}s)"
    )

    _stage("4/6 Walking scratch dir for fundus files")
    fundus_files = [
        p
        for p in scratch_dir.rglob("*")
        if p.is_file() and FUNDUS_TOKEN in p.name.lower()
    ]
    print(f"found {len(fundus_files)} fundus files on disk")

    _stage("5/6 Preprocessing all fundus images with three-tier label assignment")
    images_list: list[torch.Tensor] = []
    labels_list: list[int] = []
    metadata_list: list[dict[str, str]] = []
    biomarkers_list: list[list[float]] = []
    bcva_list: list[float] = []
    cst_list: list[float] = []
    has_clinical_list: list[bool] = []
    has_biomarkers_list: list[bool] = []
    patient_ids_list: list[int] = []
    eye_ids_list: list[int] = []
    keys_list: list[tuple[str, str, str, str]] = []
    label_counter: Counter = Counter()
    n_with_clinical = 0
    n_with_biomarkers = 0
    failed = 0

    nan_biomarkers = [float("nan")] * len(BIOMARKER_COLUMNS)

    t0 = time.time()
    for path in tqdm(fundus_files, desc="OLIVES preprocess"):
        rel_path = path.relative_to(scratch_dir).as_posix()
        key = _disk_path_to_key(rel_path)

        try:
            with Image.open(path) as img:
                tensor = transform(img)
        except Exception as exc:
            failed += 1
            print(f"  failed to load {path}: {exc}")
            continue

        meta = _parse_olives_metadata(rel_path)

        clinical = clinical_lookup.get(key) if key is not None else None
        if clinical is not None:
            bcva = _safe_float(clinical.get("BCVA"))
            cst = _safe_float(clinical.get("CST"))
            patient_id = _to_long_or_missing(clinical.get("Patient_ID"))
            eye_id = _to_long_or_missing(clinical.get("Eye_ID"))
            has_clinical = True
            n_with_clinical += 1
        else:
            bcva = float("nan")
            cst = float("nan")
            patient_id = -1
            eye_id = -1
            has_clinical = False

        biomarker_vec = biomarker_lookup.get(key) if key is not None else None
        if biomarker_vec is not None:
            biomarkers = biomarker_vec.tolist()
            label = _safe_int(biomarker_vec[DRIL_INDEX])
            has_biomarkers = True
            n_with_biomarkers += 1
            label_counter[label] += 1
        else:
            biomarkers = list(nan_biomarkers)
            label = 0
            has_biomarkers = False

        images_list.append(tensor)
        labels_list.append(label)
        metadata_list.append(meta)
        biomarkers_list.append(biomarkers)
        bcva_list.append(bcva)
        cst_list.append(cst)
        has_clinical_list.append(has_clinical)
        has_biomarkers_list.append(has_biomarkers)
        patient_ids_list.append(patient_id)
        eye_ids_list.append(eye_id)
        keys_list.append(key if key is not None else ("", "", "", ""))

    n_total = len(images_list)
    pct_clin = n_with_clinical / n_total if n_total else 0.0
    pct_bio = n_with_biomarkers / n_total if n_total else 0.0
    print(
        f"\n=== OLIVES label coverage ===\n"
        f"  Total fundus images:         {n_total}\n"
        f"  With clinical (BCVA/CST):    {n_with_clinical} ({pct_clin:.1%})\n"
        f"  With biomarkers (16 labels): {n_with_biomarkers} ({pct_bio:.1%})\n"
        f"  Clinical-only (no biomarker):"
        f" {n_with_clinical - n_with_biomarkers}\n"
        f"  Failed to load:              {failed}"
    )
    print(f"stage 5/6 complete in {time.time() - t0:.1f}s")

    if n_total == 0:
        raise RuntimeError(
            f"No fundus images were processed — found "
            f"{len(fundus_files)} files on disk, all failed to load."
        )

    _stage("6/6 Saving tensor file and metadata.json")
    t0 = time.time()
    images = torch.stack(images_list)
    labels = torch.tensor(labels_list, dtype=torch.long)
    biomarkers_t = torch.tensor(biomarkers_list, dtype=torch.float32)
    bcva_t = torch.tensor(bcva_list, dtype=torch.float32)
    cst_t = torch.tensor(cst_list, dtype=torch.float32)
    has_clinical_t = torch.tensor(has_clinical_list, dtype=torch.bool)
    has_biomarkers_t = torch.tensor(has_biomarkers_list, dtype=torch.bool)
    patient_ids_t = torch.tensor(patient_ids_list, dtype=torch.long)
    eye_ids_t = torch.tensor(eye_ids_list, dtype=torch.long)

    tmp_path = output_path.with_suffix(".pt.tmp")
    # NOTE on `labels`: this is the binary DRIL biomarker flag (one of the 16
    # biomarker columns), NOT a multi-class DR severity grade. Values are
    # 0 / 1 where biomarkers are present and 0 where they are not. Downstream
    # classification losses MUST mask by `has_biomarkers` before computing
    # loss — samples with has_biomarkers=False have label=0 by default and
    # would otherwise pollute the negative class.
    torch.save(
        {
            "images": images,
            "labels": labels,
            "metadata": metadata_list,
            "biomarkers": biomarkers_t,
            "bcva": bcva_t,
            "cst": cst_t,
            "has_clinical": has_clinical_t,
            "has_biomarkers": has_biomarkers_t,
            "patient_ids": patient_ids_t,
            "eye_ids": eye_ids_t,
            "keys": keys_list,
        },
        tmp_path,
    )
    tmp_path.replace(output_path)

    n_unlabelled = n_total - n_with_biomarkers
    class_distribution: dict[str, int] = {
        str(k): int(v) for k, v in sorted(label_counter.items())
    }
    class_distribution["unlabelled"] = n_unlabelled

    _write_metadata(
        output_dir,
        total_images=n_total,
        num_shards=1,
        image_size=image_size,
        mean=mean,
        std=std,
        class_distribution=class_distribution,
    )

    metadata_path = output_dir / METADATA_FILENAME
    with open(metadata_path) as f:
        metadata = json.load(f)
    metadata["olives_label_coverage"] = {
        "total_fundus": n_total,
        "with_clinical_labels": n_with_clinical,
        "with_biomarker_labels": n_with_biomarkers,
        "clinical_keys_in_lookup": len(clinical_lookup),
        "biomarker_keys_in_lookup": len(biomarker_lookup),
        "labels_field_semantic": (
            "Binary DRIL biomarker flag (one of the 16 biomarker columns), "
            "NOT multi-class DR severity. Samples with has_biomarkers=False "
            "have label=0 by default; downstream classification losses must "
            "mask by has_biomarkers before computing loss. class_distribution "
            "above only reflects samples where has_biomarkers=True, plus an "
            "'unlabelled' bucket for the rest."
        ),
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

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
