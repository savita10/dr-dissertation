import zipfile
from io import BytesIO
from pathlib import Path
from typing import Iterator

import pandas as pd
from omegaconf import DictConfig
from tqdm.auto import tqdm

INNER_OLIVES_ZIP_NAME = "OLIVES.zip"
TRIAL_ZIP_NAMES = ("TREX_DME.zip", "Prime_FULL.zip")
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


def _find_inner_zip(zf: zipfile.ZipFile, expected_name: str) -> str:
    target = expected_name.lower()
    for name in zf.namelist():
        if name.endswith("/"):
            continue
        if Path(name).name.lower() == target:
            return name
    raise FileNotFoundError(f"{expected_name} not found inside {zf.filename}")


def _open_inner_zip(outer: zipfile.ZipFile, inner_name: str) -> zipfile.ZipFile:
    with outer.open(inner_name) as fh:
        data = fh.read()
    return zipfile.ZipFile(BytesIO(data))


def iter_trial_zips(olive_zip_path: str) -> Iterator[tuple[str, zipfile.ZipFile]]:
    with zipfile.ZipFile(olive_zip_path) as outer:
        inner_olives_name = _find_inner_zip(outer, INNER_OLIVES_ZIP_NAME)
        olives_zf = _open_inner_zip(outer, inner_olives_name)
        try:
            for trial_zip_name in TRIAL_ZIP_NAMES:
                trial_member = _find_inner_zip(olives_zf, trial_zip_name)
                trial_zf = _open_inner_zip(olives_zf, trial_member)
                try:
                    yield trial_zip_name, trial_zf
                finally:
                    trial_zf.close()
        finally:
            olives_zf.close()


def load_olives_labels(paths_config: DictConfig) -> pd.DataFrame:
    csv_path = str(paths_config.olives_labels_csv)
    print(f"Loading OLIVES labels from {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"Shape: {df.shape}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")

    print("\nBiomarker positive-rate per column:")
    for col in BIOMARKER_COLUMNS:
        if col not in df.columns:
            continue
        positives = int(df[col].sum())
        total = int(df[col].count())
        pct = 100.0 * positives / total if total else 0.0
        print(f"  {col:<45} {positives:>6} / {total:<6} ({pct:5.2f}%)")

    if "Patient_ID" in df.columns:
        print(f"\nUnique patients: {df['Patient_ID'].nunique()}")
    if "Eye_ID" in df.columns:
        print(f"Unique eyes: {df['Eye_ID'].nunique()}")

    return df


def get_fundus_paths_from_zip(paths_config: DictConfig) -> list[str]:
    olive_zip_path = str(paths_config.olive_zip)
    print(f"Scanning {olive_zip_path} for fundus images")

    fundus_paths: list[str] = []
    per_trial_counts: dict[str, int] = {}

    for trial_name, trial_zf in iter_trial_zips(olive_zip_path):
        count = 0
        for member in trial_zf.namelist():
            if member.endswith("/"):
                continue
            if FUNDUS_TOKEN in Path(member).name.lower():
                fundus_paths.append(member)
                count += 1
        per_trial_counts[trial_name] = count

    for trial_name, count in per_trial_counts.items():
        print(f"  {trial_name}: {count} fundus images")
    print(f"  total: {len(fundus_paths)} fundus images")

    return fundus_paths


def extract_fundus_images(paths_config: DictConfig) -> str:
    olive_zip_path = str(paths_config.olive_zip)
    extracted_root = Path(str(paths_config.olives_extracted_dir))
    fundus_dir = extracted_root / "fundus"
    fundus_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting fundus images to {fundus_dir}")

    extracted = 0
    skipped = 0

    for trial_name, trial_zf in iter_trial_zips(olive_zip_path):
        members = [
            m
            for m in trial_zf.namelist()
            if not m.endswith("/") and FUNDUS_TOKEN in Path(m).name.lower()
        ]
        for member in tqdm(members, desc=f"Extracting {trial_name}"):
            target_path = fundus_dir / member
            if target_path.exists() and target_path.stat().st_size > 0:
                skipped += 1
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with trial_zf.open(member) as src, open(target_path, "wb") as dst:
                dst.write(src.read())
            extracted += 1

    print(f"Extracted {extracted} new files; skipped {skipped} already-present files.")
    return str(fundus_dir)


def join_fundus_to_labels(fundus_dir: str, labels_df: pd.DataFrame) -> pd.DataFrame:
    fundus_root = Path(fundus_dir)

    local_files = [p for p in fundus_root.rglob("*") if p.is_file()]
    print(f"Found {len(local_files)} files under {fundus_root}")

    local_by_rel: dict[str, str] = {}
    local_by_basename: dict[str, list[str]] = {}
    for path in local_files:
        rel = path.relative_to(fundus_root).as_posix()
        local_by_rel[rel] = str(path)
        local_by_basename.setdefault(path.name, []).append(str(path))

    if PATH_COLUMN not in labels_df.columns:
        raise KeyError(f"'{PATH_COLUMN}' column missing from labels dataframe")

    keep_cols = [
        c
        for c in ("Patient_ID", "Eye_ID", "BCVA", "CST", *BIOMARKER_COLUMNS)
        if c in labels_df.columns
    ]

    matched_rows: list[dict] = []
    unmatched = 0
    for _, row in labels_df.iterrows():
        csv_path = str(row[PATH_COLUMN]).lstrip("/").replace("\\", "/")
        local_path = local_by_rel.get(csv_path)
        if local_path is None:
            candidates = local_by_basename.get(Path(csv_path).name, [])
            local_path = next(
                (c for c in candidates if c.replace("\\", "/").endswith(csv_path)),
                None,
            )
        if local_path is None:
            unmatched += 1
            continue
        record = {"local_path": local_path}
        for col in keep_cols:
            record[col] = row[col]
        matched_rows.append(record)

    matched_df = pd.DataFrame(matched_rows)
    total = len(labels_df)
    matched = len(matched_df)
    print(f"Matched {matched} / {total} label rows to local files ({unmatched} unmatched)")

    return matched_df
