import zipfile
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

# ── USER CONFIG ── Update these paths before running ──────────────────
ZIP_FILES = [
    "/content/drive/MyDrive/EYEPACS.zip",
    "/content/drive/MyDrive/olive.zip",
    "/content/drive/MyDrive/OLIVES_Dataset_Labels.zip",
]
# ──────────────────────────────────────────────────────────────────────

BYTES_PER_GB = 1024**3
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
PREVIEW_COUNT = 20
DIVIDER = "=" * 72


def _get_top_level_entries(names: Iterable[str]) -> list[str]:
    top: set[str] = set()
    for name in names:
        parts = name.split("/", 1)
        if len(parts) > 1 and parts[0]:
            top.add(parts[0] + "/")
        elif parts[0]:
            top.add("<root-file>")
    return sorted(top)


def _count_extensions(names: Iterable[str]) -> Counter:
    return Counter(Path(n).suffix.lower() or "<no-ext>" for n in names)


def _try_read_image(
    zf: zipfile.ZipFile, names: Iterable[str]
) -> tuple[str, tuple[int, ...], str] | None:
    for name in names:
        if Path(name).suffix.lower() not in IMAGE_EXTS:
            continue
        try:
            with zf.open(name) as fh:
                data = fh.read()
            img = Image.open(BytesIO(data))
            img.load()
            arr = np.asarray(img)
            return name, tuple(arr.shape), str(arr.dtype)
        except Exception:
            continue
    return None


def inspect_zip(zip_path: str) -> dict[str, object]:
    path = Path(zip_path)
    print(f"ZIP: {path.name}")

    if not path.exists():
        print(f"  ERROR: file not found at {path}")
        return {"name": path.name, "size_gb": 0.0, "total": 0, "images": 0, "other": 0}

    size_gb = path.stat().st_size / BYTES_PER_GB
    print(f"Size: {size_gb:.2f} GB")

    if not zipfile.is_zipfile(path):
        print("  ERROR: not a valid zip archive")
        return {"name": path.name, "size_gb": size_gb, "total": 0, "images": 0, "other": 0}

    with zipfile.ZipFile(path) as zf:
        all_names = zf.namelist()
        file_names = [n for n in all_names if not n.endswith("/")]
        total = len(file_names)
        print(f"Total files inside: {total}")

        print("Top-level entries:")
        for entry in _get_top_level_entries(all_names):
            print(f"  {entry}")

        print(f"First {PREVIEW_COUNT} filenames:")
        for name in file_names[:PREVIEW_COUNT]:
            print(f"  {name}")

        ext_counts = _count_extensions(file_names)
        print("File extensions:")
        for ext, count in ext_counts.most_common():
            print(f"  {count} {ext}")

        sample = _try_read_image(zf, file_names)
        if sample is not None:
            name, shape, dtype = sample
            print(f"Sample image: {name}")
            print(f"  shape={shape}, dtype={dtype}")
        else:
            print("Sample image: no readable image found")

    image_count = sum(c for e, c in ext_counts.items() if e in IMAGE_EXTS)
    other_count = total - image_count
    return {
        "name": path.name,
        "size_gb": size_gb,
        "total": total,
        "images": image_count,
        "other": other_count,
    }


def print_summary(rows: list[dict[str, object]]) -> None:
    print(DIVIDER)
    print("SUMMARY")
    print(DIVIDER)
    header = f"{'Zip name':<36} {'Size (GB)':>10} {'Total':>8} {'Images':>8} {'Other':>8}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{str(row['name']):<36} "
            f"{float(row['size_gb']):>10.2f} "
            f"{int(row['total']):>8} "
            f"{int(row['images']):>8} "
            f"{int(row['other']):>8}"
        )


def main(zip_files: list[str] | None = None) -> list[dict[str, object]]:
    targets = zip_files if zip_files is not None else ZIP_FILES
    rows: list[dict[str, object]] = []
    for zip_path in targets:
        print(DIVIDER)
        rows.append(inspect_zip(zip_path))
    print_summary(rows)
    return rows


if __name__ == "__main__":
    main()
