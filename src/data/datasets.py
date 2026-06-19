"""Dataset classes for EyePACS (lazy, mmap'd shards) and OLIVES (in RAM).

Both emit per-sample dicts matching the Phase 2b batch contract; the custom
``collate_fn`` (see dataloaders.py) assembles them into batches.
"""

import shutil
from pathlib import Path
from typing import Callable, Optional, Sequence

import torch
from torch.utils.data import Dataset

# The 9 usable biomarker columns (0-based) within the 16-wide OLIVES tensor.
# Defined once here and imported everywhere else (single source of truth).
USABLE_BIOMARKER_INDICES = [1, 3, 5, 6, 7, 8, 10, 11, 12]

EYEPACS_SHARD_GLOB = "eyepacs_shard_*.pt"

TransformT = Callable[[torch.Tensor], object]


def _load_shard(path: Path) -> dict:
    return torch.load(path, map_location="cpu", mmap=True, weights_only=False)


class EyePACSDataset(Dataset):
    """Lazy EyePACS dataset.

    Shards are copied once to a local cache and accessed via memory-mapped
    ``torch.load`` so the full 35k-image set never sits in RAM. Only the small
    label vector is materialised; image slices are read on demand.
    """

    def __init__(
        self,
        eyepacs_dir: str,
        eyepacs_cache: str,
        allowed_indices: Optional[Sequence[int]] = None,
        transform: Optional[TransformT] = None,
    ) -> None:
        self.transform = transform
        self._shard_paths = self._ensure_cache(Path(eyepacs_dir), Path(eyepacs_cache))

        # Build the global index map and materialise the (tiny) labels, without
        # pulling any image tensors into RAM.
        self._index: list[tuple[int, int]] = []
        labels: list[torch.Tensor] = []
        for shard_idx, shard_path in enumerate(self._shard_paths):
            shard = _load_shard(shard_path)
            n = int(shard["labels"].shape[0])
            self._index.extend((shard_idx, local) for local in range(n))
            labels.append(shard["labels"].clone())
            del shard
        self._labels = torch.cat(labels).long()
        self.total = len(self._index)

        if allowed_indices is None:
            self._allowed = list(range(self.total))
        else:
            self._allowed = list(allowed_indices)

        # Opened lazily (per-process, so DataLoader workers each map their own).
        self._handles: dict[int, dict] = {}

    @staticmethod
    def _ensure_cache(eyepacs_dir: Path, cache_dir: Path) -> list[Path]:
        cache_dir.mkdir(parents=True, exist_ok=True)
        source_shards = sorted(eyepacs_dir.glob(EYEPACS_SHARD_GLOB))
        if not source_shards:
            raise FileNotFoundError(
                f"No EyePACS shards matching {EYEPACS_SHARD_GLOB} under {eyepacs_dir}"
            )
        copied = 0
        cached: list[Path] = []
        for src in source_shards:
            dst = cache_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
                copied += 1
            cached.append(dst)
        if copied:
            print(f"[eyepacs-cache] copied {copied}/{len(source_shards)} shards to {cache_dir}")
        else:
            print(f"[eyepacs-cache] all {len(source_shards)} shards present in {cache_dir} — skip")
        return cached

    def _handle(self, shard_idx: int) -> dict:
        handle = self._handles.get(shard_idx)
        if handle is None:
            handle = _load_shard(self._shard_paths[shard_idx])
            self._handles[shard_idx] = handle
        return handle

    @property
    def labels(self) -> torch.Tensor:
        """Global DR severity labels (length = total samples)."""
        return self._labels

    def __len__(self) -> int:
        return len(self._allowed)

    def __getitem__(self, i: int) -> dict:
        global_idx = self._allowed[i]
        shard_idx, local_idx = self._index[global_idx]
        image = self._handle(shard_idx)["images"][local_idx].clone()
        if self.transform is not None:
            image = self.transform(image)
        return {
            "images": image,
            "dataset": 0,
            "dr_labels": int(self._labels[global_idx]),
            "biomarkers": torch.zeros(9, dtype=torch.float32),
            "has_biomarkers": False,
            "bcva": float("nan"),
            "cst": float("nan"),
            "has_clinical": False,
            "patient_ids": -1,
        }


class OLIVESDataset(Dataset):
    """OLIVES dataset, fully resident in RAM.

    Selects the 9 usable biomarker columns and standardises BCVA/CST with
    passed-in stats (NaN stays NaN). Accepts either a path to the ``.pt`` file
    or an already-loaded dict (so the file is read at most once per process).
    """

    def __init__(
        self,
        olives: object,
        allowed_indices: Sequence[int],
        transform: Optional[TransformT],
        stats: dict,
    ) -> None:
        self.transform = transform
        data = self._resolve(olives)

        self.images: torch.Tensor = data["images"]
        self.has_biomarkers: torch.Tensor = data["has_biomarkers"].bool()
        self.has_clinical: torch.Tensor = data["has_clinical"].bool()
        self.patient_ids: torch.Tensor = data["patient_ids"].long()

        # Select the 9 usable biomarker columns; zero the absent (NaN) rows.
        bio9 = data["biomarkers"][:, USABLE_BIOMARKER_INDICES].float().clone()
        bio9[~self.has_biomarkers] = 0.0
        self.biomarkers = bio9

        # Standardise clinical targets; NaN propagates through and stays NaN.
        self.bcva = (data["bcva"].float() - float(stats["bcva_mean"])) / float(
            stats["bcva_std"]
        )
        self.cst = (data["cst"].float() - float(stats["cst_mean"])) / float(
            stats["cst_std"]
        )

        self._allowed = list(allowed_indices)

    @staticmethod
    def _resolve(olives: object) -> dict:
        if isinstance(olives, (str, Path)):
            return torch.load(
                str(olives), map_location="cpu", weights_only=False
            )
        if isinstance(olives, dict):
            return olives
        raise TypeError(
            f"olives must be a path or a loaded dict, got {type(olives).__name__}"
        )

    def __len__(self) -> int:
        return len(self._allowed)

    def __getitem__(self, i: int) -> dict:
        idx = self._allowed[i]
        image = self.images[idx]
        if self.transform is not None:
            image = self.transform(image)
        return {
            "images": image,
            "dataset": 1,
            "dr_labels": -1,
            "biomarkers": self.biomarkers[idx],
            "has_biomarkers": bool(self.has_biomarkers[idx]),
            "bcva": float(self.bcva[idx]),
            "cst": float(self.cst[idx]),
            "has_clinical": bool(self.has_clinical[idx]),
            "patient_ids": int(self.patient_ids[idx]),
        }
