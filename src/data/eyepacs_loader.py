from collections import Counter

from datasets import Dataset, DatasetDict, load_from_disk
from omegaconf import DictConfig

LABEL_COLUMN = "label"


def _print_class_distribution(dataset: Dataset) -> None:
    if LABEL_COLUMN not in dataset.column_names:
        print(f"Column '{LABEL_COLUMN}' not found; skipping class distribution.")
        return

    counts = Counter(dataset[LABEL_COLUMN])
    total = sum(counts.values())
    print(f"Class distribution ({total} rows):")
    for label, count in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
        pct = 100.0 * count / total if total else 0.0
        print(f"  {label:<40} {count:>7}  ({pct:5.2f}%)")


def load_eyepacs(paths_config: DictConfig) -> Dataset | DatasetDict:
    dataset_path = str(paths_config.eyepacs_hf_dir)
    print(f"Loading EyePACS from {dataset_path}")
    dataset = load_from_disk(dataset_path)

    if isinstance(dataset, DatasetDict):
        print(f"Splits: {list(dataset.keys())}")
        for split_name, split in dataset.items():
            print(f"\n[{split_name}] {len(split)} rows, columns: {split.column_names}")
            _print_class_distribution(split)
    else:
        print(f"{len(dataset)} rows, columns: {dataset.column_names}")
        _print_class_distribution(dataset)

    return dataset
