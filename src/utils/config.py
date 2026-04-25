from pathlib import Path

from omegaconf import DictConfig, OmegaConf

# paths.yaml mixes directory paths (must be created) with file paths
# (e.g. eyepacs_zip, olives_labels_csv). Calling mkdir on a file path
# raises FileExistsError if the file exists, or silently creates a
# directory with the file's name if it does not — both broken. We
# detect file paths by extension and skip them.
FILE_EXTENSIONS = frozenset(
    {".zip", ".csv", ".xlsx", ".json", ".pt", ".npy", ".tif", ".jpeg", ".png"}
)


def load_config(yaml_path: str | Path) -> DictConfig:
    cfg = OmegaConf.load(str(yaml_path))
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Expected a mapping at top level of {yaml_path}, got {type(cfg).__name__}")
    return cfg


def _is_file_path(value: str) -> bool:
    return Path(value).suffix.lower() in FILE_EXTENSIONS


def ensure_dirs(paths_config: DictConfig) -> None:
    for _, value in paths_config.items():
        if not isinstance(value, str):
            continue
        if _is_file_path(value):
            continue
        try:
            Path(value).mkdir(parents=True, exist_ok=True)
        except (FileExistsError, NotADirectoryError):
            continue
