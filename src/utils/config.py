from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_config(yaml_path: str | Path) -> DictConfig:
    cfg = OmegaConf.load(str(yaml_path))
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Expected a mapping at top level of {yaml_path}, got {type(cfg).__name__}")
    return cfg


def ensure_dirs(paths_config: DictConfig) -> None:
    for _, value in paths_config.items():
        if isinstance(value, str):
            Path(value).mkdir(parents=True, exist_ok=True)
