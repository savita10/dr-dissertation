import shutil
import subprocess
import sys
from pathlib import Path

from omegaconf import DictConfig

from src.utils.config import ensure_dirs, load_config

PATHS_CONFIG = Path("configs/paths.yaml")
REQUIREMENTS_FILE = Path("requirements.txt")
DRIVE_MOUNT_POINT = "/content/drive"
BYTES_PER_GB = 1024**3


def _mount_drive(mount_point: str) -> None:
    from google.colab import drive

    drive.mount(mount_point, force_remount=False)


def _install_requirements(requirements_file: Path) -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements_file)],
        check=True,
    )


def _print_environment_summary() -> None:
    import torch

    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_index)
        total_mem_gb = torch.cuda.get_device_properties(device_index).total_memory / BYTES_PER_GB
        print(f"GPU: {device_name} ({total_mem_gb:.1f} GB)")
    else:
        print("GPU: not available")

    try:
        import psutil

        ram_gb = psutil.virtual_memory().total / BYTES_PER_GB
        print(f"RAM: {ram_gb:.1f} GB total")
    except ImportError:
        print("RAM: psutil not installed")

    disk = shutil.disk_usage("/content")
    print(
        f"Disk (/content): {disk.free / BYTES_PER_GB:.1f} GB free / "
        f"{disk.total / BYTES_PER_GB:.1f} GB total"
    )


def setup_colab(
    paths_config: Path = PATHS_CONFIG,
    requirements_file: Path = REQUIREMENTS_FILE,
    mount_point: str = DRIVE_MOUNT_POINT,
) -> DictConfig:
    _mount_drive(mount_point)
    paths = load_config(paths_config)
    ensure_dirs(paths)
    _install_requirements(requirements_file)
    _print_environment_summary()
    return paths
