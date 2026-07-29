"""Green-channel + histogram-match harmonisation for the Variant C / Design C1
cross-modality ablation.

Design C1 hypothesis
--------------------
Zero-shot cross-modality DR grading collapses (96% grade 0 on OLIVES), and the
confident "severe" tail is an image-brightness artifact. This transform tests
whether making the EyePACS **colour** source look like the OLIVES **near-IR**
target mitigates the collapse: extract the green channel (most feature-rich for
retinal lesions/vessels), remap its intensity distribution to the OLIVES near-IR
reference via histogram matching, then replicate the single channel into three
identical channels so the 3-channel ResNet input still fits.

Only the source (EyePACS) is transformed. OLIVES stays raw at train and eval
time — the whole point is "make the source look like the target".

LABEL-PRESERVATION CONTRACT (hard requirement)
----------------------------------------------
``GreenHistMatch`` is an **image-only** transform. It receives a single image
tensor and returns a single image tensor. It never sees, receives, indexes, or
returns a label. The DR grade travels through the dataset's own immutable label
vector, completely independent of pixel values, so histogram matching cannot
reorder, drop, or relabel any sample. See ``notebooks/greyscale_experiment.ipynb``
for the element-wise + per-grade-count verification of this contract.

Value-range contract
--------------------
The saved EyePACS/OLIVES tensors are already ImageNet-normalised (values centred
near 0). Histogram matching is only meaningful in raw image space, and the
requirement is that harmonisation happens on the raw image *before* ImageNet
normalisation. This transform therefore mirrors ``src.data.transforms``:
de-normalise -> operate in [0, 1] -> re-normalise, using the same ImageNet
stats. The OLIVES reference histogram (see ``scripts/compute_olives_histogram.py``)
is computed in that same de-normalised [0, 1] space, so source and target live
in one comparable range.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch

TransformT = Callable[[torch.Tensor], torch.Tensor]

# Keys written by scripts/compute_olives_histogram.py into the reference .npy.
_REQUIRED_REF_KEYS = ("bin_centers", "cdf", "value_range")


def load_reference(reference_hist_path: str | Path) -> dict:
    """Load and validate the saved OLIVES reference histogram dict.

    The ``.npy`` holds a pickled dict with ``bin_centers`` (K,), ``cdf`` (K,)
    and ``value_range`` (2,) — the target intensity CDF in [0, 1] space.
    """
    path = Path(reference_hist_path)
    if not path.exists():
        raise FileNotFoundError(
            f"OLIVES reference histogram not found at {path}. "
            "Run scripts/compute_olives_histogram.py first."
        )
    ref = np.load(path, allow_pickle=True).item()
    missing = [k for k in _REQUIRED_REF_KEYS if k not in ref]
    if missing:
        raise KeyError(
            f"Reference histogram {path} is missing keys {missing}; "
            f"found {sorted(ref.keys())}."
        )
    return ref


def match_to_reference(
    gray: np.ndarray, ref_centers: np.ndarray, ref_cdf: np.ndarray
) -> np.ndarray:
    """CDF-based histogram matching of a single-channel image to a reference.

    Equivalent to ``skimage.exposure.match_histograms`` but against a fixed,
    pre-computed target CDF rather than a reference image: each source pixel is
    mapped to its empirical (rank-based) quantile, then to the target intensity
    at that quantile via the inverse of the reference CDF.
    """
    flat = gray.reshape(-1).astype(np.float64)
    n = flat.shape[0]
    order = np.argsort(flat, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    src_quantiles = (ranks + 0.5) / n
    matched = np.interp(src_quantiles, ref_cdf, ref_centers)
    return matched.reshape(gray.shape).astype(np.float32)


def _stat_tensor(values: object) -> torch.Tensor:
    """Turn a 3-element mean/std sequence into a broadcastable [3, 1, 1]."""
    return torch.as_tensor(list(values), dtype=torch.float32).view(-1, 1, 1)


class GreenHistMatch:
    """Green-channel + OLIVES-histogram-match + 3-channel replicate (image only).

    Slots into the existing EyePACS transform pipeline *before* ImageNet
    normalisation, gated by the ``harmonise`` config flag. When the flag is
    absent/false this class is never constructed and the colour path is
    unchanged.

    Parameters
    ----------
    reference_hist_path:
        Path to the OLIVES reference histogram ``.npy`` (near-IR target CDF).
    imagenet_mean, imagenet_std:
        The same ImageNet stats the loader normalised with, used to
        de-normalise into [0, 1] and re-normalise back. Sourced from config
        (``configs/data.yaml``), never hardcoded.
    source_channel:
        Index of the channel to extract from the colour source (1 = green).
    assume_normalised:
        If True (default), the incoming tensor is ImageNet-normalised (the
        dataset/loader contract) and is de-normalised to [0, 1] before matching
        and re-normalised after. If False, the tensor is assumed to already be
        in [0, 1] and is returned in [0, 1].

    This transform receives and returns **only** an image tensor — never a label.
    """

    def __init__(
        self,
        reference_hist_path: str | Path,
        imagenet_mean: object,
        imagenet_std: object,
        source_channel: int = 1,
        assume_normalised: bool = True,
    ) -> None:
        ref = load_reference(reference_hist_path)
        self.ref_centers = np.asarray(ref["bin_centers"], dtype=np.float64)
        self.ref_cdf = np.asarray(ref["cdf"], dtype=np.float64)
        self.reference_hist_path = str(reference_hist_path)
        self.source_channel = int(source_channel)
        self.assume_normalised = bool(assume_normalised)
        self.mean = _stat_tensor(imagenet_mean)
        self.std = _stat_tensor(imagenet_std)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Transform one image tensor [3, H, W]; return one image tensor [3, H, W]."""
        if not torch.is_tensor(image):
            raise TypeError(
                f"GreenHistMatch expects a tensor image, got {type(image).__name__}."
            )
        if image.dim() != 3 or image.shape[0] != 3:
            raise ValueError(
                f"GreenHistMatch expects a [3, H, W] tensor, got {tuple(image.shape)}."
            )

        x = image.float()
        if self.assume_normalised:
            x01 = (x * self.std + self.mean).clamp(0.0, 1.0)  # de-normalise to [0, 1]
        else:
            x01 = x.clamp(0.0, 1.0)

        green = x01[self.source_channel].cpu().numpy()  # [H, W] in [0, 1]
        matched = match_to_reference(green, self.ref_centers, self.ref_cdf)
        matched_t = torch.from_numpy(matched).to(image.device).clamp(0.0, 1.0)
        three = matched_t.unsqueeze(0).repeat(3, 1, 1)  # replicate to 3 identical channels

        if self.assume_normalised:
            return (three - self.mean.to(three.device)) / self.std.to(three.device)
        return three


class _Compose:
    """Apply ``first`` then ``second`` — a minimal, tensor-only Compose."""

    def __init__(self, first: TransformT, second: TransformT) -> None:
        self.first = first
        self.second = second

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return self.second(self.first(image))


def build_harmonised_transform(
    existing_transform: TransformT | None,
    reference_hist_path: str | Path,
    imagenet_mean: object,
    imagenet_std: object,
    source_channel: int = 1,
) -> TransformT:
    """Compose ``GreenHistMatch`` *before* the dataset's existing transform.

    Harmonisation runs first (green + histogram-match + 3ch replicate on the raw
    de-normalised image), then the existing normalised->normalised augmentation
    (if any). Both operate in normalised space at their boundaries, so the
    composite is a drop-in replacement for the EyePACS transform.
    """
    harmonise = GreenHistMatch(
        reference_hist_path,
        imagenet_mean=imagenet_mean,
        imagenet_std=imagenet_std,
        source_channel=source_channel,
        assume_normalised=True,
    )
    if existing_transform is None:
        return harmonise
    return _Compose(harmonise, existing_transform)


def _eyepacs_subdatasets(loader: object) -> list:
    """Return the EyePACS sub-datasets inside a loader's dataset.

    The train loader wraps ``ConcatDataset([eyepacs_ds, olives_ds])`` and the
    val/test loaders wrap ``ConcatDataset([eyepacs_ds, olives_ds])`` too. EyePACS
    is identified structurally by exposing the ``labels`` attribute
    (``EyePACSDataset``), so OLIVES is never touched.
    """
    dataset = getattr(loader, "dataset", None)
    members = getattr(dataset, "datasets", [dataset])
    return [d for d in members if hasattr(d, "labels") and hasattr(d, "transform")]


def harmonise_eyepacs_in_bundle(
    bundle: dict,
    reference_hist_path: str | Path,
    imagenet_mean: object,
    imagenet_std: object,
    source_channel: int = 1,
    which: tuple[str, ...] = ("train_loader", "val_loader", "test_loader"),
) -> int:
    """Wrap EyePACS-only transforms in a built dataloader bundle (in place).

    Non-invasive hook for the Design C1 run: after ``build_dataloaders(cfg)``,
    this reaches into each requested loader, finds the EyePACS sub-dataset(s),
    and replaces their ``.transform`` with ``GreenHistMatch``-then-existing.
    OLIVES sub-datasets are left as-is (raw near-IR). Returns the number of
    EyePACS datasets harmonised. The completed pipeline never calls this.
    """
    n_wrapped = 0
    for name in which:
        loader = bundle.get(name)
        if loader is None:
            continue
        for ds in _eyepacs_subdatasets(loader):
            ds.transform = build_harmonised_transform(
                ds.transform,
                reference_hist_path,
                imagenet_mean=imagenet_mean,
                imagenet_std=imagenet_std,
                source_channel=source_channel,
            )
            n_wrapped += 1
    return n_wrapped
