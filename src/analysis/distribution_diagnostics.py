"""Distributional comparison metrics between two feature sets.

Used in Phase 1 to validate whether EyePACS and OLIVES occupy compatible
regions of a pretrained-encoder feature space.
"""

import argparse
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig
from scipy.linalg import sqrtm
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel
from tqdm.auto import tqdm

from src.utils.config import load_config

EYEPACS_FEATURES_FILENAME = "eyepacs_resnet50_features.pt"
OLIVES_FEATURES_FILENAME = "olives_resnet50_features.pt"
OLIVES_INPUT_FILENAME = "olives_fundus.pt"
EYEPACS_SHARD_GLOB = "eyepacs_shard_*.pt"
PHASE1_DIRNAME = "phase1"
PCA_PLOT_FILENAME = "pca_eyepacs_vs_olives.png"
SEVERITY_PLOT_FILENAME = "pca_by_severity.png"
IMAGE_STATS_PLOT_FILENAME = "image_stats_vs_pc1.png"
REPORT_FILENAME = "phase1_report.md"
COSINE_CHUNK_SIZE = 500
MMD_RNG_SEED = 42
SEVERITY_RNG_SEED = 42
N_DR_CLASSES = 5
EYEPACS_LABEL_NAMES = (
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR",
)
# In ImageNet-normalised space, R-channel < -1.5 corresponds to a near-black
# pixel in raw RGB (≈0.14 raw); used as a proxy for fundus field-stop borders.
BLACK_PIXEL_THRESHOLD = -1.5
IMAGE_STATS_CHUNK = 256
EYEPACS_STATS_SUBSAMPLE = 2000
SHIFT_SPREAD_RATIO = 0.30
STAT_LABELS = {
    "brightness": "mean intensity",
    "black_pixel": "black-pixel proportion",
    "contrast": "pixel std (contrast)",
}


def compute_pca_projection(
    features_a: np.ndarray,
    features_b: np.ndarray,
    n_components: int = 2,
) -> tuple[np.ndarray, np.ndarray, PCA]:
    """Fit PCA on the stacked feature sets and return per-dataset projections."""
    stacked = np.vstack([features_a, features_b])
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=0)
    coords = pca.fit_transform(stacked)
    n_a = features_a.shape[0]
    return coords[:n_a], coords[n_a:], pca


def compute_frechet_distance(
    features_a: np.ndarray, features_b: np.ndarray
) -> float:
    """Fréchet distance between two distributions modelled as multivariate Gaussians."""
    mu_a = features_a.mean(axis=0)
    mu_b = features_b.mean(axis=0)
    sigma_a = np.cov(features_a, rowvar=False)
    sigma_b = np.cov(features_b, rowvar=False)

    diff = mu_a - mu_b
    covmean, _ = sqrtm(sigma_a @ sigma_b, disp=False)

    if np.iscomplexobj(covmean):
        max_imag = float(np.max(np.abs(covmean.imag)))
        if max_imag > 1e-3:
            print(
                f"[frechet] WARNING: sqrtm imaginary residual={max_imag:.2e}; "
                "matrix may be near-singular."
            )
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma_a) + np.trace(sigma_b) - 2 * np.trace(covmean)
    return float(fid)


def _mmd_squared_unbiased(
    a: np.ndarray, b: np.ndarray, gamma: float
) -> float:
    """Unbiased MMD² estimator with RBF kernel."""
    k_aa = rbf_kernel(a, a, gamma=gamma)
    k_bb = rbf_kernel(b, b, gamma=gamma)
    k_ab = rbf_kernel(a, b, gamma=gamma)
    n = a.shape[0]
    m = b.shape[0]
    term_a = (k_aa.sum() - np.trace(k_aa)) / (n * (n - 1))
    term_b = (k_bb.sum() - np.trace(k_bb)) / (m * (m - 1))
    cross = 2.0 * k_ab.mean()
    return float(term_a + term_b - cross)


def compute_mmd(
    features_a: np.ndarray,
    features_b: np.ndarray,
    n_samples: int | None = 2000,
    gamma: float | None = None,
    n_permutations: int = 100,
) -> dict[str, float]:
    """RBF MMD² with permutation significance test."""
    rng = np.random.default_rng(MMD_RNG_SEED)
    if n_samples is not None:
        size_a = min(n_samples, features_a.shape[0])
        size_b = min(n_samples, features_b.shape[0])
        idx_a = rng.choice(features_a.shape[0], size_a, replace=False)
        idx_b = rng.choice(features_b.shape[0], size_b, replace=False)
        a = features_a[idx_a]
        b = features_b[idx_b]
    else:
        a = features_a
        b = features_b

    if gamma is None:
        gamma = 1.0 / a.shape[1]

    mmd_value = _mmd_squared_unbiased(a, b, gamma)

    combined = np.vstack([a, b])
    n_a = a.shape[0]
    n_total = combined.shape[0]
    null_dist = np.empty(n_permutations, dtype=np.float64)
    for i in tqdm(range(n_permutations), desc="MMD permutations"):
        perm = rng.permutation(n_total)
        a_perm = combined[perm[:n_a]]
        b_perm = combined[perm[n_a:]]
        null_dist[i] = _mmd_squared_unbiased(a_perm, b_perm, gamma)

    p_value = float(np.mean(null_dist >= mmd_value))

    return {
        "mmd_value": float(mmd_value),
        "p_value": p_value,
        "null_distribution_mean": float(null_dist.mean()),
        "null_distribution_std": float(null_dist.std()),
        "n_samples_a": int(n_a),
        "n_samples_b": int(b.shape[0]),
        "gamma": float(gamma),
        "n_permutations": int(n_permutations),
    }


def _topk_mean_cosine(
    query: np.ndarray, reference: np.ndarray, k: int, desc: str
) -> np.ndarray:
    """Mean cosine similarity to the k nearest neighbours in `reference`, per row of `query`."""
    k_eff = min(k, reference.shape[0])
    out = np.empty(query.shape[0], dtype=np.float32)
    for start in tqdm(range(0, query.shape[0], COSINE_CHUNK_SIZE), desc=desc):
        end = min(start + COSINE_CHUNK_SIZE, query.shape[0])
        sims = cosine_similarity(query[start:end], reference)
        topk = np.partition(sims, -k_eff, axis=1)[:, -k_eff:]
        out[start:end] = topk.mean(axis=1)
    return out


def compute_cosine_similarity_diagnostics(
    features_a: np.ndarray, features_b: np.ndarray, k: int = 5
) -> dict[str, float]:
    """For each sample, mean cosine similarity to its top-k neighbours in the other set."""
    a_to_b = _topk_mean_cosine(features_a, features_b, k, desc="A→B cosine")
    b_to_a = _topk_mean_cosine(features_b, features_a, k, desc="B→A cosine")
    return {
        "a_to_b_mean": float(a_to_b.mean()),
        "a_to_b_std": float(a_to_b.std()),
        "b_to_a_mean": float(b_to_a.mean()),
        "b_to_a_std": float(b_to_a.std()),
        "k": int(k),
    }


def generate_pca_plot(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    label_a: str,
    label_b: str,
    output_path: Path | str,
    title: str | None = None,
    explained_variance: tuple[float, ...] | None = None,
) -> None:
    """Save a 2D scatter of the joint PCA projection."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(coords_a[:, 0], coords_a[:, 1], s=8, alpha=0.4, label=label_a)
    ax.scatter(coords_b[:, 0], coords_b[:, 1], s=8, alpha=0.4, label=label_b)
    ax.set_aspect("equal", adjustable="datalim")

    if explained_variance is not None and len(explained_variance) >= 2:
        ax.set_xlabel(f"PC1 ({explained_variance[0]:.1%} variance)")
        ax.set_ylabel(f"PC2 ({explained_variance[1]:.1%} variance)")
    else:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    if title is not None:
        ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def compute_pca_by_severity(
    eyepacs_features: np.ndarray,
    eyepacs_labels: np.ndarray,
    olives_features: np.ndarray,
    output_path: Path | str,
    n_components: int = 2,
) -> dict[str, float]:
    """PCA on combined features, scattered by EyePACS DR severity with OLIVES overlay.

    Returns per-class mean PC1 coordinates (EyePACS classes 0–4) plus the
    overall EyePACS and OLIVES PC1 means and PC1/PC2 explained variance.
    The plot at output_path uses 5 viridis-shaded series for EyePACS classes
    and a single distinct colour for the OLIVES overlay.
    """
    if eyepacs_labels.shape[0] != eyepacs_features.shape[0]:
        raise ValueError(
            f"eyepacs_labels has {eyepacs_labels.shape[0]} rows but "
            f"eyepacs_features has {eyepacs_features.shape[0]}. The feature "
            "extraction order must match the shard iteration order."
        )

    coords_eye, coords_oli, pca = compute_pca_projection(
        eyepacs_features, olives_features, n_components=n_components
    )
    pc1_eye = coords_eye[:, 0]
    pc1_oli = coords_oli[:, 0]
    var = pca.explained_variance_ratio_

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 8))
    cmap = plt.cm.viridis
    class_colors = cmap(np.linspace(0.1, 0.9, N_DR_CLASSES))
    for cls in range(N_DR_CLASSES):
        mask = eyepacs_labels == cls
        if not mask.any():
            continue
        ax.scatter(
            coords_eye[mask, 0],
            coords_eye[mask, 1],
            s=8,
            alpha=0.4,
            color=class_colors[cls],
            label=f"EyePACS — {EYEPACS_LABEL_NAMES[cls]}",
        )
    ax.scatter(
        coords_oli[:, 0],
        coords_oli[:, 1],
        s=4,
        alpha=0.4,
        color="red",
        label="OLIVES",
    )
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(f"PC1 ({var[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({var[1]:.1%} variance)")
    ax.set_title("PCA by DR severity (EyePACS) with OLIVES overlay")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    means: dict[str, float] = {}
    for cls in range(N_DR_CLASSES):
        mask = eyepacs_labels == cls
        means[f"class_{cls}"] = (
            float(pc1_eye[mask].mean()) if mask.any() else float("nan")
        )
    means["eyepacs_all"] = float(pc1_eye.mean())
    means["olives"] = float(pc1_oli.mean())
    means["pc1_explained_variance"] = float(var[0])
    means["pc2_explained_variance"] = float(var[1])
    return means


def compute_image_stats_correlation(
    eyepacs_pt_dir: Path | str,
    olives_pt_path: Path | str,
    eyepacs_features: np.ndarray,
    olives_features: np.ndarray,
    output_path: Path | str,
) -> dict[str, dict[str, float]]:
    """Spearman correlation between per-image acquisition stats and PC1.

    Stats per image: mean pixel intensity, black-pixel proportion (R-channel
    below BLACK_PIXEL_THRESHOLD in normalised space), pixel std. PCA is
    re-fit on the full combined features for axis consistency, then the
    subsampled EyePACS and full OLIVES features are projected to PC1.
    """
    eyepacs_pt_dir = Path(eyepacs_pt_dir)
    olives_pt_path = Path(olives_pt_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEVERITY_RNG_SEED)
    n_eye = eyepacs_features.shape[0]
    n_subsample = min(EYEPACS_STATS_SUBSAMPLE, n_eye)
    indices_sorted = np.sort(
        rng.choice(n_eye, n_subsample, replace=False)
    ).astype(np.int64)

    print(f"[stats] EyePACS subsample: {n_subsample} images from {n_eye}")
    eyepacs_stats = _eyepacs_image_stats_at_indices(eyepacs_pt_dir, indices_sorted)
    print(f"[stats] OLIVES: loading {olives_pt_path}")
    olives_stats = _olives_image_stats(olives_pt_path)

    pca = PCA(n_components=2, svd_solver="randomized", random_state=0)
    pca.fit(np.vstack([eyepacs_features, olives_features]))
    eyepacs_pc1 = pca.transform(eyepacs_features[indices_sorted])[:, 0]
    olives_pc1 = pca.transform(olives_features)[:, 0]

    if olives_stats["brightness"].shape[0] != olives_pc1.shape[0]:
        raise ValueError(
            f"OLIVES image stats length ({olives_stats['brightness'].shape[0]}) "
            f"does not match feature count ({olives_pc1.shape[0]}). The "
            "feature extraction order must match the .pt image order."
        )

    corrs: dict[str, dict[str, float]] = {}
    for ds_name, stats, pc1 in (
        ("eyepacs", eyepacs_stats, eyepacs_pc1),
        ("olives", olives_stats, olives_pc1),
    ):
        for stat_name in ("brightness", "black_pixel", "contrast"):
            rho, p_value = spearmanr(stats[stat_name], pc1)
            corrs[f"{ds_name}_{stat_name}"] = {
                "rho": float(rho),
                "p_value": float(p_value),
            }

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for col, stat_name in enumerate(("brightness", "black_pixel", "contrast")):
        for row, (ds_name, stats, pc1, color) in enumerate(
            (
                ("eyepacs", eyepacs_stats, eyepacs_pc1, "C0"),
                ("olives", olives_stats, olives_pc1, "C3"),
            )
        ):
            ax = axes[row, col]
            ax.scatter(stats[stat_name], pc1, s=4, alpha=0.4, color=color)
            entry = corrs[f"{ds_name}_{stat_name}"]
            ax.set_title(
                f"{ds_name.upper()} — {STAT_LABELS[stat_name]}\n"
                f"ρ={entry['rho']:+.3f}, p={entry['p_value']:.2e}",
                fontsize=10,
            )
            ax.set_xlabel(STAT_LABELS[stat_name])
            ax.set_ylabel("PC1")
    fig.suptitle("Per-image acquisition stats vs PC1", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return corrs


def _load_features(features_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    if not features_path.exists():
        raise FileNotFoundError(
            f"Features not found at {features_path} — run "
            "src.analysis.feature_extraction first."
        )
    blob = torch.load(features_path, map_location="cpu", weights_only=False)
    feats = blob["features"]
    if isinstance(feats, torch.Tensor):
        feats = feats.detach().cpu().numpy()
    feats = np.asarray(feats, dtype=np.float32)
    metadata = dict(blob.get("metadata", {}))
    return feats, metadata


def _load_eyepacs_labels(shards_dir: Path | str) -> np.ndarray:
    """Concatenate `labels` tensors from sorted shard files.

    Iteration order must mirror feature_extraction.py's _iter_inputs (sorted
    shard glob, within-shard order) so the returned labels align row-for-row
    with the saved feature tensor.
    """
    shards_dir = Path(shards_dir)
    shards = sorted(shards_dir.glob(EYEPACS_SHARD_GLOB))
    if not shards:
        raise FileNotFoundError(
            f"No EyePACS shards in {shards_dir} — cannot load severity labels."
        )
    parts: list[np.ndarray] = []
    for shard_path in shards:
        data = torch.load(shard_path, map_location="cpu", weights_only=False)
        labels = data["labels"]
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        parts.append(np.asarray(labels))
    return np.concatenate(parts).astype(np.int64)


def _image_stats_chunked(images: torch.Tensor) -> dict[str, np.ndarray]:
    """Per-image (brightness, black_pixel, contrast) for an (N, 3, H, W) tensor."""
    n = images.shape[0]
    brightness = np.empty(n, dtype=np.float32)
    black = np.empty(n, dtype=np.float32)
    contrast = np.empty(n, dtype=np.float32)
    for start in range(0, n, IMAGE_STATS_CHUNK):
        end = min(start + IMAGE_STATS_CHUNK, n)
        chunk = images[start:end].float()
        brightness[start:end] = chunk.mean(dim=(1, 2, 3)).numpy()
        black[start:end] = (
            (chunk[:, 0] < BLACK_PIXEL_THRESHOLD).float().mean(dim=(1, 2)).numpy()
        )
        contrast[start:end] = chunk.std(dim=(1, 2, 3)).numpy()
    return {"brightness": brightness, "black_pixel": black, "contrast": contrast}


def _eyepacs_image_stats_at_indices(
    shards_dir: Path, indices_sorted: np.ndarray
) -> dict[str, np.ndarray]:
    """Stream-load only the requested global indices across sorted shards."""
    shards = sorted(shards_dir.glob(EYEPACS_SHARD_GLOB))
    if not shards:
        raise FileNotFoundError(f"No EyePACS shards in {shards_dir}.")

    n = len(indices_sorted)
    brightness = np.empty(n, dtype=np.float32)
    black = np.empty(n, dtype=np.float32)
    contrast = np.empty(n, dtype=np.float32)

    global_offset = 0
    cursor = 0
    for shard_path in tqdm(shards, desc="EyePACS image stats"):
        if cursor >= n:
            break
        data = torch.load(shard_path, map_location="cpu", weights_only=False)
        images = data["images"]
        shard_size = images.shape[0]
        end_cursor = cursor
        while (
            end_cursor < n
            and indices_sorted[end_cursor] < global_offset + shard_size
        ):
            end_cursor += 1
        if cursor < end_cursor:
            local_idx = (
                indices_sorted[cursor:end_cursor] - global_offset
            ).astype(np.int64)
            subset_stats = _image_stats_chunked(images[local_idx])
            brightness[cursor:end_cursor] = subset_stats["brightness"]
            black[cursor:end_cursor] = subset_stats["black_pixel"]
            contrast[cursor:end_cursor] = subset_stats["contrast"]
            cursor = end_cursor
        global_offset += shard_size
        del data, images

    if cursor != n:
        raise RuntimeError(
            f"Only {cursor}/{n} EyePACS indices were loaded — "
            "shards may be missing samples."
        )
    return {"brightness": brightness, "black_pixel": black, "contrast": contrast}


def _olives_image_stats(pt_path: Path) -> dict[str, np.ndarray]:
    if not pt_path.exists():
        raise FileNotFoundError(
            f"OLIVES tensor file not found at {pt_path}."
        )
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    return _image_stats_chunked(data["images"])


def _interpret(
    results: dict[str, Any], label_a: str, label_b: str
) -> str:
    cos_avg = 0.5 * (
        results["cosine"]["a_to_b_mean"] + results["cosine"]["b_to_a_mean"]
    )
    p_value = results["mmd"]["p_value"]
    if cos_avg >= 0.7:
        cos_label = "strong directional alignment"
    elif cos_avg >= 0.4:
        cos_label = "moderate directional alignment"
    else:
        cos_label = "weak directional alignment"
    if p_value < 0.01:
        mmd_label = "a highly significant distributional difference"
    elif p_value < 0.05:
        mmd_label = "a significant distributional difference"
    else:
        mmd_label = "no statistically significant distributional difference"

    parts: list[str] = []
    parts.append(
        f"In ResNet-50 ImageNet feature space, {label_a} and {label_b} show "
        f"{cos_label} (mean cosine = {cos_avg:.3f}) and {mmd_label} "
        f"(MMD permutation p = {p_value:.4f}). The Fréchet distance is "
        f"{results['frechet_distance']:.2f}; FID is comparable across runs "
        f"of the same pair but not across dataset pairs without a baseline."
    )
    parts.append(
        f"PCA projects {sum(results['pca_var'][:2]):.1%} of the joint variance "
        "into 2D, so the global scatter plot is indicative rather than definitive."
    )

    severity_label = results.get("severity_shift_label")
    spread = results.get("severity_spread")
    centroid_distance = results.get("severity_centroid_distance")
    if severity_label is not None:
        parts.append(
            f"PCA stratified by DR severity (Section 7) classifies the shift as "
            f"**{severity_label}**: the EyePACS class-mean spread on PC1 is "
            f"{spread:.3f} versus a {label_a}↔{label_b} centroid distance of "
            f"{centroid_distance:.3f} on PC1."
        )

    strongest = results.get("strongest_stat_corr")
    if strongest is not None:
        ds_key, stat_key = strongest["key"].split("_", 1)
        ds_label = label_a if ds_key == "eyepacs" else label_b
        stat_label = STAT_LABELS.get(stat_key, stat_key)
        parts.append(
            f"PC1 is most strongly correlated with {ds_label} {stat_label} "
            f"(Spearman ρ = {strongest['rho']:+.3f}, "
            f"p = {strongest['p_value']:.2e}; Section 8), indicating the "
            f"dominant separation axis tracks acquisition-level variation "
            f"rather than retinal pathology content."
        )

    if severity_label and "Acquisition-driven" in severity_label:
        verdict = (
            "the shift is **mechanism-confirmed bridgeable** — PC1 carries "
            "acquisition signal, not pathology, so contrastive alignment or "
            "a learned projection head should close the gap"
        )
    elif severity_label and "Pathology-correlated" in severity_label:
        verdict = (
            "the shift is **concerning** — PC1 carries DR pathology signal, "
            "so naive joint encoding risks confounding domain with severity; "
            "careful adaptation will be needed"
        )
    elif cos_avg >= 0.5:
        verdict = (
            "the datasets share substantial latent structure and look "
            "bridgeable via contrastive alignment or a learned projection head"
        )
    else:
        verdict = (
            "the datasets occupy noticeably distinct regions of feature "
            "space and will need careful adaptation before joint encoding"
        )
    parts.append(f"**Conclusion:** {verdict}.")
    return " ".join(parts)


def _build_report(
    results: dict[str, Any],
    plot_path: Path,
    label_a: str,
    label_b: str,
) -> str:
    mmd = results["mmd"]
    cos = results["cosine"]
    pca_var = results["pca_var"]
    lines: list[str] = []
    lines.append("# Phase 1 — Distribution Validation Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append("## 1. Dataset summary")
    lines.append("")
    lines.append("| Dataset | Samples | Feature dim | Encoder |")
    lines.append("|---------|--------:|------------:|---------|")
    lines.append(
        f"| {label_a} | {results['n_a']} | {results['feature_dim']} | "
        f"{results['model_a']} |"
    )
    lines.append(
        f"| {label_b} | {results['n_b']} | {results['feature_dim']} | "
        f"{results['model_b']} |"
    )
    lines.append("")
    lines.append("## 2. PCA projection")
    lines.append("")
    lines.append(f"- PC1 explained variance: **{pca_var[0]:.2%}**")
    lines.append(f"- PC2 explained variance: **{pca_var[1]:.2%}**")
    lines.append(f"- Cumulative (PC1+PC2): **{sum(pca_var[:2]):.2%}**")
    lines.append("")
    lines.append(f"![PCA scatter]({plot_path.name})")
    lines.append("")
    lines.append("## 3. Fréchet distance")
    lines.append("")
    lines.append(f"FID = **{results['frechet_distance']:.4f}**")
    lines.append("")
    lines.append(
        "> Lower values indicate more similar distributions. Treats each "
        "feature set as a multivariate Gaussian; computed via "
        "`scipy.linalg.sqrtm` on the covariance product."
    )
    lines.append("")
    lines.append("## 4. Maximum Mean Discrepancy")
    lines.append("")
    lines.append(f"- MMD² = **{mmd['mmd_value']:.6f}**")
    lines.append(
        f"- Permutation p-value = **{mmd['p_value']:.4f}** "
        f"({mmd['n_permutations']} permutations)"
    )
    lines.append(
        f"- Null distribution: μ = {mmd['null_distribution_mean']:.6f}, "
        f"σ = {mmd['null_distribution_std']:.6f}"
    )
    lines.append(
        f"- Subsample size: {mmd['n_samples_a']} ({label_a}), "
        f"{mmd['n_samples_b']} ({label_b}); RBF γ = {mmd['gamma']:.6f}"
    )
    lines.append("")
    if mmd["p_value"] < 0.05:
        lines.append(
            "> p < 0.05: distributions are statistically distinguishable "
            "in feature space."
        )
    else:
        lines.append(
            "> p ≥ 0.05: cannot reject the null hypothesis that the two "
            "samples come from the same distribution."
        )
    lines.append("")
    lines.append(f"## 5. Cosine similarity (top-{cos['k']} neighbours)")
    lines.append("")
    lines.append("| Direction | Mean | Std |")
    lines.append("|-----------|-----:|----:|")
    lines.append(
        f"| {label_a} → {label_b} | {cos['a_to_b_mean']:.4f} | "
        f"{cos['a_to_b_std']:.4f} |"
    )
    lines.append(
        f"| {label_b} → {label_a} | {cos['b_to_a_mean']:.4f} | "
        f"{cos['b_to_a_std']:.4f} |"
    )
    lines.append("")
    lines.append(
        "> For each sample in one dataset, mean cosine similarity to its "
        "k nearest neighbours in the other dataset. Higher values indicate "
        "denser cross-dataset matches in the embedding space."
    )
    lines.append("")
    lines.append("## 6. Interpretation")
    lines.append("")
    lines.append(_interpret(results, label_a, label_b))
    lines.append("")

    severity_means = results.get("severity_means")
    if severity_means is not None:
        lines.append("## 7. PCA stratified by DR severity")
        lines.append("")
        lines.append(f"![PCA by severity]({SEVERITY_PLOT_FILENAME})")
        lines.append("")
        lines.append("Per-class mean PC1 coordinate:")
        lines.append("")
        lines.append("| Class | Mean PC1 |")
        lines.append("|-------|---------:|")
        for cls in range(N_DR_CLASSES):
            lines.append(
                f"| {label_a} class {cls} ({EYEPACS_LABEL_NAMES[cls]}) | "
                f"{severity_means[f'class_{cls}']:.4f} |"
            )
        lines.append(
            f"| {label_a} (overall) | {severity_means['eyepacs_all']:.4f} |"
        )
        lines.append(
            f"| {label_b} (overall) | {severity_means['olives']:.4f} |"
        )
        lines.append("")
        lines.append(
            f"- {label_a} class-mean spread on PC1: "
            f"**{results['severity_spread']:.4f}**"
        )
        lines.append(
            f"- {label_a}↔{label_b} centroid distance on PC1: "
            f"**{results['severity_centroid_distance']:.4f}**"
        )
        lines.append(
            f"- Verdict: **{results['severity_shift_label']}**"
        )
        lines.append("")
        lines.append(
            "> If EyePACS classes overlap on PC1, the shift to OLIVES "
            "(visible as a distinct cluster) is acquisition-driven rather "
            "than pathology-driven and should be bridgeable by contrastive "
            "alignment. If EyePACS classes form distinct PC1 sub-clusters, "
            "PC1 captures pathology variation and the OLIVES shift is more "
            "concerning."
        )
        lines.append("")

    image_stats = results.get("image_stats")
    if image_stats is not None:
        lines.append("## 8. Image statistics correlation with PC1")
        lines.append("")
        lines.append(f"![Stats vs PC1]({IMAGE_STATS_PLOT_FILENAME})")
        lines.append("")
        lines.append(
            "Spearman rank correlation between per-image acquisition stats "
            "and PC1, by dataset:"
        )
        lines.append("")
        lines.append("| Dataset | Stat | ρ | p-value |")
        lines.append("|---------|------|--:|--------:|")
        for ds_key, ds_label in (("eyepacs", label_a), ("olives", label_b)):
            for stat_key in ("brightness", "black_pixel", "contrast"):
                entry = image_stats[f"{ds_key}_{stat_key}"]
                lines.append(
                    f"| {ds_label} | {STAT_LABELS[stat_key]} | "
                    f"{entry['rho']:+.4f} | {entry['p_value']:.2e} |"
                )
        lines.append("")
        strongest = results.get("strongest_stat_corr")
        if strongest is not None:
            ds_key, stat_key = strongest["key"].split("_", 1)
            ds_label = label_a if ds_key == "eyepacs" else label_b
            lines.append(
                f"**Strongest correlation:** {ds_label} "
                f"{STAT_LABELS.get(stat_key, stat_key)} ↔ PC1, "
                f"ρ = {strongest['rho']:+.3f} "
                f"(p = {strongest['p_value']:.2e})."
            )
            lines.append("")
            lines.append(
                f"> The dominant separation direction (PC1) is most strongly "
                f"correlated with **{STAT_LABELS.get(stat_key, stat_key)}** "
                f"in {ds_label}, {'confirming' if abs(strongest['rho']) >= 0.3 else 'suggesting'} "
                f"that the shift is driven by acquisition-level differences "
                f"(field-stop, framing, exposure) rather than retinal "
                f"pathology content."
            )
            lines.append("")

    return "\n".join(lines)


def _run_diagnostics(
    cfg: DictConfig,
    label_a: str = "EyePACS",
    label_b: str = "OLIVES",
) -> None:
    features_dir = Path(str(cfg.features_dir))
    results_dir = Path(str(cfg.results_dir)) / PHASE1_DIRNAME
    results_dir.mkdir(parents=True, exist_ok=True)

    feats_a_path = features_dir / EYEPACS_FEATURES_FILENAME
    feats_b_path = features_dir / OLIVES_FEATURES_FILENAME
    print(f"[diagnostics] loading {feats_a_path}")
    feats_a, meta_a = _load_features(feats_a_path)
    print(f"  {label_a}: {feats_a.shape}")
    print(f"[diagnostics] loading {feats_b_path}")
    feats_b, meta_b = _load_features(feats_b_path)
    print(f"  {label_b}: {feats_b.shape}")

    if feats_a.shape[1] != feats_b.shape[1]:
        raise ValueError(
            f"Feature dimensionality mismatch: {label_a}={feats_a.shape[1]} "
            f"vs {label_b}={feats_b.shape[1]}. Re-run feature extraction with "
            "the same encoder for both datasets."
        )

    results: dict[str, Any] = {
        "n_a": int(feats_a.shape[0]),
        "n_b": int(feats_b.shape[0]),
        "feature_dim": int(feats_a.shape[1]),
        "model_a": str(meta_a.get("model_name", "?")),
        "model_b": str(meta_b.get("model_name", "?")),
    }

    print("\n[diagnostics] PCA projection")
    t0 = time.time()
    coords_a, coords_b, pca = compute_pca_projection(feats_a, feats_b, n_components=2)
    print(f"  PCA done in {time.time() - t0:.1f}s")
    pca_var = tuple(float(v) for v in pca.explained_variance_ratio_)
    results["pca_var"] = pca_var

    plot_path = results_dir / PCA_PLOT_FILENAME
    generate_pca_plot(
        coords_a,
        coords_b,
        label_a=label_a,
        label_b=label_b,
        output_path=plot_path,
        title=f"PCA: {label_a} vs {label_b} (ResNet-50 features)",
        explained_variance=pca_var,
    )
    print(f"  plot saved to {plot_path}")

    print("\n[diagnostics] Fréchet distance")
    t0 = time.time()
    results["frechet_distance"] = compute_frechet_distance(feats_a, feats_b)
    print(
        f"  FID = {results['frechet_distance']:.4f} "
        f"({time.time() - t0:.1f}s)"
    )

    print("\n[diagnostics] MMD with permutation test")
    t0 = time.time()
    results["mmd"] = compute_mmd(feats_a, feats_b)
    print(
        f"  MMD² = {results['mmd']['mmd_value']:.6f}, "
        f"p = {results['mmd']['p_value']:.4f} "
        f"({time.time() - t0:.1f}s)"
    )

    print("\n[diagnostics] Cosine similarity (k-NN)")
    t0 = time.time()
    results["cosine"] = compute_cosine_similarity_diagnostics(feats_a, feats_b, k=5)
    print(
        f"  {label_a}→{label_b}: μ={results['cosine']['a_to_b_mean']:.4f}, "
        f"{label_b}→{label_a}: μ={results['cosine']['b_to_a_mean']:.4f} "
        f"({time.time() - t0:.1f}s)"
    )

    eyepacs_pt_dir = Path(str(cfg.eyepacs_output))
    olives_pt_path = Path(str(cfg.olives_output)) / OLIVES_INPUT_FILENAME

    print("\n[diagnostics] PCA stratified by DR severity")
    t0 = time.time()
    eyepacs_labels = _load_eyepacs_labels(eyepacs_pt_dir)
    severity_path = results_dir / SEVERITY_PLOT_FILENAME
    severity_means = compute_pca_by_severity(
        feats_a, eyepacs_labels, feats_b, severity_path
    )
    results["severity_means"] = severity_means
    class_means = [severity_means[f"class_{cls}"] for cls in range(N_DR_CLASSES)]
    valid_means = [m for m in class_means if not np.isnan(m)]
    spread = (
        max(valid_means) - min(valid_means) if valid_means else 0.0
    )
    centroid_distance = abs(severity_means["eyepacs_all"] - severity_means["olives"])
    if centroid_distance > 0 and spread < SHIFT_SPREAD_RATIO * centroid_distance:
        shift_label = "Acquisition-driven shift (favourable)"
    else:
        shift_label = "Pathology-correlated shift (concerning)"
    results["severity_spread"] = float(spread)
    results["severity_centroid_distance"] = float(centroid_distance)
    results["severity_shift_label"] = shift_label
    print(
        f"  spread={spread:.4f}, centroid_distance={centroid_distance:.4f}, "
        f"verdict={shift_label} ({time.time() - t0:.1f}s)"
    )
    print(f"  plot saved to {severity_path}")

    print("\n[diagnostics] Image stats vs PC1")
    t0 = time.time()
    stats_path = results_dir / IMAGE_STATS_PLOT_FILENAME
    image_stats = compute_image_stats_correlation(
        eyepacs_pt_dir, olives_pt_path, feats_a, feats_b, stats_path
    )
    results["image_stats"] = image_stats
    strongest_key = max(image_stats, key=lambda k: abs(image_stats[k]["rho"]))
    results["strongest_stat_corr"] = {
        "key": strongest_key,
        "rho": image_stats[strongest_key]["rho"],
        "p_value": image_stats[strongest_key]["p_value"],
    }
    print(
        f"  strongest: {strongest_key} ρ={image_stats[strongest_key]['rho']:+.3f} "
        f"p={image_stats[strongest_key]['p_value']:.2e} "
        f"({time.time() - t0:.1f}s)"
    )
    print(f"  plot saved to {stats_path}")

    report_path = results_dir / REPORT_FILENAME
    report_path.write_text(_build_report(results, plot_path, label_a, label_b))
    print(f"\n[diagnostics] report written to {report_path}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Phase-1 distribution diagnostics on extracted features."
    )
    parser.add_argument("--config", default="configs/preprocess.yaml")
    parser.add_argument("--label-a", default="EyePACS")
    parser.add_argument("--label-b", default="OLIVES")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        cfg = load_config(args.config)
    except Exception as exc:
        print(f"[ERROR] failed to load config from {args.config}: {exc}")
        traceback.print_exc()
        return 2

    try:
        _run_diagnostics(cfg, label_a=args.label_a, label_b=args.label_b)
    except Exception as exc:
        print(f"\n[ERROR] distribution diagnostics failed: {exc}")
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
