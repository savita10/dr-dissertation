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
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel
from tqdm.auto import tqdm

from src.utils.config import load_config

EYEPACS_FEATURES_FILENAME = "eyepacs_resnet50_features.pt"
OLIVES_FEATURES_FILENAME = "olives_resnet50_features.pt"
PHASE1_DIRNAME = "phase1"
PCA_PLOT_FILENAME = "pca_eyepacs_vs_olives.png"
REPORT_FILENAME = "phase1_report.md"
COSINE_CHUNK_SIZE = 500
MMD_RNG_SEED = 42


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
    bridge = (
        "the datasets share substantial latent structure and look bridgeable "
        "via contrastive alignment or a learned projection head"
        if cos_avg >= 0.5
        else "the datasets occupy noticeably distinct regions of feature "
        "space and will need careful adaptation before joint encoding"
    )
    return (
        f"In ResNet-50 ImageNet feature space, {label_a} and {label_b} show "
        f"{cos_label} (mean cosine = {cos_avg:.3f}) and {mmd_label} "
        f"(MMD permutation p = {p_value:.4f}). The Fréchet distance is "
        f"{results['frechet_distance']:.2f}; FID is comparable across runs "
        f"of the same pair but not across dataset pairs without a baseline. "
        f"PCA projects {sum(results['pca_var'][:2]):.1%} of the joint "
        f"variance into 2D, so the scatter plot is indicative rather than "
        f"definitive. Taken together, these results suggest "
        f"{bridge}."
    )


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
