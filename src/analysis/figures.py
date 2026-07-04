"""Dissertation figure generation from frozen Phase 1 / Phase 2 artefacts.

This module is *plotting only*. It reads the feature tensors, diagnostics JSON,
and training-history JSON that Phases 1 and 2 already wrote to Google Drive and
renders a consistent, colourblind-safe set of dissertation figures. It never
re-extracts features, never loads the encoder, and has no side effects on
import.

Artefact schema (confirmed from the writers ``feature_extraction.py`` /
``trained_feature_extraction.py`` / ``phase2_diagnostics.py``):

* Phase 1 ``{eyepacs,olives}_resnet50_features.pt``
    ``{"features": Tensor[N, 2048], "metadata": {...}}`` — no labels/stats.
* Phase 2 ``phase2_{arm}_{split}_{dataset}_features.pt``
    ``{"features": Tensor[N, 2048], "labels": Tensor[N] | None,
       "image_stats": {"brightness","black_pixel","contrast": ndarray[N]},
       "metadata": {...}}``.
  The ``full`` split mirrors Phase 1's iteration order, so the Phase 2 EyePACS
  ``labels`` / ``image_stats`` are row-aligned with the Phase 1 features too.
* ``reports/phase2_results.json``
    ``{"phase1_baseline": {...}, "targets": {...},
       "results": {arm: {split: {fid, cosine_mean, black_border_rho,
                                  severity: {means: {...}, rank_rho}, ...}}}}``.
* ``checkpoints/phase2_{arm}_history.json``
    list of per-epoch records ``{"epoch", "train": {"loss": {...}}, "val":
    {"dr_qwk", ...}}``.

Every figure is saved as both PNG (dpi 200, for the slide deck / preview) and
PDF (vector, for the dissertation) to ``cfg.figures_dir``. Figure functions
degrade gracefully: if an optional array is absent they render what they can
and annotate the omission rather than raising.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

# --------------------------------------------------------------------- palette
# Okabe-Ito colourblind-safe hues for the two datasets.
COLOR_EYEPACS = "#0072B2"  # blue
COLOR_OLIVES = "#E69F00"  # orange
COLOR_BASELINE = "#999999"  # neutral grey for "before" bars
COLOR_TARGET = "#000000"  # target reference lines

# Perceptually-uniform sequential map for the 5 DR severity grades (0..4).
N_DR_CLASSES = 5
DR_LABEL_NAMES = ("No DR", "Mild", "Moderate", "Severe", "Proliferative")
SEVERITY_COLORS = plt.cm.viridis(np.linspace(0.10, 0.90, N_DR_CLASSES))

FIG_DPI = 200
FORMATS = ("png", "pdf")

# Phase 1 baseline provenance (matches configs/diagnostics.yaml). Used only as a
# fallback label; the live numbers are read from phase2_results.json.
PHASE1_FID = 240.65
PHASE1_RHO = -0.274

# ImageNet normalisation, for de-normalising preprocessed image tensors.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Hard-coded dataset-chapter counts (not stored in the feature artefacts).
EYEPACS_N = 35108
OLIVES_N = 3142
OLIVES_TIERS = {"tier0 (fundus only)": 1546, "tier1 (+clinical)": 1404, "tier2 (+biomarker)": 190}
OLIVES_PATIENTS = 96


# ---------------------------------------------------------------------- config
@dataclass
class FiguresConfig:
    """Resolved Drive paths for the artefacts and the figure output directory.

    Build with :meth:`from_diagnostics_yaml` (preferred, reuses the frozen
    ``configs/diagnostics.yaml``) or :meth:`default` (hard-coded Drive layout).
    """

    features_dir: Path
    reports_dir: Path
    checkpoint_dir: Path
    figures_dir: Path
    eyepacs_dir: Path
    olives_file: Path
    arms: tuple[str, ...] = ("coral_on", "coral_off")

    @classmethod
    def default(cls) -> "FiguresConfig":
        root = Path("/content/drive/MyDrive/dissertation")
        return cls(
            features_dir=root / "features",
            reports_dir=root / "reports",
            checkpoint_dir=root / "checkpoints",
            figures_dir=root / "figures",
            eyepacs_dir=root / "preprocessed" / "eyepacs",
            olives_file=root / "preprocessed" / "olives" / "olives_fundus.pt",
        )

    @classmethod
    def from_diagnostics_yaml(
        cls, path: str | Path = "configs/diagnostics.yaml"
    ) -> "FiguresConfig":
        """Build from the frozen diagnostics config, deriving ``figures_dir``."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(str(path))
        features_dir = Path(str(cfg.features_dir))
        return cls(
            features_dir=features_dir,
            reports_dir=Path(str(cfg.report_dir)),
            checkpoint_dir=Path(str(cfg.checkpoint_dir)),
            figures_dir=features_dir.parent / "figures",
            eyepacs_dir=Path(str(cfg.eyepacs_dir)),
            olives_file=Path(str(cfg.olives_file)),
            arms=tuple(str(a) for a in cfg.arms),
        )


# --------------------------------------------------------------------- styling
def set_style() -> None:
    """Apply the shared print-friendly matplotlib style (idempotent).

    White background, no top/right spines, readable ~11-12 pt labels. Called at
    the top of every figure function so all figures match.
    """
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#DDDDDD",
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 110,
        }
    )


# ------------------------------------------------------------------- io helpers
def _load_pt(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def _features_np(blob: Any) -> np.ndarray:
    feats = blob["features"] if isinstance(blob, dict) else blob
    if isinstance(feats, torch.Tensor):
        feats = feats.detach().cpu().numpy()
    return np.asarray(feats, dtype=np.float32)


def _phase1_path(cfg: FiguresConfig, dataset: str) -> Path:
    return cfg.features_dir / f"{dataset}_resnet50_features.pt"


def _phase2_path(cfg: FiguresConfig, arm: str, split: str, dataset: str) -> Path:
    return cfg.features_dir / f"phase2_{arm}_{split}_{dataset}_features.pt"


def _load_results(cfg: FiguresConfig) -> dict[str, Any]:
    with open(cfg.reports_dir / "phase2_results.json", encoding="utf-8") as fh:
        return json.load(fh)


def _load_history(cfg: FiguresConfig, arm: str) -> Optional[list[dict[str, Any]]]:
    path = cfg.checkpoint_dir / f"phase2_{arm}_history.json"
    if not path.exists():
        print(f"[figures] history not found for arm={arm}: {path}")
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _labels_np(blob: Any) -> Optional[np.ndarray]:
    labels = blob.get("labels") if isinstance(blob, dict) else None
    if labels is None:
        return None
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    return np.asarray(labels).astype(np.int64)


def _border_np(blob: Any) -> Optional[np.ndarray]:
    stats = blob.get("image_stats") if isinstance(blob, dict) else None
    if not stats or "black_pixel" not in stats:
        return None
    return np.asarray(stats["black_pixel"], dtype=np.float32)


def _save(fig: Figure, cfg: FiguresConfig, stem: str) -> dict[str, Path]:
    """Save ``fig`` to ``figures_dir`` as PNG (dpi 200) and PDF; close it."""
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for ext in FORMATS:
        path = cfg.figures_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
        out[ext] = path
    plt.close(fig)
    print(f"[figures] wrote {stem}: " + ", ".join(p.name for p in out.values()))
    return out


def _joint_pca(
    eye: np.ndarray, oli: np.ndarray, n_components: int = 2
) -> tuple[np.ndarray, np.ndarray, PCA]:
    """Fit PCA on the stacked feature sets (matches the diagnostics module)."""
    stacked = np.vstack([eye, oli])
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=0)
    coords = pca.fit_transform(stacked)
    n = eye.shape[0]
    return coords[:n], coords[n:], pca


# ================================================================== FIGURE 1 ==
def fig_dataset_summary_table(cfg: FiguresConfig) -> dict[str, Path]:
    """Figure 1 — dataset summary table (EyePACS + OLIVES).

    Renders a styled table PNG/PDF and a companion CSV. EyePACS DR-grade counts
    are read from the Phase 2 full-split labels when available; OLIVES three-tier
    counts and patient totals are taken from the data chapter (not in the .pt
    artefacts). Writes ``fig01_dataset_summary.{png,pdf,csv}``.
    """
    set_style()

    # EyePACS DR-grade distribution from labels (Phase 2 full split), if present.
    grade_counts: Optional[dict[int, int]] = None
    try:
        blob = _load_pt(_phase2_path(cfg, "coral_on", "full", "eyepacs"))
        labels = _labels_np(blob)
        if labels is not None:
            counts = Counter(int(v) for v in labels.tolist())
            grade_counts = {c: counts.get(c, 0) for c in range(N_DR_CLASSES)}
    except FileNotFoundError:
        grade_counts = None

    rows: list[tuple[str, str, str]] = [
        ("Total images", f"{EYEPACS_N:,}", f"{OLIVES_N:,}"),
        ("Modality", "Fundus only", "Fundus + OCT + clinical"),
        ("Patients", "n/a (image-level)", f"{OLIVES_PATIENTS}"),
    ]
    if grade_counts is not None:
        for c in range(N_DR_CLASSES):
            n = grade_counts[c]
            pct = 100.0 * n / max(1, sum(grade_counts.values()))
            rows.append(
                (f"DR grade {c} ({DR_LABEL_NAMES[c]})", f"{n:,} ({pct:.1f}%)", "—")
            )
    else:
        rows.append(("DR grade distribution", "labels unavailable", "—"))
    for tier, n in OLIVES_TIERS.items():
        rows.append((f"OLIVES {tier}", "—", f"{n:,}"))

    # CSV companion.
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cfg.figures_dir / "fig01_dataset_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Attribute", "EyePACS", "OLIVES"])
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(9, 0.55 * len(rows) + 1.4))
    ax.axis("off")
    ax.set_title("Dataset summary: EyePACS vs OLIVES", pad=16)
    table = ax.table(
        cellText=[list(r) for r in rows],
        colLabels=["Attribute", "EyePACS", "OLIVES"],
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        if row == 0:
            cell.set_facecolor("#F0F0F0")
            cell.set_text_props(fontweight="bold")
        if col == 1 and row == 0:
            cell.set_text_props(color=COLOR_EYEPACS, fontweight="bold")
        if col == 2 and row == 0:
            cell.set_text_props(color=COLOR_OLIVES, fontweight="bold")

    out = _save(fig, cfg, "fig01_dataset_summary")
    out["csv"] = csv_path
    return out


# ================================================================== FIGURE 2 ==
def fig_acquisition_pca(cfg: FiguresConfig) -> dict[str, Path]:
    """Figure 2 — Phase 1 acquisition PCA (before training).

    PCA fit on the combined Phase 1 ImageNet features of both datasets; scatter
    coloured by dataset. Annotates the PC1 <-> black-border Spearman ρ from the
    baseline (-0.274). If the per-image border proportion is available (from the
    row-aligned Phase 2 full-split ``image_stats``), a second panel plots PC1 vs
    border proportion. Writes ``fig02_acquisition_pca.{png,pdf}``.
    """
    set_style()
    eye = _features_np(_load_pt(_phase1_path(cfg, "eyepacs")))
    oli = _features_np(_load_pt(_phase1_path(cfg, "olives")))
    coords_eye, coords_oli, pca = _joint_pca(eye, oli)
    var = pca.explained_variance_ratio_

    # Optional row-aligned border proportion from the Phase 2 full split.
    border_eye = rho = None
    try:
        b2 = _load_pt(_phase2_path(cfg, "coral_on", "full", "eyepacs"))
        cand = _border_np(b2)
        if cand is not None and cand.shape[0] == eye.shape[0]:
            border_eye = cand
            rho = float(spearmanr(border_eye, coords_eye[:, 0])[0])
    except FileNotFoundError:
        border_eye = None

    ncols = 2 if border_eye is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6.5 * ncols, 6))
    ax0 = axes[0] if ncols == 2 else axes

    ax0.scatter(
        coords_eye[:, 0], coords_eye[:, 1], s=7, alpha=0.35,
        color=COLOR_EYEPACS, label="EyePACS", edgecolors="none",
    )
    ax0.scatter(
        coords_oli[:, 0], coords_oli[:, 1], s=7, alpha=0.45,
        color=COLOR_OLIVES, label="OLIVES", edgecolors="none",
    )
    ax0.set_xlabel(f"PC1 ({var[0]:.1%} variance)")
    ax0.set_ylabel(f"PC2 ({var[1]:.1%} variance)")
    ax0.set_title("Phase 1 (ImageNet features): acquisition-separated")
    ax0.legend(loc="best", markerscale=2)
    rho_text = f"{PHASE1_RHO:+.3f}" if rho is None else f"{rho:+.3f}"
    ax0.annotate(
        f"PC1 ↔ black-border ρ = {rho_text}",
        xy=(0.02, 0.98), xycoords="axes fraction", va="top", ha="left",
        fontsize=10, bbox=dict(boxstyle="round", fc="white", ec="#CCCCCC"),
    )

    if border_eye is not None:
        ax1 = axes[1]
        ax1.scatter(
            border_eye, coords_eye[:, 0], s=7, alpha=0.35,
            color=COLOR_EYEPACS, edgecolors="none",
        )
        ax1.set_xlabel("Black-border proportion (EyePACS)")
        ax1.set_ylabel("PC1")
        ax1.set_title(f"PC1 tracks acquisition geometry (ρ = {rho:+.3f})")

    fig.suptitle(
        "Before training, the dominant axis is camera geometry, not pathology",
        y=1.02, fontsize=13,
    )
    return _save(fig, cfg, "fig02_acquisition_pca")


# ================================================================== FIGURE 3 ==
def fig_example_fundus(
    cfg: FiguresConfig, n_per_dataset: int = 4
) -> Optional[dict[str, Path]]:
    """Figure 3 — example fundus images (EyePACS vs OLIVES).

    Optional / best-effort: loads a few preprocessed image tensors (first
    EyePACS shard + OLIVES fundus file), de-normalises ImageNet stats, and shows
    them side by side to illustrate the field-stop/border difference. If the
    tensors are not readily loadable (missing files or memory), prints a note and
    returns ``None`` rather than forcing it. Writes ``fig03_example_fundus.*``.
    """
    set_style()

    def _first_images(path: Path, k: int) -> Optional[np.ndarray]:
        if not path.exists():
            return None
        data = _load_pt(path)
        imgs = data["images"] if isinstance(data, dict) else data
        imgs = imgs[:k].float().cpu().numpy()  # (k, 3, H, W)
        imgs = imgs.transpose(0, 2, 3, 1)  # (k, H, W, 3)
        imgs = imgs * IMAGENET_STD + IMAGENET_MEAN
        return np.clip(imgs, 0.0, 1.0)

    try:
        shards = sorted(cfg.eyepacs_dir.glob("eyepacs_shard_*.pt"))
        eye_path = shards[0] if shards else None
        eye_imgs = _first_images(eye_path, n_per_dataset) if eye_path else None
        oli_imgs = _first_images(cfg.olives_file, n_per_dataset)
    except (OSError, RuntimeError, ValueError, MemoryError) as exc:
        print(f"[figures] fig_example_fundus skipped — could not load images: {exc}")
        return None

    if eye_imgs is None and oli_imgs is None:
        print("[figures] fig_example_fundus skipped — no image tensors found.")
        return None

    n = n_per_dataset
    fig, axes = plt.subplots(2, n, figsize=(2.4 * n, 5.2))
    for row, (imgs, name, color) in enumerate(
        ((eye_imgs, "EyePACS", COLOR_EYEPACS), (oli_imgs, "OLIVES", COLOR_OLIVES))
    ):
        for col in range(n):
            ax = axes[row, col]
            ax.axis("off")
            if imgs is not None and col < imgs.shape[0]:
                ax.imshow(imgs[col])
            if col == 0:
                ax.set_title(name, loc="left", color=color, fontweight="bold")
    fig.suptitle(
        "Preprocessed fundus examples: EyePACS field-stop borders vs OLIVES framing",
        y=1.02,
    )
    return _save(fig, cfg, "fig03_example_fundus")


# ================================================================== FIGURE 4 ==
def fig_training_curves(cfg: FiguresConfig) -> dict[str, dict[str, Path]]:
    """Figure 4 — training curves.

    (a) Validation DR QWK per epoch for both arms on one axis, best epoch marked.
    (b) Per-term training losses (dr/biomarker/regression/supcon/coral/total)
    over epochs for the ``coral_on`` arm, best epoch marked. Reads both
    ``phase2_{arm}_history.json`` files. Writes ``fig04a_training_qwk.*`` and
    ``fig04b_training_losses.*``.
    """
    set_style()
    out: dict[str, dict[str, Path]] = {}

    histories = {arm: _load_history(cfg, arm) for arm in cfg.arms}

    # (a) DR QWK per epoch, both arms.
    fig, ax = plt.subplots(figsize=(8, 5))
    arm_colors = {"coral_on": COLOR_EYEPACS, "coral_off": COLOR_OLIVES}
    for arm, hist in histories.items():
        if not hist:
            continue
        epochs = [r["epoch"] for r in hist]
        qwk = [float(r["val"]["dr_qwk"]) for r in hist]
        color = arm_colors.get(arm, None)
        ax.plot(epochs, qwk, marker="o", ms=3, lw=1.6, color=color, label=arm)
        best_i = int(np.nanargmax(qwk))
        ax.scatter(
            [epochs[best_i]], [qwk[best_i]], s=90, facecolors="none",
            edgecolors=color, linewidths=2, zorder=5,
        )
        ax.annotate(
            f"best {qwk[best_i]:.3f} @ e{epochs[best_i]}",
            xy=(epochs[best_i], qwk[best_i]), xytext=(4, 6),
            textcoords="offset points", fontsize=9, color=color,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation DR QWK")
    ax.set_title("DR quadratic weighted kappa per epoch")
    ax.legend(loc="best")
    out["qwk"] = _save(fig, cfg, "fig04a_training_qwk")

    # (b) Per-term training losses (coral_on).
    hist = histories.get("coral_on")
    if hist:
        terms = ["dr", "biomarker", "regression", "supcon", "coral", "total"]
        term_cmap = plt.cm.tab10(np.linspace(0, 1, len(terms)))
        epochs = [r["epoch"] for r in hist]
        val_qwk = [float(r["val"]["dr_qwk"]) for r in hist]
        best_epoch = epochs[int(np.nanargmax(val_qwk))]

        fig, ax = plt.subplots(figsize=(8, 5))
        for term, color in zip(terms, term_cmap):
            series = [float(r["train"]["loss"].get(term, np.nan)) for r in hist]
            if np.all(np.isnan(series)):
                continue
            lw = 2.4 if term == "total" else 1.4
            ls = "-" if term == "total" else "--"
            ax.plot(epochs, series, lw=lw, ls=ls, color=color, label=term)
        ax.axvline(best_epoch, color="#999999", ls=":", lw=1.2)
        ax.annotate(
            f"best QWK @ e{best_epoch}", xy=(best_epoch, ax.get_ylim()[1]),
            xytext=(4, -12), textcoords="offset points", fontsize=9, color="#666666",
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training loss (per term)")
        ax.set_title("Per-term training losses — coral_on")
        ax.legend(loc="best", ncol=2)
        out["losses"] = _save(fig, cfg, "fig04b_training_losses")
    else:
        print("[figures] fig_training_curves: coral_on history missing — losses skipped")
    return out


# ================================================================== FIGURE 5 ==
def fig_fid_bar(cfg: FiguresConfig) -> dict[str, Path]:
    """Figure 5 — cross-dataset FID before/after alignment.

    Bar chart Phase 1 baseline -> coral_off -> coral_on from
    ``phase2_results.json``, annotated with the overall percentage drop.
    Writes ``fig05_fid_bar.{png,pdf}``.
    """
    set_style()
    res = _load_results(cfg)
    baseline = float(res["phase1_baseline"]["fid"])
    off = float(res["results"]["coral_off"]["full"]["fid"])
    on = float(res["results"]["coral_on"]["full"]["fid"])
    target = float(res["targets"]["fid_max"])

    labels = ["Phase 1\n(ImageNet)", "coral_off\n(trained)", "coral_on\n(trained)"]
    values = [baseline, off, on]
    colors = [COLOR_BASELINE, COLOR_OLIVES, COLOR_EYEPACS]

    fig, ax = plt.subplots(figsize=(7, 5.2))
    bars = ax.bar(labels, values, color=colors, width=0.62)
    ax.axhline(target, ls="--", color=COLOR_TARGET, lw=1.2, label=f"target < {target:.0f}")
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, value + 3, f"{value:.2f}",
            ha="center", va="bottom", fontsize=10,
        )
    drop = 100.0 * (on - baseline) / baseline
    ax.annotate(
        f"{drop:+.0f}% vs Phase 1",
        xy=(2, on), xytext=(1.15, baseline * 0.72),
        arrowprops=dict(arrowstyle="->", color="#444444"),
        fontsize=11, fontweight="bold", color="#444444",
    )
    ax.set_ylabel("Fréchet distance (FID)")
    ax.set_title("Cross-dataset FID collapses after training")
    ax.set_ylim(0, baseline * 1.12)
    ax.legend(loc="upper right")
    # Honest framing: the CORAL term contributes only marginally.
    ax.annotate(
        "CORAL marginal (off vs on ≈ %.1f FID); augmentation is the main driver"
        % abs(off - on),
        xy=(0.5, -0.16), xycoords="axes fraction", ha="center",
        fontsize=8.5, color="#666666",
    )
    return _save(fig, cfg, "fig05_fid_bar")


# ================================================================== FIGURE 6 ==
def fig_pca_before_after(cfg: FiguresConfig) -> dict[str, dict[str, Path]]:
    """Figure 6 — PCA before vs after (the marquee figure).

    Two panels sharing style: before = Phase 1 features (acquisition-separated),
    after = Phase 2 ``coral_on`` features, both coloured by dataset. A second
    figure re-colours the "after" panel by EyePACS DR severity to show the
    pathology ordering. Writes ``fig06a_pca_before_after.*`` and
    ``fig06b_pca_after_severity.*``.
    """
    set_style()
    out: dict[str, dict[str, Path]] = {}

    eye_b = _features_np(_load_pt(_phase1_path(cfg, "eyepacs")))
    oli_b = _features_np(_load_pt(_phase1_path(cfg, "olives")))
    after_eye_blob = _load_pt(_phase2_path(cfg, "coral_on", "full", "eyepacs"))
    eye_a = _features_np(after_eye_blob)
    oli_a = _features_np(_load_pt(_phase2_path(cfg, "coral_on", "full", "olives")))

    ce_b, co_b, pca_b = _joint_pca(eye_b, oli_b)
    ce_a, co_a, pca_a = _joint_pca(eye_a, oli_a)

    # Panel figure: before vs after, coloured by dataset.
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, (ce, co, pca, title) in zip(
        axes,
        (
            (ce_b, co_b, pca_b, "Before — Phase 1 (ImageNet)"),
            (ce_a, co_a, pca_a, "After — Phase 2 (coral_on)"),
        ),
    ):
        var = pca.explained_variance_ratio_
        ax.scatter(ce[:, 0], ce[:, 1], s=7, alpha=0.35, color=COLOR_EYEPACS,
                   label="EyePACS", edgecolors="none")
        ax.scatter(co[:, 0], co[:, 1], s=7, alpha=0.45, color=COLOR_OLIVES,
                   label="OLIVES", edgecolors="none")
        ax.set_xlabel(f"PC1 ({var[0]:.1%})")
        ax.set_ylabel(f"PC2 ({var[1]:.1%})")
        ax.set_title(title)
        ax.legend(loc="best", markerscale=2)
    fig.suptitle(
        "PCA before/after: the dominant axis flips from acquisition to alignment",
        y=1.02, fontsize=13,
    )
    out["by_dataset"] = _save(fig, cfg, "fig06a_pca_before_after")

    # "After" panel coloured by DR severity (pathology ordering).
    labels = _labels_np(after_eye_blob)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    var = pca_a.explained_variance_ratio_
    if labels is not None and labels.shape[0] == ce_a.shape[0]:
        for c in range(N_DR_CLASSES):
            mask = labels == c
            if not mask.any():
                continue
            ax.scatter(
                ce_a[mask, 0], ce_a[mask, 1], s=8, alpha=0.5,
                color=SEVERITY_COLORS[c], edgecolors="none",
                label=f"EyePACS — {DR_LABEL_NAMES[c]}",
            )
        ax.scatter(co_a[:, 0], co_a[:, 1], s=6, alpha=0.35, color=COLOR_OLIVES,
                   edgecolors="none", label="OLIVES")
        ax.legend(loc="best", fontsize=9, markerscale=2)
    else:
        ax.scatter(ce_a[:, 0], ce_a[:, 1], s=7, alpha=0.4, color=COLOR_EYEPACS)
        ax.annotate("DR labels unavailable — severity colouring omitted",
                    xy=(0.02, 0.98), xycoords="axes fraction", va="top", fontsize=9)
    ax.set_xlabel(f"PC1 ({var[0]:.1%})")
    ax.set_ylabel(f"PC2 ({var[1]:.1%})")
    ax.set_title("After (coral_on): EyePACS coloured by DR severity")
    out["by_severity"] = _save(fig, cfg, "fig06b_pca_after_severity")
    return out


# ================================================================== FIGURE 7 ==
def fig_tsne_umap_joint(
    cfg: FiguresConfig, eyepacs_cap: int = 6000, seed: int = 42
) -> Optional[dict[str, dict[str, Path]]]:
    """Figure 7 — joint 2-D embedding of the coral_on features.

    Computes one 2-D embedding of the combined ``coral_on`` full features and
    plots it twice: coloured by dataset (should intermix = alignment) and by DR
    severity (should separate = pathology). Uses UMAP if ``umap-learn`` imports,
    else falls back to ``sklearn.manifold.TSNE`` (prints which). EyePACS is
    subsampled to ``eyepacs_cap`` points for tractability; all OLIVES kept.
    Writes ``fig07a_embed_by_dataset.*`` and ``fig07b_embed_by_severity.*``.
    """
    set_style()
    eye_blob = _load_pt(_phase2_path(cfg, "coral_on", "full", "eyepacs"))
    eye = _features_np(eye_blob)
    oli = _features_np(_load_pt(_phase2_path(cfg, "coral_on", "full", "olives")))
    labels = _labels_np(eye_blob)

    rng = np.random.default_rng(seed)
    if eye.shape[0] > eyepacs_cap:
        idx = np.sort(rng.choice(eye.shape[0], eyepacs_cap, replace=False))
        eye = eye[idx]
        labels = labels[idx] if labels is not None else None
    n_eye = eye.shape[0]
    combined = np.vstack([eye, oli])

    # PCA pre-reduction (standard for t-SNE/UMAP speed & stability).
    pre = PCA(n_components=min(50, combined.shape[1]), svd_solver="randomized",
              random_state=0).fit_transform(combined)

    try:
        import umap  # type: ignore

        method = "UMAP"
        embed = umap.UMAP(
            n_neighbors=30, min_dist=0.1, random_state=seed
        ).fit_transform(pre)
    except Exception as exc:  # ImportError or backend failure
        from sklearn.manifold import TSNE

        method = "t-SNE"
        print(f"[figures] UMAP unavailable ({exc}); falling back to t-SNE")
        embed = TSNE(
            n_components=2, perplexity=30, init="pca", random_state=seed
        ).fit_transform(pre)
    print(f"[figures] fig_tsne_umap_joint using {method} "
          f"(EyePACS n={n_eye}, OLIVES n={oli.shape[0]})")

    emb_eye, emb_oli = embed[:n_eye], embed[n_eye:]
    caption = f"{method}; EyePACS subsampled to {n_eye:,}, all {oli.shape[0]:,} OLIVES"
    out: dict[str, dict[str, Path]] = {}

    # By dataset.
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.scatter(emb_eye[:, 0], emb_eye[:, 1], s=6, alpha=0.35, color=COLOR_EYEPACS,
               label="EyePACS", edgecolors="none")
    ax.scatter(emb_oli[:, 0], emb_oli[:, 1], s=8, alpha=0.5, color=COLOR_OLIVES,
               label="OLIVES", edgecolors="none")
    ax.set_xlabel(f"{method}-1")
    ax.set_ylabel(f"{method}-2")
    ax.set_title(f"{method} of coral_on features — by dataset (intermix = alignment)")
    ax.legend(loc="best", markerscale=2)
    ax.annotate(caption, xy=(0.5, -0.12), xycoords="axes fraction", ha="center",
                fontsize=8.5, color="#666666")
    out["by_dataset"] = _save(fig, cfg, "fig07a_embed_by_dataset")

    # By severity.
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    if labels is not None and labels.shape[0] == n_eye:
        for c in range(N_DR_CLASSES):
            mask = labels == c
            if not mask.any():
                continue
            ax.scatter(emb_eye[mask, 0], emb_eye[mask, 1], s=7, alpha=0.5,
                       color=SEVERITY_COLORS[c], edgecolors="none",
                       label=f"EyePACS — {DR_LABEL_NAMES[c]}")
        ax.scatter(emb_oli[:, 0], emb_oli[:, 1], s=6, alpha=0.3, color=COLOR_OLIVES,
                   edgecolors="none", label="OLIVES")
        ax.legend(loc="best", fontsize=9, markerscale=2)
    else:
        ax.scatter(emb_eye[:, 0], emb_eye[:, 1], s=6, alpha=0.4, color=COLOR_EYEPACS)
        ax.annotate("DR labels unavailable — severity colouring omitted",
                    xy=(0.02, 0.98), xycoords="axes fraction", va="top", fontsize=9)
    ax.set_xlabel(f"{method}-1")
    ax.set_ylabel(f"{method}-2")
    ax.set_title(f"{method} of coral_on features — by DR severity (separate = pathology)")
    ax.annotate(caption, xy=(0.5, -0.12), xycoords="axes fraction", ha="center",
                fontsize=8.5, color="#666666")
    out["by_severity"] = _save(fig, cfg, "fig07b_embed_by_severity")
    return out


# ================================================================== FIGURE 8 ==
def fig_severity_ordering(cfg: FiguresConfig) -> dict[str, Path]:
    """Figure 8 — PC1 mean per DR grade (monotonic severity ordering).

    Bar chart of the mean PC1 coordinate per EyePACS DR grade (0..4) from
    ``phase2_results.json`` (``results.coral_on.full.severity.means``), showing
    the monotonic ordering (rank ρ from the same block). Writes
    ``fig08_severity_ordering.{png,pdf}``.
    """
    set_style()
    res = _load_results(cfg)
    sev = res["results"]["coral_on"]["full"]["severity"]
    means = sev["means"]
    values = [float(means[f"class_{c}"]) for c in range(N_DR_CLASSES)]
    rank_rho = float(sev.get("rank_rho", float("nan")))

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    bars = ax.bar(range(N_DR_CLASSES), values, color=SEVERITY_COLORS, width=0.68)
    for bar, value in zip(bars, values):
        va = "bottom" if value >= 0 else "top"
        offset = 0.02 * (max(values) - min(values) + 1e-9)
        ax.text(bar.get_x() + bar.get_width() / 2, value + (offset if value >= 0 else -offset),
                f"{value:.2f}", ha="center", va=va, fontsize=10)
    ax.axhline(0, color="#888888", lw=0.8)
    ax.set_xticks(range(N_DR_CLASSES))
    ax.set_xticklabels([f"{c}\n{DR_LABEL_NAMES[c]}" for c in range(N_DR_CLASSES)])
    ax.set_ylabel("Mean PC1 coordinate")
    ax.set_xlabel("EyePACS DR grade")
    ax.set_title("PC1 orders monotonically by DR severity (coral_on)")
    ax.annotate(f"rank ρ = {rank_rho:.2f}", xy=(0.02, 0.95),
                xycoords="axes fraction", va="top", fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round", fc="white", ec="#CCCCCC"))
    return _save(fig, cfg, "fig08_severity_ordering")


# ================================================================== FIGURE 9 ==
def fig_acquisition_rho_before_after(cfg: FiguresConfig) -> dict[str, Path]:
    """Figure 9 — PC1 <-> black-border ρ before vs after.

    Bar chart of the PC1<->black-border Spearman ρ (EyePACS): Phase 1 baseline
    (-0.274) vs Phase 2 ``coral_on`` (from ``phase2_results.json``), with the
    ±|ρ| target band drawn. Writes ``fig09_acquisition_rho.{png,pdf}``.
    """
    set_style()
    res = _load_results(cfg)
    before = float(res["phase1_baseline"]["pc1_black_border_rho"])
    after = float(res["results"]["coral_on"]["full"]["black_border_rho"]["eyepacs"]["rho"])
    target = float(res["targets"]["rho_abs_max"])

    labels = ["Phase 1\n(ImageNet)", "Phase 2\n(coral_on)"]
    values = [before, after]
    colors = [COLOR_BASELINE, COLOR_EYEPACS]

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    bars = ax.bar(labels, values, color=colors, width=0.55)
    ax.axhspan(-target, target, color="#2ca02c", alpha=0.12,
               label=f"target band |ρ| < {target:.2f}")
    ax.axhline(0, color="#888888", lw=0.8)
    for bar, value in zip(bars, values):
        va = "bottom" if value >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2,
                value + (0.01 if value >= 0 else -0.01),
                f"{value:+.3f}", ha="center", va=va, fontsize=11)
    ax.set_ylabel("PC1 ↔ black-border Spearman ρ (EyePACS)")
    ax.set_title("Acquisition correlation collapses into the target band")
    ax.legend(loc="best")
    lo = min(values + [-target]) - 0.05
    hi = max(values + [target]) + 0.05
    ax.set_ylim(lo, hi)
    return _save(fig, cfg, "fig09_acquisition_rho")


# ---------------------------------------------------------------- orchestration
def list_outputs(cfg: FiguresConfig) -> list[Path]:
    """Return the sorted list of files currently in ``figures_dir``."""
    if not cfg.figures_dir.exists():
        return []
    return sorted(p for p in cfg.figures_dir.iterdir() if p.is_file())
