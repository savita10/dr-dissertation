"""Phase 4a — zero-shot cross-modality DR grading of OLIVES (near-IR fundus).

The DR grader was trained on EyePACS *colour* fundus; OLIVES is *near-infrared*
fundus — a different imaging modality with no DR grade of its own. Phase 3a
validated (gate = PASS) that the TTA confidence signal tracks accuracy on
labelled EyePACS. This phase applies that **exact, unchanged** engine
(``uncertainty.py``) to all 3,142 OLIVES images to produce, per image, a
predicted grade + uncertainty.

This is the **prediction** step, not validation: OLIVES has no ground-truth DR
grade, so these grades are *unvalidated until Phase 4b* (indirect validation
against OLIVES clinical labels). Because DR grading leans partly on colour cues
that near-IR renders differently, whether the transfer is meaningful is an open
empirical question. A degenerate result (collapsed grade distribution, or much
lower confidence than EyePACS) is a **legitimate, reportable finding** — the key
early signal is whether the model reports lower confidence / higher entropy on
these out-of-modality images.

Run:
    python -m src.analysis.zero_shot_grading --config configs/zero_shot.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig
from scipy.stats import mannwhitneyu
from torch.utils.data import DataLoader

from src.analysis import figures
from src.analysis.uncertainty import (
    NUM_DR_CLASSES,
    load_full_model,
    read_merged_config,
    run_tta_over_loader,
)
from src.data.dataloaders import collate_fn
from src.data.datasets import USABLE_BIOMARKER_INDICES, OLIVESDataset
from src.data.transforms import build_train_transform, build_val_transform
from src.utils.config import load_config

TAG = "olives"
CACHE_NAME = "tta_olives.pt"
CSV_NAME = "olives_dr_predictions.csv"

# Per-image scalar columns for the Phase 4b input table (probs_mean stays only in
# the .pt cache — it is [N,5], not a scalar).
CSV_COLUMNS = (
    "index", "patient_id", "eye_id", "visit", "tier", "has_clinical",
    "has_biomarkers", "pred_grade", "pred_grade_mean", "confidence_agreement",
    "confidence_maxprob", "entropy", "entropy_norm", "grade_std",
    "within_one_frac", "bcva", "cst", "biomarker_burden",
)

# Correct/incorrect-style hues reused for the modality comparison overlays.
COLOR_EYEPACS = figures.COLOR_EYEPACS
COLOR_OLIVES = figures.COLOR_OLIVES


# ------------------------------------------------------------------- data
def build_olives_loader(
    data_cfg: DictConfig, batch_size: int
) -> tuple[DataLoader, dict[str, Any]]:
    """Build a loader over **all** OLIVES images (all tiers) + return the raw dict.

    Reuses the existing ``OLIVESDataset`` with the deterministic val transform
    (identity — TTA supplies its own augmentation on top). Identity clinical
    stats are passed because this phase reads raw BCVA/CST/eye_id/visit straight
    from the on-disk dict (index-aligned), not the dataset's z-scored copies.
    """
    olives = torch.load(
        str(data_cfg.paths.olives_file), map_location="cpu", weights_only=False
    )
    n = int(olives["images"].shape[0])
    identity_stats = {
        "bcva_mean": 0.0, "bcva_std": 1.0, "cst_mean": 0.0, "cst_std": 1.0
    }
    val_tf = build_val_transform(data_cfg)
    dataset = OLIVESDataset(olives, list(range(n)), val_tf, identity_stats)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )
    print(f"[data] OLIVES loader: {n} images (all tiers)")
    return loader, olives


# -------------------------------------------------------------- table assembly
def assemble_table(
    tta: dict[str, Any], olives: dict[str, Any]
) -> dict[str, np.ndarray]:
    """Merge the per-image TTA arrays with OLIVES metadata into one table.

    ``biomarker_burden`` = number of positive usable biomarkers (>0.5) per image
    where ``has_biomarkers``, else NaN. ``tier`` is 2 (biomarkers) / 1 (clinical)
    / 0 (fundus only). Raw (un-standardised) BCVA/CST are carried for Phase 4b.
    """
    n = int(olives["images"].shape[0])
    pred = tta["pred_grade"]
    if pred.shape[0] != n:
        raise RuntimeError(
            f"TTA rows ({pred.shape[0]}) != OLIVES images ({n}) — order mismatch."
        )

    has_clinical = olives["has_clinical"].numpy().astype(bool)
    has_bio = olives["has_biomarkers"].numpy().astype(bool)
    tier = np.where(has_bio, 2, np.where(has_clinical, 1, 0)).astype(np.int64)

    bio16 = olives["biomarkers"].numpy()  # [N,16], NaN where absent
    usable = bio16[:, USABLE_BIOMARKER_INDICES]
    burden = np.where(has_bio, (usable > 0.5).sum(axis=1), np.nan).astype(np.float64)

    metadata = olives["metadata"]
    visit = np.array([str(m.get("visit", "")) for m in metadata], dtype=object)

    table: dict[str, np.ndarray] = {
        "index": np.arange(n, dtype=np.int64),
        "patient_id": olives["patient_ids"].numpy().astype(np.int64),
        "eye_id": olives["eye_ids"].numpy().astype(np.int64),
        "visit": visit,
        "tier": tier,
        "has_clinical": has_clinical,
        "has_biomarkers": has_bio,
        "pred_grade": pred.astype(np.int64),
        "pred_grade_mean": tta["pred_grade_mean"].astype(np.float64),
        "confidence_agreement": tta["confidence_agreement"].astype(np.float64),
        "confidence_maxprob": tta["confidence_maxprob"].astype(np.float64),
        "entropy": tta["entropy"].astype(np.float64),
        "entropy_norm": tta["entropy_norm"].astype(np.float64),
        "grade_std": tta["grade_std"].astype(np.float64),
        "within_one_frac": tta["within_one_frac"].astype(np.float64),
        "bcva": olives["bcva"].numpy().astype(np.float64),
        "cst": olives["cst"].numpy().astype(np.float64),
        "biomarker_burden": burden,
    }
    return table


def _write_csv(table: dict[str, np.ndarray], path: Path) -> None:
    """Write the scalar per-image table to CSV in a stable column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = table["index"].shape[0]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_COLUMNS)
        for i in range(n):
            row = []
            for col in CSV_COLUMNS:
                v = table[col][i]
                if isinstance(v, (np.floating, float)) and np.isnan(v):
                    row.append("")
                elif isinstance(v, (np.bool_, bool)):
                    row.append(int(v))
                else:
                    row.append(v)
            writer.writerow(row)
    print(f"[table] wrote {n} rows -> {path}")


def save_predictions(
    tta: dict[str, Any], table: dict[str, np.ndarray], cfg: DictConfig
) -> tuple[Path, Path]:
    """Persist the enriched table to ``tta_olives.pt`` (with the TTA arrays) + CSV.

    ``tta_olives.pt`` keeps every original TTA key (so ``run_tta_over_loader``'s
    idempotent cache-hit still works next run) plus the metadata columns, giving
    Phase 4b a single clean artefact.
    """
    features_dir = Path(str(cfg.features_dir))
    report_dir = Path(str(cfg.report_dir))
    pt_path = features_dir / CACHE_NAME
    csv_path = report_dir / CSV_NAME

    enriched: dict[str, Any] = dict(tta)
    enriched.update(table)  # metadata columns win on any name overlap (none expected)
    tmp = pt_path.with_suffix(".pt.tmp")
    torch.save(enriched, tmp)
    tmp.replace(pt_path)
    print(f"[table] wrote enriched cache -> {pt_path}")

    _write_csv(table, csv_path)
    return pt_path, csv_path


# ------------------------------------------------------------------- summaries
def _dist(values: np.ndarray) -> dict[str, float]:
    return {"mean": float(np.mean(values)), "median": float(np.median(values))}


def compute_summary(
    table: dict[str, np.ndarray], eyepacs_cache: Optional[dict[str, Any]]
) -> dict[str, Any]:
    """Grade distribution, OLIVES confidence/entropy, and the EyePACS comparison."""
    pred = table["pred_grade"]
    conf = table["confidence_maxprob"]
    ent = table["entropy_norm"]
    n = int(pred.shape[0])

    counts = np.bincount(pred, minlength=NUM_DR_CLASSES).astype(int)
    grade_distribution = {
        f"grade_{g}": int(counts[g]) for g in range(NUM_DR_CLASSES)
    }
    n_distinct = int((counts > 0).sum())
    dominant_frac = float(counts.max() / n) if n else 0.0
    collapsed = bool(n_distinct <= 1 or dominant_frac >= 0.95)
    near_uniform_conf = bool(float(conf.std()) < 0.02)

    summary: dict[str, Any] = {
        "n_images": n,
        "grade_distribution": grade_distribution,
        "n_distinct_grades": n_distinct,
        "dominant_grade_fraction": dominant_frac,
        "confidence_maxprob": _dist(conf),
        "entropy_norm": _dist(ent),
        "flags": {
            "collapsed_distribution": collapsed,
            "near_uniform_confidence": near_uniform_conf,
        },
    }

    if eyepacs_cache is not None:
        eye_conf = np.asarray(eyepacs_cache["confidence_maxprob"], dtype=np.float64)
        eye_ent = np.asarray(eyepacs_cache["entropy_norm"], dtype=np.float64)
        # One-sided test: is OLIVES confidence systematically LOWER than EyePACS?
        u_stat, p_lower = mannwhitneyu(conf, eye_conf, alternative="less")
        conf_diff = float(conf.mean() - eye_conf.mean())
        ent_diff = float(ent.mean() - eye_ent.mean())
        summary["comparison"] = {
            "eyepacs_confidence_maxprob": _dist(eye_conf),
            "eyepacs_entropy_norm": _dist(eye_ent),
            "confidence_mean_diff_olives_minus_eyepacs": conf_diff,
            "entropy_mean_diff_olives_minus_eyepacs": ent_diff,
            "mannwhitney_u": float(u_stat),
            "p_olives_confidence_lower": float(p_lower),
            "olives_confidence_lower": bool(conf_diff < 0 and p_lower < 0.05),
        }
    else:
        summary["comparison"] = None
    return summary


# --------------------------------------------------------------------- figures
def _save_fig(fig: "plt.Figure", figures_dir: Path, stem: str) -> dict[str, Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for ext in ("png", "pdf"):
        path = figures_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        out[ext] = path
    plt.close(fig)
    print(f"[figures] wrote {stem}: " + ", ".join(p.name for p in out.values()))
    return out


def fig_grade_distribution(
    summary: dict[str, Any], figures_dir: Path
) -> dict[str, Path]:
    """Figure — OLIVES predicted-grade distribution (counts per grade 0–4)."""
    figures.set_style()
    counts = [summary["grade_distribution"][f"grade_{g}"] for g in range(NUM_DR_CLASSES)]
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    bars = ax.bar(range(NUM_DR_CLASSES), counts, color=figures.SEVERITY_COLORS, width=0.68)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, c, f"{c}", ha="center",
                va="bottom", fontsize=10)
    ax.set_xticks(range(NUM_DR_CLASSES))
    ax.set_xticklabels([f"{g}\n{figures.DR_LABEL_NAMES[g]}" for g in range(NUM_DR_CLASSES)])
    ax.set_xlabel("Predicted DR grade (zero-shot; unvalidated)")
    ax.set_ylabel("OLIVES image count")
    verdict = ("COLLAPSED" if summary["flags"]["collapsed_distribution"]
               else "plausibly spread")
    ax.set_title(f"OLIVES zero-shot predicted-grade distribution — {verdict}")
    ax.annotate(
        f"{summary['n_distinct_grades']}/5 grades used; "
        f"dominant grade = {summary['dominant_grade_fraction']:.0%}",
        xy=(0.5, -0.18), xycoords="axes fraction", ha="center",
        fontsize=8.5, color="#666666",
    )
    return _save_fig(fig, figures_dir, "phase4a_fig1_grade_distribution")


def _overlay_hist(
    ax: "plt.Axes", eye: np.ndarray, oli: np.ndarray, xlabel: str
) -> None:
    bins = np.linspace(
        float(min(eye.min(), oli.min())), float(max(eye.max(), oli.max())), 26
    )
    ax.hist(eye, bins=bins, color=COLOR_EYEPACS, alpha=0.55, density=True,
            label=f"EyePACS (n={eye.shape[0]})")
    ax.hist(oli, bins=bins, color=COLOR_OLIVES, alpha=0.55, density=True,
            label=f"OLIVES (n={oli.shape[0]})")
    ax.axvline(float(eye.mean()), color=COLOR_EYEPACS, ls="--", lw=1.4)
    ax.axvline(float(oli.mean()), color=COLOR_OLIVES, ls="--", lw=1.4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.legend(loc="best")


def fig_confidence_compare(
    table: dict[str, np.ndarray], eyepacs_cache: dict[str, Any], figures_dir: Path
) -> dict[str, Path]:
    """Figure — confidence (max-prob) EyePACS vs OLIVES; the key modality signal."""
    figures.set_style()
    oli = table["confidence_maxprob"]
    eye = np.asarray(eyepacs_cache["confidence_maxprob"], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    _overlay_hist(ax, eye, oli, "Confidence (max mean-softmax probability)")
    diff = float(oli.mean() - eye.mean())
    lower = "lower" if diff < 0 else "higher"
    ax.set_title(
        f"Confidence: EyePACS vs OLIVES (OLIVES {lower} by {abs(diff):.3f})"
    )
    return _save_fig(fig, figures_dir, "phase4a_fig2_confidence_compare")


def fig_entropy_compare(
    table: dict[str, np.ndarray], eyepacs_cache: dict[str, Any], figures_dir: Path
) -> dict[str, Path]:
    """Figure — normalised predictive entropy EyePACS vs OLIVES."""
    figures.set_style()
    oli = table["entropy_norm"]
    eye = np.asarray(eyepacs_cache["entropy_norm"], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    _overlay_hist(ax, eye, oli, "Normalised predictive entropy (÷ ln 5)")
    diff = float(oli.mean() - eye.mean())
    higher = "higher" if diff > 0 else "lower"
    ax.set_title(
        f"Entropy: EyePACS vs OLIVES (OLIVES {higher} by {abs(diff):.3f})"
    )
    return _save_fig(fig, figures_dir, "phase4a_fig3_entropy_compare")


def fig_examples(
    olives: dict[str, Any],
    table: dict[str, np.ndarray],
    figures_dir: Path,
    n_high: int = 3,
    n_low: int = 3,
) -> Optional[dict[str, Path]]:
    """Figure — example OLIVES near-IR images with predicted grade + confidence.

    Mixes the most- and least-confident predictions; annotates BCVA/CST where
    available. Labels clearly that there is **no ground-truth grade**. Best-effort:
    skips with a note if the image tensors cannot be read.
    """
    figures.set_style()
    conf = table["confidence_maxprob"]
    order = np.argsort(-conf)
    picks = list(order[:n_high]) + list(order[-n_low:])
    if not picks:
        print("[figures] fig_examples skipped — no predictions.")
        return None

    mean = figures.IMAGENET_MEAN.reshape(3, 1, 1)
    std = figures.IMAGENET_STD.reshape(3, 1, 1)
    try:
        imgs = olives["images"]
        views = [
            np.clip(imgs[int(i)].cpu().numpy() * std + mean, 0, 1).transpose(1, 2, 0)
            for i in picks
        ]
    except (OSError, RuntimeError, ValueError, IndexError, MemoryError) as exc:
        print(f"[figures] fig_examples skipped — could not load images: {exc}")
        return None

    ncol = n_high + n_low
    fig, axes = plt.subplots(1, ncol, figsize=(2.5 * ncol, 3.2))
    for ax, i, img in zip(np.atleast_1d(axes), picks, views):
        ax.imshow(img)
        ax.axis("off")
        i = int(i)
        clin = ""
        if bool(table["has_clinical"][i]) and not np.isnan(table["bcva"][i]):
            clin = f"\nBCVA {table['bcva'][i]:.0f} / CST {table['cst'][i]:.0f}"
        ax.set_title(
            f"pred {int(table['pred_grade'][i])} (no GT)\n"
            f"conf {conf[i]:.2f}{clin}",
            fontsize=9, color=figures.COLOR_OLIVES,
        )
    fig.suptitle(
        "Example OLIVES near-IR predictions — no ground-truth DR grade "
        "(high-confidence → low-confidence)",
        y=1.06,
    )
    return _save_fig(fig, figures_dir, "phase4a_fig4_examples")


# ---------------------------------------------------------------------- report
def build_report(summary: dict[str, Any]) -> str:
    """Cautious, honest markdown: predictions only, pending Phase 4b validation."""
    gd = summary["grade_distribution"]
    flags = summary["flags"]
    cmp = summary["comparison"]

    lines = ["# Phase 4a — Zero-Shot Cross-Modality DR Grading of OLIVES", ""]
    lines.append(
        "The DR grader was trained on EyePACS **colour** fundus; OLIVES is "
        "**near-IR** fundus with no DR grade of its own. Using the frozen "
        "`phase2_coral_on_best.pt` and the Phase-3a-validated TTA engine "
        "(30 augmented passes + 1 clean pass per image), this produces a "
        "predicted grade + uncertainty for all "
        f"{summary['n_images']} OLIVES images."
    )
    lines.append("")
    lines.append(
        "> **These predictions are NOT yet validated.** OLIVES has no ground-truth "
        "DR grade; Phase 4b tests whether they concord with OLIVES's clinical "
        "labels. Everything below is *predicted grades and their uncertainty, "
        "pending indirect validation* — not a claim that the grades are correct."
    )
    lines.append("")
    lines.append("## Predicted-grade distribution")
    lines.append("")
    lines.append("| Grade | Name | Count |")
    lines.append("|------:|------|------:|")
    for g in range(NUM_DR_CLASSES):
        lines.append(f"| {g} | {figures.DR_LABEL_NAMES[g]} | {gd[f'grade_{g}']} |")
    lines.append("")
    lines.append(
        f"- Distinct grades used: **{summary['n_distinct_grades']}/5**; "
        f"dominant grade fraction: **{summary['dominant_grade_fraction']:.1%}**."
    )
    if flags["collapsed_distribution"]:
        lines.append(
            "- ⚠️ **Collapsed distribution** — the predictions concentrate on one "
            "grade. This is a plausible out-of-modality failure mode and must be "
            "read as such, not as a genuine severity spread."
        )
    else:
        lines.append("- The distribution is plausibly spread across grades.")
    lines.append("")
    lines.append("## Uncertainty on OLIVES")
    lines.append("")
    lines.append(
        f"- Confidence (max-prob): mean **{summary['confidence_maxprob']['mean']:.3f}**, "
        f"median {summary['confidence_maxprob']['median']:.3f}."
    )
    lines.append(
        f"- Normalised entropy: mean **{summary['entropy_norm']['mean']:.3f}**, "
        f"median {summary['entropy_norm']['median']:.3f}."
    )
    if flags["near_uniform_confidence"]:
        lines.append(
            "- ⚠️ **Near-uniform confidence** (std < 0.02) — the confidence signal "
            "barely varies across OLIVES, limiting its discriminative value here."
        )
    lines.append("")
    lines.append("## EyePACS vs OLIVES confidence — the main early signal")
    lines.append("")
    if cmp is None:
        lines.append(
            "_EyePACS TTA cache not found — comparison skipped. Run Phase 3a first "
            "to produce `tta_eyepacs_test.pt`._"
        )
    else:
        lines.append(
            f"- EyePACS confidence mean {cmp['eyepacs_confidence_maxprob']['mean']:.3f} "
            f"vs OLIVES {summary['confidence_maxprob']['mean']:.3f} "
            f"(difference {cmp['confidence_mean_diff_olives_minus_eyepacs']:+.3f})."
        )
        lines.append(
            f"- EyePACS entropy mean {cmp['eyepacs_entropy_norm']['mean']:.3f} "
            f"vs OLIVES {summary['entropy_norm']['mean']:.3f} "
            f"(difference {cmp['entropy_mean_diff_olives_minus_eyepacs']:+.3f})."
        )
        lines.append(
            f"- Mann–Whitney one-sided (OLIVES < EyePACS confidence): "
            f"U = {cmp['mannwhitney_u']:.0f}, p = {cmp['p_olives_confidence_lower']:.2e}."
        )
        lines.append("")
        if cmp["olives_confidence_lower"]:
            lines.append(
                "> OLIVES confidence is **significantly lower** (and entropy higher) "
                "than EyePACS. Read charitably, this is the model **appropriately "
                "signalling that near-IR is out of its training modality** — a good, "
                "honest behaviour, not a failure. Phase 4b still decides whether the "
                "grades themselves carry usable signal."
            )
        else:
            lines.append(
                "> OLIVES confidence is **not** systematically lower than EyePACS. "
                "The model reports similar certainty on out-of-modality near-IR "
                "images — note this openly; similar confidence does not imply the "
                "grades are correct, and could equally reflect miscalibrated "
                "over-confidence out of modality."
            )
    lines.append("")
    lines.append("## What feeds Phase 4b")
    lines.append("")
    lines.append(
        "`features/tta_olives.pt` and `reports/olives_dr_predictions.csv` carry, per "
        "image: patient_id, eye_id, visit, predicted grade + all confidence/ordinal "
        "measures, raw BCVA/CST, biomarker_burden, and tier/mask flags — the direct "
        "input to the indirect validation in Phase 4b."
    )
    lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------- orchestrate
def run_zero_shot(cfg: DictConfig, device: str) -> dict[str, Any]:
    """End-to-end Phase 4a: TTA over OLIVES, assemble table, summarise, write."""
    report_dir = Path(str(cfg.report_dir))
    figures_dir = Path(str(cfg.figures_dir))
    report_dir.mkdir(parents=True, exist_ok=True)

    print("[1/7] loading frozen model + train-time config")
    model = load_full_model(cfg.checkpoint, device)
    data_cfg = read_merged_config(cfg.checkpoint)

    print("[2/7] building OLIVES loader (all tiers) + TTA augmentation")
    loader, olives = build_olives_loader(data_cfg, int(cfg.tta.batch_size))
    augment_fn = build_train_transform(data_cfg)  # same full-strength TTA as EyePACS

    print("[3/7] running TTA over OLIVES (idempotent cache)")
    tta = run_tta_over_loader(model, loader, cfg, device, augment_fn, TAG)

    print("[4/7] assembling the per-image prediction table")
    table = assemble_table(tta, olives)
    pt_path, csv_path = save_predictions(tta, table, cfg)

    print("[5/7] loading EyePACS TTA cache for the confidence comparison")
    eye_path = Path(str(cfg.eyepacs_tta_cache))
    eyepacs_cache = (
        torch.load(eye_path, map_location="cpu", weights_only=False)
        if eye_path.exists() else None
    )
    if eyepacs_cache is None:
        print(f"[warn] EyePACS cache not found at {eye_path} — comparison skipped")

    print("[6/7] computing summary + rendering figures")
    summary = compute_summary(table, eyepacs_cache)
    fig_grade_distribution(summary, figures_dir)
    if eyepacs_cache is not None:
        fig_confidence_compare(table, eyepacs_cache, figures_dir)
        fig_entropy_compare(table, eyepacs_cache, figures_dir)
    fig_examples(olives, table, figures_dir)

    print("[7/7] writing JSON summary + report")
    summary["artefacts"] = {"cache": str(pt_path), "csv": str(csv_path)}
    (report_dir / "phase4a_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (report_dir / "phase4a_report.md").write_text(
        build_report(summary), encoding="utf-8"
    )
    gd = summary["grade_distribution"]
    print(f"[done] OLIVES grade distribution: {gd}")
    if summary["comparison"] is not None:
        print(
            "[done] confidence diff (OLIVES - EyePACS): "
            f"{summary['comparison']['confidence_mean_diff_olives_minus_eyepacs']:+.3f}"
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 4a — zero-shot cross-modality DR grading of OLIVES."
    )
    parser.add_argument("--config", default="configs/zero_shot.yaml")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except Exception as exc:
        print(f"[ERROR] failed to load config from {args.config}: {exc}")
        traceback.print_exc()
        return 2

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[phase4a] cuda requested but unavailable — falling back to cpu")
        device = "cpu"

    try:
        run_zero_shot(cfg, device=device)
    except Exception as exc:
        print(f"\n[ERROR] phase 4a zero-shot grading failed: {exc}")
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
