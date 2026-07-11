"""Phase 3a — EyePACS calibration of the TTA confidence signal, and the gate.

Runs the dataset-agnostic TTA engine (``uncertainty.py``) over the held-out
EyePACS test split (real 5-class DR labels) and asks the one question Phase 4
depends on: **is the confidence signal trustworthy?** i.e. are confident
predictions actually more accurate than unconfident ones? If yes, Phase 4a may
apply the same TTA confidence to unlabelled OLIVES; if not, that is an honest,
reportable finding.

Computes accuracy / QWK (with bootstrap CIs), per-class P/R/F1, ECE + reliability
data, a selective-prediction curve, and a confidence-vs-correctness check;
writes ``phase3a_calibration.json`` + ``phase3a_report.md`` and five
colourblind-safe figures (PNG + PDF), reusing ``figures.set_style()``.

Honest caveat carried into the report: TTA characterises input-perturbation /
predictive uncertainty, not pure epistemic uncertainty.

Run:
    python -m src.analysis.calibration_eyepacs --config configs/uncertainty.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

from src.analysis import figures
from src.analysis.uncertainty import (
    NUM_DR_CLASSES,
    load_full_model,
    read_merged_config,
    run_tta_over_loader,
)
from src.data.dataloaders import collate_fn
from src.data.datasets import EyePACSDataset
from src.data.samplers import random_split
from src.data.transforms import build_train_transform, build_val_transform
from src.utils.config import load_config

TAG = "eyepacs_test"
COVERAGES = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)

# Correct/incorrect hues (Okabe-Ito bluish-green / vermillion — colourblind-safe).
COLOR_CORRECT = "#009E73"
COLOR_INCORRECT = "#D55E00"


# ------------------------------------------------------------------- utilities
def safe_qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Quadratic-weighted Cohen kappa; 0.0 when undefined (single class)."""
    if len(y_true) == 0:
        return 0.0
    try:
        value = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    except Exception:
        return 0.0
    return float(value) if value is not None and math.isfinite(value) else 0.0


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_pred == y_true).mean()) if len(y_true) else 0.0


def _bootstrap_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_resamples: int,
    seed: int,
) -> dict[str, float]:
    """Percentile 95% CI of ``metric_fn`` via ``n_resamples`` bootstrap draws."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        vals[i] = metric_fn(y_true[idx], y_pred[idx])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"lo": float(lo), "hi": float(hi), "std": float(vals.std())}


def _ece(
    conf: np.ndarray, correct: np.ndarray, n_bins: int
) -> tuple[float, list[dict[str, Any]]]:
    """Expected Calibration Error over ``n_bins`` equal-width confidence bins.

    Returns (ECE, per-bin rows) where each row carries the bin edges, count,
    mean confidence, and empirical accuracy (None for empty bins).
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(conf)
    ece = 0.0
    rows: list[dict[str, Any]] = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (conf > lo) & (conf <= hi) if b > 0 else (conf >= lo) & (conf <= hi)
        count = int(mask.sum())
        if count == 0:
            rows.append(
                {"bin_lo": float(lo), "bin_hi": float(hi), "count": 0,
                 "mean_conf": None, "accuracy": None}
            )
            continue
        mean_conf = float(conf[mask].mean())
        accuracy = float(correct[mask].mean())
        ece += (count / n) * abs(accuracy - mean_conf)
        rows.append(
            {"bin_lo": float(lo), "bin_hi": float(hi), "count": count,
             "mean_conf": mean_conf, "accuracy": accuracy}
        )
    return float(ece), rows


def _selective_curve(
    pred: np.ndarray, labels: np.ndarray, conf: np.ndarray
) -> list[dict[str, float]]:
    """Accuracy + QWK of the most-confident c% for each coverage in COVERAGES."""
    order = np.argsort(-conf)  # most confident first
    pred_s, lab_s = pred[order], labels[order]
    n = len(conf)
    curve: list[dict[str, float]] = []
    for c in COVERAGES:
        k = max(1, int(round(c * n)))
        curve.append(
            {"coverage": float(c), "n": int(k),
             "accuracy": _accuracy(lab_s[:k], pred_s[:k]),
             "qwk": safe_qwk(lab_s[:k], pred_s[:k])}
        )
    return curve


# -------------------------------------------------------------- data / loaders
def build_eyepacs_test_loader(
    data_cfg: DictConfig, batch_size: int
) -> tuple[DataLoader, EyePACSDataset, list[int]]:
    """Rebuild the exact Phase 2b EyePACS test split as an unaugmented loader.

    Uses the same ``random_split`` ratios/seed the checkpoint was trained under
    (recovered from its config), the deterministic val transform (identity — the
    tensors are already 224² and normalised), and the shared ``collate_fn``. TTA
    supplies its own augmentation on top, so the loader itself must be clean.
    """
    paths = data_cfg.paths
    ratios = (
        float(data_cfg.split.train),
        float(data_cfg.split.val),
        float(data_cfg.split.test),
    )
    seed = int(data_cfg.split.seed)

    # A throwaway full dataset gives the total count (and warms the shard cache).
    ep_full = EyePACSDataset(
        str(paths.eyepacs_dir), str(paths.eyepacs_cache), allowed_indices=None
    )
    _, _, ep_test = random_split(ep_full.total, ratios, seed)

    val_tf = build_val_transform(data_cfg)
    test_ds = EyePACSDataset(
        str(paths.eyepacs_dir),
        str(paths.eyepacs_cache),
        allowed_indices=ep_test,
        transform=val_tf,
    )
    loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )
    print(f"[data] EyePACS test loader: {len(test_ds)} images (seed={seed})")
    return loader, test_ds, list(ep_test)


# ------------------------------------------------------------------- metrics
def compute_calibration_metrics(
    pred: np.ndarray,
    labels: np.ndarray,
    conf: np.ndarray,
    agreement: np.ndarray,
    within_one: np.ndarray,
    cal_cfg: DictConfig,
) -> dict[str, Any]:
    """All EyePACS calibration numbers + the gate verdict, as a JSON-safe dict."""
    n_bins = int(cal_cfg.n_bins)
    boot_n = int(cal_cfg.bootstrap_n)
    seed = int(cal_cfg.seed)
    correct = (pred == labels).astype(np.float64)

    accuracy = _accuracy(labels, pred)
    qwk = safe_qwk(labels, pred)
    acc_ci = _bootstrap_ci(_accuracy, labels, pred, boot_n, seed)
    qwk_ci = _bootstrap_ci(safe_qwk, labels, pred, boot_n, seed)

    precision, recall, f1, support = precision_recall_fscore_support(
        labels, pred, labels=list(range(NUM_DR_CLASSES)), zero_division=0
    )
    per_class = [
        {"grade": c, "name": figures.DR_LABEL_NAMES[c],
         "precision": float(precision[c]), "recall": float(recall[c]),
         "f1": float(f1[c]), "support": int(support[c])}
        for c in range(NUM_DR_CLASSES)
    ]

    ece, reliability = _ece(conf, correct, n_bins)
    mean_conf = float(conf.mean())
    overconfidence = mean_conf - accuracy  # >0 overconfident, <0 underconfident
    if abs(overconfidence) < 0.05:
        calib_label = "well-calibrated"
    elif overconfidence > 0:
        calib_label = "overconfident"
    else:
        calib_label = "underconfident"

    selective = _selective_curve(pred, labels, conf)
    acc_full = selective[0]["accuracy"]
    acc_low_cov = selective[-1]["accuracy"]
    selective_accs = np.array([row["accuracy"] for row in selective])
    # AURC-style summary: mean risk (1-acc) as coverage falls; lower is better.
    aurc = float(np.mean(1.0 - selective_accs))

    conf_correct = float(conf[correct.astype(bool)].mean()) if correct.any() else float("nan")
    incorrect_mask = ~correct.astype(bool)
    conf_incorrect = float(conf[incorrect_mask].mean()) if incorrect_mask.any() else float("nan")

    # ---- the gate --------------------------------------------------------
    gate_confidence_separates = bool(
        math.isfinite(conf_correct) and math.isfinite(conf_incorrect)
        and conf_correct > conf_incorrect
    )
    gate_selective_rises = bool(acc_low_cov > acc_full)
    if gate_confidence_separates and gate_selective_rises:
        gate_verdict = "PASS"
    elif gate_confidence_separates or gate_selective_rises:
        gate_verdict = "WEAK"
    else:
        gate_verdict = "FAIL"

    return {
        "n_images": int(len(labels)),
        "accuracy": accuracy,
        "accuracy_ci95": acc_ci,
        "qwk": qwk,
        "qwk_ci95": qwk_ci,
        "per_class": per_class,
        "calibration": {
            "n_bins": n_bins,
            "ece": ece,
            "mean_confidence": mean_conf,
            "overconfidence": float(overconfidence),
            "label": calib_label,
            "reliability_bins": reliability,
        },
        "selective_prediction": {
            "curve": selective,
            "accuracy_full_coverage": float(acc_full),
            "accuracy_low_coverage": float(acc_low_cov),
            "aurc_style_mean_risk": aurc,
        },
        "confidence_vs_error": {
            "mean_confidence_correct": conf_correct,
            "mean_confidence_incorrect": conf_incorrect,
            "separation": float(conf_correct - conf_incorrect),
        },
        "ordinal": {
            "mean_agreement": float(agreement.mean()),
            "mean_within_one": float(within_one.mean()),
        },
        "gate": {
            "confidence_separates_correct_incorrect": gate_confidence_separates,
            "selective_accuracy_rises": gate_selective_rises,
            "verdict": gate_verdict,
        },
    }


# --------------------------------------------------------------------- figures
def _save_fig(fig: plt.Figure, figures_dir: Path, stem: str) -> dict[str, Path]:
    """Save PNG (dpi 200) + PDF to ``figures_dir`` and close the figure."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for ext in ("png", "pdf"):
        path = figures_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        out[ext] = path
    plt.close(fig)
    print(f"[figures] wrote {stem}: " + ", ".join(p.name for p in out.values()))
    return out


def fig_reliability(metrics: dict[str, Any], figures_dir: Path) -> dict[str, Path]:
    """Figure — reliability diagram (mean confidence vs empirical accuracy)."""
    figures.set_style()
    cal = metrics["calibration"]
    rows = [r for r in cal["reliability_bins"] if r["count"] > 0]
    xs = [r["mean_conf"] for r in rows]
    ys = [r["accuracy"] for r in rows]
    sizes = [max(20.0, 400.0 * r["count"] / metrics["n_images"]) for r in rows]

    fig, ax = plt.subplots(figsize=(6.5, 6.2))
    ax.plot([0, 1], [0, 1], ls="--", color="#888888", lw=1.2, label="perfect calibration")
    ax.scatter(xs, ys, s=sizes, color=figures.COLOR_EYEPACS, alpha=0.8,
               edgecolors="white", linewidths=0.5, zorder=5, label="confidence bins")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(
        f"Reliability — {cal['label']} (ECE = {cal['ece']:.3f})"
    )
    ax.annotate(
        f"mean conf {cal['mean_confidence']:.3f} vs acc {metrics['accuracy']:.3f}\n"
        f"({cal['overconfidence']:+.3f})",
        xy=(0.02, 0.98), xycoords="axes fraction", va="top", fontsize=9,
        bbox=dict(boxstyle="round", fc="white", ec="#CCCCCC"),
    )
    ax.legend(loc="lower right")
    return _save_fig(fig, figures_dir, "phase3a_fig1_reliability")


def fig_selective(metrics: dict[str, Any], figures_dir: Path) -> dict[str, Path]:
    """Figure — selective-prediction curve: accuracy & QWK vs coverage."""
    figures.set_style()
    curve = metrics["selective_prediction"]["curve"]
    cov = [row["coverage"] * 100 for row in curve]
    acc = [row["accuracy"] for row in curve]
    qwk = [row["qwk"] for row in curve]

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.plot(cov, acc, marker="o", ms=4, lw=1.8, color=figures.COLOR_EYEPACS,
            label="accuracy")
    ax.plot(cov, qwk, marker="s", ms=4, lw=1.8, color=figures.COLOR_OLIVES,
            label="QWK")
    ax.invert_xaxis()  # abstain the least-confident as we move right
    ax.set_xlabel("Coverage (%) — most-confident predictions retained")
    ax.set_ylabel("Retained-set metric")
    gate = metrics["gate"]
    ax.set_title(
        "Selective prediction: does accuracy rise as we abstain? "
        f"[{gate['verdict']}]"
    )
    ax.legend(loc="best")
    ax.annotate(
        f"acc {metrics['selective_prediction']['accuracy_full_coverage']:.3f} "
        f"(100%) → {metrics['selective_prediction']['accuracy_low_coverage']:.3f} "
        f"(10%)",
        xy=(0.5, -0.16), xycoords="axes fraction", ha="center",
        fontsize=8.5, color="#666666",
    )
    return _save_fig(fig, figures_dir, "phase3a_fig2_selective_prediction")


def fig_confidence_hist(
    conf: np.ndarray, pred: np.ndarray, labels: np.ndarray, figures_dir: Path
) -> dict[str, Path]:
    """Figure — confidence distribution for correct vs incorrect predictions."""
    figures.set_style()
    correct = pred == labels
    bins = np.linspace(0, 1, 26)

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.hist(conf[correct], bins=bins, color=COLOR_CORRECT, alpha=0.6,
            density=True, label=f"correct (n={int(correct.sum())})")
    ax.hist(conf[~correct], bins=bins, color=COLOR_INCORRECT, alpha=0.6,
            density=True, label=f"incorrect (n={int((~correct).sum())})")
    ax.axvline(float(conf[correct].mean()), color=COLOR_CORRECT, ls="--", lw=1.4)
    ax.axvline(float(conf[~correct].mean()), color=COLOR_INCORRECT, ls="--", lw=1.4)
    ax.set_xlabel("Confidence (max mean-softmax probability)")
    ax.set_ylabel("Density")
    ax.set_title("Confidence by correctness (correct should skew higher)")
    ax.legend(loc="best")
    return _save_fig(fig, figures_dir, "phase3a_fig3_confidence_by_correctness")


def fig_examples(
    test_ds: EyePACSDataset,
    pred: np.ndarray,
    labels: np.ndarray,
    conf: np.ndarray,
    figures_dir: Path,
    n_high: int = 3,
    n_low: int = 3,
) -> Optional[dict[str, Path]]:
    """Figure — example EyePACS test images with pred / true grade / confidence.

    Mixes the most-confident correct predictions with the least-confident cases;
    images are de-normalised for display. Best-effort: skips with a note if the
    image tensors cannot be read.
    """
    figures.set_style()
    correct = pred == labels
    correct_idx = np.where(correct)[0]
    high = correct_idx[np.argsort(-conf[correct_idx])[:n_high]] if len(correct_idx) else np.array([], int)
    low = np.argsort(conf)[:n_low]
    picks = list(high) + list(low)
    if not picks:
        print("[figures] fig_examples skipped — no predictions to show.")
        return None

    mean = figures.IMAGENET_MEAN.reshape(3, 1, 1)
    std = figures.IMAGENET_STD.reshape(3, 1, 1)
    try:
        imgs = []
        for i in picks:
            x = test_ds[int(i)]["images"].cpu().numpy()  # normalised [3,H,W]
            imgs.append(np.clip(x * std + mean, 0, 1).transpose(1, 2, 0))
    except (OSError, RuntimeError, ValueError, IndexError, MemoryError) as exc:
        print(f"[figures] fig_examples skipped — could not load images: {exc}")
        return None

    ncol = n_high + n_low
    fig, axes = plt.subplots(1, ncol, figsize=(2.5 * ncol, 3.0))
    for ax, i, img in zip(np.atleast_1d(axes), picks, imgs):
        ax.imshow(img)
        ax.axis("off")
        ok = pred[int(i)] == labels[int(i)]
        ax.set_title(
            f"pred {int(pred[int(i)])} / true {int(labels[int(i)])}\n"
            f"conf {conf[int(i)]:.2f}",
            fontsize=9, color=COLOR_CORRECT if ok else COLOR_INCORRECT,
        )
    fig.suptitle(
        "Example EyePACS test predictions (high-confidence correct → low-confidence)",
        y=1.05,
    )
    return _save_fig(fig, figures_dir, "phase3a_fig4_examples")


def fig_simple_vs_ordinal(
    agreement: np.ndarray, within_one: np.ndarray, figures_dir: Path
) -> dict[str, Path]:
    """Figure — category-agreement vs within-±1 ordinal agreement per prediction.

    Every point sits on or above the diagonal (within-one >= agreement by
    construction); the vertical gap is the mass of pass disagreements that are in
    fact ±1 near-misses, motivating the ordinal-aware confidence measures.
    """
    figures.set_style()
    fig, ax = plt.subplots(figsize=(6.5, 6.2))
    ax.plot([0, 1], [0, 1], ls="--", color="#888888", lw=1.2, label="equal")
    ax.scatter(agreement, within_one, s=8, alpha=0.25,
               color=figures.COLOR_EYEPACS, edgecolors="none")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_aspect("equal")
    ax.set_xlabel("Category agreement (fraction of passes == pred grade)")
    ax.set_ylabel("Within-±1 agreement (fraction within one grade)")
    ax.set_title(
        f"Many disagreements are ±1 near-misses "
        f"(mean {agreement.mean():.2f} → {within_one.mean():.2f})"
    )
    ax.legend(loc="lower right")
    return _save_fig(fig, figures_dir, "phase3a_fig5_simple_vs_ordinal")


# ---------------------------------------------------------------------- report
def _fmt_ci(ci: dict[str, float]) -> str:
    return f"[{ci['lo']:.4f}, {ci['hi']:.4f}]"


def build_report(metrics: dict[str, Any]) -> str:
    """Honest markdown interpretation of the calibration numbers + gate."""
    gate = metrics["gate"]
    cal = metrics["calibration"]
    cve = metrics["confidence_vs_error"]
    sp = metrics["selective_prediction"]

    lines = ["# Phase 3a — EyePACS TTA Calibration & Gate", ""]
    lines.append(
        "Uncertainty is obtained by **test-time augmentation (TTA)** on the frozen "
        "`phase2_coral_on_best.pt` — many augmented forward passes per image, no "
        "retraining, no architecture change. This report calibrates the confidence "
        "signal against the **real** EyePACS test labels before it is ever applied "
        "to unlabelled OLIVES in Phase 4."
    )
    lines.append("")
    lines.append("## Point performance")
    lines.append("")
    lines.append(f"- Images (EyePACS test): **{metrics['n_images']}**")
    lines.append(
        f"- Accuracy: **{metrics['accuracy']:.4f}** (95% CI {_fmt_ci(metrics['accuracy_ci95'])})"
    )
    lines.append(
        f"- Quadratic weighted kappa (QWK): **{metrics['qwk']:.4f}** "
        f"(95% CI {_fmt_ci(metrics['qwk_ci95'])})"
    )
    lines.append("")
    lines.append("| Grade | Name | Precision | Recall | F1 | Support |")
    lines.append("|------:|------|----------:|-------:|---:|--------:|")
    for pc in metrics["per_class"]:
        lines.append(
            f"| {pc['grade']} | {pc['name']} | {pc['precision']:.3f} | "
            f"{pc['recall']:.3f} | {pc['f1']:.3f} | {pc['support']} |"
        )
    lines.append("")
    lines.append("## Calibration")
    lines.append("")
    lines.append(
        f"- ECE ({cal['n_bins']} bins): **{cal['ece']:.4f}** — the model is "
        f"**{cal['label']}** (mean confidence {cal['mean_confidence']:.3f} vs "
        f"accuracy {metrics['accuracy']:.3f}, gap {cal['overconfidence']:+.3f})."
    )
    lines.append("")
    lines.append("## The gate — does confidence track accuracy?")
    lines.append("")
    lines.append(
        f"- Mean confidence of **correct** predictions: "
        f"{cve['mean_confidence_correct']:.4f}"
    )
    lines.append(
        f"- Mean confidence of **incorrect** predictions: "
        f"{cve['mean_confidence_incorrect']:.4f} "
        f"(separation {cve['separation']:+.4f})"
    )
    lines.append(
        f"- Selective accuracy: {sp['accuracy_full_coverage']:.4f} at 100% coverage "
        f"→ {sp['accuracy_low_coverage']:.4f} at 10% (most-confident) coverage; "
        f"AURC-style mean risk {sp['aurc_style_mean_risk']:.4f}."
    )
    lines.append("")
    checks = [
        ("Confident predictions are more accurate than unconfident ones",
         gate["confidence_separates_correct_incorrect"]),
        ("Selective accuracy rises as low-confidence cases are abstained",
         gate["selective_accuracy_rises"]),
    ]
    for text, ok in checks:
        lines.append(f"- {'✅' if ok else '❌'} {text}")
    lines.append("")
    verdict = gate["verdict"]
    if verdict == "PASS":
        lines.append(
            "> **GATE: PASS.** The confidence signal is real — confident TTA "
            "predictions are meaningfully more accurate. Phase 4a may use TTA "
            "confidence on OLIVES."
        )
    elif verdict == "WEAK":
        lines.append(
            "> **GATE: WEAK.** Only one of the two checks holds; the confidence "
            "signal is partial. Phase 4 should use it cautiously and report the "
            "weakness openly."
        )
    else:
        lines.append(
            "> **GATE: FAIL.** Confidence does not track accuracy on EyePACS "
            "(accuracy is roughly flat across confidence). This is an important, "
            "honest finding — do not rely on TTA confidence for OLIVES; report it."
        )
    lines.append("")
    lines.append("## Ordinal note")
    lines.append("")
    lines.append(
        f"Mean per-pass category agreement is {metrics['ordinal']['mean_agreement']:.3f}, "
        f"but mean within-±1 agreement is {metrics['ordinal']['mean_within_one']:.3f} — "
        "many pass 'disagreements' are ±1 near-misses, as expected for an ordinal "
        "grade (see fig 5)."
    )
    lines.append("")
    lines.append("## Caveat")
    lines.append("")
    lines.append(
        "> TTA characterises **input-perturbation / predictive** uncertainty — how "
        "stable the grade is under realistic image variation — **not pure epistemic** "
        "uncertainty (what the model fundamentally does not know). Confidence here "
        "should be read in that light."
    )
    lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------- orchestrate
def run_calibration(cfg: DictConfig, device: str) -> dict[str, Any]:
    """End-to-end Phase 3a: TTA over EyePACS test, calibrate, write outputs."""
    report_dir = Path(str(cfg.report_dir))
    figures_dir = Path(str(cfg.figures_dir))
    report_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] loading frozen model + train-time config")
    model = load_full_model(cfg.checkpoint, device)
    data_cfg = read_merged_config(cfg.checkpoint)

    print("[2/6] building EyePACS test loader + TTA augmentation")
    loader, test_ds, _ = build_eyepacs_test_loader(
        data_cfg, int(cfg.tta.batch_size)
    )
    augment_fn = build_train_transform(data_cfg)  # full training-strength TTA

    print("[3/6] running TTA over the EyePACS test split")
    tta = run_tta_over_loader(model, loader, cfg, device, augment_fn, TAG)
    if "labels" not in tta:
        raise RuntimeError("EyePACS TTA produced no labels — cannot calibrate.")

    pred = tta["pred_grade"].astype(np.int64)
    labels = tta["labels"].astype(np.int64)
    conf = tta["confidence_maxprob"].astype(np.float64)
    agreement = tta["confidence_agreement"].astype(np.float64)
    within_one = tta["within_one_frac"].astype(np.float64)

    print("[4/6] computing calibration metrics + gate")
    metrics = compute_calibration_metrics(
        pred, labels, conf, agreement, within_one, cfg.calibration
    )

    print("[5/6] rendering figures")
    fig_reliability(metrics, figures_dir)
    fig_selective(metrics, figures_dir)
    fig_confidence_hist(conf, pred, labels, figures_dir)
    fig_examples(test_ds, pred, labels, conf, figures_dir)
    fig_simple_vs_ordinal(agreement, within_one, figures_dir)

    print("[6/6] writing JSON + report")
    json_path = report_dir / "phase3a_calibration.json"
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report_path = report_dir / "phase3a_report.md"
    report_path.write_text(build_report(metrics), encoding="utf-8")
    print(f"[done] gate = {metrics['gate']['verdict']}")
    print(f"[done] wrote {json_path.name} and {report_path.name} to {report_dir}")
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 3a — EyePACS TTA calibration and the confidence gate."
    )
    parser.add_argument("--config", default="configs/uncertainty.yaml")
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
        print("[phase3a] cuda requested but unavailable — falling back to cpu")
        device = "cpu"

    try:
        run_calibration(cfg, device=device)
    except Exception as exc:
        print(f"\n[ERROR] phase 3a calibration failed: {exc}")
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
