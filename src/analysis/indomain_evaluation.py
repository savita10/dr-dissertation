"""Phase 4c — in-domain multi-task evaluation of the shared model on OLIVES.

Phase 4a showed that *cross-modality* zero-shot DR grading (EyePACS colour →
OLIVES near-IR) collapses. This phase instead evaluates the tasks the shared
encoder was actually **trained for on OLIVES near-IR** — the biomarker head and
the BCVA/CST regression head. Train and test are both OLIVES near-IR, so there is
**no modality jump**; this is a clean in-domain measurement against real labels.

Critical validity: evaluation is on the **held-out OLIVES test split only**
(patient-aware, seed 42, reconstructed with the Phase 2b split logic — the split
the model never trained on). The full OLIVES set is never used.

Expectation ranking (prior work): biomarkers strongest, CST moderate, BCVA weak.
Biomarker outputs are **feature-presence calls**, not a DR severity grade — no
biomarker→grade mapping is claimed.

Run:
    python -m src.analysis.indomain_evaluation --config configs/indomain_eval.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from torch.amp import autocast
from torch.utils.data import DataLoader

from src.analysis import figures
from src.analysis.uncertainty import load_full_model, read_merged_config
from src.data.dataloaders import collate_fn, compute_regression_stats
from src.data.datasets import USABLE_BIOMARKER_INDICES, OLIVESDataset
from src.data.olives_loader import BIOMARKER_COLUMNS
from src.data.samplers import patient_aware_split
from src.data.transforms import build_val_transform
from src.utils.config import load_config

# The biomarker-head's 9 outputs, in order (single source of truth).
BIOMARKER_NAMES = [BIOMARKER_COLUMNS[i] for i in USABLE_BIOMARKER_INDICES]
COLOR_OLIVES = figures.COLOR_OLIVES
COLOR_REF = "#009E73"  # benchmark reference line (Okabe-Ito bluish green)


# --------------------------------------------------------------- data / model
def build_olives_test_loader(
    data_cfg: DictConfig, batch_size: int, seed: int
) -> tuple[DataLoader, dict[str, Any], dict[str, float], dict[str, int]]:
    """Reconstruct the Phase 2b patient-aware OLIVES **test** split as a loader.

    Returns the loader, the raw OLIVES dict, the (train-computed) regression
    stats for de-standardising to real units, and a coverage dict (test size,
    patient count, tier-1/tier-2 counts). The full set is never evaluated.
    """
    olives = torch.load(
        str(data_cfg.paths.olives_file), map_location="cpu", weights_only=False
    )
    ratios = (
        float(data_cfg.split.train),
        float(data_cfg.split.val),
        float(data_cfg.split.test),
    )
    ol_train, _, ol_test = patient_aware_split(
        olives["patient_ids"], olives["has_clinical"], olives["has_biomarkers"],
        ratios, seed,
    )
    regression_stats = compute_regression_stats(
        olives, ol_train, str(data_cfg.paths.regression_stats)
    )

    idx = torch.tensor(ol_test, dtype=torch.long)
    pids = olives["patient_ids"][idx]
    coverage = {
        "n_test": int(len(ol_test)),
        "n_patients": int(torch.unique(pids[pids >= 0]).numel()),
        "n_tier2_biomarkers": int(olives["has_biomarkers"][idx].sum()),
        "n_tier1_clinical": int(olives["has_clinical"][idx].sum()),
        "split": "patient_aware_test",
        "seed": int(seed),
    }
    print(
        f"[split] OLIVES TEST split (seed={seed}): {coverage['n_test']} images, "
        f"{coverage['n_patients']} patients, tier1={coverage['n_tier1_clinical']}, "
        f"tier2={coverage['n_tier2_biomarkers']}"
    )

    val_tf = build_val_transform(data_cfg)
    test_ds = OLIVESDataset(olives, ol_test, val_tf, regression_stats)
    loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )
    return loader, olives, dict(regression_stats), coverage


@torch.no_grad()
def forward_collect(
    model: torch.nn.Module, loader: DataLoader, device: str
) -> dict[str, np.ndarray]:
    """Single forward pass over the test loader; collect head outputs + targets.

    Returns biomarker sigmoid probabilities [N,9] and standardised regression
    outputs [N,2], plus the (standardised) BCVA/CST targets and the tier masks.
    """
    use_amp = device.startswith("cuda")
    bio_probs, bio_true, bio_mask = [], [], []
    reg_pred, bcva_true, cst_true, clin_mask = [], [], [], []
    n_batches = len(loader)
    for i, batch in enumerate(loader, start=1):
        x = batch["images"].to(device, non_blocking=True)
        ctx = autocast("cuda") if use_amp else nullcontext()
        with ctx:
            out = model(x)
        bio_probs.append(torch.sigmoid(out["biomarker_logits"].float()).cpu())
        reg_pred.append(out["regression"].float().cpu())
        bio_true.append(batch["biomarkers"].float())
        bio_mask.append(batch["has_biomarkers"].bool())
        bcva_true.append(batch["bcva"].float())
        cst_true.append(batch["cst"].float())
        clin_mask.append(batch["has_clinical"].bool())
        if i % max(1, n_batches // 5) == 0 or i == n_batches:
            print(f"[forward] batch {i}/{n_batches}")

    return {
        "bio_probs": torch.cat(bio_probs).numpy(),
        "bio_true": torch.cat(bio_true).numpy(),
        "bio_mask": torch.cat(bio_mask).numpy(),
        "reg_pred": torch.cat(reg_pred).numpy(),
        "bcva_true_z": torch.cat(bcva_true).numpy(),
        "cst_true_z": torch.cat(cst_true).numpy(),
        "clin_mask": torch.cat(clin_mask).numpy(),
    }


# ------------------------------------------------------------------- metrics
def _boot_ci(
    fn: Callable[..., float], arrays: tuple[np.ndarray, ...], n_boot: int, seed: int,
    require_two_classes: bool = False,
) -> dict[str, Optional[float]]:
    """Percentile 95% CI of ``fn(*resampled_arrays)`` over paired bootstrap draws."""
    n = arrays[0].shape[0]
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        parts = tuple(a[idx] for a in arrays)
        if require_two_classes and len(np.unique(parts[0])) < 2:
            continue
        try:
            v = fn(*parts)
        except Exception:
            continue
        if v is not None and math.isfinite(v):
            vals.append(float(v))
    if not vals:
        return {"lo": None, "hi": None}
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"lo": float(lo), "hi": float(hi)}


def _auroc(y: np.ndarray, s: np.ndarray) -> Optional[float]:
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else None


def compute_biomarker_metrics(
    probs: np.ndarray, true: np.ndarray, mask: np.ndarray,
    n_boot: int, seed: int, f1_thr: float,
) -> dict[str, Any]:
    """Per-biomarker AUROC/AP/F1 (+CIs) on tier-2 test samples, plus macro/micro."""
    Y = true[mask]  # [n2, 9] in {0,1}
    P = probs[mask]  # [n2, 9]
    n2 = int(Y.shape[0])

    per: list[dict[str, Any]] = []
    aurocs: list[float] = []
    for j, name in enumerate(BIOMARKER_NAMES):
        yj, pj = Y[:, j], P[:, j]
        n_pos = int(yj.sum())
        auroc = _auroc(yj, pj)
        entry: dict[str, Any] = {
            "biomarker": name, "n": n2, "n_positives": n_pos,
            "auroc": auroc,
            "auroc_ci": _boot_ci(_auroc, (yj, pj), n_boot, seed + j,
                                 require_two_classes=True) if auroc is not None else {"lo": None, "hi": None},
            "average_precision": float(average_precision_score(yj, pj)) if n_pos > 0 else None,
            "f1": float(f1_score(yj, (pj >= f1_thr).astype(int), zero_division=0)),
        }
        if auroc is None:
            entry["note"] = f"AUROC undefined (n_positives={n_pos}, single class)"
        else:
            aurocs.append(auroc)
        per.append(entry)

    # Micro: pool all biomarker (label, score) pairs.
    y_flat, p_flat = Y.ravel(), P.ravel()

    def _macro_auroc(yy: np.ndarray, pp: np.ndarray) -> Optional[float]:
        vals = [_auroc(yy[:, j], pp[:, j]) for j in range(yy.shape[1])]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None

    macro = float(np.mean(aurocs)) if aurocs else None
    return {
        "n_tier2_test": n2,
        "per_biomarker": per,
        "macro_auroc": macro,
        "macro_auroc_ci": _boot_ci(_macro_auroc, (Y, P), n_boot, seed),
        "micro_auroc": _auroc(y_flat, p_flat),
        "micro_auroc_ci": _boot_ci(_auroc, (y_flat, p_flat), n_boot, seed,
                                   require_two_classes=True),
        "micro_average_precision": float(average_precision_score(y_flat, p_flat))
        if y_flat.sum() > 0 else None,
        "micro_f1": float(f1_score(y_flat, (p_flat >= f1_thr).astype(int), zero_division=0)),
    }


def compute_regression_metrics(
    pred_real: np.ndarray, true_real: np.ndarray, unit: str, n_boot: int, seed: int,
) -> dict[str, Any]:
    """R², MAE, RMSE (real units) with bootstrap CIs on valid paired samples."""
    n = int(pred_real.shape[0])
    if n < 2:
        return {"n": n, "note": "too few samples", "r2": None, "mae": None, "rmse": None}

    def _rmse(t: np.ndarray, p: np.ndarray) -> float:
        return float(math.sqrt(mean_squared_error(t, p)))

    return {
        "n": n, "unit": unit,
        "r2": float(r2_score(true_real, pred_real)),
        "r2_ci": _boot_ci(lambda t, p: r2_score(t, p), (true_real, pred_real), n_boot, seed),
        "mae": float(mean_absolute_error(true_real, pred_real)),
        "mae_ci": _boot_ci(lambda t, p: mean_absolute_error(t, p),
                           (true_real, pred_real), n_boot, seed),
        "rmse": _rmse(true_real, pred_real),
    }


def _destd(z: np.ndarray, mean: float, std: float) -> np.ndarray:
    return z * std + mean


def evaluate(
    collected: dict[str, np.ndarray], stats: dict[str, float], cfg: DictConfig,
) -> dict[str, Any]:
    """Assemble biomarker + CST + BCVA metrics from the collected head outputs."""
    n_boot = int(cfg.eval.bootstrap_n)
    seed = int(cfg.get("seed", 42))
    f1_thr = float(cfg.eval.f1_threshold)

    biomarkers = compute_biomarker_metrics(
        collected["bio_probs"], collected["bio_true"], collected["bio_mask"],
        n_boot, seed, f1_thr,
    )

    # Regression: de-standardise predictions and targets to real units.
    clin = collected["clin_mask"]
    bcva_pred = _destd(collected["reg_pred"][:, 0], stats["bcva_mean"], stats["bcva_std"])
    cst_pred = _destd(collected["reg_pred"][:, 1], stats["cst_mean"], stats["cst_std"])
    bcva_true = _destd(collected["bcva_true_z"], stats["bcva_mean"], stats["bcva_std"])
    cst_true = _destd(collected["cst_true_z"], stats["cst_mean"], stats["cst_std"])

    valid_bcva = clin & np.isfinite(bcva_true)
    valid_cst = clin & np.isfinite(cst_true)
    cst = compute_regression_metrics(
        cst_pred[valid_cst], cst_true[valid_cst], "µm", n_boot, seed
    )
    bcva = compute_regression_metrics(
        bcva_pred[valid_bcva], bcva_true[valid_bcva], "ETDRS letters", n_boot, seed
    )
    # Keep the paired points for the scatter figures.
    cst["points"] = {"pred": cst_pred[valid_cst].tolist(), "true": cst_true[valid_cst].tolist()}
    bcva["points"] = {"pred": bcva_pred[valid_bcva].tolist(), "true": bcva_true[valid_bcva].tolist()}
    return {"biomarkers": biomarkers, "cst": cst, "bcva": bcva}


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


def _short(name: str) -> str:
    return name if len(name) <= 22 else name[:20] + "…"


def fig_biomarker_auroc(
    res: dict[str, Any], benchmarks: dict[str, float], figures_dir: Path
) -> dict[str, Path]:
    """Figure 1 — per-biomarker AUROC with 95% CI whiskers; the 'what works' figure."""
    figures.set_style()
    per = res["biomarkers"]["per_biomarker"]
    names, vals, los, his, defined = [], [], [], [], []
    for e in per:
        names.append(f"{_short(e['biomarker'])}\n(n+={e['n_positives']})")
        if e["auroc"] is None:
            vals.append(0.0); los.append(0.0); his.append(0.0); defined.append(False)
        else:
            vals.append(e["auroc"])
            lo = e["auroc_ci"]["lo"] if e["auroc_ci"]["lo"] is not None else e["auroc"]
            hi = e["auroc_ci"]["hi"] if e["auroc_ci"]["hi"] is not None else e["auroc"]
            los.append(e["auroc"] - lo); his.append(hi - e["auroc"]); defined.append(True)

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(8, 1.0 * len(names)), 5.6))
    colors = [COLOR_OLIVES if d else "#CCCCCC" for d in defined]
    ax.bar(x, vals, color=colors, width=0.7)
    yerr = np.array([los, his])
    ax.errorbar(x[defined], np.array(vals)[defined], yerr=yerr[:, defined],
                fmt="none", ecolor="black", capsize=3, lw=1)
    ax.axhline(benchmarks["biomarker_auroc"], ls="--", color=COLOR_REF, lw=1.4,
               label=f"benchmark {benchmarks['biomarker_auroc']:.2f} (Kokilepersaud 2023)")
    ax.axhline(0.5, ls=":", color="#888888", lw=1.2, label="chance 0.50")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("AUROC (tier-2 OLIVES test)")
    ax.set_ylim(0, 1.0)
    macro = res["biomarkers"]["macro_auroc"]
    ax.set_title(f"Per-biomarker AUROC — macro {macro:.3f} (grey = undefined, too few positives)"
                 if macro is not None else "Per-biomarker AUROC")
    ax.legend(loc="lower right", fontsize=8)
    return _save_fig(fig, figures_dir, "phase4c_fig1_biomarker_auroc")


def _scatter(
    task: dict[str, Any], title: str, unit: str, color: str,
    figures_dir: Path, stem: str,
) -> dict[str, Path]:
    figures.set_style()
    pred = np.asarray(task["points"]["pred"], dtype=np.float64)
    true = np.asarray(task["points"]["true"], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    if pred.shape[0] > 0:
        lo = float(min(pred.min(), true.min()))
        hi = float(max(pred.max(), true.max()))
        ax.plot([lo, hi], [lo, hi], ls="--", color="#888888", lw=1.2, label="identity")
        ax.scatter(true, pred, s=22, alpha=0.6, color=color, edgecolors="white",
                   linewidths=0.4)
        ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(f"Actual {unit}")
    ax.set_ylabel(f"Predicted {unit}")
    r2 = task.get("r2")
    mae = task.get("mae")
    ax.set_title(f"{title}: R² = {_fmt(r2)}, MAE = {_fmt(mae, 1)} {unit} (n={task['n']})")
    ax.legend(loc="best")
    return _save_fig(fig, figures_dir, stem)


def fig_cst_scatter(res: dict[str, Any], figures_dir: Path) -> dict[str, Path]:
    """Figure 2 — CST predicted vs actual (real µm)."""
    return _scatter(res["cst"], "CST predicted vs actual", "µm", COLOR_OLIVES,
                    figures_dir, "phase4c_fig2_cst_scatter")


def fig_bcva_scatter(res: dict[str, Any], figures_dir: Path) -> dict[str, Path]:
    """Figure 3 — BCVA predicted vs actual (real ETDRS letters); framed honestly."""
    return _scatter(res["bcva"], "BCVA predicted vs actual", "ETDRS letters",
                    figures.COLOR_EYEPACS, figures_dir, "phase4c_fig3_bcva_scatter")


def fig_summary_table(
    res: dict[str, Any], benchmarks: dict[str, float], figures_dir: Path
) -> dict[str, Path]:
    """Figure 4 — summary table: each task vs metric vs benchmark, with n and CI."""
    figures.set_style()
    rows: list[list[str]] = []
    for e in res["biomarkers"]["per_biomarker"]:
        ci = e["auroc_ci"]
        auroc = (f"{e['auroc']:.3f} [{_fmt(ci['lo'])},{_fmt(ci['hi'])}]"
                 if e["auroc"] is not None else "n/a")
        rows.append([_short(e["biomarker"]), "AUROC", auroc,
                     f"{benchmarks['biomarker_auroc']:.2f}",
                     f"{e['n']} (+{e['n_positives']})"])
    bm = res["biomarkers"]
    rows.append(["MACRO biomarker", "AUROC",
                 _fmt(bm["macro_auroc"]), f"{benchmarks['biomarker_auroc']:.2f}",
                 str(bm["n_tier2_test"])])
    cst, bcva = res["cst"], res["bcva"]
    rows.append(["CST", "R² / MAE",
                 f"{_fmt(cst['r2'])} / {_fmt(cst.get('mae'), 1)} µm",
                 f"{benchmarks['cst_r2']:.2f}", str(cst["n"])])
    rows.append(["BCVA", "R² / MAE",
                 f"{_fmt(bcva['r2'])} / {_fmt(bcva.get('mae'), 1)} let.",
                 "weak (Sükei 2024)", str(bcva["n"])])

    fig, ax = plt.subplots(figsize=(11, 0.45 * len(rows) + 1.4))
    ax.axis("off")
    ax.set_title("Phase 4c — in-domain OLIVES test metrics (held-out, patient-aware)", pad=14)
    table = ax.table(
        cellText=rows,
        colLabels=["Task", "Metric", "Value [95% CI]", "Benchmark", "n (+pos)"],
        cellLoc="left", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.35)
    for (r, _c), cell in table.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        if r == 0:
            cell.set_facecolor("#F0F0F0")
            cell.set_text_props(fontweight="bold")
    return _save_fig(fig, figures_dir, "phase4c_fig4_summary_table")


# ---------------------------------------------------------------------- report
def _fmt(x: Any, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def _ci(d: Optional[dict[str, Any]]) -> str:
    if not d or d.get("lo") is None:
        return ""
    return f" [{_fmt(d['lo'])}, {_fmt(d['hi'])}]"


def build_report(
    res: dict[str, Any], coverage: dict[str, int], benchmarks: dict[str, float]
) -> str:
    bm = res["biomarkers"]
    cst, bcva = res["cst"], res["bcva"]
    lines = ["# Phase 4c — In-Domain Multi-Task Evaluation (OLIVES near-IR)", ""]
    lines.append(
        "These are **in-domain** results: the biomarker and BCVA/CST regression "
        "heads were trained on OLIVES near-IR and are here evaluated on the "
        "**held-out OLIVES test split** (patient-aware, seed 42 — the split the "
        "model never trained on). **No modality jump**, unlike the collapsed "
        "cross-modality DR-grade transfer (Phase 4a). Biomarker outputs are "
        "**feature-presence calls**, not a DR severity grade — no biomarker→grade "
        "mapping is claimed."
    )
    lines.append("")
    lines.append(
        f"**Test split:** {coverage['n_test']} images from "
        f"{coverage['n_patients']} patients · tier-1 (clinical) "
        f"{coverage['n_tier1_clinical']} · tier-2 (biomarkers) "
        f"{coverage['n_tier2_biomarkers']}. Small samples → **wide CIs**; treat as "
        "auxiliary evidence."
    )
    lines.append("")
    lines.append("## Biomarkers (primary) — tier-2 test")
    lines.append("")
    lines.append("| Biomarker | AUROC [95% CI] | AP | F1@0.5 | n (+pos) |")
    lines.append("|-----------|---------------:|---:|-------:|---------:|")
    for e in bm["per_biomarker"]:
        auroc = f"{_fmt(e['auroc'])}{_ci(e['auroc_ci'])}" if e["auroc"] is not None else "n/a"
        lines.append(
            f"| {e['biomarker']} | {auroc} | {_fmt(e['average_precision'])} | "
            f"{_fmt(e['f1'])} | {e['n']} (+{e['n_positives']}) |"
        )
    lines.append(
        f"\n- **Macro AUROC {_fmt(bm['macro_auroc'])}{_ci(bm['macro_auroc_ci'])}**, "
        f"micro AUROC {_fmt(bm['micro_auroc'])}{_ci(bm['micro_auroc_ci'])} "
        f"(benchmark ≈ {benchmarks['biomarker_auroc']:.2f}, Kokilepersaud et al. 2023)."
    )
    lines.append(
        "- Biomarkers with too few positives to score AUROC are marked n/a "
        "(n_positives shown); these are not failures, just unscoreable at this n."
    )
    lines.append("")
    lines.append("## CST (moderate) — tier-1 test")
    lines.append("")
    lines.append(
        f"- **R² {_fmt(cst['r2'])}{_ci(cst.get('r2_ci'))}**, "
        f"MAE {_fmt(cst.get('mae'), 1)}{_ci(cst.get('mae_ci'))} µm, "
        f"RMSE {_fmt(cst.get('rmse'), 1)} µm (n = {cst['n']}); "
        f"benchmark ≈ {benchmarks['cst_r2']:.2f} R² (Sükei et al. 2024)."
    )
    lines.append("")
    lines.append("## BCVA (expected weak) — tier-1 test")
    lines.append("")
    lines.append(
        f"- R² {_fmt(bcva['r2'])}{_ci(bcva.get('r2_ci'))}, "
        f"MAE {_fmt(bcva.get('mae'), 1)}{_ci(bcva.get('mae_ci'))} ETDRS letters "
        f"(n = {bcva['n']})."
    )
    lines.append(
        "- A weak or negative R² is **consistent with prior work** — BCVA is poorly "
        "predictable from fundus alone (Sükei et al. 2024). Reported honestly, not "
        "as a headline result."
    )
    lines.append("")
    lines.append("## Reading these results")
    lines.append("")
    lines.append(
        "The expected ranking is **biomarkers > CST > BCVA**. These in-domain "
        "numbers are the fair measure of what the shared encoder learned on OLIVES "
        "near-IR, and stand apart from the (negative) cross-modality DR-grade "
        "transfer. All metrics are in real units with bootstrapped 95% CIs; given "
        "the small tier-1/tier-2 test sizes, they are auxiliary evidence with wide "
        "intervals, not definitive benchmarks."
    )
    lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------- orchestrate
def run(cfg: DictConfig, device: str) -> dict[str, Any]:
    """End-to-end Phase 4c: load model, eval on held-out OLIVES test, write outputs."""
    report_dir = Path(str(cfg.report_dir))
    figures_dir = Path(str(cfg.figures_dir))
    report_dir.mkdir(parents=True, exist_ok=True)
    benchmarks = {
        "biomarker_auroc": float(cfg.benchmarks.biomarker_auroc),
        "cst_r2": float(cfg.benchmarks.cst_r2),
    }

    print("[1/6] loading frozen model + train-time config")
    model = load_full_model(cfg.checkpoint, device)
    data_cfg = read_merged_config(cfg.checkpoint)

    print("[2/6] reconstructing held-out OLIVES test split (patient-aware, seed 42)")
    loader, _olives, stats, coverage = build_olives_test_loader(
        data_cfg, int(cfg.eval.batch_size), int(cfg.get("seed", 42))
    )

    print("[3/6] forward pass over the test split (no TTA — point metrics)")
    collected = forward_collect(model, loader, device)

    print("[4/6] computing biomarker + CST + BCVA metrics (real units, bootstrap CIs)")
    res = evaluate(collected, stats, cfg)
    res["coverage"] = coverage
    res["benchmarks"] = benchmarks

    print("[5/6] rendering figures")
    fig_biomarker_auroc(res, benchmarks, figures_dir)
    fig_cst_scatter(res, figures_dir)
    fig_bcva_scatter(res, figures_dir)
    fig_summary_table(res, benchmarks, figures_dir)

    print("[6/6] writing JSON + report")
    # Drop the bulky scatter points from the JSON payload (kept only for figures).
    payload = json.loads(json.dumps(res))
    payload["cst"].pop("points", None)
    payload["bcva"].pop("points", None)
    (report_dir / "phase4c_indomain.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (report_dir / "phase4c_report.md").write_text(
        build_report(res, coverage, benchmarks), encoding="utf-8"
    )
    print(
        f"[done] macro biomarker AUROC={_fmt(res['biomarkers']['macro_auroc'])}, "
        f"CST R²={_fmt(res['cst']['r2'])}, BCVA R²={_fmt(res['bcva']['r2'])}"
    )
    return res


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 4c — in-domain multi-task evaluation on OLIVES near-IR."
    )
    parser.add_argument("--config", default="configs/indomain_eval.yaml")
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
        print("[phase4c] cuda requested but unavailable — falling back to cpu")
        device = "cpu"

    try:
        run(cfg, device=device)
    except Exception as exc:
        print(f"\n[ERROR] phase 4c evaluation failed: {exc}")
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
