"""Greyscale ablation — per-biomarker (categorical) concordance for the harmonised model.

Prompt 3 established that the harmonised model's predicted grade has genuine
*continuous* concordance with CST (ρ = 0.332, correct sign) and biomarker burden
(ρ = 0.486). This module runs the stronger *categorical* check: does the
harmonised predicted grade detect the presence of individual OLIVES biomarkers
(per-biomarker AUC), directly paralleling Phase 4b's per-label approach — and it
compares harmonised vs original throughout.

Per-biomarker labels are not in the predictions CSV; they are read from the raw
``olives_fundus.pt`` biomarkers tensor ([N, 16]), index-aligned to the CSV via the
``index`` column and masked to the tier-2 subset (``has_biomarkers``). Column
indices below index that full 16-wide tensor (verified via positive-rate check),
NOT the model's 9-column usable subset.

Reuses helpers from ``src.analysis.greyscale_concordance`` and
``dr_failure_characterisation`` by import — it modifies neither, nor any Phase 1-4
module. All outputs are NEW, under ``greyscale_experiment/``. Analysis only, CPU fine.

Run:
    python -m src.analysis.greyscale_categorical_concordance --config configs/greyscale_eval.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score, roc_curve

from src.analysis import figures
from src.analysis.dr_failure_characterisation import _fmt, _jsonable
from src.analysis.greyscale_concordance import (
    COLOR_HARMONISED,
    COLOR_ORIGINAL,
    CSV_NAME,
    _load_csv,
)
from src.utils.config import load_config

# ---- biomarker mapping over the raw [N,16] tensor (from the verified table) ----
BIOMARKER_NAMES = {
    0: "Atrophy/thinning", 1: "EZ disruption", 2: "DRIL", 3: "IR hemorrhages",
    4: "IR HRF", 5: "Partial vitreous face", 6: "Full vitreous face",
    7: "Preretinal tissue/hem", 8: "Vitreous debris", 9: "VMT", 10: "DRT/ME",
    11: "Fluid (IRF)", 12: "Fluid (SRF)", 13: "RPE disruption", 14: "PED (serous)",
    15: "SHRM",
}
# Expected tier-2 positive rates (%) for the STEP-1 cross-check.
EXPECTED_POS_RATE = {
    0: 8.9, 1: 35.4, 2: 2.6, 3: 34.9, 4: 100.0, 5: 57.3, 6: 69.8, 7: 25.0,
    8: 85.4, 9: 1.6, 10: 59.4, 11: 90.6, 12: 15.1, 13: 3.1, 14: 1.0, 15: 8.9,
}
EXCLUDED_COLS = (2, 4, 9, 13, 14)                        # too rare / saturated
ANALYSABLE_COLS = (0, 1, 3, 5, 6, 7, 8, 10, 11, 12, 15)  # 11 biomarkers
PRIMARY_COLS = (1, 3, 10, 12)                            # EZ, IR hem, DRT/ME, SRF
GRADE2_COLS = (10, 11, 12, 3)                            # item-3 confusion set
DME_ANY_COLS = (10, 11, 12)                              # DRT/ME OR IRF OR SRF
DME_MODSEV_COLS = (10, 12)                               # DRT/ME OR SRF (better balance)

GRADE2_THRESHOLD = 2
N_DR_CLASSES = 5
FORMATS = ("png", "pdf")


# --------------------------------------------------------------------- paths
def resolve_paths(cfg: DictConfig, args: argparse.Namespace) -> dict[str, Path]:
    report_dir = Path(str(cfg.report_dir))
    figures_dir = Path(str(cfg.figures_dir))
    greyscale_root = report_dir.parent
    dissertation_root = greyscale_root.parent
    return {
        "report_dir": report_dir,
        "figures_dir": figures_dir,
        "greyscale_root": greyscale_root,
        "harmonised_csv": report_dir / CSV_NAME,
        "original_csv": (
            Path(args.original_predictions) if args.original_predictions
            else dissertation_root / "reports" / CSV_NAME
        ),
        "olives_file": (
            Path(args.olives_file) if args.olives_file
            else dissertation_root / "preprocessed" / "olives" / "olives_fundus.pt"
        ),
        "prompt3_concordance": greyscale_root / "greyscale_concordance.json",
        "original_phase4b": dissertation_root / "reports" / "phase4b_failure.json",
        "out_json": greyscale_root / "greyscale_categorical_concordance.json",
        "out_report": greyscale_root / "greyscale_categorical_concordance_report.md",
    }


# ------------------------------------------------------------------ STEP 1
def _load_biomarkers(olives_file: Path) -> tuple[np.ndarray, np.ndarray, list]:
    """Return (biomarkers[N,16] float, has_biomarkers[N] bool, tensor keys)."""
    if not olives_file.exists():
        raise FileNotFoundError(f"OLIVES tensor not found: {olives_file}")
    olives = torch.load(olives_file, map_location="cpu", weights_only=False)
    bio = np.asarray(olives["biomarkers"], dtype=np.float64)
    has_bio = np.asarray(olives["has_biomarkers"]).astype(bool)
    return bio, has_bio, list(olives.keys())


def introspect(paths: dict[str, Path]) -> dict[str, Any]:
    """STEP 1 (mandatory): print CSV/tensor facts + confirm comparison inputs."""
    print("\n" + "=" * 72)
    print("STEP 1 — introspection (wire to actual columns; do not assume)")
    print("=" * 72)

    df = _load_csv(paths["harmonised_csv"], "harmonised")
    facts: dict[str, Any] = {
        "harmonised_csv": paths["harmonised_csv"].name,
        "n_rows": int(len(df)),
    }
    print(f"[step1] (1) predictions CSV: {paths['harmonised_csv'].name}  rows={len(df)}")
    if "tier" in df.columns:
        tb = df["tier"].value_counts().sort_index().to_dict()
        tb = {int(k): int(v) for k, v in tb.items()}
        facts["tier_breakdown"] = tb
        print(f"[step1]     tier breakdown (0/1/2): {tb}")

    bio, has_bio, keys = _load_biomarkers(paths["olives_file"])
    n_tier2 = int(has_bio.sum())
    facts.update({
        "olives_tensor_keys": keys,
        "biomarkers_shape": list(bio.shape),
        "has_biomarkers_count": n_tier2,
    })
    print(f"[step1] (2) OLIVES tensor keys: {keys}")
    print(f"[step1]     biomarkers shape: {bio.shape}  has_biomarkers count: {n_tier2}")

    print("[step1] (3) per-biomarker positive rate on the tier-2 subset "
          "(vs expected table):")
    tier2_bio = bio[has_bio]  # [n_tier2, 16]
    pos_rates: dict[int, float] = {}
    for col in range(16):
        rate = float(100.0 * (tier2_bio[:, col] > 0.5).mean())
        pos_rates[col] = rate
        exp = EXPECTED_POS_RATE[col]
        flag = "" if abs(rate - exp) <= 3.0 else "  <-- differs from table"
        tag = ("EXCLUDE" if col in EXCLUDED_COLS
               else "PRIMARY" if col in PRIMARY_COLS else "include")
        print(f"[step1]     col{col:>2} {BIOMARKER_NAMES[col]:<22} "
              f"{rate:5.1f}%  (exp {exp:5.1f}%) [{tag}]{flag}")
    facts["tier2_positive_rates"] = pos_rates
    facts["n_analysable_biomarkers"] = len(ANALYSABLE_COLS)

    p3 = paths["prompt3_concordance"]
    facts["prompt3_concordance_readable"] = p3.exists()
    print(f"[step1] (4) Prompt 3 concordance JSON readable: {p3.exists()}  ({p3})")
    op = paths["original_phase4b"]
    facts["original_phase4b_readable"] = op.exists()
    print(f"[step1] (5) original Phase 4b JSON readable: {op.exists()}  ({op})")
    oc = paths["original_csv"]
    facts["original_csv_readable"] = oc.exists()
    print(f"[step1]     original predictions CSV readable: {oc.exists()}  ({oc})")
    print("=" * 72 + "\n")
    return facts


# ------------------------------------------------------------------ core stats
def _score(df: pd.DataFrame) -> np.ndarray:
    """Continuous grade score for AUC (prefer pred_grade_mean, else pred_grade)."""
    if "pred_grade_mean" in df.columns and df["pred_grade_mean"].notna().any():
        return df["pred_grade_mean"].to_numpy(dtype=np.float64)
    return df["pred_grade"].to_numpy(dtype=np.float64)


def _auc_ci(
    score: np.ndarray, label: np.ndarray, n_boot: int, seed: int
) -> dict[str, Any]:
    """AUC + stratified bootstrap 95% CI (resample within pos/neg separately)."""
    label = label.astype(bool)
    n_pos = int(label.sum())
    n_neg = int((~label).sum())
    out: dict[str, Any] = {"n_pos": n_pos, "n_neg": n_neg, "auc": None,
                           "ci_lo": None, "ci_hi": None, "ci_excludes_half": False}
    if n_pos < 3 or n_neg < 3:
        out["note"] = "too few positives/negatives for AUC"
        return out
    out["auc"] = float(roc_auc_score(label, score))
    pos_idx = np.flatnonzero(label)
    neg_idx = np.flatnonzero(~label)
    rng = np.random.default_rng(seed)
    boots: list[float] = []
    for _ in range(n_boot):
        p = rng.choice(pos_idx, n_pos, replace=True)
        q = rng.choice(neg_idx, n_neg, replace=True)
        idx = np.concatenate([p, q])
        boots.append(roc_auc_score(label[idx], score[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    out.update({"ci_lo": float(lo), "ci_hi": float(hi),
                "ci_excludes_half": bool(lo > 0.5 or hi < 0.5)})
    return out


def _operating_points(score: np.ndarray, label: np.ndarray) -> dict[str, Any]:
    """Sensitivity @ 90% specificity + Youden-J optimal threshold."""
    label = label.astype(bool)
    if label.sum() < 3 or (~label).sum() < 3:
        return {"sens_at_90spec": None, "youden_threshold": None,
                "youden_sens": None, "youden_spec": None}
    fpr, tpr, thr = roc_curve(label, score)
    spec = 1.0 - fpr
    mask = spec >= 0.90
    sens90 = float(tpr[mask].max()) if mask.any() else None
    j = tpr - fpr
    ji = int(np.argmax(j))
    return {"sens_at_90spec": sens90, "youden_threshold": float(thr[ji]),
            "youden_sens": float(tpr[ji]), "youden_spec": float(spec[ji])}


def _grade2_confusion(grade: np.ndarray, label: np.ndarray) -> dict[str, Any]:
    """grade>=2 as the positive predictor; grade==0 specificity reported too."""
    label = label.astype(bool)
    pred_pos = grade >= GRADE2_THRESHOLD
    tp = int((pred_pos & label).sum())
    fn = int((~pred_pos & label).sum())
    fp = int((pred_pos & ~label).sum())
    tn = int((~pred_pos & ~label).sum())
    n_pos = tp + fn
    n_neg = fp + tn
    sens = tp / n_pos if n_pos else None
    spec = tn / n_neg if n_neg else None
    spec0 = float((grade[~label] == 0).mean()) if n_neg else None  # specificity of grade==0
    ppv = tp / (tp + fp) if (tp + fp) else None
    npv = tn / (tn + fn) if (tn + fn) else None
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "sens_grade_ge2": sens, "spec_grade_ge2": spec,
            "spec_grade_eq0": spec0, "ppv": ppv, "npv": npv,
            "n_pos": n_pos, "n_neg": n_neg}


def _labels_for(bio_tier2: np.ndarray, cols: tuple[int, ...]) -> np.ndarray:
    """Composite positive if ANY of the given columns is positive (>0.5)."""
    return (bio_tier2[:, list(cols)] > 0.5).any(axis=1)


# ------------------------------------------------------------------ analysis
def analyse_model(
    df: pd.DataFrame, bio: np.ndarray, has_bio: np.ndarray, seed: int, n_boot: int
) -> dict[str, Any]:
    """All categorical tests for one model, over its tier-2 predictions."""
    idx = df["index"].to_numpy()
    row_has_bio = has_bio[idx]
    t2 = df.loc[row_has_bio].reset_index(drop=True)
    t2_idx = idx[row_has_bio]
    bio_t2 = bio[t2_idx]  # [n_tier2, 16], aligned to t2 rows

    grade = t2["pred_grade"].to_numpy(dtype=np.int64)
    score = _score(t2)

    per_biomarker: dict[str, Any] = {}
    for col in ANALYSABLE_COLS:
        label = bio_t2[:, col] > 0.5
        auc = _auc_ci(score, label, n_boot, seed)
        ops = _operating_points(score, label)
        conf = _grade2_confusion(grade, label)
        per_biomarker[str(col)] = {
            "name": BIOMARKER_NAMES[col], "column": col,
            "primary": col in PRIMARY_COLS,
            "positive_rate": float((label).mean()),
            **auc, **ops, "grade2": conf,
        }

    composites: dict[str, Any] = {}
    for name, cols in (("any_dme", DME_ANY_COLS), ("moderate_severe", DME_MODSEV_COLS)):
        label = _labels_for(bio_t2, cols)
        composites[name] = {
            "columns": list(cols),
            "definition": " OR ".join(BIOMARKER_NAMES[c] for c in cols),
            "positive_rate": float(label.mean()),
            **_auc_ci(score, label, n_boot, seed),
            "grade2": _grade2_confusion(grade, label),
            "roc": _roc_points(label, score),
        }

    grade2_primary: dict[str, Any] = {}
    for col in GRADE2_COLS:
        label = bio_t2[:, col] > 0.5
        grade2_primary[str(col)] = {
            "name": BIOMARKER_NAMES[col], "column": col,
            **_grade2_confusion(grade, label),
        }

    return {
        "n_tier2": int(len(t2)),
        "score_column": "pred_grade_mean" if "pred_grade_mean" in t2.columns else "pred_grade",
        "per_biomarker": per_biomarker,
        "composites": composites,
        "grade2_primary": grade2_primary,
    }


def _roc_points(label: np.ndarray, score: np.ndarray) -> Optional[dict[str, list]]:
    label = label.astype(bool)
    if label.sum() < 3 or (~label).sum() < 3:
        return None
    fpr, tpr, _ = roc_curve(label, score)
    return {"fpr": [float(v) for v in fpr], "tpr": [float(v) for v in tpr]}


def patient_agreement(
    df: pd.DataFrame, has_bio: np.ndarray, seed: int, n_shuffle: int = 1000
) -> dict[str, Any]:
    """Within patient-eye consistency of predicted grade across tier-2 visits."""
    idx = df["index"].to_numpy()
    t2 = df.loc[has_bio[idx]].copy()
    t2 = t2[(t2["patient_id"] >= 0) & (t2["eye_id"] >= 0)]
    grades = t2["pred_grade"].to_numpy(dtype=np.float64)
    keys = list(zip(t2["patient_id"].to_numpy(), t2["eye_id"].to_numpy()))

    groups: dict[tuple, list[int]] = {}
    for pos, k in enumerate(keys):
        groups.setdefault(k, []).append(pos)
    multi = {k: v for k, v in groups.items() if len(v) >= 2}

    if not multi:
        return {"n_tier2_eyes": len(groups), "n_multi_visit_eyes": 0,
                "note": "no patient-eyes with >=2 tier-2 visits"}

    def mean_within_var(g: np.ndarray) -> float:
        return float(np.mean([np.var(g[v]) for v in multi.values()]))

    def frac_within_pm1(g: np.ndarray) -> float:
        return float(np.mean([(g[v].max() - g[v].min()) <= 1 for v in multi.values()]))

    obs_var = mean_within_var(grades)
    obs_pm1 = frac_within_pm1(grades)

    rng = np.random.default_rng(seed)
    shuffled = np.empty(n_shuffle, dtype=np.float64)
    for i in range(n_shuffle):
        perm = rng.permutation(grades)
        shuffled[i] = mean_within_var(perm)
    baseline = float(np.mean(shuffled))
    p_consistent = float(np.mean(shuffled <= obs_var))  # obs more consistent => low var

    return {
        "n_tier2_eyes": len(groups),
        "n_multi_visit_eyes": len(multi),
        "n_visits_total": int(len(t2)),
        "observed_mean_within_eye_variance": obs_var,
        "shuffled_baseline_mean_variance": baseline,
        "variance_ratio_obs_over_shuffled": (obs_var / baseline) if baseline else None,
        "p_more_consistent_than_chance": p_consistent,
        "frac_eyes_all_visits_within_pm1_grade": obs_pm1,
    }


# --------------------------------------------------------------------- figures
def _save_fig(fig: "plt.Figure", figures_dir: Path, stem: str) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    for ext in FORMATS:
        fig.savefig(figures_dir / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] wrote {stem}.{{png,pdf}}")


def fig_biomarker_auc(
    harm: dict[str, Any], orig: dict[str, Any], figures_dir: Path
) -> None:
    figures.set_style()
    cols = list(ANALYSABLE_COLS)
    names = [BIOMARKER_NAMES[c] + (" *" if c in PRIMARY_COLS else "") for c in cols]
    hb = [harm["per_biomarker"][str(c)] for c in cols]
    ob = [orig["per_biomarker"][str(c)] for c in cols]

    def vals(bb):
        auc = np.array([b["auc"] if b["auc"] is not None else np.nan for b in bb])
        lo = np.array([b["ci_lo"] if b["ci_lo"] is not None else np.nan for b in bb])
        hi = np.array([b["ci_hi"] if b["ci_hi"] is not None else np.nan for b in bb])
        return auc, np.abs(auc - lo), np.abs(hi - auc)

    h_auc, h_lo, h_hi = vals(hb)
    o_auc, o_lo, o_hi = vals(ob)
    y = np.arange(len(cols))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 0.62 * len(cols) + 1.6))
    ax.barh(y + w / 2, h_auc, w, xerr=[h_lo, h_hi], color=COLOR_HARMONISED,
            label="Harmonised", capsize=2, error_kw={"lw": 0.8})
    ax.barh(y - w / 2, o_auc, w, xerr=[o_lo, o_hi], color=COLOR_ORIGINAL,
            label="Original", capsize=2, error_kw={"lw": 0.8})
    ax.axvline(0.5, color="#B00020", ls="--", lw=1.2, label="chance (0.5)")
    ax.axvline(0.65, color="#2ca02c", ls=":", lw=1.0, label="0.65 target")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("AUC (predicted grade → biomarker presence)")
    ax.set_xlim(0.3, 1.0)
    ax.set_title("Per-biomarker AUC — harmonised vs original (* = primary DR-relevant)")
    ax.legend(loc="lower right", fontsize=9)
    _save_fig(fig, figures_dir, "greyscale_biomarker_auc_barchart")


def fig_composite_roc(
    harm: dict[str, Any], orig: dict[str, Any], figures_dir: Path
) -> None:
    figures.set_style()
    comps = [("any_dme", "Any DME (DRT/ME ∨ IRF ∨ SRF)"),
             ("moderate_severe", "Moderate–severe (DRT/ME ∨ SRF)")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    for ax, (key, title) in zip(axes, comps):
        for res, color, name in ((harm, COLOR_HARMONISED, "Harmonised"),
                                 (orig, COLOR_ORIGINAL, "Original")):
            c = res["composites"][key]
            roc = c.get("roc")
            auc = c.get("auc")
            if roc is None:
                continue
            ax.plot(roc["fpr"], roc["tpr"], color=color, lw=1.8,
                    label=f"{name} (AUC {_fmt(auc, 2)})")
        ax.plot([0, 1], [0, 1], color="#999999", ls="--", lw=1.0)
        ax.set_xlabel("False positive rate (1 − specificity)")
        ax.set_ylabel("True positive rate (sensitivity)")
        ax.set_title(title, fontsize=11)
        ax.legend(loc="lower right", fontsize=9)
    fig.suptitle("Composite pathology ROC — harmonised vs original", y=1.02, fontsize=13)
    _save_fig(fig, figures_dir, "greyscale_composite_roc")


def fig_confusion(harm: dict[str, Any], orig: dict[str, Any], figures_dir: Path) -> None:
    figures.set_style()
    cols = list(GRADE2_COLS)
    fig, axes = plt.subplots(2, len(cols), figsize=(3.0 * len(cols), 6.2))
    for r, (res, tag) in enumerate(((harm, "Harmonised"), (orig, "Original"))):
        for cpos, col in enumerate(cols):
            ax = axes[r, cpos]
            m = res["grade2_primary"][str(col)]
            mat = np.array([[m["tp"], m["fn"]], [m["fp"], m["tn"]]], dtype=float)
            ax.imshow(mat, cmap="Blues", vmin=0)
            for (i, j), val in np.ndenumerate(mat):
                ax.text(j, i, f"{int(val)}", ha="center", va="center",
                        color="black", fontsize=11, fontweight="bold")
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(["grade≥2", "grade<2"], fontsize=8)
            ax.set_yticklabels(["bm +", "bm −"], fontsize=8)
            sens = m["sens_grade_ge2"]
            title = BIOMARKER_NAMES[col] if r == 0 else ""
            ax.set_title(f"{title}\n{tag}: sens {_fmt(sens, 2)}", fontsize=9)
    fig.suptitle("grade≥2 confusion matrices — primary biomarkers (harmonised vs original)",
                 y=1.02, fontsize=12)
    _save_fig(fig, figures_dir, "greyscale_biomarker_confusion")


# ---------------------------------------------------------------------- verdict
def decide_verdict(harm: dict[str, Any]) -> dict[str, Any]:
    """categorical-recovered / aggregate-only / categorical-fails."""
    prim = [harm["per_biomarker"][str(c)] for c in PRIMARY_COLS]
    prim_auc = [p["auc"] for p in prim if p["auc"] is not None]
    prim_strong = [p for p in prim
                   if p["auc"] is not None and p["auc"] > 0.65 and p["ci_excludes_half"]]
    dme = harm["composites"]["any_dme"]["grade2"]
    dme_sens = dme["sens_grade_ge2"]

    n_strong = len(prim_strong)
    median_prim = float(np.median(prim_auc)) if prim_auc else None

    if n_strong >= 2 and dme_sens is not None and dme_sens > 0.60:
        label = "categorical-recovered"
        prose = (
            "Primary biomarkers show harmonised AUC > 0.65 with CIs excluding 0.5, "
            "and grade≥2 detects >60% of the DME composite. Individual pathology "
            "detection generalises across the colour→near-IR modality gap — a strong "
            "categorical result. Still indirect (no ground-truth grade on OLIVES)."
        )
    elif median_prim is not None and median_prim >= 0.55 and n_strong <= 1:
        label = "aggregate-only"
        prose = (
            "Per-biomarker AUCs sit mostly in the 0.55–0.65 band with wide CIs, "
            "while continuous concordance (Prompt 3) was strong. The model detects "
            "that SOME pathology is present but does not reliably identify specific "
            "pathology types — consistent with the 'grade 0 vs everything else' "
            "structure. The continuous signal rides largely on biomarker COUNT."
        )
    else:
        label = "categorical-fails"
        prose = (
            "Per-biomarker AUCs are mostly ≤ 0.55 with CIs including 0.5: the "
            "predicted grade does not detect individual pathologies. The Prompt-3 "
            "continuous concordance reflected aggregate correlation with biomarker "
            "burden, not specific pathology detection."
        )
    return {
        "label": label, "prose": prose,
        "n_primary_strong": n_strong,
        "primary_median_auc": median_prim,
        "dme_composite_grade2_sensitivity": dme_sens,
        "dme_composite_grade2_specificity": dme["spec_grade_ge2"],
    }


# ---------------------------------------------------------------------- report
def _auc_str(b: dict[str, Any]) -> str:
    if b.get("auc") is None:
        return "n/a"
    star = " *" if b.get("ci_excludes_half") else ""
    return f"{_fmt(b['auc'], 3)} [{_fmt(b['ci_lo'], 3)}, {_fmt(b['ci_hi'], 3)}]{star}"


def build_report(
    facts: dict[str, Any], harm: dict[str, Any], orig: dict[str, Any],
    patient: dict[str, Any], verdict: dict[str, Any],
) -> str:
    lines = ["# Greyscale ablation — per-biomarker (categorical) concordance", ""]
    lines.append(
        "Prompt 3 found strong *continuous* concordance (grade↔CST ρ = 0.332, "
        "grade↔burden ρ = 0.486). This is the stronger *categorical* test: can the "
        "harmonised predicted grade detect individual OLIVES biomarkers? Per-biomarker "
        "AUC on the tier-2 subset, harmonised vs original, with stratified bootstrap CIs."
    )
    lines.append("")
    lines.append(f"> ## Verdict: **{verdict['label']}**")
    lines.append(">")
    lines.append(f"> {verdict['prose']}")
    lines.append("")

    lines.append("## 1. Per-biomarker AUC (harmonised vs original)")
    lines.append("")
    lines.append("| Biomarker | Pos. rate | Harmonised AUC (95% CI) | Original AUC (95% CI) | Δ AUC |")
    lines.append("|-----------|----------:|-------------------------|-----------------------|------:|")
    for col in ANALYSABLE_COLS:
        h = harm["per_biomarker"][str(col)]
        o = orig["per_biomarker"][str(col)]
        star = " *" if col in PRIMARY_COLS else ""
        d = (h["auc"] - o["auc"]) if (h["auc"] is not None and o["auc"] is not None) else None
        lines.append(
            f"| {BIOMARKER_NAMES[col]}{star} | {h['positive_rate'] * 100:.1f}% | "
            f"{_auc_str(h)} | {_auc_str(o)} | {_fmt(d, 3)} |"
        )
    lines.append("")
    lines.append("`*` biomarker = primary DR-relevant; `*` on AUC = bootstrap CI excludes 0.5.")
    lines.append("")

    lines.append("## 2. Composite pathology detection")
    lines.append("")
    for key, title in (("any_dme", "Any DME (DRT/ME ∨ IRF ∨ SRF)"),
                       ("moderate_severe", "Moderate–severe (DRT/ME ∨ SRF)")):
        h = harm["composites"][key]
        o = orig["composites"][key]
        g = h["grade2"]
        lines.append(f"### {title} — positive rate {h['positive_rate'] * 100:.1f}%")
        lines.append("")
        lines.append(f"- AUC: harmonised {_auc_str(h)} vs original {_auc_str(o)}")
        lines.append(
            f"- grade≥2 operating point (harmonised): sensitivity "
            f"**{_fmt(g['sens_grade_ge2'], 3)}**, specificity {_fmt(g['spec_grade_ge2'], 3)}, "
            f"PPV {_fmt(g['ppv'], 3)}, NPV {_fmt(g['npv'], 3)}."
        )
        lines.append(
            f"- Specificity of grade=0 for excluding composite-positive: "
            f"{_fmt(g['spec_grade_eq0'], 3)}."
        )
        lines.append("")

    lines.append("## 3. grade≥2 operating point — primary biomarkers")
    lines.append("")
    lines.append("| Biomarker | Model | Sens (grade≥2) | Spec (grade=0) | TP | FN | FP | TN |")
    lines.append("|-----------|-------|---------------:|---------------:|---:|---:|---:|---:|")
    for col in GRADE2_COLS:
        for res, tag in ((harm, "harm"), (orig, "orig")):
            m = res["grade2_primary"][str(col)]
            lines.append(
                f"| {BIOMARKER_NAMES[col]} | {tag} | {_fmt(m['sens_grade_ge2'], 3)} | "
                f"{_fmt(m['spec_grade_eq0'], 3)} | {m['tp']} | {m['fn']} | {m['fp']} | {m['tn']} |"
            )
    lines.append("")

    lines.append("## 4. Patient-level agreement (multi-visit consistency)")
    lines.append("")
    if patient.get("n_multi_visit_eyes"):
        lines.append(
            f"- {patient['n_multi_visit_eyes']} patient-eyes with ≥2 tier-2 visits "
            f"(of {patient['n_tier2_eyes']} tier-2 eyes)."
        )
        lines.append(
            f"- Within-eye grade variance: observed **{_fmt(patient['observed_mean_within_eye_variance'], 3)}** "
            f"vs shuffled baseline {_fmt(patient['shuffled_baseline_mean_variance'], 3)} "
            f"(ratio {_fmt(patient['variance_ratio_obs_over_shuffled'], 3)}, "
            f"p_more_consistent = {_fmt(patient['p_more_consistent_than_chance'], 3)})."
        )
        lines.append(
            f"- **{patient['frac_eyes_all_visits_within_pm1_grade'] * 100:.0f}%** of "
            "patient-eyes have all visits within ±1 grade."
        )
    else:
        lines.append(f"_{patient.get('note', 'no multi-visit tier-2 eyes')}._")
    lines.append("")

    dme = harm["composites"]["any_dme"]["grade2"]
    lines.append("## Clinically-interpretable claim")
    lines.append("")
    lines.append(
        f"Screening the harmonised model at **grade ≥ 2** detects "
        f"**{_fmt((dme['sens_grade_ge2'] or 0) * 100, 0)}%** of DME-composite-positive eyes "
        f"at **{_fmt((dme['spec_grade_ge2'] or 0) * 100, 0)}%** specificity "
        f"(PPV {_fmt(dme['ppv'], 2)}, NPV {_fmt(dme['npv'], 2)})."
    )
    lines.append("")
    lines.append("## Framing (honest)")
    lines.append("")
    lines.append(
        "Tier-2 labels exist for only ~192 samples, so per-biomarker CIs are wide "
        "and imbalanced biomarkers (IRF 90.6%, vitreous debris 85.4%) carry little "
        "discriminative headroom. OLIVES has no ground-truth DR grade; positive "
        "categorical detection here is strong indirect evidence, not proof. The "
        "verdict rests on the primary-biomarker AUCs and the DME grade≥2 operating "
        "point above, not on the continuous concordance alone."
    )
    lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------- orchestrate
def run(cfg: DictConfig, args: argparse.Namespace) -> dict[str, Any]:
    paths = resolve_paths(cfg, args)
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    n_boot = int(args.bootstrap_n)

    facts = introspect(paths)

    for d in (paths["greyscale_root"], paths["figures_dir"]):
        d.mkdir(parents=True, exist_ok=True)
    print(f"[save-safety] output root: {paths['greyscale_root']} "
          f"(exists={paths['greyscale_root'].exists()}); figures: "
          f"{paths['figures_dir']} (exists={paths['figures_dir'].exists()})")

    print("[1/6] loading predictions + raw biomarker labels")
    harm_df = _load_csv(paths["harmonised_csv"], "harmonised")
    orig_df = _load_csv(paths["original_csv"], "original")
    bio, has_bio, _ = _load_biomarkers(paths["olives_file"])

    print("[2/6] per-biomarker + composite + grade≥2 analysis (harmonised)")
    harm = analyse_model(harm_df, bio, has_bio, seed, n_boot)
    print("[3/6] same for the original model")
    orig = analyse_model(orig_df, bio, has_bio, seed, n_boot)

    print("[4/6] patient-level agreement (harmonised)")
    patient = patient_agreement(harm_df, has_bio, seed)

    print("[5/6] rendering figures")
    fd = paths["figures_dir"]
    fig_biomarker_auc(harm, orig, fd)
    fig_composite_roc(harm, orig, fd)
    fig_confusion(harm, orig, fd)

    print("[6/6] verdict + writing JSON + report")
    verdict = decide_verdict(harm)
    out = _jsonable({
        "step1_introspection": facts,
        "config": {"seed": seed, "bootstrap_n": n_boot,
                   "analysable_cols": list(ANALYSABLE_COLS),
                   "primary_cols": list(PRIMARY_COLS),
                   "grade2_threshold": GRADE2_THRESHOLD},
        "harmonised": harm,
        "original": orig,
        "patient_agreement": patient,
        "verdict": verdict,
    })
    paths["out_json"].write_text(json.dumps(out, indent=2), encoding="utf-8")
    paths["out_report"].write_text(
        build_report(facts, harm, orig, patient, verdict), encoding="utf-8"
    )
    print(f"[done] wrote {paths['out_json']}")
    print(f"[done] wrote {paths['out_report']}")
    dme = harm["composites"]["any_dme"]["grade2"]
    print(f"[done] VERDICT = {verdict['label']} | primary strong AUCs = "
          f"{verdict['n_primary_strong']}/4 | DME grade≥2 sens = "
          f"{_fmt(dme['sens_grade_ge2'], 3)}, spec = {_fmt(dme['spec_grade_ge2'], 3)}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Greyscale ablation — per-biomarker categorical concordance."
    )
    parser.add_argument("--config", default="configs/greyscale_eval.yaml")
    parser.add_argument("--original-predictions", default=None)
    parser.add_argument("--olives-file", default=None)
    parser.add_argument("--bootstrap-n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except Exception as exc:
        print(f"[ERROR] failed to load config from {args.config}: {exc}")
        traceback.print_exc()
        return 2

    try:
        run(cfg, args)
    except Exception as exc:
        print(f"\n[ERROR] categorical concordance analysis failed: {exc}")
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
