"""Greyscale ablation — clinical-concordance decider for the harmonised model.

The input-harmonisation experiment (Design C1: EyePACS green channel →
histogram-matched to the OLIVES near-IR intensity distribution → 3-channel
replicate; OLIVES raw) *softened* the zero-shot collapse: the harmonised OLIVES
grade distribution spreads across grades (≈53% grade 0 vs the original 96%), at a
cost of only ~0.02 in-domain validation QWK. The open question this module
decides:

    Is the harmonised spread genuine, transferable DR-severity signal, or a NEW
    brightness-driven appearance artifact?

It mirrors Phase 4b (``src.analysis.dr_failure_characterisation``) but on the
harmonised predictions and, crucially, **compares original vs harmonised on every
test**. The verdict rests on two things, not on the softened distribution alone:
(1) the SIGN of grade↔CST concordance (genuine severity ⇒ higher grade with
higher CST ⇒ positive ρ; the original was ρ ≈ −0.492, the clinically wrong
direction), and (2) the clinical-vs-brightness head-to-head (does predicted grade
track clinical values more than image brightness, with clinical correctly signed).

Reuses the Phase 4b statistics/plot helpers by import — it does NOT modify
``dr_failure_characterisation.py`` or any Phase 1-4 module. All outputs are NEW,
under ``greyscale_experiment/``. Analysis only: no model inference, CPU fine.

Run:
    python -m src.analysis.greyscale_concordance --config configs/greyscale_eval.yaml
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
from omegaconf import DictConfig

from src.analysis import figures
from src.analysis.dr_failure_characterisation import (
    _fmt,
    _grade_boxplot,
    _jsonable,
    compute_brightness,
    mannwhitney,
    spearman_ci,
)
from src.utils.config import load_config

N_DR_CLASSES = 5
CSV_NAME = "olives_dr_predictions.csv"
HIGH_GRADE_MIN = 2   # the harmonised spread lives in grades 2–4
SEVERE_MIN = 3       # mirrors Phase 4b's severe(>=3) definition

# The original (colour model) reference, from Phase 4b (reports/phase4b_failure.json).
# Recomputed live from the original CSV below; this is only a labelling fallback.
ORIGINAL_GRADE_CST_RHO_REF = -0.492

COLOR_ORIGINAL = figures.COLOR_BASELINE  # neutral grey — the "before"/original model
COLOR_HARMONISED = figures.COLOR_EYEPACS  # blue — the harmonised model


# --------------------------------------------------------------------- paths
def resolve_paths(cfg: DictConfig, args: argparse.Namespace) -> dict[str, Path]:
    """Resolve all read/write paths from the eval config + optional overrides."""
    report_dir = Path(str(cfg.report_dir))                 # greyscale_experiment/reports
    figures_dir = Path(str(cfg.figures_dir))               # greyscale_experiment/figures
    greyscale_root = report_dir.parent                     # greyscale_experiment
    dissertation_root = greyscale_root.parent              # dissertation

    harmonised_csv = report_dir / CSV_NAME
    original_csv = (
        Path(args.original_predictions)
        if args.original_predictions
        else dissertation_root / "reports" / CSV_NAME
    )
    original_phase4b = (
        Path(args.original_phase4b)
        if args.original_phase4b
        else dissertation_root / "reports" / "phase4b_failure.json"
    )
    olives_file = (
        Path(args.olives_file)
        if args.olives_file
        else dissertation_root / "preprocessed" / "olives" / "olives_fundus.pt"
    )
    return {
        "report_dir": report_dir,
        "figures_dir": figures_dir,
        "greyscale_root": greyscale_root,
        "harmonised_csv": harmonised_csv,
        "original_csv": original_csv,
        "original_phase4b": original_phase4b,
        "olives_file": olives_file,
        "out_json": greyscale_root / "greyscale_concordance.json",
        "out_report": greyscale_root / "greyscale_concordance_report.md",
    }


def _load_csv(path: Path, tag: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{tag} predictions not found at {path} — "
            "run the corresponding zero_shot_grading eval first."
        )
    df = pd.read_csv(path).sort_values("index").reset_index(drop=True)
    print(f"[data] {tag}: {len(df)} predictions from {path}")
    return df


# ------------------------------------------------------------------ STEP 1
def introspect(paths: dict[str, Path]) -> dict[str, Any]:
    """STEP 1 (mandatory): print the harmonised schema + counts; confirm readables."""
    print("\n" + "=" * 72)
    print("STEP 1 — introspection (wire to actual columns; do not assume)")
    print("=" * 72)

    hp = paths["harmonised_csv"]
    df = _load_csv(hp, "harmonised")
    cols = list(df.columns)
    print(f"[step1] harmonised predictions file : {hp.name}")
    print(f"[step1] full path                   : {hp}")
    print(f"[step1] columns ({len(cols)})        : {cols}")
    print(f"[step1] row count                   : {len(df)}")

    facts: dict[str, Any] = {
        "harmonised_csv": str(hp),
        "columns": cols,
        "n_rows": int(len(df)),
    }
    for col in ("cst", "bcva", "biomarker_burden"):
        if col in df.columns:
            n_valid = int(df[col].notna().sum())
            print(f"[step1] non-NaN {col:<17}: {n_valid}")
            facts[f"n_valid_{col}"] = n_valid
        else:
            print(f"[step1] WARNING: column '{col}' absent from harmonised CSV")
            facts[f"n_valid_{col}"] = None

    if "pred_grade" in df.columns:
        counts = np.bincount(df["pred_grade"].to_numpy(np.int64), minlength=N_DR_CLASSES)
        n = int(len(df))
        dist = {int(g): int(counts[g]) for g in range(N_DR_CLASSES)}
        pct = {int(g): float(100.0 * counts[g] / n) for g in range(N_DR_CLASSES)}
        print(f"[step1] harmonised grade counts     : {dist}")
        print("[step1] harmonised grade %          : "
              + ", ".join(f"g{g}={pct[g]:.1f}%" for g in range(N_DR_CLASSES)))
        print(f"[step1] grade-0 share               : {pct[0]:.1f}%  (expect ~53%)")
        facts["harmonised_grade_counts"] = dist
        facts["harmonised_grade_pct"] = pct
        facts["harmonised_grade0_pct"] = pct[0]

    # Confirm comparison inputs are readable (read-only).
    oc = paths["original_csv"]
    facts["original_csv"] = str(oc)
    facts["original_csv_readable"] = oc.exists()
    print(f"[step1] original predictions readable: {oc.exists()}  ({oc})")
    op = paths["original_phase4b"]
    facts["original_phase4b"] = str(op)
    facts["original_phase4b_readable"] = op.exists()
    print(f"[step1] original phase4b json readable: {op.exists()}  ({op})")
    of = paths["olives_file"]
    facts["olives_file"] = str(of)
    facts["olives_file_readable"] = of.exists()
    print(f"[step1] OLIVES tensor (brightness)   : {of.exists()}  ({of})")
    print("=" * 72 + "\n")
    return facts


# ------------------------------------------------------------------ analysis
def _arr(df: pd.DataFrame, col: str) -> np.ndarray:
    return df[col].to_numpy(dtype=np.float64)


def concordance_block(
    df: pd.DataFrame, brightness: Optional[np.ndarray], seed: int, n_boot: int
) -> dict[str, Any]:
    """All concordance tests for one model's predictions (harmonised or original)."""
    grade = df["pred_grade"].to_numpy(dtype=np.int64)
    grade_mean = _arr(df, "pred_grade_mean")
    cst = _arr(df, "cst")
    bcva = _arr(df, "bcva")
    burden = _arr(df, "biomarker_burden")
    conf = _arr(df, "confidence_maxprob")
    n = int(len(df))

    bright = None
    if brightness is not None:
        bright = brightness[df["index"].to_numpy()]

    counts = np.bincount(grade, minlength=N_DR_CLASSES).astype(int)
    dist = {
        "counts": {int(g): int(counts[g]) for g in range(N_DR_CLASSES)},
        "pct": {int(g): float(100.0 * counts[g] / n) for g in range(N_DR_CLASSES)},
        "dominant_grade": int(counts.argmax()),
        "dominant_share": float(counts.max() / n),
        "n_distinct_grades": int((counts > 0).sum()),
    }

    tail = grade >= 1
    high = grade >= HIGH_GRADE_MIN
    severe = grade >= SEVERE_MIN
    g0 = grade == 0

    block: dict[str, Any] = {
        "n_images": n,
        "grade_distribution": dist,
        # -- clinical concordance (SIGN is the headline) --
        "grade_vs_cst_full": spearman_ci(grade, cst, n_boot, seed, "grade~CST (all)"),
        "grade_vs_cst_tail": spearman_ci(grade[tail], cst[tail], n_boot, seed, "grade~CST (tail>=1)"),
        "grademean_vs_cst_full": spearman_ci(grade_mean, cst, n_boot, seed, "grade_mean~CST (all)"),
        "grade_vs_burden": spearman_ci(grade, burden, n_boot, seed, "grade~biomarker_burden"),
        "grade_vs_bcva_full": spearman_ci(grade, bcva, n_boot, seed, "grade~BCVA (de-emphasised)"),
        # -- do "severe" eyes have worse clinical values? (median_a = high/severe) --
        "cst_high_vs_g0": mannwhitney(cst[high], cst[g0], f"CST: grade>={HIGH_GRADE_MIN} vs grade0"),
        "cst_severe_vs_g0": mannwhitney(cst[severe], cst[g0], f"CST: grade>={SEVERE_MIN} vs grade0"),
        "burden_high_vs_g0": mannwhitney(burden[high], burden[g0], f"burden: grade>={HIGH_GRADE_MIN} vs grade0"),
    }

    # -- competing hypothesis: brightness artifact --
    if bright is not None:
        block["grade_vs_brightness_full"] = spearman_ci(
            grade, bright, n_boot, seed, "grade~brightness (all)"
        )
        block["grade_vs_brightness_tail"] = spearman_ci(
            grade[tail], bright[tail], n_boot, seed, "grade~brightness (tail>=1)"
        )
        block["brightness_high_vs_g0"] = mannwhitney(
            bright[high], bright[g0], f"brightness: grade>={HIGH_GRADE_MIN} vs grade0"
        )
        block["brightness_available"] = True
    else:
        block["grade_vs_brightness_full"] = {"rho": None, "note": "brightness unavailable"}
        block["brightness_available"] = False

    # -- confidence-stratified concordance --
    thr = float(np.median(conf))
    hi = conf >= thr
    lo = ~hi
    block["confidence"] = {
        "threshold_median": thr,
        "n_confident": int(hi.sum()),
        "n_unconfident": int(lo.sum()),
        "confident_grade_vs_cst": spearman_ci(grade[hi], cst[hi], n_boot, seed, "grade~CST (confident)"),
        "unconfident_grade_vs_cst": spearman_ci(grade[lo], cst[lo], n_boot, seed, "grade~CST (unconfident)"),
        "confident_grade_vs_burden": spearman_ci(grade[hi], burden[hi], n_boot, seed, "grade~burden (confident)"),
        "unconfident_grade_vs_burden": spearman_ci(grade[lo], burden[lo], n_boot, seed, "grade~burden (unconfident)"),
    }
    return block


def _rho(entry: Optional[dict[str, Any]]) -> Optional[float]:
    return entry.get("rho") if isinstance(entry, dict) else None


def _sig(entry: Optional[dict[str, Any]]) -> bool:
    return bool(entry.get("ci_excludes_zero")) if isinstance(entry, dict) else False


def _correct_signed_strength(entry: Optional[dict[str, Any]]) -> float:
    """|ρ| if positive (correct severity sign), else 0."""
    r = _rho(entry)
    return float(r) if isinstance(r, (int, float)) and r is not None and r > 0 else 0.0


def decide_verdict(harm: dict[str, Any]) -> dict[str, Any]:
    """Recovered-signal / new-artifact / partial — from sign + head-to-head."""
    cst = harm["grade_vs_cst_full"]
    bur = harm["grade_vs_burden"]
    cst_rho = _rho(cst)
    cst_pos_sig = cst_rho is not None and cst_rho > 0 and _sig(cst)
    bur_rho = _rho(bur)
    bur_pos_sig = bur_rho is not None and bur_rho > 0 and _sig(bur)

    clin_correct_strength = max(_correct_signed_strength(cst), _correct_signed_strength(bur))
    bright = harm.get("grade_vs_brightness_full")
    bright_abs = abs(_rho(bright)) if _rho(bright) is not None else 0.0
    bright_sig = _sig(bright)

    mw = harm["cst_high_vs_g0"]
    severe_worse = (
        mw.get("median_a") is not None
        and mw.get("median_b") is not None
        and mw["median_a"] > mw["median_b"]
    )
    clinical_correct = cst_pos_sig or bur_pos_sig
    clinical_beats_brightness = clin_correct_strength >= bright_abs

    if clinical_correct and severe_worse and clinical_beats_brightness:
        label = "recovered-signal"
        prose = (
            "Harmonised predicted grade now concords with clinical severity in the "
            "CORRECT direction (higher grade ↔ higher CST / more biomarkers, CI "
            "excluding zero), the high-grade eyes genuinely have worse clinical "
            "values, and this clinical concordance exceeds any brightness "
            "association. Input harmonisation recovered transferable DR-severity "
            "signal. On a modality with no ground-truth grade this is strong "
            "INDIRECT evidence, not proof."
        )
    elif (not clinical_correct) and bright_sig and bright_abs >= clin_correct_strength and not severe_worse:
        label = "new-artifact"
        prose = (
            "Harmonised predicted grade still fails to concord with clinical "
            "severity in the correct direction, brightness remains at least as "
            "strong a driver, and high-grade eyes are not clinically worse. "
            "Harmonisation changed the failure mode (it spread the distribution) "
            "but did not recover genuine grading — the spread is a NEW "
            "appearance-driven artifact. Consistent with the modality gap being "
            "deeper than intensity (lesions render differently; Xiao et al. 2024)."
        )
    else:
        label = "partial"
        prose = (
            "Some correct-signed clinical concordance emerges but it is weak or "
            "brightness still co-dominates. Partial recovery: the softened "
            "distribution is not, on its own, evidence of grading — the sign and "
            "the clinical-vs-brightness head-to-head are only partly favourable. "
            "Report honestly as partial; clinical-label concordance (CST / "
            "biomarker burden) would need to strengthen before claiming recovery."
        )

    return {
        "label": label,
        "prose": prose,
        "grade_cst_rho": cst_rho,
        "grade_cst_positive": bool(cst_rho is not None and cst_rho > 0),
        "grade_cst_ci_excludes_zero": _sig(cst),
        "grade_burden_rho": bur_rho,
        "grade_burden_positive_sig": bool(bur_pos_sig),
        "clinical_correct_signed_strength": clin_correct_strength,
        "brightness_abs_rho": bright_abs,
        "brightness_ci_excludes_zero": bright_sig,
        "clinical_beats_brightness": bool(clinical_beats_brightness and clinical_correct),
        "high_grade_eyes_worse_cst": bool(severe_worse),
    }


def _cst_diff(mw: dict[str, Any]) -> Optional[float]:
    """median(high) − median(grade0); positive ⇒ high-grade eyes have higher CST."""
    a, b = mw.get("median_a"), mw.get("median_b")
    if a is None or b is None:
        return None
    return float(a - b)


def build_comparison(orig: dict[str, Any], harm: dict[str, Any]) -> dict[str, Any]:
    """Side-by-side original vs harmonised for each decisive test."""
    def pair(key: str) -> dict[str, Any]:
        o, h = orig.get(key), harm.get(key)
        return {
            "original_rho": _rho(o),
            "original_ci_excludes_zero": _sig(o),
            "harmonised_rho": _rho(h),
            "harmonised_ci_excludes_zero": _sig(h),
        }

    return {
        "grade_vs_cst_full": {**pair("grade_vs_cst_full"),
                              "sign_flip_to_positive": bool(
                                  (_rho(orig.get("grade_vs_cst_full")) or 0) < 0
                                  and (_rho(harm.get("grade_vs_cst_full")) or 0) > 0)},
        "grade_vs_cst_tail": pair("grade_vs_cst_tail"),
        "grade_vs_burden": pair("grade_vs_burden"),
        "grade_vs_brightness_full": pair("grade_vs_brightness_full"),
        "high_grade_vs_grade0_cst_diff": {
            "original": _cst_diff(orig["cst_high_vs_g0"]),
            "harmonised": _cst_diff(harm["cst_high_vs_g0"]),
            "note": f"median CST(grade>={HIGH_GRADE_MIN}) − median CST(grade0); "
                    "positive ⇒ high-grade eyes have higher (worse) CST",
        },
        "clinical_vs_brightness_headtohead": {
            "original": {
                "clinical_correct_signed_strength": max(
                    _correct_signed_strength(orig.get("grade_vs_cst_full")),
                    _correct_signed_strength(orig.get("grade_vs_burden")),
                ),
                "brightness_abs_rho": abs(_rho(orig.get("grade_vs_brightness_full")) or 0.0),
            },
            "harmonised": {
                "clinical_correct_signed_strength": max(
                    _correct_signed_strength(harm.get("grade_vs_cst_full")),
                    _correct_signed_strength(harm.get("grade_vs_burden")),
                ),
                "brightness_abs_rho": abs(_rho(harm.get("grade_vs_brightness_full")) or 0.0),
            },
        },
    }


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


def _sign_word(rho: Optional[float]) -> str:
    if rho is None:
        return "n/a"
    if rho > 0:
        return "POSITIVE (correct: higher grade ↔ higher CST)"
    if rho < 0:
        return "NEGATIVE (wrong sign)"
    return "zero"


def _compare_boxplots(
    orig_df: pd.DataFrame, harm_df: pd.DataFrame, values_col: str,
    orig_block: dict[str, Any], harm_block: dict[str, Any],
    rho_key: str, ylabel: str, suptitle: str, figures_dir: Path, stem: str,
    brightness: Optional[np.ndarray] = None, annotate_sign: bool = False,
) -> dict[str, Path]:
    figures.set_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
    for ax, df, block, name in (
        (axes[0], orig_df, orig_block, "Original (colour model)"),
        (axes[1], harm_df, harm_block, "Harmonised (Design C1)"),
    ):
        grade = df["pred_grade"].to_numpy(np.int64)
        if brightness is not None:
            values = brightness[df["index"].to_numpy()]
        else:
            values = df[values_col].to_numpy(np.float64)
        _grade_boxplot(ax, grade, values, ylabel)
        rho = _rho(block.get(rho_key))
        star = " *" if _sig(block.get(rho_key)) else ""
        title = f"{name}\nρ = {_fmt(rho)}{star}"
        if annotate_sign:
            title += f"\n{_sign_word(rho)}"
        ax.set_title(title, fontsize=11)
    fig.suptitle(suptitle, y=1.03, fontsize=13)
    return _save_fig(fig, figures_dir, stem)


def fig_rho_summary(
    orig: dict[str, Any], harm: dict[str, Any], figures_dir: Path
) -> dict[str, Path]:
    """Money figure: signed ρ (orig vs harm) + clinical-vs-brightness |ρ| both models."""
    figures.set_style()
    metrics = [
        ("grade_vs_cst_full", "grade~CST"),
        ("grade_vs_burden", "grade~burden"),
        ("grade_vs_brightness_full", "grade~brightness"),
    ]
    orig_vals = [(_rho(orig.get(k)) or 0.0) for k, _ in metrics]
    harm_vals = [(_rho(harm.get(k)) or 0.0) for k, _ in metrics]
    labels = [lbl for _, lbl in metrics]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))

    # Panel A — signed ρ, grouped by metric.
    x = np.arange(len(metrics))
    w = 0.38
    ax = axes[0]
    ax.bar(x - w / 2, orig_vals, w, color=COLOR_ORIGINAL, label="Original")
    ax.bar(x + w / 2, harm_vals, w, color=COLOR_HARMONISED, label="Harmonised")
    ax.axhline(0, color="#888888", lw=0.9)
    for xi, (ov, hv) in enumerate(zip(orig_vals, harm_vals)):
        ax.text(xi - w / 2, ov + (0.01 if ov >= 0 else -0.01), f"{ov:+.2f}",
                ha="center", va="bottom" if ov >= 0 else "top", fontsize=8)
        ax.text(xi + w / 2, hv + (0.01 if hv >= 0 else -0.01), f"{hv:+.2f}",
                ha="center", va="bottom" if hv >= 0 else "top", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Spearman ρ (signed)")
    ax.set_title("Signed ρ — sign is the point\n(clinical: + = correct severity direction)")
    ax.legend(loc="best")

    # Panel B — clinical (correct-signed) vs brightness |ρ|, both models.
    ax = axes[1]
    def clin(m: dict[str, Any]) -> float:
        return max(_correct_signed_strength(m.get("grade_vs_cst_full")),
                   _correct_signed_strength(m.get("grade_vs_burden")))
    def brt(m: dict[str, Any]) -> float:
        return abs(_rho(m.get("grade_vs_brightness_full")) or 0.0)
    groups = ["Original", "Harmonised"]
    clin_vals = [clin(orig), clin(harm)]
    brt_vals = [brt(orig), brt(harm)]
    xg = np.arange(len(groups))
    ax.bar(xg - w / 2, clin_vals, w, color=figures.COLOR_OLIVES, label="clinical |ρ| (correct sign)")
    ax.bar(xg + w / 2, brt_vals, w, color="#B00020", label="brightness |ρ|")
    for xi, (cv, bv) in enumerate(zip(clin_vals, brt_vals)):
        ax.text(xi - w / 2, cv, f"{cv:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(xi + w / 2, bv, f"{bv:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xg)
    ax.set_xticklabels(groups)
    ax.set_ylabel("|Spearman ρ| with predicted grade")
    ax.set_title("Head-to-head: clinical vs brightness\n(recovery ⇒ clinical ≥ brightness)")
    ax.legend(loc="best")

    fig.suptitle("Original vs harmonised — the deciding comparison", y=1.03, fontsize=13)
    return _save_fig(fig, figures_dir, "greyscale_concordance_fig4_rho_summary")


# ---------------------------------------------------------------------- report
def _rho_line(entry: Optional[dict[str, Any]]) -> str:
    if not isinstance(entry, dict) or entry.get("rho") is None:
        return "ρ = n/a"
    ci = f"[{_fmt(entry.get('ci_lo'))}, {_fmt(entry.get('ci_hi'))}]"
    star = " *" if entry.get("ci_excludes_zero") else ""
    return (f"ρ = {_fmt(entry['rho'])} (95% CI {ci}, p = {_fmt(entry.get('p'))}, "
            f"n = {entry.get('n')}){star}")


def build_report(
    facts: dict[str, Any], orig: dict[str, Any], harm: dict[str, Any],
    comparison: dict[str, Any], verdict: dict[str, Any],
    orig_phase4b: Optional[dict[str, Any]],
) -> str:
    lines = ["# Greyscale ablation — clinical-concordance decider (harmonised vs original)", ""]
    lines.append(
        "Design C1 softened the zero-shot collapse (harmonised OLIVES grade-0 share "
        f"**{facts.get('harmonised_grade0_pct', float('nan')):.1f}%** vs the original "
        "96%). This report decides whether that spread is **genuine transferable "
        "severity signal** or a **new brightness-driven artifact**. The verdict rests "
        "on the SIGN of clinical concordance and the clinical-vs-brightness "
        "head-to-head — not on the softened distribution alone."
    )
    lines.append("")
    lines.append(f"> ## Verdict: **{verdict['label']}**")
    lines.append(">")
    lines.append(f"> {verdict['prose']}")
    lines.append("")

    lines.append("## 1. Is the spread real severity signal? (clinical concordance)")
    lines.append("")
    lines.append(f"- Harmonised **grade ↔ CST (all)**: {_rho_line(harm['grade_vs_cst_full'])} "
                 f"→ sign **{_sign_word(_rho(harm['grade_vs_cst_full']))}**")
    lines.append(f"- Harmonised grade ↔ CST (tail≥1): {_rho_line(harm['grade_vs_cst_tail'])}")
    lines.append(f"- Harmonised **grade ↔ biomarker burden**: {_rho_line(harm['grade_vs_burden'])} "
                 "(positive expected if real)")
    lines.append(f"- Harmonised grade ↔ BCVA (de-emphasised): {_rho_line(harm['grade_vs_bcva_full'])}")
    mw = harm["cst_high_vs_g0"]
    lines.append(
        f"- CST, grade≥{HIGH_GRADE_MIN} vs grade-0: median "
        f"{_fmt(mw.get('median_a'), 1)} vs {_fmt(mw.get('median_b'), 1)} µm "
        f"(Mann–Whitney p = {_fmt(mw.get('p'))}, rank-biserial {_fmt(mw.get('rank_biserial'))}) — "
        f"high-grade eyes {'HAVE' if verdict['high_grade_eyes_worse_cst'] else 'do NOT have'} "
        "higher (worse) CST."
    )
    lines.append("")

    lines.append("## 2. Is it still a brightness artifact? (competing hypothesis)")
    lines.append("")
    if harm.get("brightness_available"):
        lines.append(f"- Harmonised **grade ↔ brightness (all)**: {_rho_line(harm['grade_vs_brightness_full'])}")
        lines.append(f"- Harmonised grade ↔ brightness (tail≥1): {_rho_line(harm.get('grade_vs_brightness_tail'))}")
        lines.append(
            f"- **Head-to-head (harmonised):** clinical |ρ| (correct sign) = "
            f"**{_fmt(verdict['clinical_correct_signed_strength'])}** vs brightness |ρ| = "
            f"**{_fmt(verdict['brightness_abs_rho'])}** → clinical "
            f"{'WINS' if verdict['clinical_beats_brightness'] else 'does NOT win'}."
        )
    else:
        lines.append("_Brightness unavailable (OLIVES tensor not found) — artifact test skipped._")
    lines.append("")

    lines.append("## 3. Confidence-stratified concordance")
    lines.append("")
    cf = harm["confidence"]
    lines.append(f"Split at median confidence ({_fmt(cf['threshold_median'])}); "
                 f"{cf['n_confident']} confident / {cf['n_unconfident']} unconfident.")
    lines.append(f"- Confident, grade ↔ CST: {_rho_line(cf['confident_grade_vs_cst'])}")
    lines.append(f"- Unconfident, grade ↔ CST: {_rho_line(cf['unconfident_grade_vs_cst'])}")
    lines.append(f"- Confident, grade ↔ burden: {_rho_line(cf['confident_grade_vs_burden'])}")
    lines.append("")

    lines.append("## 4. Original vs harmonised — the deciding comparison")
    lines.append("")
    lines.append("| Test | Original | Harmonised |")
    lines.append("|------|---------:|-----------:|")

    def row(name: str, key: str) -> str:
        o = _rho(orig.get(key))
        h = _rho(harm.get(key))
        os_ = " *" if _sig(orig.get(key)) else ""
        hs = " *" if _sig(harm.get(key)) else ""
        return f"| {name} | {_fmt(o)}{os_} | {_fmt(h)}{hs} |"

    lines.append(row("grade ↔ CST (all) ρ", "grade_vs_cst_full"))
    lines.append(row("grade ↔ CST (tail) ρ", "grade_vs_cst_tail"))
    lines.append(row("grade ↔ biomarker burden ρ", "grade_vs_burden"))
    lines.append(row("grade ↔ brightness (all) ρ", "grade_vs_brightness_full"))
    cd = comparison["high_grade_vs_grade0_cst_diff"]
    lines.append(f"| CST(grade≥{HIGH_GRADE_MIN}) − CST(grade0), µm | "
                 f"{_fmt(cd['original'], 1)} | {_fmt(cd['harmonised'], 1)} |")
    h2h = comparison["clinical_vs_brightness_headtohead"]
    lines.append(f"| clinical vs brightness \\|ρ\\| | "
                 f"{_fmt(h2h['original']['clinical_correct_signed_strength'])} vs "
                 f"{_fmt(h2h['original']['brightness_abs_rho'])} | "
                 f"{_fmt(h2h['harmonised']['clinical_correct_signed_strength'])} vs "
                 f"{_fmt(h2h['harmonised']['brightness_abs_rho'])} |")
    lines.append("")
    lines.append("`*` = bootstrap 95% CI excludes zero. Sign is decisive for the clinical "
                 "tests: genuine severity ⇒ **positive** grade↔CST. The original "
                 f"grade↔CST was ρ ≈ {ORIGINAL_GRADE_CST_RHO_REF} (wrong sign).")
    if comparison["grade_vs_cst_full"].get("sign_flip_to_positive"):
        lines.append("")
        lines.append("> **Harmonisation flipped grade↔CST from the wrong (negative) sign to the "
                     "correct (positive) direction.**")
    lines.append("")

    if orig_phase4b is not None:
        v = orig_phase4b.get("verdict", {})
        lines.append(f"_Original Phase 4b verdict (provenance): **{v.get('label', 'n/a')}** — "
                     f"{v.get('prose', '')}_")
        lines.append("")

    lines.append("## Framing (honest)")
    lines.append("")
    lines.append(
        "The ~0.02 in-domain QWK drop (0.615 → 0.595) confirms the model still grades "
        "colour fundus competently, so a genuine positive here would be credible. But "
        "the verdict rests on the concordance **sign** and the **clinical-vs-brightness** "
        "head-to-head above — a softened distribution alone is **not** validated grading. "
        "OLIVES has no ground-truth DR grade; a positive result is strong indirect "
        "evidence, not proof, and the natural next check is stronger clinical-label "
        "concordance (CST / biomarker burden), as in Phase 4b."
    )
    lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------- orchestrate
def run(cfg: DictConfig, args: argparse.Namespace) -> dict[str, Any]:
    paths = resolve_paths(cfg, args)
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    n_boot = int(args.bootstrap_n)

    # STEP 1 — introspect first.
    facts = introspect(paths)

    # Save-safety: create every output dir before writing anything.
    for d in (paths["greyscale_root"], paths["figures_dir"]):
        d.mkdir(parents=True, exist_ok=True)
    print(f"[save-safety] output root ready: {paths['greyscale_root']} "
          f"(exists={paths['greyscale_root'].exists()})")
    print(f"[save-safety] figures dir ready: {paths['figures_dir']} "
          f"(exists={paths['figures_dir'].exists()})")

    print("[1/6] loading harmonised + original predictions")
    harm_df = _load_csv(paths["harmonised_csv"], "harmonised")
    orig_df = _load_csv(paths["original_csv"], "original")

    print("[2/6] computing per-image brightness from the OLIVES tensor (index-aligned)")
    n_expected = int(max(len(harm_df), len(orig_df)))
    brightness = compute_brightness(paths["olives_file"], n_expected)

    print("[3/6] running concordance blocks (harmonised + original)")
    harm_block = concordance_block(harm_df, brightness, seed, n_boot)
    orig_block = concordance_block(orig_df, brightness, seed, n_boot)

    print("[4/6] deciding verdict + building the comparison")
    verdict = decide_verdict(harm_block)
    comparison = build_comparison(orig_block, harm_block)
    orig_phase4b = None
    if paths["original_phase4b"].exists():
        orig_phase4b = json.loads(paths["original_phase4b"].read_text(encoding="utf-8"))

    print("[5/6] rendering figures")
    fd = paths["figures_dir"]
    _compare_boxplots(
        orig_df, harm_df, "cst", orig_block, harm_block, "grade_vs_cst_full",
        "CST (µm, raw)", "Predicted grade vs CST — does CST rise with grade?",
        fd, "greyscale_concordance_fig1_grade_vs_cst", annotate_sign=True,
    )
    _compare_boxplots(
        orig_df, harm_df, "biomarker_burden", orig_block, harm_block, "grade_vs_burden",
        "Biomarker burden (count positive)", "Predicted grade vs biomarker burden",
        fd, "greyscale_concordance_fig2_grade_vs_burden",
    )
    if brightness is not None:
        _compare_boxplots(
            orig_df, harm_df, "cst", orig_block, harm_block, "grade_vs_brightness_full",
            "Mean image intensity (0–1)",
            "Predicted grade vs image brightness — the artifact test",
            fd, "greyscale_concordance_fig3_grade_vs_brightness", brightness=brightness,
        )
    fig_rho_summary(orig_block, harm_block, fd)

    print("[6/6] writing JSON + report")
    out = _jsonable({
        "step1_introspection": facts,
        "config": {"seed": seed, "bootstrap_n": n_boot,
                   "high_grade_min": HIGH_GRADE_MIN, "severe_min": SEVERE_MIN},
        "harmonised": harm_block,
        "original": orig_block,
        "comparison": comparison,
        "verdict": verdict,
        "original_phase4b_verdict": (orig_phase4b or {}).get("verdict") if orig_phase4b else None,
    })
    paths["out_json"].write_text(json.dumps(out, indent=2), encoding="utf-8")
    paths["out_report"].write_text(
        build_report(facts, orig_block, harm_block, comparison, verdict, orig_phase4b),
        encoding="utf-8",
    )
    print(f"[done] wrote {paths['out_json']}")
    print(f"[done] wrote {paths['out_report']}")
    print(f"[done] VERDICT = {verdict['label']} | "
          f"grade~CST ρ = {_fmt(verdict['grade_cst_rho'])} "
          f"({'positive' if verdict['grade_cst_positive'] else 'not positive'}), "
          f"brightness |ρ| = {_fmt(verdict['brightness_abs_rho'])}, "
          f"clinical beats brightness = {verdict['clinical_beats_brightness']}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Greyscale ablation — clinical-concordance decider (harmonised vs original)."
    )
    parser.add_argument("--config", default="configs/greyscale_eval.yaml")
    parser.add_argument("--original-predictions", default=None,
                        help="original OLIVES predictions CSV (default: <drive>/reports/olives_dr_predictions.csv)")
    parser.add_argument("--original-phase4b", default=None,
                        help="original Phase 4b JSON (default: <drive>/reports/phase4b_failure.json)")
    parser.add_argument("--olives-file", default=None,
                        help="olives_fundus.pt for brightness (default: <drive>/preprocessed/olives/olives_fundus.pt)")
    parser.add_argument("--bootstrap-n", type=int, default=1000)
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
        print(f"\n[ERROR] greyscale concordance analysis failed: {exc}")
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
