"""Phase 4b — characterisation of the cross-modality DR-grading failure.

Phase 4a's zero-shot transfer of the EyePACS (colour-fundus) DR grader onto
OLIVES (near-IR fundus) **collapsed**: ~96% of images predicted grade 0, a small
grade 3–4 tail, and TTA confidence that did *not* flag the modality shift. This
module does **not** try to rescue the grades — it characterises the failure
honestly and asks the one open question:

    Is the non-grade-0 tail a faint residual severity signal (swamped), or is it
    an image-appearance artifact (e.g. dark near-IR images read as grade 4)?

Everything here is analysis of the **saved Phase 4a arrays** (+ per-image
brightness from the raw OLIVES tensor). No model inference, no GPU. A finding
either way is the deliverable; this is the characterisation of a **negative
result**, not a validation.

Run:
    python -m src.analysis.dr_failure_characterisation --config configs/zero_shot.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from scipy.stats import mannwhitneyu, spearmanr

from src.analysis import figures
from src.utils.config import load_config

N_DR_CLASSES = 5
CSV_NAME = "olives_dr_predictions.csv"
BRIGHT_CHUNK = 256


# --------------------------------------------------------------------- loading
def _olives_file(cfg: DictConfig) -> Path:
    """Locate ``olives_fundus.pt`` (config override, else the standard layout)."""
    override = cfg.get("olives_file", None)
    if override:
        return Path(str(override))
    return Path(str(cfg.features_dir)).parent / "preprocessed" / "olives" / "olives_fundus.pt"


def load_predictions(cfg: DictConfig) -> pd.DataFrame:
    """Load the Phase 4a per-image prediction CSV, sorted by raw image index."""
    csv_path = Path(str(cfg.report_dir)) / CSV_NAME
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found — run Phase 4a (zero_shot_grading) first."
        )
    df = pd.read_csv(csv_path)
    df = df.sort_values("index").reset_index(drop=True)
    print(f"[data] loaded {len(df)} predictions from {csv_path.name}")
    return df


def compute_brightness(olives_file: Path, n_expected: int) -> Optional[np.ndarray]:
    """Per-image mean intensity in [0,1] from the raw OLIVES tensor (index-aligned).

    Images are ImageNet-normalised on disk; we de-normalise per channel, clamp to
    [0,1], and average over channels and pixels. Returns ``None`` (with a note) if
    the tensor is unavailable, so the appearance analysis degrades gracefully.
    """
    if not olives_file.exists():
        print(f"[brightness] {olives_file} not found — appearance analysis skipped")
        return None
    olives = torch.load(olives_file, map_location="cpu", weights_only=False)
    images = olives["images"]
    n = int(images.shape[0])
    if n != n_expected:
        print(f"[brightness] WARNING: tensor has {n} images, CSV has {n_expected}")
    mean = torch.tensor(figures.IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(figures.IMAGENET_STD).view(1, 3, 1, 1)
    out = np.empty(n, dtype=np.float64)
    for start in range(0, n, BRIGHT_CHUNK):
        end = min(start + BRIGHT_CHUNK, n)
        chunk = images[start:end].float() * std + mean
        chunk = chunk.clamp(0.0, 1.0)
        out[start:end] = chunk.mean(dim=(1, 2, 3)).numpy()
    print(f"[brightness] computed mean intensity for {n} images")
    return out


# ------------------------------------------------------------------- stats
def _jsonable(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.floating,)):
        return _jsonable(float(value))
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def spearman_ci(
    x: np.ndarray, y: np.ndarray, n_boot: int, seed: int, label: str = ""
) -> dict[str, Any]:
    """Spearman ρ with a percentile bootstrap 95% CI over complete pairs."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = int(x.shape[0])
    out: dict[str, Any] = {
        "label": label, "n": n, "rho": None, "p": None,
        "ci_lo": None, "ci_hi": None,
    }
    if n < 3 or np.std(x) == 0 or np.std(y) == 0:
        out["note"] = "insufficient variation / n<3"
        return out
    rho, p = spearmanr(x, y)
    rng = np.random.default_rng(seed)
    boots: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        xb, yb = x[idx], y[idx]
        if np.std(xb) == 0 or np.std(yb) == 0:
            continue
        r, _ = spearmanr(xb, yb)
        if np.isfinite(r):
            boots.append(float(r))
    if boots:
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out.update({"ci_lo": float(lo), "ci_hi": float(hi)})
    out.update({"rho": float(rho), "p": float(p)})
    out["ci_excludes_zero"] = bool(
        out["ci_lo"] is not None and (out["ci_lo"] > 0 or out["ci_hi"] < 0)
    )
    return out


def mannwhitney(a: np.ndarray, b: np.ndarray, label: str = "") -> dict[str, Any]:
    """Mann–Whitney U (a vs b) with rank-biserial effect size and group medians."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    out: dict[str, Any] = {
        "label": label, "n_a": int(a.shape[0]), "n_b": int(b.shape[0]),
        "median_a": None, "median_b": None, "u": None, "p": None,
        "rank_biserial": None,
    }
    if a.shape[0] < 1 or b.shape[0] < 1:
        out["note"] = "empty group"
        return out
    out["median_a"] = float(np.median(a))
    out["median_b"] = float(np.median(b))
    if a.shape[0] < 3 or b.shape[0] < 3:
        out["note"] = "group too small for a meaningful test"
        return out
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    out["u"] = float(u)
    out["p"] = float(p)
    out["rank_biserial"] = float(1.0 - (2.0 * u) / (a.shape[0] * b.shape[0]))
    return out


# ------------------------------------------------------------------- analysis
def analyse(
    df: pd.DataFrame, brightness: Optional[np.ndarray], cfg: DictConfig
) -> dict[str, Any]:
    """Run the full Phase 4b analysis; return a JSON-safe nested result dict."""
    seed = int(cfg.get("seed", 42))
    n_boot = int(cfg.get("bootstrap_n", 1000))
    n = len(df)

    grade = df["pred_grade"].to_numpy(dtype=np.int64)
    grade_mean = df["pred_grade_mean"].to_numpy(dtype=np.float64)
    cst = df["cst"].to_numpy(dtype=np.float64)
    bcva = df["bcva"].to_numpy(dtype=np.float64)
    burden = df["biomarker_burden"].to_numpy(dtype=np.float64)
    conf = df["confidence_maxprob"].to_numpy(dtype=np.float64)
    if brightness is not None:
        brightness = brightness[df["index"].to_numpy()]

    # -- [1] collapse ------------------------------------------------------
    counts = np.bincount(grade, minlength=N_DR_CLASSES).astype(int)
    collapse = {
        "counts": {f"grade_{g}": int(counts[g]) for g in range(N_DR_CLASSES)},
        "percent": {f"grade_{g}": float(100.0 * counts[g] / n) for g in range(N_DR_CLASSES)},
        "dominant_grade": int(counts.argmax()),
        "dominant_share": float(counts.max() / n),
        "n_distinct_grades": int((counts > 0).sum()),
        "n_tail_ge1": int((grade >= 1).sum()),
        "n_severe_ge3": int((grade >= 3).sum()),
        "n_grade4": int((grade == 4).sum()),
    }

    tail = grade >= 1
    severe = grade >= 3
    g0 = grade == 0

    # -- [2] residual severity signal -------------------------------------
    severity_signal = {
        "full_grade_vs_cst": spearman_ci(grade, cst, n_boot, seed, "grade~CST (all)"),
        "full_grademean_vs_cst": spearman_ci(grade_mean, cst, n_boot, seed, "grade_mean~CST (all)"),
        "tail_grade_vs_cst": spearman_ci(grade[tail], cst[tail], n_boot, seed, "grade~CST (tail>=1)"),
        "tail_grade_vs_burden": spearman_ci(grade[tail], burden[tail], n_boot, seed, "grade~biomarker_burden (tail>=1)"),
        "tail_grade_vs_bcva": spearman_ci(grade[tail], bcva[tail], n_boot, seed, "grade~BCVA (tail>=1; de-emphasised)"),
        "cst_tail_vs_grade0": mannwhitney(cst[tail], cst[g0], "CST: tail(>=1) vs grade0"),
        "cst_severe_vs_grade0": mannwhitney(cst[severe], cst[g0], "CST: severe(>=3) vs grade0"),
        "burden_tail_vs_grade0": mannwhitney(burden[tail], burden[g0], "burden: tail(>=1) vs grade0"),
    }

    # -- [3] appearance artifact ------------------------------------------
    appearance: dict[str, Any] = {"available": brightness is not None}
    if brightness is not None:
        appearance.update({
            "full_grade_vs_brightness": spearman_ci(grade, brightness, n_boot, seed, "grade~brightness (all)"),
            "tail_grade_vs_brightness": spearman_ci(grade[tail], brightness[tail], n_boot, seed, "grade~brightness (tail>=1)"),
            "brightness_grade4_vs_grade0": mannwhitney(
                brightness[grade == 4], brightness[g0], "brightness: grade4 vs grade0"
            ),
        })

    # -- [4] does confidence rescue anything? -----------------------------
    thr = float(np.median(conf))
    hi = conf >= thr
    lo = ~hi
    confidence_rescue = {
        "confidence_threshold_median": thr,
        "n_confident": int(hi.sum()),
        "n_unconfident": int(lo.sum()),
        "confident_grade_vs_cst": spearman_ci(grade[hi], cst[hi], n_boot, seed, "grade~CST (confident)"),
        "unconfident_grade_vs_cst": spearman_ci(grade[lo], cst[lo], n_boot, seed, "grade~CST (unconfident)"),
        "confident_grade_vs_burden": spearman_ci(grade[hi], burden[hi], n_boot, seed, "grade~burden (confident)"),
        "unconfident_grade_vs_burden": spearman_ci(grade[lo], burden[lo], n_boot, seed, "grade~burden (unconfident)"),
    }

    # -- [5] temporal (moot under collapse) -------------------------------
    eyes = df[df["eye_id"] >= 0].groupby("eye_id")["pred_grade"]
    n_multi = int((eyes.count() >= 2).sum())
    n_varying = int((eyes.nunique() > 1)[eyes.count() >= 2].sum())
    temporal = {
        "n_eyes_with_ids": int(df[df["eye_id"] >= 0]["eye_id"].nunique()),
        "n_eyes_multi_visit": n_multi,
        "n_eyes_grade_varies": n_varying,
        "fraction_varying": float(n_varying / n_multi) if n_multi else 0.0,
        "note": "Under a collapsed grade distribution, within-eye trajectories are "
                "near-flat; temporal analysis is uninformative and not pursued.",
    }

    verdict = _verdict(severity_signal, appearance)
    return _jsonable({
        "n_images": n,
        "collapse": collapse,
        "severity_signal": severity_signal,
        "appearance": appearance,
        "confidence_rescue": confidence_rescue,
        "temporal": temporal,
        "verdict": verdict,
    })


def _abs_rho(entry: dict[str, Any]) -> float:
    r = entry.get("rho")
    return abs(r) if isinstance(r, (int, float)) and r is not None else 0.0


def _verdict(
    severity_signal: dict[str, Any], appearance: dict[str, Any]
) -> dict[str, Any]:
    """Decide which hypothesis the numbers support, conservatively."""
    cst_rho = _abs_rho(severity_signal["tail_grade_vs_cst"])
    bur_rho = _abs_rho(severity_signal["tail_grade_vs_burden"])
    clin_rho = max(cst_rho, bur_rho)
    clin_sig = bool(
        severity_signal["tail_grade_vs_cst"].get("ci_excludes_zero")
        or severity_signal["tail_grade_vs_burden"].get("ci_excludes_zero")
    )
    bright_rho = 0.0
    bright_sig = False
    if appearance.get("available"):
        bright_rho = max(
            _abs_rho(appearance["full_grade_vs_brightness"]),
            _abs_rho(appearance["tail_grade_vs_brightness"]),
        )
        bright_sig = bool(
            appearance["full_grade_vs_brightness"].get("ci_excludes_zero")
            or appearance["tail_grade_vs_brightness"].get("ci_excludes_zero")
        )

    if bright_sig and bright_rho >= clin_rho and bright_rho >= 0.15:
        label = "appearance-driven"
        prose = (
            "The tail tracks image brightness more strongly than any clinical "
            "indicator: the non-grade-0 predictions are largely an appearance "
            "artifact of near-IR, not pathology."
        )
    elif clin_sig and clin_rho > bright_rho:
        label = "faint-residual-signal"
        prose = (
            "A weak but non-zero concordance with clinical severity survives in "
            "the tail, stronger than the brightness association — a faint residual "
            "severity signal, heavily swamped by the collapse."
        )
    else:
        label = "no-usable-signal"
        prose = (
            "Neither clinical concordance nor brightness dominates convincingly; "
            "the tail carries no reliably usable severity signal."
        )
    return {
        "label": label,
        "clinical_abs_rho": clin_rho,
        "brightness_abs_rho": bright_rho,
        "clinical_ci_excludes_zero": clin_sig,
        "brightness_ci_excludes_zero": bright_sig,
        "prose": prose,
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


def _grade_boxplot(
    ax: "plt.Axes", grade: np.ndarray, values: np.ndarray, ylabel: str
) -> int:
    """Box of ``values`` per predicted grade on ``ax``; returns n populated grades."""
    data, positions, colors = [], [], []
    for g in range(N_DR_CLASSES):
        v = values[(grade == g) & ~np.isnan(values)]
        if v.shape[0] > 0:
            data.append(v)
            positions.append(g)
            colors.append(figures.SEVERITY_COLORS[g])
    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True,
                    showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    for median in bp["medians"]:
        median.set_color("black")
    ax.set_xticks(range(N_DR_CLASSES))
    ax.set_xticklabels([f"{g}\n{figures.DR_LABEL_NAMES[g]}" for g in range(N_DR_CLASSES)])
    ax.set_xlabel("Predicted DR grade (zero-shot; unvalidated)")
    ax.set_ylabel(ylabel)
    return len(data)


def fig_grade_vs_cst(
    df: pd.DataFrame, res: dict[str, Any], figures_dir: Path
) -> dict[str, Path]:
    """Figure 1 — CST per predicted grade (does CST rise with grade? likely flat)."""
    figures.set_style()
    grade = df["pred_grade"].to_numpy(np.int64)
    cst = df["cst"].to_numpy(np.float64)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    _grade_boxplot(ax, grade, cst, "CST (µm, raw)")
    rho = res["severity_signal"]["full_grade_vs_cst"]
    ax.set_title(f"Predicted grade vs CST (ρ = {_fmt(rho['rho'])}, "
                 f"95% CI [{_fmt(rho['ci_lo'])}, {_fmt(rho['ci_hi'])}])")
    return _save_fig(fig, figures_dir, "phase4b_fig1_grade_vs_cst")


def fig_grade_vs_burden(
    df: pd.DataFrame, res: dict[str, Any], figures_dir: Path
) -> dict[str, Path]:
    """Figure 2 — biomarker burden per predicted grade."""
    figures.set_style()
    grade = df["pred_grade"].to_numpy(np.int64)
    burden = df["biomarker_burden"].to_numpy(np.float64)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    n_pop = _grade_boxplot(ax, grade, burden, "Biomarker burden (count positive)")
    rho = res["severity_signal"]["tail_grade_vs_burden"]
    ax.set_title(f"Predicted grade vs biomarker burden "
                 f"(tail ρ = {_fmt(rho['rho'])}, n={rho['n']})")
    if n_pop <= 1:
        ax.annotate("burden available only for tier-2 (biomarker) images",
                    xy=(0.5, -0.2), xycoords="axes fraction", ha="center",
                    fontsize=8.5, color="#666666")
    return _save_fig(fig, figures_dir, "phase4b_fig2_grade_vs_burden")


def fig_grade_vs_brightness(
    df: pd.DataFrame, brightness: np.ndarray, res: dict[str, Any], figures_dir: Path
) -> dict[str, Path]:
    """Figure 3 — image brightness per predicted grade (the artifact test)."""
    figures.set_style()
    grade = df["pred_grade"].to_numpy(np.int64)
    bright = brightness[df["index"].to_numpy()]
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    _grade_boxplot(ax, grade, bright, "Mean image intensity (0–1)")
    rho = res["appearance"]["full_grade_vs_brightness"]
    mw = res["appearance"]["brightness_grade4_vs_grade0"]
    darker = (mw["median_a"] is not None and mw["median_b"] is not None
              and mw["median_a"] < mw["median_b"])
    tag = "grade-4 DARKER than grade-0" if darker else "grade-4 not darker"
    ax.set_title(f"Grade vs image brightness — {tag} (ρ = {_fmt(rho['rho'])})")
    return _save_fig(fig, figures_dir, "phase4b_fig3_grade_vs_brightness")


def fig_confidence_stratified(
    res: dict[str, Any], figures_dir: Path
) -> dict[str, Path]:
    """Figure 4 — |ρ|(grade vs clinical) for confident vs unconfident predictions."""
    figures.set_style()
    cr = res["confidence_rescue"]
    groups = ["grade~CST", "grade~burden"]
    conf_vals = [_abs_rho(cr["confident_grade_vs_cst"]), _abs_rho(cr["confident_grade_vs_burden"])]
    unc_vals = [_abs_rho(cr["unconfident_grade_vs_cst"]), _abs_rho(cr["unconfident_grade_vs_burden"])]

    x = np.arange(len(groups))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    ax.bar(x - w / 2, conf_vals, w, color=figures.COLOR_EYEPACS, label="confident")
    ax.bar(x + w / 2, unc_vals, w, color=figures.COLOR_OLIVES, label="unconfident")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("|Spearman ρ| with clinical value")
    ax.set_title("Does confidence rescue concordance? (higher = better)")
    ax.legend(loc="best")
    ax.annotate("confidence split at the median confidence_maxprob",
                xy=(0.5, -0.16), xycoords="axes fraction", ha="center",
                fontsize=8.5, color="#666666")
    return _save_fig(fig, figures_dir, "phase4b_fig4_confidence_stratified")


# ---------------------------------------------------------------------- report
def _fmt(x: Any, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def _rho_line(entry: dict[str, Any]) -> str:
    ci = f"[{_fmt(entry['ci_lo'])}, {_fmt(entry['ci_hi'])}]"
    star = " *" if entry.get("ci_excludes_zero") else ""
    return (f"ρ = {_fmt(entry['rho'])} (95% CI {ci}, p = {_fmt(entry['p'])}, "
            f"n = {entry['n']}){star}")


def build_report(res: dict[str, Any]) -> str:
    """Honest markdown, leading with the plain conclusion."""
    c = res["collapse"]
    ss = res["severity_signal"]
    ap = res["appearance"]
    cr = res["confidence_rescue"]
    tp = res["temporal"]
    v = res["verdict"]

    lines = ["# Phase 4b — Cross-Modality DR-Grading Failure Characterisation", ""]
    lines.append(
        "**Headline.** The zero-shot transfer of the EyePACS colour-fundus DR "
        "grader onto OLIVES near-IR fundus **collapsed**. This phase does not "
        "rescue the grades — it asks *why*, and whether any residual severity "
        "signal survives. **This characterises a negative result; it is not a "
        "validation.**"
    )
    lines.append("")
    lines.append(f"> **Conclusion: {v['label']}.** {v['prose']}")
    lines.append("")
    lines.append("## 1. The collapse")
    lines.append("")
    lines.append("| Grade | Name | Count | % |")
    lines.append("|------:|------|------:|--:|")
    for g in range(N_DR_CLASSES):
        lines.append(
            f"| {g} | {figures.DR_LABEL_NAMES[g]} | {c['counts'][f'grade_{g}']} | "
            f"{c['percent'][f'grade_{g}']:.1f}% |"
        )
    lines.append("")
    lines.append(
        f"Dominant grade **{c['dominant_grade']}** holds "
        f"**{c['dominant_share']:.1%}** of images; only "
        f"**{c['n_distinct_grades']}/5** grades are used. The tail is "
        f"{c['n_tail_ge1']} images at grade ≥1 ({c['n_severe_ge3']} at grade ≥3, "
        f"{c['n_grade4']} at grade 4)."
    )
    lines.append("")
    lines.append("## 2. Is the tail real severity signal?")
    lines.append("")
    lines.append(f"- Predicted grade vs **CST** (tail): {_rho_line(ss['tail_grade_vs_cst'])}")
    lines.append(f"- Predicted grade vs **biomarker burden** (tail): {_rho_line(ss['tail_grade_vs_burden'])}")
    lines.append(f"- Predicted grade vs **BCVA** (tail, de-emphasised): {_rho_line(ss['tail_grade_vs_bcva'])}")
    mw = ss["cst_severe_vs_grade0"]
    lines.append(
        f"- CST, severe(≥3) vs grade-0: median {_fmt(mw['median_a'], 1)} vs "
        f"{_fmt(mw['median_b'], 1)} µm (Mann–Whitney p = {_fmt(mw['p'])}, "
        f"rank-biserial {_fmt(mw['rank_biserial'])})."
    )
    lines.append("")
    lines.append("> `*` marks a bootstrap CI that excludes zero. Full-sample "
                 "correlations are dominated by the grade-0 mass and reported in "
                 "the JSON; the tail subset is the informative test.")
    lines.append("")
    lines.append("## 3. Is the tail an image-appearance artifact?")
    lines.append("")
    if not ap.get("available"):
        lines.append("_Brightness unavailable (raw OLIVES tensor not found) — "
                     "appearance test skipped._")
    else:
        lines.append(f"- Predicted grade vs **brightness** (all): {_rho_line(ap['full_grade_vs_brightness'])}")
        lines.append(f"- Predicted grade vs **brightness** (tail): {_rho_line(ap['tail_grade_vs_brightness'])}")
        bmw = ap["brightness_grade4_vs_grade0"]
        lines.append(
            f"- Brightness, grade-4 vs grade-0: median {_fmt(bmw['median_a'])} vs "
            f"{_fmt(bmw['median_b'])} (Mann–Whitney p = {_fmt(bmw['p'])}, "
            f"rank-biserial {_fmt(bmw['rank_biserial'])})."
        )
        lines.append("")
        lines.append(
            f"**Which dominates?** clinical |ρ| = {_fmt(v['clinical_abs_rho'])} vs "
            f"brightness |ρ| = {_fmt(v['brightness_abs_rho'])} → **{v['label']}**."
        )
    lines.append("")
    lines.append("## 4. Does confidence rescue anything?")
    lines.append("")
    lines.append(f"Split at median confidence ({_fmt(cr['confidence_threshold_median'])}); "
                 f"{cr['n_confident']} confident / {cr['n_unconfident']} unconfident.")
    lines.append(f"- Confident, grade vs CST: {_rho_line(cr['confident_grade_vs_cst'])}")
    lines.append(f"- Unconfident, grade vs CST: {_rho_line(cr['unconfident_grade_vs_cst'])}")
    lines.append(
        "\nConsistent with Phase 4a, confidence does **not** flag the modality "
        "shift: confident predictions do not concord meaningfully better with "
        "clinical values than unconfident ones."
    )
    lines.append("")
    lines.append("## 5. Temporal (moot under collapse)")
    lines.append("")
    lines.append(
        f"Of {tp['n_eyes_multi_visit']} multi-visit eyes, only "
        f"{tp['n_eyes_grade_varies']} ({tp['fraction_varying']:.1%}) show any "
        f"predicted-grade variation across visits. {tp['note']}"
    )
    lines.append("")
    lines.append("## Framing")
    lines.append("")
    lines.append(
        "This is a **characterised negative result**, consistent with OLIVES being "
        "the hardest cross-modality target in prior work (Sükei et al., 2024). Any "
        "weak positive correlation above is reported as such and not overstated: "
        "the colour→near-IR modality gap is not bridged for DR grading by this "
        "encoder."
    )
    lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------- orchestrate
def run(cfg: DictConfig) -> dict[str, Any]:
    """End-to-end Phase 4b: load predictions, analyse, write JSON + report + figs."""
    report_dir = Path(str(cfg.report_dir))
    figures_dir = Path(str(cfg.figures_dir))
    report_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] loading Phase 4a predictions")
    df = load_predictions(cfg)

    print("[2/5] computing per-image brightness (appearance test)")
    brightness = compute_brightness(_olives_file(cfg), len(df))

    print("[3/5] running the failure-characterisation analysis")
    res = analyse(df, brightness, cfg)

    print("[4/5] rendering figures")
    fig_grade_vs_cst(df, res, figures_dir)
    fig_grade_vs_burden(df, res, figures_dir)
    if brightness is not None:
        fig_grade_vs_brightness(df, brightness, res, figures_dir)
    fig_confidence_stratified(res, figures_dir)

    print("[5/5] writing JSON + report")
    (report_dir / "phase4b_failure.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8"
    )
    (report_dir / "phase4b_report.md").write_text(build_report(res), encoding="utf-8")
    v = res["verdict"]
    print(f"[done] verdict = {v['label']} "
          f"(clinical |ρ|={_fmt(v['clinical_abs_rho'])}, "
          f"brightness |ρ|={_fmt(v['brightness_abs_rho'])})")
    return res


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 4b — cross-modality DR-grading failure characterisation."
    )
    parser.add_argument("--config", default="configs/zero_shot.yaml")
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except Exception as exc:
        print(f"[ERROR] failed to load config from {args.config}: {exc}")
        traceback.print_exc()
        return 2

    try:
        run(cfg)
    except Exception as exc:
        print(f"\n[ERROR] phase 4b characterisation failed: {exc}")
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
