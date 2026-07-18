"""Compare the Design C1 harmonised zero-shot result against the original collapse.

Reads the harmonised eval summary written by the UNMODIFIED
``src.analysis.zero_shot_grading`` (``greyscale_experiment/reports/phase4a_summary.json``),
compares its OLIVES predicted-grade distribution and confidence/entropy against
the original zero-shot result, and writes:

    greyscale_experiment/greyscale_comparison.json
    greyscale_experiment/greyscale_report.md
    greyscale_experiment/figures/greyscale_grade_distribution_compare.{png,pdf}

The original result is read from the read-only Phase 4a summary if present,
otherwise from the documented fallback distribution below. This script only
writes NEW artefacts under ``greyscale_experiment/`` and never modifies Phase 1-4
code, configs, checkpoints, or reports.

Run (after training + `python -m src.analysis.zero_shot_grading --config configs/greyscale_eval.yaml`):
    python -m scripts.greyscale_compare --eval-config configs/greyscale_eval.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils.config import load_config

NUM_DR_CLASSES = 5
DR_LABEL_NAMES = ("No DR", "Mild", "Moderate", "Severe", "Proliferative")

# Original (colour-model) zero-shot OLIVES grade distribution — the 96% collapse.
# Used only if the read-only Phase 4a summary is unavailable.
ORIGINAL_GRADE_DISTRIBUTION = {0: 3021, 1: 15, 2: 13, 3: 25, 4: 68}

COLOR_ORIGINAL = "#B00020"   # collapse red
COLOR_HARMONISED = "#1f77b4"  # harmonised blue


def _grade_counts(summary: dict) -> dict[int, int]:
    gd = summary["grade_distribution"]
    return {g: int(gd[f"grade_{g}"]) for g in range(NUM_DR_CLASSES)}


def _dist_stats(counts: dict[int, int]) -> dict[str, Any]:
    total = sum(counts.values())
    pct = {g: (100.0 * counts[g] / total if total else 0.0) for g in counts}
    dominant_grade = max(counts, key=lambda g: counts[g]) if total else 0
    dominant_frac = counts[dominant_grade] / total if total else 0.0
    n_distinct = sum(1 for c in counts.values() if c > 0)
    return {
        "total": total,
        "counts": counts,
        "pct": pct,
        "dominant_grade": int(dominant_grade),
        "dominant_frac": float(dominant_frac),
        "n_distinct_grades": int(n_distinct),
    }


def _load_original(original_report: Path) -> tuple[dict[int, int], Optional[dict]]:
    """Original grade counts + (optionally) the full original summary for conf/entropy."""
    if original_report.exists():
        original_summary = json.loads(original_report.read_text(encoding="utf-8"))
        print(f"[compare] original summary loaded (read-only): {original_report}")
        return _grade_counts(original_summary), original_summary
    print(
        f"[compare] original summary not found at {original_report} — "
        "using documented fallback grade distribution (conf/entropy unavailable)"
    )
    return dict(ORIGINAL_GRADE_DISTRIBUTION), None


def _interpretation(harm: dict[str, Any]) -> tuple[str, str]:
    """Select the pre-agreed honest frame from the harmonised dominant fraction."""
    dom = harm["dominant_frac"]
    n_distinct = harm["n_distinct_grades"]
    if dom < 0.60 and n_distinct >= 3:
        return (
            "substantial spread",
            "The harmonised distribution is spread across grades with the dominant "
            "grade well below the original 96%. Input-level harmonisation appears to "
            "**meaningfully mitigate the zero-shot collapse**. This is a positive "
            "signal, but a spread distribution alone is **not** validated grading — "
            "concordance with OLIVES clinical labels (CST / biomarker burden, as in "
            "Phase 4b) is the necessary next check before claiming grading works.",
        )
    if dom < 0.90:
        return (
            "partial spread",
            "The collapse eases but a single grade still dominates. This is consistent "
            "with **brightness/intensity being one driver of the collapse while the "
            "modality gap is not fully bridged** — the most likely outcome. Harmonising "
            "the intensity distribution helps, but colour/near-IR differences in how "
            "lesions render remain. Clinical-label concordance would be needed to judge "
            "whether any of the spread is meaningful.",
        )
    return (
        "little change",
        "The distribution stays essentially collapsed after harmonisation. This "
        "indicates the modality gap is **deeper than intensity** — lesions render "
        "differently across colour and near-IR modalities (consistent with Xiao et al. "
        "2024), so matching the source intensity distribution is not enough to transfer "
        "DR grading. A legitimate, reportable negative result.",
    )


def _fig_compare(
    orig: dict[str, Any], harm: dict[str, Any], figures_dir: Path
) -> dict[str, Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    grades = np.arange(NUM_DR_CLASSES)
    orig_pct = [orig["pct"][g] for g in range(NUM_DR_CLASSES)]
    harm_pct = [harm["pct"][g] for g in range(NUM_DR_CLASSES)]

    width = 0.4
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    b1 = ax.bar(grades - width / 2, orig_pct, width, label="Original (colour model)",
                color=COLOR_ORIGINAL, alpha=0.85)
    b2 = ax.bar(grades + width / 2, harm_pct, width, label="Harmonised (Design C1)",
                color=COLOR_HARMONISED, alpha=0.85)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.0f}%", ha="center",
                    va="bottom", fontsize=8)
    ax.set_xticks(grades)
    ax.set_xticklabels([f"{g}\n{DR_LABEL_NAMES[g]}" for g in range(NUM_DR_CLASSES)])
    ax.set_xlabel("Predicted DR grade (zero-shot; unvalidated)")
    ax.set_ylabel("Share of OLIVES images (%)")
    ax.set_title(
        f"OLIVES zero-shot grade distribution: original {orig['dominant_frac']:.0%} "
        f"vs harmonised {harm['dominant_frac']:.0%} dominant"
    )
    ax.legend(loc="upper right")
    ax.set_ylim(0, 100)

    out: dict[str, Path] = {}
    for ext in ("png", "pdf"):
        path = figures_dir / f"greyscale_grade_distribution_compare.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        out[ext] = path
    plt.close(fig)
    print("[compare] wrote figure: " + ", ".join(p.name for p in out.values()))
    return out


def _conf_entropy(summary: Optional[dict]) -> Optional[dict[str, float]]:
    if summary is None:
        return None
    return {
        "confidence_maxprob_mean": float(summary["confidence_maxprob"]["mean"]),
        "confidence_maxprob_median": float(summary["confidence_maxprob"]["median"]),
        "entropy_norm_mean": float(summary["entropy_norm"]["mean"]),
        "entropy_norm_median": float(summary["entropy_norm"]["median"]),
    }


def _report_md(
    orig: dict[str, Any],
    harm: dict[str, Any],
    orig_ce: Optional[dict[str, float]],
    harm_ce: dict[str, float],
    frame: str,
    frame_text: str,
    fig_paths: dict[str, Path],
) -> str:
    lines = ["# Greyscale ablation (Variant C / Design C1) — harmonised vs original", ""]
    lines.append(
        "Zero-shot cross-modality DR grading of OLIVES near-IR, comparing the **original** "
        "colour-trained model against the **Design C1 harmonised** model (EyePACS green "
        "channel → histogram-matched to the OLIVES intensity distribution → 3-channel "
        "replicate; OLIVES left raw). Both use the identical frozen TTA engine and "
        "`zero_shot_grading.py`."
    )
    lines.append("")
    lines.append("## Predicted-grade distribution")
    lines.append("")
    lines.append("| Grade | Name | Original count (%) | Harmonised count (%) |")
    lines.append("|------:|------|-------------------:|---------------------:|")
    for g in range(NUM_DR_CLASSES):
        lines.append(
            f"| {g} | {DR_LABEL_NAMES[g]} | {orig['counts'][g]} "
            f"({orig['pct'][g]:.1f}%) | {harm['counts'][g]} ({harm['pct'][g]:.1f}%) |"
        )
    lines.append("")
    lines.append(
        f"- **Dominant grade:** original grade {orig['dominant_grade']} at "
        f"**{orig['dominant_frac']:.1%}** → harmonised grade {harm['dominant_grade']} at "
        f"**{harm['dominant_frac']:.1%}**."
    )
    lines.append(
        f"- **Distinct grades used:** original {orig['n_distinct_grades']}/5 → "
        f"harmonised {harm['n_distinct_grades']}/5."
    )
    lines.append("")
    lines.append("## Confidence / entropy")
    lines.append("")
    if orig_ce is not None:
        lines.append(
            f"- Confidence (max-prob) mean: original **{orig_ce['confidence_maxprob_mean']:.3f}** "
            f"→ harmonised **{harm_ce['confidence_maxprob_mean']:.3f}**."
        )
        lines.append(
            f"- Normalised entropy mean: original **{orig_ce['entropy_norm_mean']:.3f}** "
            f"→ harmonised **{harm_ce['entropy_norm_mean']:.3f}**."
        )
    else:
        lines.append(
            f"- Harmonised confidence (max-prob) mean **{harm_ce['confidence_maxprob_mean']:.3f}**, "
            f"normalised entropy mean **{harm_ce['entropy_norm_mean']:.3f}**. "
            "_(Original confidence/entropy unavailable — Phase 4a summary not found.)_"
        )
    lines.append("")
    lines.append(f"## Interpretation — {frame}")
    lines.append("")
    lines.append(frame_text)
    lines.append("")
    lines.append(
        "> **Not validated grading.** These are zero-shot predictions on a modality with no "
        "ground-truth DR grade. Any spread must be checked against OLIVES clinical labels "
        "(CST / biomarker concordance, Phase 4b) before it can be read as genuine severity."
    )
    lines.append("")
    lines.append("## Figure")
    lines.append("")
    png = fig_paths.get("png")
    if png is not None:
        lines.append(f"![grade distribution comparison]({png.name})")
    lines.append("")
    return "\n".join(lines)


def run(eval_cfg_path: str, original_report: Optional[str]) -> None:
    eval_cfg = load_config(eval_cfg_path)
    report_dir = Path(str(eval_cfg.report_dir))
    figures_dir = Path(str(eval_cfg.figures_dir))
    greyscale_root = report_dir.parent  # .../greyscale_experiment

    # Save-safety: ensure every output dir exists before writing.
    for d in (report_dir, figures_dir, greyscale_root):
        d.mkdir(parents=True, exist_ok=True)
    print(f"[compare] output dirs ready: root={greyscale_root} (exists={greyscale_root.exists()}), "
          f"figures={figures_dir} (exists={figures_dir.exists()})")

    harm_summary_path = report_dir / "phase4a_summary.json"
    if not harm_summary_path.exists():
        raise FileNotFoundError(
            f"harmonised eval summary not found: {harm_summary_path} — run "
            "`python -m src.analysis.zero_shot_grading --config configs/greyscale_eval.yaml` first."
        )
    harm_summary = json.loads(harm_summary_path.read_text(encoding="utf-8"))

    orig_report_path = (
        Path(original_report)
        if original_report
        else greyscale_root.parent / "reports" / "phase4a_summary.json"
    )
    orig_counts, orig_summary = _load_original(orig_report_path)

    harm = _dist_stats(_grade_counts(harm_summary))
    orig = _dist_stats(orig_counts)
    harm_ce = _conf_entropy(harm_summary)
    orig_ce = _conf_entropy(orig_summary)

    frame, frame_text = _interpretation(harm)
    fig_paths = _fig_compare(orig, harm, figures_dir)

    comparison = {
        "design": "C1 (EyePACS green + histogram-match to OLIVES; OLIVES raw)",
        "original": {
            "grade_counts": orig["counts"],
            "grade_pct": orig["pct"],
            "dominant_grade": orig["dominant_grade"],
            "dominant_frac": orig["dominant_frac"],
            "n_distinct_grades": orig["n_distinct_grades"],
            "total": orig["total"],
            "confidence_entropy": orig_ce,
            "source": str(orig_report_path) if orig_summary is not None else "fallback_constant",
        },
        "harmonised": {
            "grade_counts": harm["counts"],
            "grade_pct": harm["pct"],
            "dominant_grade": harm["dominant_grade"],
            "dominant_frac": harm["dominant_frac"],
            "n_distinct_grades": harm["n_distinct_grades"],
            "total": harm["total"],
            "confidence_entropy": harm_ce,
            "source": str(harm_summary_path),
        },
        "interpretation": {"frame": frame, "text": frame_text},
        "figure": {ext: str(p) for ext, p in fig_paths.items()},
    }

    (greyscale_root / "greyscale_comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    report_md = _report_md(orig, harm, orig_ce, harm_ce, frame, frame_text, fig_paths)
    (greyscale_root / "greyscale_report.md").write_text(report_md, encoding="utf-8")

    print(f"[compare] wrote {greyscale_root / 'greyscale_comparison.json'}")
    print(f"[compare] wrote {greyscale_root / 'greyscale_report.md'}")
    print(
        f"[compare] original dominant {orig['dominant_frac']:.1%} "
        f"→ harmonised dominant {harm['dominant_frac']:.1%}  [{frame}]"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare Design C1 harmonised zero-shot result vs the original collapse."
    )
    parser.add_argument("--eval-config", default="configs/greyscale_eval.yaml")
    parser.add_argument(
        "--original-report",
        default=None,
        help="path to the original Phase 4a summary.json (read-only). "
        "Defaults to <drive>/dissertation/reports/phase4a_summary.json.",
    )
    args = parser.parse_args(argv)

    try:
        run(args.eval_config, args.original_report)
    except Exception as exc:
        print(f"\n[ERROR] greyscale comparison failed: {exc}")
        traceback.print_exc()
        return 1

    print("\n[DONE] greyscale comparison written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
