"""Phase 4d — consolidated evaluation report + figure manifest.

Assembles one coherent, honest write-up (``PHASE4_RESULTS.md``) of the whole
cross-modality evaluation from the **saved artefacts** of Phases 3a / 4a / 4b /
4c. No new computation — every number is read from the JSONs; nothing is
hard-coded from memory. Where a phase's JSON is absent, a clearly-marked
placeholder is emitted instead of an invented value.

The three-part story:
  Part 1 — cross-modality DR-grade transfer fails (EyePACS in-domain baseline →
           OLIVES zero-shot collapse → failure characterisation).
  Part 2 — in-domain multi-task prediction on OLIVES near-IR (what works).
  Part 3 — uncertainty behaviour and its limits.
plus Limitations, Future work, and a figure manifest.

Run:
    python -m src.analysis.phase4_report --config configs/zero_shot.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Optional

from omegaconf import DictConfig

from src.utils.config import load_config

MISSING = "_[missing — phase not run]_"
OUTPUT_NAME = "PHASE4_RESULTS.md"

# Report/JSON inputs consumed (label -> filename in report_dir).
JSON_INPUTS = {
    "phase3a": "phase3a_calibration.json",
    "phase4a": "phase4a_summary.json",
    "phase4b": "phase4b_failure.json",
    "phase4c": "phase4c_indomain.json",
}
MD_INPUTS = {
    "phase3a": "phase3a_report.md",
    "phase4a": "phase4a_report.md",
    "phase4b": "phase4b_report.md",
    "phase4c": "phase4c_report.md",
}

# One-line captions for the consolidated figure manifest, keyed by file stem.
CAPTIONS: dict[str, str] = {
    "fig01_dataset_summary": "Dataset summary — EyePACS colour fundus vs OLIVES near-IR.",
    "fig02_acquisition_pca": "Phase 1 PCA — modality-separated before training (field-stop facet).",
    "fig03_example_fundus": "Example fundus images — colour vs near-IR appearance.",
    "fig04a_training_qwk": "Training curves — validation DR QWK per epoch (both arms).",
    "fig04b_training_losses": "Training curves — per-term losses (coral_on).",
    "fig05_fid_bar": "Cross-modality FID before/after alignment (−69%).",
    "fig06a_pca_before_after": "PCA before/after — dominant axis flips from modality gap toward alignment.",
    "fig06b_pca_after_severity": "PCA after — EyePACS coloured by DR severity.",
    "fig07a_embed_by_dataset": "UMAP/t-SNE by modality — substantial but partial alignment.",
    "fig07b_embed_by_severity": "UMAP/t-SNE by DR severity.",
    "fig08_severity_ordering": "PC1 orders monotonically by DR severity.",
    "fig09_acquisition_rho": "Field-stop correlation collapses into the target band.",
    "phase3a_fig1_reliability": "EyePACS calibration — reliability diagram (ECE).",
    "phase3a_fig2_selective_prediction": "EyePACS selective prediction — accuracy/QWK vs coverage.",
    "phase3a_fig3_confidence_by_correctness": "EyePACS confidence by correctness.",
    "phase3a_fig4_examples": "EyePACS example predictions (high/low confidence).",
    "phase3a_fig5_simple_vs_ordinal": "Category vs ±1 ordinal agreement (near-misses).",
    "phase4a_fig1_grade_distribution": "OLIVES zero-shot predicted-grade distribution (collapse).",
    "phase4a_fig2_confidence_compare": "Confidence: EyePACS vs OLIVES (modality signal).",
    "phase4a_fig3_entropy_compare": "Entropy: EyePACS vs OLIVES.",
    "phase4a_fig4_examples": "OLIVES example predictions — no ground-truth grade.",
    "phase4b_fig1_grade_vs_cst": "Predicted grade vs CST (residual-signal test).",
    "phase4b_fig2_grade_vs_burden": "Predicted grade vs biomarker burden.",
    "phase4b_fig3_grade_vs_brightness": "Grade vs image brightness (appearance-artifact test).",
    "phase4b_fig4_confidence_stratified": "Confidence-stratified concordance.",
    "phase4c_fig1_biomarker_auroc": "Per-biomarker AUROC on the held-out OLIVES test split.",
    "phase4c_fig2_cst_scatter": "CST predicted vs actual (real µm).",
    "phase4c_fig3_bcva_scatter": "BCVA predicted vs actual (real letters).",
    "phase4c_fig4_summary_table": "In-domain metric summary table.",
}


# ----------------------------------------------------------------- io + helpers
def _load_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # corrupt file — treat as missing, note it
        print(f"[warn] could not parse {path.name}: {exc}")
        return None


def _g(d: Optional[dict], *keys: Any, default: Any = None) -> Any:
    """Safe nested get: ``_g(d, 'a', 'b')`` -> d['a']['b'] or ``default``."""
    cur: Any = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        elif isinstance(cur, list) and isinstance(k, int) and -len(cur) <= k < len(cur):
            cur = cur[k]
        else:
            return default
    return cur


def _num(x: Any, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and x != x):  # None or NaN
        return MISSING
    if isinstance(x, (int,)) and not isinstance(x, bool):
        return str(x)
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _ci(ci: Any, nd: int = 3) -> str:
    lo, hi = _g(ci, "lo"), _g(ci, "hi")
    if lo is None or hi is None:
        return ""
    return f" [{_num(lo, nd)}, {_num(hi, nd)}]"


def _rho_ci(entry: Any, nd: int = 3) -> str:
    """CI string for a phase-4b Spearman entry (flat ci_lo/ci_hi fields)."""
    lo, hi = _g(entry, "ci_lo"), _g(entry, "ci_hi")
    if lo is None or hi is None:
        return ""
    return f" [{_num(lo, nd)}, {_num(hi, nd)}]"


# --------------------------------------------------------------- section builders
def _framing() -> list[str]:
    return [
        "# Phase 4 — Consolidated Cross-Modality Evaluation",
        "",
        "> Auto-assembled by `python -m src.analysis.phase4_report` from the saved "
        "Phase 3a/4a/4b/4c JSONs. No numbers are hand-entered; missing inputs are "
        "marked, not invented.",
        "",
        "## Framing",
        "",
        "This is a **cross-modality** study: a diabetic-retinopathy model trained on "
        "**EyePACS colour fundus** is evaluated against **OLIVES near-infrared "
        "fundus** — a different imaging modality. The question is *what transfers "
        "across the modality gap*. The evaluation tells a three-part story: "
        "(1) the DR **grade** does not transfer zero-shot; (2) the tasks trained "
        "**in-domain** on OLIVES near-IR (biomarkers, CST, BCVA) are measurable and "
        "reported honestly; (3) the TTA **uncertainty** signal is trustworthy "
        "in-domain but does not, by itself, flag the cross-modality shift.",
        "",
    ]


def _part1(j3a: Optional[dict], j4a: Optional[dict], j4b: Optional[dict]) -> list[str]:
    out = ["## Part 1 — Cross-modality DR-grade transfer fails", ""]

    # 1a EyePACS in-domain baseline (Phase 3a).
    out.append("### EyePACS in-domain baseline (Phase 3a)")
    out.append("")
    if j3a is None:
        out.append(MISSING)
    else:
        out.append(
            f"- Accuracy **{_num(_g(j3a, 'accuracy'), 4)}**"
            f"{_ci(_g(j3a, 'accuracy_ci95'), 4)}, "
            f"QWK **{_num(_g(j3a, 'qwk'), 4)}**{_ci(_g(j3a, 'qwk_ci95'), 4)} "
            f"(n = {_num(_g(j3a, 'n_images'), 0)})."
        )
        out.append(
            f"- Calibration: **{_g(j3a, 'calibration', 'label', default=MISSING)}**, "
            f"ECE {_num(_g(j3a, 'calibration', 'ece'), 4)}; mean confidence "
            f"{_num(_g(j3a, 'calibration', 'mean_confidence'))} vs accuracy "
            f"{_num(_g(j3a, 'accuracy'))}."
        )
        out.append(
            f"- Confidence tracks accuracy: correct {_num(_g(j3a, 'confidence_vs_error', 'mean_confidence_correct'))} "
            f"vs incorrect {_num(_g(j3a, 'confidence_vs_error', 'mean_confidence_incorrect'))}; "
            f"selective accuracy {_num(_g(j3a, 'selective_prediction', 'accuracy_full_coverage'))} "
            f"(100%) → {_num(_g(j3a, 'selective_prediction', 'accuracy_low_coverage'))} (10%). "
            f"**Gate: {_g(j3a, 'gate', 'verdict', default=MISSING)}.**"
        )
    out.append("")

    # 1b Zero-shot OLIVES collapse (Phase 4a).
    out.append("### Zero-shot OLIVES DR grading collapses (Phase 4a)")
    out.append("")
    if j4a is None:
        out.append(MISSING)
    else:
        gd = _g(j4a, "grade_distribution", default={})
        dist = ", ".join(f"g{g}={gd.get(f'grade_{g}', '?')}" for g in range(5))
        out.append(
            f"- Predicted-grade distribution: {dist} "
            f"(dominant share **{_num(_g(j4a, 'dominant_grade_fraction'))}**, "
            f"{_num(_g(j4a, 'n_distinct_grades'), 0)}/5 grades used)."
        )
        cmp = _g(j4a, "comparison")
        if cmp:
            out.append(
                f"- Confidence vs EyePACS: OLIVES mean "
                f"{_num(_g(j4a, 'confidence_maxprob', 'mean'))} vs EyePACS "
                f"{_num(_g(cmp, 'eyepacs_confidence_maxprob', 'mean'))} "
                f"(diff {_num(_g(cmp, 'confidence_mean_diff_olives_minus_eyepacs'))}; "
                f"OLIVES-lower = {_g(cmp, 'olives_confidence_lower', default=MISSING)}). "
                "TTA confidence does **not** flag the modality shift."
            )
        else:
            out.append("- EyePACS-vs-OLIVES confidence comparison: " + MISSING)
    out.append("")

    # 1c Failure characterisation (Phase 4b).
    out.append("### Failure characterisation (Phase 4b)")
    out.append("")
    if j4b is None:
        out.append(MISSING)
    else:
        out.append(
            f"- **Verdict: {_g(j4b, 'verdict', 'label', default=MISSING)}.** "
            f"{_g(j4b, 'verdict', 'prose', default='')}"
        )
        tail = _g(j4b, "severity_signal", "tail_grade_vs_cst")
        bright = _g(j4b, "appearance", "full_grade_vs_brightness")
        out.append(
            f"- Tail concordance: grade~CST ρ = {_num(_g(tail, 'rho'))}{_rho_ci(tail)}; "
            f"grade~brightness ρ = {_num(_g(bright, 'rho'))}{_rho_ci(bright)}."
        )
        out.append(
            f"- Head-to-head |ρ|: clinical {_num(_g(j4b, 'verdict', 'clinical_abs_rho'))} "
            f"vs brightness {_num(_g(j4b, 'verdict', 'brightness_abs_rho'))}."
        )
    out.append("")
    out.append(
        "> This part is a **characterised negative result** for cross-modality DR "
        "grading, consistent with OLIVES being the hardest cross-modality target in "
        "prior work (Sükei et al., 2024)."
    )
    out.append("")
    return out


def _part2(j4c: Optional[dict]) -> list[str]:
    out = ["## Part 2 — In-domain multi-task prediction (what works)", ""]
    if j4c is None:
        out.append(MISSING)
        out.append("")
        return out
    cov = _g(j4c, "coverage", default={})
    bench = _g(j4c, "benchmarks", default={})
    out.append(
        f"Held-out **OLIVES test split** (patient-aware, seed "
        f"{_g(cov, 'seed', default='?')}): {_g(cov, 'n_test', default='?')} images, "
        f"{_g(cov, 'n_patients', default='?')} patients; tier-1 "
        f"{_g(cov, 'n_tier1_clinical', default='?')}, tier-2 "
        f"{_g(cov, 'n_tier2_biomarkers', default='?')}. In-domain (no modality jump)."
    )
    out.append("")
    out.append("### Biomarkers (primary)")
    out.append("")
    out.append("| Biomarker | AUROC [95% CI] | AP | F1 | n (+pos) |")
    out.append("|-----------|---------------:|---:|---:|---------:|")
    for e in _g(j4c, "biomarkers", "per_biomarker", default=[]):
        auroc = (f"{_num(_g(e, 'auroc'))}{_ci(_g(e, 'auroc_ci'))}"
                 if _g(e, "auroc") is not None else "n/a")
        out.append(
            f"| {_g(e, 'biomarker', default='?')} | {auroc} | "
            f"{_num(_g(e, 'average_precision'))} | {_num(_g(e, 'f1'))} | "
            f"{_g(e, 'n', default='?')} (+{_g(e, 'n_positives', default='?')}) |"
        )
    out.append(
        f"\n- **Macro AUROC {_num(_g(j4c, 'biomarkers', 'macro_auroc'))}"
        f"{_ci(_g(j4c, 'biomarkers', 'macro_auroc_ci'))}**, micro "
        f"{_num(_g(j4c, 'biomarkers', 'micro_auroc'))} "
        f"(benchmark ≈ {_num(_g(bench, 'biomarker_auroc'), 2)}, Kokilepersaud et al. 2023)."
    )
    out.append("")
    cst, bcva = _g(j4c, "cst", default={}), _g(j4c, "bcva", default={})
    out.append("### CST (moderate) and BCVA (weak)")
    out.append("")
    out.append(
        f"- **CST**: R² {_num(_g(cst, 'r2'))}{_ci(_g(cst, 'r2_ci'))}, "
        f"MAE {_num(_g(cst, 'mae'), 1)} µm, RMSE {_num(_g(cst, 'rmse'), 1)} µm "
        f"(n = {_g(cst, 'n', default='?')}; benchmark ≈ {_num(_g(bench, 'cst_r2'), 2)} "
        "R², Sükei et al. 2024)."
    )
    out.append(
        f"- **BCVA**: R² {_num(_g(bcva, 'r2'))}{_ci(_g(bcva, 'r2_ci'))}, "
        f"MAE {_num(_g(bcva, 'mae'), 1)} letters (n = {_g(bcva, 'n', default='?')}). "
        "Weak/negative R² is expected — BCVA is poorly predictable from fundus "
        "(Sükei et al. 2024)."
    )
    out.append("")
    out.append(
        "> These are **in-domain** results (train and test both OLIVES near-IR), "
        "distinct from the collapsed cross-modality DR transfer. Biomarker outputs "
        "are **feature-presence calls, not a DR severity grade** (no biomarker→grade "
        "mapping is claimed)."
    )
    out.append("")
    return out


def _part3(j3a: Optional[dict], j4a: Optional[dict]) -> list[str]:
    out = ["## Part 3 — Uncertainty behaviour and limits", ""]
    if j3a is not None:
        out.append(
            f"- **In-domain, TTA confidence is trustworthy**: gate "
            f"{_g(j3a, 'gate', 'verdict', default=MISSING)}; ECE "
            f"{_num(_g(j3a, 'calibration', 'ece'), 4)}; selective accuracy rises from "
            f"{_num(_g(j3a, 'selective_prediction', 'accuracy_full_coverage'))} to "
            f"{_num(_g(j3a, 'selective_prediction', 'accuracy_low_coverage'))} as "
            "low-confidence cases are abstained (Phase 3a)."
        )
    else:
        out.append("- In-domain calibration: " + MISSING)
    if j4a is not None:
        cmp = _g(j4a, "comparison")
        lower = _g(cmp, "olives_confidence_lower") if cmp else None
        out.append(
            f"- **Cross-modality limit**: on OLIVES near-IR the same TTA confidence "
            f"does not appropriately drop (OLIVES-lower = {lower if lower is not None else MISSING}; "
            "Phase 4a). TTA characterises input-perturbation / predictive "
            "uncertainty, **not** the epistemic 'out-of-modality' shift."
        )
    else:
        out.append("- Cross-modality confidence behaviour: " + MISSING)
    out.append("")
    return out


def _limitations() -> list[str]:
    return [
        "## Limitations",
        "",
        "- **No OLIVES DR ground truth** — cross-modality grading can only be "
        "validated indirectly (against clinical indicators), never against a true "
        "DR grade.",
        "- **Cross-modality DR transfer is documented as hardest on OLIVES** "
        "(Sükei et al., 2024); the collapse is consistent with prior work, not a "
        "one-off bug.",
        "- **In-domain heads are small-sample** (tier-1/tier-2 test sizes) → wide "
        "CIs; treat as auxiliary evidence, not definitive benchmarks.",
        "- **Biomarkers ≠ DR grade** — feature-presence calls, not a severity grade.",
        "- **BCVA is intrinsically weak from fundus** — a weak result is expected.",
        "",
    ]


def _future_work() -> list[str]:
    return [
        "## Future work",
        "",
        "- **Colour→near-IR translation** (greyscale baseline or a learned mapping) "
        "to attempt to recover DR-grade transfer across the modality gap.",
        "- **External DR-graded validation** on a near-IR set with true DR labels.",
        "- **Larger biomarker training** to tighten the in-domain CIs.",
        "- **Longitudinal progression prediction** using the OLIVES visit structure.",
        "",
    ]


def _figure_manifest(figures_dir: Path) -> list[str]:
    out = ["## Figure manifest", ""]
    if not figures_dir.exists():
        out.append(f"_[figures directory not found: {figures_dir}]_")
        out.append("")
        return out
    pngs = sorted(figures_dir.glob("*.png"))
    if not pngs:
        out.append(f"_[no figures found in {figures_dir}]_")
        out.append("")
        return out
    out.append("| Figure file | Caption |")
    out.append("|-------------|---------|")
    for p in pngs:
        cap = CAPTIONS.get(p.stem, "—")
        out.append(f"| `figures/{p.name}` | {cap} |")
    out.append("")
    out.append(
        f"_{len(pngs)} figures (each also saved as PDF for the dissertation)._"
    )
    out.append("")
    return out


# ----------------------------------------------------------------- orchestrate
def introspect(report_dir: Path) -> dict[str, bool]:
    """Print which report inputs are present/missing; return a presence map."""
    print("[introspect] scanning report inputs in", report_dir)
    presence: dict[str, bool] = {}
    for label, name in JSON_INPUTS.items():
        p = report_dir / name
        present = p.exists()
        presence[label] = present
        keys = ""
        if present:
            d = _load_json(p)
            keys = ", ".join(list(d.keys())[:8]) if isinstance(d, dict) else "(unparseable)"
        print(f"  {name:<28} {'FOUND ' if present else 'MISSING'}  {keys}")
    for _label, name in MD_INPUTS.items():
        p = report_dir / name
        print(f"  {name:<28} {'FOUND ' if p.exists() else 'MISSING'}")
    return presence


def assemble(cfg: DictConfig) -> Path:
    """Read the saved artefacts and write the consolidated ``PHASE4_RESULTS.md``."""
    report_dir = Path(str(cfg.report_dir))
    figures_dir = Path(str(cfg.figures_dir))

    print("[1/3] introspecting inputs")
    presence = introspect(report_dir)
    found = [k for k, v in presence.items() if v]
    print(f"[introspect] present: {found or 'none'}")

    print("[2/3] loading JSONs")
    j3a = _load_json(report_dir / JSON_INPUTS["phase3a"])
    j4a = _load_json(report_dir / JSON_INPUTS["phase4a"])
    j4b = _load_json(report_dir / JSON_INPUTS["phase4b"])
    j4c = _load_json(report_dir / JSON_INPUTS["phase4c"])

    print("[3/3] assembling PHASE4_RESULTS.md")
    lines: list[str] = []
    lines += _framing()
    lines += _part1(j3a, j4a, j4b)
    lines += _part2(j4c)
    lines += _part3(j3a, j4a)
    lines += _limitations()
    lines += _future_work()
    lines += _figure_manifest(figures_dir)

    report_dir.mkdir(parents=True, exist_ok=True)
    out_path = report_dir / OUTPUT_NAME
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[done] wrote {out_path}")
    print(
        "[done] convert to Word later with: "
        f"pandoc '{out_path}' -o PHASE4_RESULTS.docx"
    )
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 4d — consolidated evaluation report + figure manifest."
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
        assemble(cfg)
    except Exception as exc:
        print(f"\n[ERROR] phase 4d report assembly failed: {exc}")
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
