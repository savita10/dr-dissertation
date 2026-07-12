# Phase 4 — Consolidated Cross-Modality Evaluation

> **Skeleton / placeholder.** This committed copy is the authored structure of the
> consolidated report. The live, number-filled version is produced on Drive by
> `python -m src.analysis.phase4_report --config configs/zero_shot.yaml`, which
> reads the saved Phase 3a/4a/4b/4c JSONs and fills every `_[to be filled …]_`
> below. No numbers are hand-entered here; run the module in Colab (Drive mounted)
> and copy the generated `PHASE4_RESULTS.md` back over this file before the final
> commit. Missing phases are marked, never invented.

## Framing

Cross-modality study: a diabetic-retinopathy model trained on **EyePACS colour
fundus** is evaluated against **OLIVES near-infrared fundus** — a different
imaging modality. The question is *what transfers across the modality gap*. The
evaluation tells a three-part story: (1) the DR **grade** does not transfer
zero-shot; (2) the tasks trained **in-domain** on OLIVES near-IR (biomarkers,
CST, BCVA) are measurable and reported honestly; (3) the TTA **uncertainty**
signal is trustworthy in-domain but does not, by itself, flag the cross-modality
shift.

## Part 1 — Cross-modality DR-grade transfer fails

### EyePACS in-domain baseline (Phase 3a)

- Accuracy, QWK (with 95% CIs), calibration label + ECE, and the confidence gate
  verdict: _[to be filled from `phase3a_calibration.json`]_

### Zero-shot OLIVES DR grading collapses (Phase 4a)

- Predicted-grade distribution, dominant-class share, and the EyePACS-vs-OLIVES
  confidence comparison: _[to be filled from `phase4a_summary.json`]_

### Failure characterisation (Phase 4b)

- Verdict (residual-signal vs appearance-artifact), tail grade~CST and
  grade~brightness ρ (+CIs), clinical vs brightness |ρ|:
  _[to be filled from `phase4b_failure.json`]_

> Characterised negative result for cross-modality DR grading, consistent with
> OLIVES being the hardest cross-modality target in prior work (Sükei et al., 2024).

## Part 2 — In-domain multi-task prediction (what works)

Held-out OLIVES test split (patient-aware, seed 42); in-domain, no modality jump.

- Per-biomarker AUROC / AP / F1 (+ macro/micro, vs ~0.80 benchmark), CST R²/MAE
  (vs ~0.29), BCVA R²/MAE (honest weak): _[to be filled from `phase4c_indomain.json`]_

> In-domain results, distinct from the collapsed cross-modality DR transfer.
> Biomarker outputs are feature-presence calls, not a DR severity grade.

## Part 3 — Uncertainty behaviour and limits

- In-domain: TTA confidence is trustworthy (gate PASS, ECE, selective prediction).
- Cross-modality limit: the same TTA confidence does not flag the modality shift.
  _[to be filled from `phase3a_calibration.json` + `phase4a_summary.json`]_

## Limitations

- No OLIVES DR ground truth — cross-modality grading validated only indirectly.
- Cross-modality DR transfer documented as hardest on OLIVES (Sükei et al., 2024).
- In-domain heads are small-sample (wide CIs; auxiliary evidence).
- Biomarkers ≠ DR grade (feature-presence calls).
- BCVA intrinsically weak from fundus (a weak result is expected).

## Future work

- Colour→near-IR translation (greyscale baseline or learned mapping) to attempt to
  recover DR-grade transfer.
- External DR-graded validation on a near-IR set with true DR labels.
- Larger biomarker training to tighten in-domain CIs.
- Longitudinal progression prediction using the OLIVES visit structure.

## Figure manifest

_[to be filled — one row per figure in `figures/` with a one-line caption, emitted
by the module from the live figures directory.]_
