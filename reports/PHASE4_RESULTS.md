# Phase 4 — Consolidated Cross-Modality Evaluation

> Auto-assembled by `python -m src.analysis.phase4_report` from the saved Phase 3a/4a/4b/4c JSONs. No numbers are hand-entered; missing inputs are marked, not invented.

## Framing

This is a **cross-modality** study: a diabetic-retinopathy model trained on **EyePACS colour fundus** is evaluated against **OLIVES near-infrared fundus** — a different imaging modality. The question is *what transfers across the modality gap*. The evaluation tells a three-part story: (1) the DR **grade** does not transfer zero-shot; (2) the tasks trained **in-domain** on OLIVES near-IR (biomarkers, CST, BCVA) are measurable and reported honestly; (3) the TTA **uncertainty** signal is trustworthy in-domain but does not, by itself, flag the cross-modality shift.

## Part 1 — Cross-modality DR-grade transfer fails

### EyePACS in-domain baseline (Phase 3a)

- Accuracy **0.7359** [0.7243, 0.7473], QWK **0.6416** [0.6137, 0.6645] (n = 5266).
- Calibration: **underconfident**, ECE 0.1198; mean confidence 0.619 vs accuracy 0.736.
- Confidence tracks accuracy: correct 0.648 vs incorrect 0.541; selective accuracy 0.736 (100%) → 0.903 (10%). **Gate: PASS.**

### Zero-shot OLIVES DR grading collapses (Phase 4a)

- Predicted-grade distribution: g0=3021, g1=15, g2=13, g3=25, g4=68 (dominant share **0.961**, 5/5 grades used).
- Confidence vs EyePACS: OLIVES mean 0.675 vs EyePACS 0.619 (diff 0.056; OLIVES-lower = False). TTA confidence does **not** flag the modality shift.

### Failure characterisation (Phase 4b)

- **Conclusion: the tail is an image-appearance artifact, not preserved severity signal.** Although the tail's grade–CST correlation (|ρ| = 0.492) marginally exceeds the grade–brightness correlation (|ρ| = 0.436), the CST association is in the **clinically wrong direction** — higher predicted grade → *lower* CST (ρ = −0.492), whereas genuine DR severity would *raise* CST — and the direct comparison confirms severe-predicted images do **not** have higher CST (median 268 vs 276 µm, n.s.). Grade-4 images are meanwhile markedly darker than grade-0 (median brightness 0.388 vs 0.515, large effect). Biomarker concordance in the tail is absent (n = 6, n.s.) and confidence does not rescue concordance. The confident non-grade-0 predictions are therefore best explained as a reaction to near-IR image darkness, consistent with the confident false-positives on dark images seen in Phase 4a — not surviving severity signal.
- Tail concordance: grade~CST ρ = -0.492 [-0.683, -0.251] (wrong sign); grade~brightness ρ = -0.132 [-0.161, -0.102] (all), -0.436 [-0.574, -0.281] (tail).
- Head-to-head |ρ|: clinical 0.492 vs brightness 0.436 — but the clinical association is directionally invalid, so the appearance artifact is the operative explanation.

> This part is a **characterised negative result** for cross-modality DR grading, consistent with OLIVES being the hardest cross-modality target in prior work (Sükei et al., 2024). (Note: the automated `verdict` field in `phase4b_failure.json` reads "faint-residual-signal" on raw |ρ| magnitude; it is overridden here because the clinical correlation is in the wrong direction and thus not evidence of preserved severity.)

## Part 2 — In-domain multi-task prediction (what works)

Held-out **OLIVES test split** (patient-aware, seed 42): 259 images, 13 patients; tier-1 259, tier-2 28. In-domain (no modality jump).

### Biomarkers (primary)

The model detects the clinically-critical **fluid** biomarkers well from near-IR fundus: intraretinal fluid (IRF) AUROC **0.958** [0.870, 1.000], subretinal fluid (SRF) **0.904** [0.750, 1.000], and vitreous debris **0.875** [0.631, 1.000] — all meeting or exceeding the ~0.80 benchmark. Because IRF and SRF are the features most relevant to diabetic macular edema, this indicates near-IR fundus carries recoverable, DME-relevant structural signal *in-domain*. The macro AUROC (0.712) is lower only because several biomarkers with few positives sit near chance; per-biomarker results — with wide CIs given the small tier-2 test set (n = 28) — are the honest unit of reporting.

| Biomarker | AUROC [95% CI] | AP | F1 | n (+pos) |
|-----------|---------------:|---:|---:|---------:|
| Disruption of EZ | 0.725 [0.478, 0.919] | 0.590 | 0.400 | 28 (+8) |
| IR hemorrhages | 0.487 [0.266, 0.719] | 0.655 | 0.400 | 28 (+17) |
| Partially attached vitreous face | 0.631 [0.412, 0.850] | 0.627 | 0.588 | 28 (+13) |
| Fully attached vitreous face | 0.617 [0.293, 0.893] | 0.863 | 0.791 | 28 (+23) |
| Preretinal tissue/hemorrhage | 0.568 [0.278, 0.841] | 0.387 | 0.316 | 28 (+6) |
| Vitreous debris | 0.875 [0.631, 1.000] | 0.977 | 0.923 | 28 (+24) |
| DRT/ME | 0.641 [0.415, 0.856] | 0.709 | 0.649 | 28 (+15) |
| Fluid (IRF) | 0.958 [0.870, 1.000] | 0.994 | 0.902 | 28 (+24) |
| Fluid (SRF) | 0.904 [0.750, 1.000] | 0.725 | 0.571 | 28 (+5) |

- **Macro AUROC 0.712 [0.621, 0.796]**, micro 0.771 (benchmark ≈ 0.80, Kokilepersaud et al. 2023) — reported for completeness; the fluid-biomarker results above are the substantive finding.

### CST (moderate) and BCVA (weak)

- **CST**: R² 0.198 [0.055, 0.307], MAE 53.9 µm, RMSE 78.7 µm (n = 259; benchmark ≈ 0.29 R², Sükei et al. 2024).
- **BCVA**: R² -0.194 [-0.307, -0.098], MAE 11.0 letters (n = 259). Weak/negative R² is expected — BCVA is poorly predictable from fundus (Sükei et al. 2024).

> These are **in-domain** results (train and test both OLIVES near-IR), distinct from the collapsed cross-modality DR transfer. Biomarker outputs are **feature-presence calls, not a DR severity grade** (no biomarker→grade mapping is claimed).

## Part 3 — Uncertainty behaviour and limits

- **In-domain, TTA confidence is trustworthy**: gate PASS; ECE 0.1198; selective accuracy rises from 0.736 to 0.903 as low-confidence cases are abstained (Phase 3a).
- **Cross-modality limit**: on OLIVES near-IR the same TTA confidence does not appropriately drop (OLIVES-lower = False; Phase 4a). TTA characterises input-perturbation / predictive uncertainty, **not** the epistemic 'out-of-modality' shift.

## Limitations

- **No OLIVES DR ground truth** — cross-modality grading can only be validated indirectly (against clinical indicators), never against a true DR grade.
- **Cross-modality DR transfer is documented as hardest on OLIVES** (Sükei et al., 2024); the collapse is consistent with prior work, not a one-off bug.
- **In-domain heads are small-sample** (tier-1/tier-2 test sizes) → wide CIs; treat as auxiliary evidence, not definitive benchmarks.
- **Biomarkers ≠ DR grade** — feature-presence calls, not a severity grade.
- **BCVA is intrinsically weak from fundus** — a weak result is expected.

## Future work

- **Colour→near-IR translation** (greyscale baseline or a learned mapping) to attempt to recover DR-grade transfer across the modality gap.
- **External DR-graded validation** on a near-IR set with true DR labels.
- **Larger biomarker training** to tighten the in-domain CIs.
- **Longitudinal progression prediction** using the OLIVES visit structure.

## Figure manifest

| Figure file | Caption |
|-------------|---------|
| `figures/fig01_dataset_summary.png` | Dataset summary — EyePACS colour fundus vs OLIVES near-IR. |
| `figures/fig02_acquisition_pca.png` | Phase 1 PCA — modality-separated before training (field-stop facet). |
| `figures/fig03_example_fundus.png` | Example fundus images — colour vs near-IR appearance. |
| `figures/fig04a_training_qwk.png` | Training curves — validation DR QWK per epoch (both arms). |
| `figures/fig04b_training_losses.png` | Training curves — per-term losses (coral_on). |
| `figures/fig05_fid_bar.png` | Cross-modality FID before/after alignment (−69%). |
| `figures/fig06a_pca_before_after.png` | PCA before/after — dominant axis flips from modality gap toward alignment. |
| `figures/fig06b_pca_after_severity.png` | PCA after — EyePACS coloured by DR severity. |
| `figures/fig07a_embed_by_dataset.png` | UMAP/t-SNE by modality — substantial but partial alignment. |
| `figures/fig07b_embed_by_severity.png` | UMAP/t-SNE by DR severity. |
| `figures/fig08_severity_ordering.png` | PC1 orders monotonically by DR severity. |
| `figures/fig09_acquisition_rho.png` | Field-stop correlation collapses into the target band. |
| `figures/phase3a_fig1_reliability.png` | EyePACS calibration — reliability diagram (ECE). |
| `figures/phase3a_fig2_selective_prediction.png` | EyePACS selective prediction — accuracy/QWK vs coverage. |
| `figures/phase3a_fig3_confidence_by_correctness.png` | EyePACS confidence by correctness. |
| `figures/phase3a_fig4_examples.png` | EyePACS example predictions (high/low confidence). |
| `figures/phase3a_fig5_simple_vs_ordinal.png` | Category vs ±1 ordinal agreement (near-misses). |
| `figures/phase4a_fig1_grade_distribution.png` | OLIVES zero-shot predicted-grade distribution (collapse). |
| `figures/phase4a_fig2_confidence_compare.png` | Confidence: EyePACS vs OLIVES (modality signal). |
| `figures/phase4a_fig3_entropy_compare.png` | Entropy: EyePACS vs OLIVES. |
| `figures/phase4a_fig4_examples.png` | OLIVES example predictions — no ground-truth grade. |
| `figures/phase4b_fig1_grade_vs_cst.png` | Predicted grade vs CST (residual-signal test). |
| `figures/phase4b_fig2_grade_vs_burden.png` | Predicted grade vs biomarker burden. |
| `figures/phase4b_fig3_grade_vs_brightness.png` | Grade vs image brightness (appearance-artifact test). |
| `figures/phase4b_fig4_confidence_stratified.png` | Confidence-stratified concordance. |
| `figures/phase4c_fig1_biomarker_auroc.png` | Per-biomarker AUROC on the held-out OLIVES test split. |
| `figures/phase4c_fig2_cst_scatter.png` | CST predicted vs actual (real µm). |
| `figures/phase4c_fig3_bcva_scatter.png` | BCVA predicted vs actual (real letters). |
| `figures/phase4c_fig4_summary_table.png` | In-domain metric summary table. |

_29 figures (each also saved as PDF for the dissertation)._
