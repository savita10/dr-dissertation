# Post-Training Reference Guide

**Project**: Adaptive Multimodal ML for Diabetic Retinopathy Progression Prediction
**Institution**: University of Bath, MSc dissertation
**Submission deadline**: August 2026
**Document purpose**: Step-by-step reference for everything that happens *after* Phase 2 training completes — covering Phase 3 (uncertainty quantification), Phase 4 (evaluation), and dissertation writing.

This document is designed to be uploaded to a new Claude chat at the start of Phase 3 or Phase 4. It works alongside `METHODOLOGY.md`, `PREPROCESSING_DOCUMENTATION.md`, and `PHASE2_KICKOFF.md` (which together describe what came before).

---

## 1. Where you are when you start using this document

You should reach this point after Phase 2 has produced:

- A trained model checkpoint at `/content/drive/MyDrive/dissertation/checkpoints/phase2_best.pt`
- New feature embeddings at `/content/drive/MyDrive/dissertation/features/` extracted using the trained encoder
- A `phase2_report.md` showing the before/after distributional comparison
- Per-head training curves (DR classification, biomarker BCE, BCVA/CST regression)
- `PHASE2_RESULTS.md` documenting architecture, hyperparameters, final metrics

If any of these are missing, Phase 2 isn't truly complete. Go back and finish those deliverables before starting Phase 3.

**Sanity checks before proceeding**:
- The trained model loads without errors via `torch.load()`
- Each output head produces sensible predictions on a few test samples
- Post-training FID is meaningfully lower than 240 (Phase 1 baseline)
- At least 3 of the 5 Phase 2 success criteria from the methodology document are met

---

## 2. Phase 3: Uncertainty Quantification

### 2.1 What Phase 3 produces

The trained model from Phase 2 makes predictions, but every prediction comes with the same implicit confidence — even when the model is essentially guessing. Phase 3 adds explicit confidence estimates to every prediction.

**Concrete deliverables**:
- Modified model that outputs both a prediction AND a confidence number for each output head
- A calibration analysis showing the confidence numbers are reliable (when the model says 80%, it's actually right ~80% of the time)
- A risk-stratification capability: flag low-confidence predictions for human review
- Updated documentation: `PHASE3_RESULTS.md`

### 2.2 Three approaches to consider

The approach choice depends on training time available, inference latency requirements, and how much architectural change you're willing to make. Pick one as primary; the others can be mentioned as "alternative approaches considered" in the dissertation.

**Approach A — Monte Carlo Dropout (recommended starting point)**

- Enable dropout layers at inference time (normally only active during training)
- Run the same input through the model N times (typically 30-50) with different random dropout patterns each time
- The mean of the N predictions is your final prediction
- The variance/standard deviation across the N predictions is your uncertainty estimate

Pros:
- Minimal architectural change (just enable dropout at test time)
- No retraining needed if the model already has dropout layers
- Computationally tractable

Cons:
- Requires N forward passes per prediction (30-50× inference latency)
- Quality depends on where dropout layers exist in the architecture
- Calibration can be imperfect

Implementation effort: 1-2 days

**Approach B — Deep Ensembles**

- Train K independent copies of the same model architecture (typically K=5 to K=10), each with different random initialisation
- At inference: run all K models on the same input, take the mean as the prediction and the disagreement as uncertainty

Pros:
- Often produces the best-calibrated uncertainty estimates in the literature
- Each model is independently usable

Cons:
- K× training time (multiplies your Phase 2 effort)
- K× storage for checkpoints
- K× inference latency

Implementation effort: 2-3 weeks (mostly waiting for K training runs)

**Approach C — Variational / Bayesian methods**

- Replace point-estimate weights in the network with weight distributions (means and variances per weight)
- Train using variational inference or Laplace approximation
- At inference, sample from the weight distributions to get prediction distributions

Pros:
- Theoretically principled
- Single model artefact

Cons:
- Significantly more complex implementation
- Training stability can be challenging
- Less standardised tooling

Implementation effort: 3-4 weeks

**Recommendation for this dissertation**: Approach A (MC Dropout) as primary. Mention Approach B in the dissertation's "alternative methods considered" subsection. Skip Approach C unless you have unusual amounts of time.

### 2.3 Phase 3 step-by-step implementation

#### Step 3.1 — Verify dropout exists in the architecture

```python
# In Colab, load your Phase 2 model and check for dropout layers
import torch
model = torch.load('/content/drive/MyDrive/dissertation/checkpoints/phase2_best.pt')

dropout_layers = []
for name, module in model.named_modules():
    if 'dropout' in name.lower() or isinstance(module, torch.nn.Dropout):
        dropout_layers.append((name, module))

print(f"Found {len(dropout_layers)} dropout layers")
for name, module in dropout_layers:
    print(f"  {name}: p={module.p}")
```

ResNet-50 by default has minimal dropout. You may need to add dropout layers to the projection head and output heads before Phase 3 works well. If so, this becomes a Phase 2 architecture revision rather than a pure Phase 3 task.

**If dropout is missing or insufficient**: send Claude Code a prompt to add `nn.Dropout(p=0.3)` after each linear layer in the projection head and output heads, then retrain. This is a Phase 2.5 step.

#### Step 3.2 — Implement the MC Dropout inference wrapper

New file: `src/inference/mc_dropout.py`

The wrapper should:
- Accept a trained model and a batch of inputs
- Set the model to `train()` mode (counterintuitively — this enables dropout) BUT only for the dropout layers; keep batch norm in `eval()` mode using `model.apply()` selectively
- Run N forward passes (default N=30)
- Collect per-prediction means and standard deviations for each output head
- Return both prediction tensors and uncertainty tensors

#### Step 3.3 — Implement calibration evaluation

New file: `src/evaluation/calibration.py`

For each output head, compute:
- **Reliability diagram**: plot predicted confidence vs actual accuracy in bins (e.g. 0-10%, 10-20%, etc.)
- **Expected Calibration Error (ECE)**: weighted average gap between confidence and accuracy across bins
- **Brier score** for classification heads
- **Negative log-likelihood (NLL)** as overall calibration metric

A well-calibrated model has a reliability diagram close to the diagonal y=x line and a low ECE (typically <0.05 is considered well-calibrated).

#### Step 3.4 — Run calibration on validation set

Apply the MC Dropout inference wrapper to the validation set, compute calibration metrics, generate reliability diagrams.

Expected outcomes:
- DR severity classifier: should be reasonably calibrated since EyePACS has enough samples
- Biomarker classifier: may be poorly calibrated due to small Tier-2 sample count (192) — important to report honestly
- BCVA/CST regressor: report as prediction intervals (e.g. 95% confidence intervals from the MC sample distribution)

#### Step 3.5 — Risk-stratification analysis

Useful additional analysis: take all predictions, sort by uncertainty, and ask "what happens to accuracy if we only act on the most confident X% of predictions?"

Plot a curve: x-axis = "fraction of cases acted on" (10% to 100%), y-axis = "accuracy on those cases". A good uncertainty estimator produces a curve where accuracy increases sharply as you act on fewer, more-confident cases.

This translates uncertainty into clinical utility: "if a clinician reviews the 20% of cases the model is least confident about, the remaining 80% can be triaged automatically with 95% accuracy."

### 2.4 Phase 3 deliverables checklist

Before declaring Phase 3 complete:

- [ ] `src/inference/mc_dropout.py` — MC Dropout wrapper implemented
- [ ] `src/evaluation/calibration.py` — calibration metrics module
- [ ] Reliability diagrams generated for all three output heads
- [ ] ECE numbers computed and reported per head
- [ ] Risk-stratification curve generated
- [ ] `PHASE3_RESULTS.md` documenting the approach, metrics, and findings
- [ ] All artefacts saved to `/content/drive/MyDrive/dissertation/results/phase3/`

### 2.5 What "good" looks like for Phase 3

You don't need state-of-the-art calibration. You need honest, documented calibration with reasonable numbers:

- DR severity classifier ECE < 0.10 (decent)
- Biomarker classifier ECE may be higher (0.10-0.20) due to small sample size — this is OK if reported honestly
- Reliability diagrams show monotonically-increasing calibration curves
- The risk-stratification curve shows uncertainty has predictive value (high-uncertainty cases ARE the ones the model gets wrong more often)

If the model is extremely poorly calibrated (e.g. ECE > 0.30 with no monotonic relationship between confidence and accuracy), that's worth investigating — it likely means dropout isn't well-placed in the architecture, and Phase 3 may need to revisit Phase 2 design.

### 2.6 Realistic Phase 3 timeline

1-2 weeks total:
- Days 1-2: implement MC Dropout wrapper
- Days 3-5: implement calibration metrics
- Days 6-8: run analysis, generate plots, write `PHASE3_RESULTS.md`
- Buffer: 2-3 days for unexpected issues, architecture tweaks, calibration debugging

---

## 3. Phase 4: Evaluation

This is the final research phase before writing. Phase 4 produces all the numbers and figures that will appear in the dissertation's Results and Discussion chapters.

### 3.1 What Phase 4 produces

**Three categories of evaluation**:
1. Standard performance metrics on a held-out test set
2. The longitudinal trajectory test (the dissertation's key scientific claim)
3. Cross-dataset transfer validation (proves Phase 2's architecture worked)

**Concrete deliverables**:
- Final test-set results table with all metrics
- Longitudinal trajectory analysis plots and numbers
- Cross-dataset transfer comparison
- Per-biomarker analysis showing what the model can and cannot predict from fundus
- Failure mode analysis: when does the model get things wrong, and why?
- `PHASE4_RESULTS.md` summarising everything

### 3.2 Critical: the held-out test set

**The test set must remain untouched throughout Phases 2 and 3**. Touch it for the first time only in Phase 4.

This is the rule for honest evaluation: if you tune hyperparameters or make design decisions based on test set performance, you've "leaked" the test set into your model and the final numbers become inflated.

The patient-aware split from Phase 2 should have produced three splits: train, validation, test. During Phases 2 and 3 you only use train (for fitting) and validation (for monitoring and hyperparameter tuning). Phase 4 is the first and only time you evaluate on test.

**Reality check before starting Phase 4**: confirm you didn't accidentally use the test set during Phase 2 hyperparameter tuning or Phase 3 calibration evaluation. If you did, the test set is contaminated and you need to construct a fresh held-out set (likely from data you haven't seen yet, possibly compromising the original split structure).

### 3.3 Standard performance metrics

#### 3.3.1 DR severity classifier metrics

Run inference on test-set EyePACS samples. Compute:

- **Overall accuracy**
- **Per-class precision, recall, F1**
- **Macro-averaged F1** (treats all classes equally — important given EyePACS class imbalance)
- **Weighted F1** (weighted by class frequency — gives an overall accuracy-like number)
- **Quadratic Weighted Kappa** (the standard metric for DR severity grading, since classes are ordinal — predicting Severe when truth is Mild is a worse error than predicting Moderate)
- **Confusion matrix** (a 5×5 grid showing where errors happen)
- **ROC curves and AUC** for each class in one-vs-rest framing

Realistic targets:
- Overall accuracy: 65-80% (depending on how well you handled class imbalance)
- Macro F1: 0.45-0.65 (lower than overall accuracy due to minority class performance)
- Quadratic Weighted Kappa: 0.65-0.85 (the standard literature benchmark range)

#### 3.3.2 Biomarker classifier metrics

Run inference on test-set OLIVES samples with `has_biomarkers=True`. Note this will be a small subset of the test set (maybe 20-30 samples depending on split).

Compute per-biomarker:
- **AUC** (treats each biomarker independently)
- **Sensitivity at 90% specificity** (clinically interpretable threshold)
- **Specificity at 90% sensitivity** (alternative clinical framing)
- **Average precision (AP)** for highly imbalanced biomarkers

Combined metrics:
- **Macro-averaged AUC** across the 9 usable biomarkers
- **Mean Average Precision (mAP)**

Realistic targets:
- For fundus-grounded biomarkers (DRT/ME, IRF, IR hemorrhages, SRF): AUC 0.70-0.85
- For OCT-derived biomarkers (vitreous-related, EZ Disruption): AUC 0.55-0.70 (the model can only predict these weakly from fundus alone)

Honest reporting here is important — your dissertation should explicitly note that some biomarkers are inherently harder to predict from fundus and that the model's limitations on those targets are expected.

#### 3.3.3 BCVA/CST regressor metrics

Run inference on test-set OLIVES samples with `has_clinical=True` and `~isnan(bcva)`.

For BCVA:
- **Mean Absolute Error (MAE)** in ETDRS letters
- **Root Mean Square Error (RMSE)**
- **R²** (coefficient of determination)
- **Concordance index** (c-index) — rank-based metric standard for clinical prediction

For CST:
- Same metrics, in microns instead of letters

Realistic targets:
- BCVA MAE: 8-15 letters (the test set is small so noise is high; literature reports vary widely)
- CST MAE: 30-80 microns
- BCVA c-index: 0.60-0.75
- CST c-index: 0.55-0.70

### 3.4 The longitudinal trajectory test (the dissertation's key claim)

This is what makes the dissertation distinctive. Other DR papers predict current severity from a single image; this dissertation claims the model can predict *future* trajectories.

#### 3.4.1 The test design

Take OLIVES test-set patients. For each patient:
- Find their baseline (first) visit and their last available visit
- The model sees only the baseline visit's fundus image
- The model predicts BCVA and CST
- Compare the model's "predicted-at-baseline" against the actual BCVA/CST at later visits

#### 3.4.2 Three specific hypotheses to test

**Hypothesis 1 — Same-baseline-different-outcome discrimination**

Stratify patients by baseline DR severity. Within each stratum, identify:
- "Decliners" — patients whose BCVA dropped >5 letters over follow-up
- "Stable" — patients whose BCVA stayed within ±5 letters
- "Improvers" — patients whose BCVA gained >5 letters

Question: at baseline (when the future outcome is unknown), can the model's predicted BCVA at follow-up correctly rank decliners below stable below improvers?

Metric: pairwise ranking accuracy within each baseline stratum.

**Hypothesis 2 — Time-to-vision-loss prediction**

For patients who eventually lost significant vision (>10 letters), how well does the model predict the time to that event from baseline?

Metric: concordance index for time-to-event prediction.

**Hypothesis 3 — Treatment response prediction (OLIVES-specific)**

OLIVES patients received treatment during the trial. Can the model predict which baseline-features patients are likely to respond well to treatment?

Metric: AUC for predicting "responder" vs "non-responder" status.

#### 3.4.3 Realistic timeline for Phase 4.4

This is where Phase 4 takes the bulk of its time:
- 1 week: defining outcome groups (decliner/stable/improver thresholds, what counts as "significant" vision change)
- 1 week: implementing the test infrastructure
- 1 week: running tests, analysing results, generating plots

### 3.5 Cross-dataset transfer validation

This directly measures whether Phase 2's contrastive alignment achieved its purpose.

#### 3.5.1 Two tests to run

**Test A — EyePACS-only baseline**: train a new model using only EyePACS data (no contrastive alignment, no OLIVES). Evaluate this baseline model on the OLIVES test set.

**Test B — Your full model**: take your Phase 2 shared encoder and evaluate it on the same OLIVES test set.

Compare: how much better is your model than the baseline on OLIVES tasks? If significantly better, the contrastive alignment is doing real work. If similar, the alignment didn't help meaningfully (still worth reporting honestly).

Metrics for comparison:
- DR severity prediction accuracy on OLIVES (using EyePACS-style DR labels derived from the biomarker presence patterns, or from clinical notes if available)
- Feature space alignment metrics from Phase 1 (FID, MMD, cosine similarity) — both before and after training

#### 3.5.2 Reverse transfer test

**Test C — OLIVES-only baseline**: train a model using only OLIVES (small dataset). Evaluate on EyePACS test set.

**Test D — Your full model**: evaluate your shared model on EyePACS test set.

This reveals whether the EyePACS-OLIVES combined training improved OLIVES feature learning (Test B vs Test D comparison would show this).

### 3.6 Failure mode analysis

For the dissertation's Discussion chapter, identify systematic errors:

- Visualise the 50 most-wrong predictions per head: are they similar in some way?
- Plot prediction errors against image properties (mean intensity, black-pixel proportion, contrast): does the model fail more on certain image types?
- Plot errors against patient demographics if available: does the model perform worse on specific subgroups?
- For DR severity: which classes get confused with which? (5×5 confusion matrix)
- For biomarkers: which biomarkers have the highest false-positive vs false-negative rates?

The dissertation should report these failure modes honestly. Knowing where a model fails is more valuable than claiming it always works.

### 3.7 Phase 4 deliverables checklist

- [ ] Test-set inference completed for all three output heads
- [ ] Standard performance metrics computed and tabulated
- [ ] Longitudinal trajectory test executed for all three hypotheses
- [ ] Cross-dataset transfer comparison completed
- [ ] Failure mode analysis with examples
- [ ] All figures generated at publication quality (300 DPI minimum)
- [ ] `PHASE4_RESULTS.md` summarising findings
- [ ] All artefacts saved to `/content/drive/MyDrive/dissertation/results/phase4/`

### 3.8 Realistic Phase 4 timeline

3-4 weeks:
- Week 1: Standard performance metrics on test set
- Week 2: Longitudinal trajectory test (the most complex piece)
- Week 3: Cross-dataset transfer + failure analysis
- Week 4: Plot polishing, results writing, sanity checks

---

## 4. Dissertation Writing (after Phase 4)

Phase 4 produces all the empirical content. Writing assembles it into a coherent narrative.

### 4.1 Standard dissertation structure

```
1. Introduction
   - Diabetic retinopathy as a public health problem
   - Limitations of current screening approaches
   - Research questions and contributions

2. Background and Related Work
   - DR detection literature
   - Multi-modal learning in medical imaging
   - Contrastive learning approaches
   - Uncertainty quantification in clinical ML

3. Datasets
   - EyePACS description (use content from PREPROCESSING_DOCUMENTATION.md)
   - OLIVES description
   - Three-tier label structure
   - Preprocessing pipeline and reconciliation against OLIVES paper

4. Methodology
   - Phase 1 distributional validation (use METHODOLOGY.md content)
   - Phase 2 architecture (shared encoder + heads + contrastive loss)
   - Phase 3 uncertainty quantification approach
   - Phase 4 evaluation design

5. Results
   - Phase 1 findings (FID, MMD, mechanism diagnosis)
   - Phase 2 training results (curves, per-head metrics)
   - Phase 2 distributional re-evaluation (before/after comparison)
   - Phase 3 calibration results
   - Phase 4 standard metrics
   - Phase 4 longitudinal trajectory results
   - Phase 4 cross-dataset transfer results
   - Failure mode analysis

6. Discussion
   - What the model can do well
   - What it cannot do (and why — biomarker analysis honest about OCT vs fundus limitations)
   - How findings compare to prior work
   - Limitations of the study
   - Future work

7. Conclusion
   - Summary of contributions
   - Practical implications

8. References

9. Appendices
   - Detailed hyperparameters
   - Additional plots
   - Bug fixes documented in PREPROCESSING_DOCUMENTATION.md (mention as methodological rigour)
```

### 4.2 Word count target

University of Bath MSc dissertations are typically 15,000-20,000 words. Allocate roughly:
- Introduction: 1,500
- Background: 3,000
- Datasets: 2,000
- Methodology: 4,000
- Results: 4,000
- Discussion: 2,500
- Conclusion: 500
- (References and appendices not counted)

### 4.3 Writing timeline

4-6 weeks before submission (early-mid July 2026 if August submission):
- Week 1: Outline + Introduction + Background
- Week 2: Datasets + Methodology
- Week 3: Results
- Week 4: Discussion + Conclusion
- Week 5: Editing, formatting, citations
- Week 6: Final review, proofreading, submission

### 4.4 Style and tone

Aim for:
- Concise, evidence-led prose (avoid filler)
- Specific numbers wherever possible (not "the model performs well" but "achieved 78% accuracy on test set")
- Honest reporting of limitations (a dissertation that acknowledges weaknesses is stronger than one that hides them)
- Appropriate use of figures — every figure must have a clear takeaway

### 4.5 Reference management

Use a reference manager (Zotero, Mendeley, or BibTeX-based) from day 1 of writing. Maintaining ad-hoc citation lists in Word becomes painful at 60+ references.

Key papers to ensure citation of:
- Prabhushankar et al. NeurIPS 2022 (OLIVES dataset)
- Sükei et al. Nature 2024 (SSL contrastive on OLIVES — primary baseline)
- Kokilepersaud et al. 2023 (clinically-labelled contrastive)
- Lan et al. MICCAI 2025 (OT-based missing modality)
- He et al. 2016 (ResNet)
- Khosla et al. 2020 (Supervised Contrastive Learning)
- Gal & Ghahramani 2016 (MC Dropout)
- The methodology paper for any other technique used (PCA, FID, MMD, etc.)

---

## 5. Workflow context (carry-over from earlier phases)

### 5.1 Repository structure expected by Phase 3/4

```
dr-dissertation/
├── src/
│   ├── data/
│   │   ├── preprocess.py            # (complete from Phase 0)
│   │   ├── datasets.py              # (from Phase 2b)
│   │   ├── samplers.py              # (from Phase 2b)
│   │   └── dataloaders.py           # (from Phase 2b)
│   ├── models/
│   │   ├── encoder.py               # (from Phase 2a)
│   │   ├── heads.py                 # (from Phase 2a)
│   │   ├── model.py                 # (from Phase 2a)
│   │   └── losses.py                # (from Phase 2a)
│   ├── training/
│   │   ├── trainer.py               # (from Phase 2c)
│   │   └── train.py                 # (from Phase 2c)
│   ├── analysis/
│   │   ├── feature_extraction.py    # (from Phase 1)
│   │   └── distribution_diagnostics.py # (from Phase 1)
│   ├── inference/
│   │   └── mc_dropout.py            # (NEW — Phase 3)
│   └── evaluation/
│       ├── calibration.py           # (NEW — Phase 3)
│       ├── metrics.py               # (NEW — Phase 4)
│       ├── longitudinal.py          # (NEW — Phase 4)
│       ├── transfer.py              # (NEW — Phase 4)
│       └── failure_analysis.py      # (NEW — Phase 4)
├── configs/
│   ├── preprocess.yaml              # (existing)
│   ├── train.yaml                   # (from Phase 2c)
│   └── eval.yaml                    # (NEW — Phase 4)
└── docs/
    ├── METHODOLOGY.md
    ├── PREPROCESSING_DOCUMENTATION.md
    ├── PHASE2_KICKOFF.md
    ├── POST_TRAINING_REFERENCE.md    # (this document)
    ├── PHASE2_RESULTS.md             # (from Phase 2 completion)
    ├── PHASE3_RESULTS.md             # (NEW)
    └── PHASE4_RESULTS.md             # (NEW)
```

### 5.2 Code conventions to maintain

- OmegaConf for config loading (consistent with `preprocess.py` and `train.py`)
- CLI via `argparse` with `--config configs/xxxx.yaml`
- Numbered stage logging (`[1/6 ...]`)
- tqdm progress bars
- Early-exit guards on output paths
- Persistent outputs to Drive, not `/content/`

### 5.3 Workflow cycle (same as Phase 1 and Phase 2)

1. **VS Code (Claude Code)**: write/edit code, modify configs
2. **PowerShell**: `git add . && git commit -m "..." && git push`
3. **Colab notebook**: `!git -C /content/dr-dissertation pull` then `!python -m src.module --config ...`

### 5.4 Compute requirements

- Phase 3 (MC Dropout): ~1-3 hours total compute (no retraining if dropout exists)
- Phase 4 (Inference + analysis): ~2-5 hours total compute
- Phase 4 (Cross-dataset transfer baseline training): ~8-16 hours (training a new EyePACS-only baseline)

These are GPU-hours on Colab T4. Significantly less than Phase 2's training compute.

---

## 6. Risk register — what might go wrong

### 6.1 Phase 3 risks

**Risk**: Calibration is poor (ECE > 0.20) across all heads.
**Mitigation**: Likely dropout placement issue. Re-architect projection head with explicit dropout layers, retrain Phase 2 partially (just the heads, freezing the encoder). 1-week delay if it happens.

**Risk**: MC Dropout inference is too slow for practical use.
**Mitigation**: Reduce N (number of samples) from 30 to 10, accept slightly noisier uncertainty estimates. Or batch inference better.

### 6.2 Phase 4 risks

**Risk**: Test set is too small for stable metrics (likely for biomarker head with ~20-30 tier-2 test samples).
**Mitigation**: Report metrics with bootstrap confidence intervals, acknowledge sample size in dissertation, frame as preliminary evidence rather than definitive results.

**Risk**: Longitudinal trajectory test shows the model has no predictive value beyond baseline severity.
**Mitigation**: Report honestly. This is still a valuable finding — it would mean "fundus imaging at baseline doesn't predict trajectory beyond what's already captured by current severity grading." A null result framed correctly is a legitimate contribution.

**Risk**: Cross-dataset transfer test shows the contrastive alignment didn't help.
**Mitigation**: Report honestly. Compare your full model against the EyePACS-only baseline; if they're equivalent, the contrastive contribution was minimal. Discussion chapter analyses why (e.g. ImageNet pretraining was already sufficient, OLIVES added marginal value).

### 6.3 Writing-phase risks

**Risk**: Late August submission rush with too much unwritten.
**Mitigation**: Start writing Methods and Datasets chapters early — these can be written based on documents you already have (METHODOLOGY.md, PREPROCESSING_DOCUMENTATION.md), without waiting for Phase 4 results.

**Risk**: Plots/figures don't meet university formatting requirements.
**Mitigation**: Check Bath's MSc dissertation template/guidelines early (during Phase 3). Apply formatting consistently from the start rather than reformatting at the end.

---

## 7. Reference numbers for sanity-checking

When working through Phase 3 and Phase 4, these numbers from earlier phases should remain consistent:

| Quantity | Value |
|---|---|
| EyePACS total samples | 35,108 |
| OLIVES total samples | 3,142 |
| OLIVES tier-1 (has clinical) | 1,594 |
| OLIVES tier-2 (has biomarker) | 192 |
| OLIVES patients | 96 |
| OLIVES patient-eyes | 204 |
| OLIVES visits per eye (mean) | 15.4 |
| Test set size (assuming 10% split) | ~10 OLIVES patients, ~3,500 EyePACS samples |
| Test set tier-2 size (estimated) | ~20-30 samples |
| Phase 1 baseline FID | 240.65 |
| Phase 2 target FID | < 100 |
| Phase 2 target MMD p-value | > 0.05 |
| Phase 2 target cosine similarity | > 0.70 |
| Usable biomarkers | 9 (DRT/ME, IRF, SRF, IR hemorrhages, EZ Disruption, plus 4 vitreous-related) |

If your code produces results inconsistent with these, something has gone wrong.

---

## 8. What to do as the new Claude reading this

If this document is uploaded to a new chat at the start of Phase 3 or Phase 4:

1. Read the relevant section (Phase 3 or Phase 4) thoroughly before proposing code.
2. Read `METHODOLOGY.md` and `PHASE2_KICKOFF.md` for context on what was built earlier.
3. **For Phase 3**: confirm that the user wants MC Dropout as the primary approach, then propose the implementation plan before generating code.
4. **For Phase 4**: confirm with the user that the test set has remained untouched throughout Phases 2 and 3. If not, help them construct an evaluation strategy that compensates.
5. Generate Claude Code prompts in the same style as the earlier phases — step-by-step, with validation checkpoints between sub-tasks.

The user prefers:
- Step-by-step prompts to paste into Claude Code in VS Code
- Push/pull workflow between PowerShell and Colab
- Validation steps after each sub-task
- Plain-language explanations alongside technical content

---

## 9. The end state

When all four phases complete, you have:

- A trained DR severity + biomarker classifier + BCVA/CST regressor model
- Confidence estimates accompanying every prediction
- Honest evaluation showing what the model can and cannot do
- The longitudinal trajectory test outcome (whatever direction it goes — still publishable)
- Cross-dataset transfer evidence supporting (or qualifying) the architectural contribution
- A documented dissertation contribution: combining EyePACS and OLIVES via shared encoder with contrastive alignment, with concrete before/after evidence

That's a complete MSc dissertation. From here, the only remaining work is writing it up.

Good luck with the remaining phases.
