# Dissertation Methodology

**Project**: Adaptive Multimodal Machine Learning for Diabetic Retinopathy Progression Prediction
**Institution**: University of Bath
**Submission deadline**: August 2026
**Document version**: Updated 27 April 2026 (revised after preprocessing reconciliation)

---

## Overview

This project is a purely computational study using two publicly available, de-identified ophthalmic datasets. No human participants are recruited, no new data is collected, and no clinical interventions are performed.

**Companion document**: see `docs/PREPROCESSING_DOCUMENTATION.md` for the data preprocessing pipeline that underpins this methodology, including dataset structure, three-tier label organisation, design decisions made during preprocessing, and reconciliation against the OLIVES paper's reported cohort.

The methodology proceeds in four phases:

1. **Distributional validation** — quantitative assessment of whether the two datasets share compatible feature distributions
2. **Model development** — shared encoder architecture with contrastive alignment and dual prediction heads
3. **Uncertainty quantification** — Bayesian or ensemble-based confidence estimates
4. **Evaluation** — standard classification and regression metrics, plus a longitudinal trajectory test

---

## Datasets and label structure

The study uses two datasets with structurally different label availability — a tiered structure that arose naturally from preprocessing reconciliation rather than being imposed externally.

**EyePACS** (35,108 fundus images): cross-sectional dataset with 5-class DR severity grading (No DR / Mild / Moderate / Severe / Proliferative). Used for shared encoder pretraining on the DR classification task.

**OLIVES** (3,142 unique fundus images from 96 patients across two clinical trials): multi-tier label structure:
- **Tier 0** — fundus only, 1,546 samples (49.2%): no clinical or biomarker labels. Used for self-supervised contrastive pretraining.
- **Tier 1** — fundus + clinical labels, 1,404 samples (44.7%): adds BCVA (Best Corrected Visual Acuity) and CST (Central Subfield Thickness) continuous regression targets.
- **Tier 2** — full labels, 190 samples (6.0%): adds 16 binary biomarker labels derived from OCT B-scan annotations.
- **Anomalous** — biomarker-only, 2 samples (0.06%): edge case handled by mask-based loss formulation.

Critically, OLIVES does NOT provide per-image DR severity grades. This means the DR severity classifier head trains on EyePACS only; OLIVES contributes longitudinal regression (BCVA, CST) and multi-label biomarker classification. DR severity features transfer to the OLIVES branch through the shared encoder via cross-dataset contrastive alignment.

**Important distinction on multimodality**: although OLIVES is described as a multimodal dataset (fundus + OCT + clinical), this study uses **fundus images as the only input modality**. OCT-derived biomarkers serve as *labels* (supervision targets), not as additional model inputs. The model never directly consumes OCT data — it learns to predict OCT-derived labels from fundus features alone. This is a clinically valuable capability since OCT acquisition is more expensive and not universally available.

The "missing modalities as a first-class constraint" framing — handling label availability that varies systematically across samples — is realised concretely through the three-tier structure rather than via synthetic missingness.

---

## Phase 1: Distributional validation

Before committing to the shared encoder architecture, the project assesses whether the two datasets share compatible feature distributions. This is a prerequisite — if EyePACS and OLIVES occupy entirely separate regions of feature space, no amount of contrastive alignment can bridge them effectively.

### Methodology

Features are extracted from fundus images in both datasets using a pretrained convolutional neural network (ResNet-50 with ImageNet1K-V2 weights). Each image is mapped to a 2048-dimensional feature vector by stripping the final classification layer. Six diagnostics are then computed on the extracted feature distributions:

1. **PCA visualisation** — projection of both feature distributions to a 2D plane.
2. **Fréchet distance (FID-style)** — multivariate-Gaussian distance between the two feature distributions.
3. **Maximum Mean Discrepancy (MMD)** — kernel-based distribution distance with permutation-test significance.
4. **k-NN cosine similarity** — average cosine similarity of samples in one dataset to their k=5 nearest neighbours in the other.
5. **PCA stratified by DR severity** — colours EyePACS samples by their 5-class DR severity grade with OLIVES overlay, to determine whether the dominant separation axis (PC1) reflects pathology or acquisition.
6. **Image statistics correlation with PC1** — Spearman rank correlation between PC1 coordinates and per-image acquisition statistics (mean intensity, black-pixel proportion, contrast).

Diagnostics 1-4 measure the magnitude of the domain shift; diagnostics 5-6 attribute the shift mechanistically.

### Findings

The Phase 1 analysis confirms a substantial but mechanistically explained domain shift:

| Diagnostic | Result | Interpretation |
|---|---|---|
| Fréchet distance | 240.65 | Substantial separation (>200 indicates significant shift) |
| MMD² (RBF kernel) | 0.1528, p<0.0001 | Statistically significant distributional difference |
| EyePACS→OLIVES k-NN cosine | 0.526 ± 0.068 | Moderate cross-dataset alignment in full 2048-dim space |
| OLIVES→EyePACS k-NN cosine | 0.557 ± 0.070 | Slight asymmetry favouring EyePACS-side density |
| PCA explained variance (PC1+PC2) | 35.3% | 2D projection is indicative, not definitive |

The mechanism-attribution diagnostics produce the decisive finding:

| Mechanism evidence | Result |
|---|---|
| EyePACS DR severity class-mean spread on PC1 | 0.82 |
| EyePACS↔OLIVES centroid distance on PC1 | 13.06 |
| Ratio (centroid distance / severity spread) | 16× — PC1 captures dataset-of-origin, not pathology |
| Strongest PC1 correlation in EyePACS | Black-pixel proportion: ρ=-0.274, p=1.12×10⁻³⁵ |

The dominant separation axis (PC1, 23.3% of variance) is acquisition-driven rather than pathology-driven: DR severity classes intermix on PC1, while black-pixel proportion (a proxy for field-stop border) correlates extremely strongly with PC1 in EyePACS. The shift between datasets is therefore primarily an artefact of image-acquisition geometry (camera field of view, framing, exposure) rather than a difference in retinal disease content.

### Implication for Phase 2

This is the favourable scenario for cross-dataset contrastive alignment. Acquisition-driven shifts are precisely what contrastive learning suppresses when paired samples (here: same DR severity from different datasets) are pulled together — the encoder is forced to identify and discard variation that doesn't correspond to the pairing signal. Pathology-relevant features, which already live orthogonally to PC1 in raw ResNet space, can be preserved and amplified by the alignment objective.

Phase 1 also establishes a quantitative success criterion for Phase 2. After training the shared encoder, the same six diagnostics will be re-run on the *learned* features. Successful contrastive alignment should produce: substantially reduced FID (target <100), MMD non-significant or much weaker (target p>0.05 or MMD² <0.05), increased cross-dataset cosine similarity (target >0.7), and reduced PC1-vs-acquisition-stat correlation (target |ρ|<0.1). DR severity classes should now spread on the dominant axis as pathology-relevant features take over the principal direction.

A pixel-level domain shift between the datasets had already been characterised at the preprocessing stage (per-channel mean shift of ~0.7 driven by EyePACS's ~20% black-border field-stop geometry vs OLIVES's ~5%). The Phase 1 ResNet-feature analysis confirms this pixel-level acquisition difference propagates to feature space and is the dominant source of separation — providing strong, mechanism-level motivation for the cross-dataset alignment in Phase 2.

---

## Phase 2: Model development

A shared encoder architecture processes both EyePACS and OLIVES fundus images, projecting them into a joint latent space where contrastive alignment via cosine similarity enforces that similar disease patterns from both datasets produce similar representations.

**Shared encoder**: a single CNN backbone (ResNet-50 fine-tuned from ImageNet pretraining is the baseline choice; alternatives like EfficientNet or Vision Transformer may be explored). The same weights process images from both datasets, which forces the encoder to learn features generalisable across acquisition equipment and patient populations.

**Joint latent space**: the high-dimensional feature space output by the encoder. Samples from both datasets project into this shared space, enabling direct comparison and contrastive learning across datasets.

**Contrastive alignment loss**: pulls together latent representations of related samples (e.g. fundus images sharing the same DR severity, even from different datasets) and pushes apart unrelated samples. Cosine similarity is the chosen distance metric for measuring proximity in latent space.

**Dual output heads**: attached to the shared encoder's output:
- **Softmax DR severity classifier** (5 classes): trained on EyePACS labels. The OLIVES branch inherits DR severity features through the shared encoder rather than via direct supervision.
- **Multi-label biomarker classifier** (BCE loss): trained on OLIVES tier-2 samples (190 samples), with masking via `has_biomarkers`. Predicts presence/absence of biomarkers visible on OCT but inferable from fundus features.
- **FCN regressor** (BCVA, CST): trained on OLIVES tier-1 samples (1,404 samples), with masking via `has_clinical & ~isnan`. Predicts continuous clinical outcome measurements.

**Mask-based loss formulation**: the three-tier label structure means each loss component applies only to the subset of samples with the relevant labels. Mask tensors (`has_clinical`, `has_biomarkers`) are stored alongside images in the preprocessed `.pt` file and consumed at training time to gate the per-sample loss contributions. This handles the missing-modalities constraint cleanly without requiring data augmentation or label imputation.

**Patient-aware data splitting**: train/val/test splits are constructed at the patient level (not visit-eye or sample level) to prevent leakage of the same eye's longitudinal trajectory across splits. With 96 patients in OLIVES, candidate splits include 70/20/10 (~67/19/10 patients), 70/15/15, or 80/10/10 — chosen with attention to validation set adequacy given the small cohort.

---

## Phase 3: Uncertainty quantification

Bayesian or ensemble-based methods are integrated to provide calibrated confidence estimates alongside predictions, distinguishing cases where the model is confident from those requiring clinical review.

Candidate approaches:
- **Monte Carlo Dropout**: enable dropout at inference time and run multiple forward passes; the variance across passes provides an uncertainty estimate.
- **Deep Ensembles**: train multiple instances of the model with different random initialisations; the disagreement across ensemble members provides uncertainty.
- **Bayesian Neural Networks** (e.g. variational inference, Laplace approximation): explicitly model weight distributions rather than point estimates.

The choice depends on computational cost (training time, inference latency) and calibration quality. Calibration is assessed via reliability diagrams and expected calibration error (ECE).

---

## Phase 4: Evaluation

Model performance is assessed using standard metrics for each output head, plus a longitudinal trajectory test that tests the dissertation's core hypothesis.

**Classification metrics** (DR severity head, biomarker head):
- Accuracy, AUC (area under ROC curve), F1-score
- Per-class precision/recall to capture minority-class performance
- For multi-label biomarker classification: per-biomarker AUC, micro/macro F1

**Regression metrics** (BCVA, CST):
- Mean Absolute Error (MAE) — direct interpretability in clinical units
- Concordance index (c-index) — rank-based metric standard for clinical prediction tasks
- Coefficient of determination (R²)

**Longitudinal trajectory test** (the core evaluation):
The key dissertation hypothesis is that the model can distinguish patients with the **same current DR grade** but **different future trajectories**. Operationalisation: stratify OLIVES patients by baseline severity (similar grade at first visit), then assess whether predicted BCVA/CST at a held-out future visit ranks correctly relative to actual outcomes. This tests genuine prognostic capability beyond cross-sectional classification.

**Cross-dataset transfer evaluation**: separate from the OLIVES-specific tests, the encoder's transferability is evaluated by training on EyePACS and probing performance on OLIVES (and vice versa) without retraining. This quantifies whether the contrastive alignment has produced genuinely cross-domain features.

---

## Implementation environment

All computation is performed using Python (PyTorch) on GPU-enabled cloud infrastructure (Google Colab Pro). Both EyePACS and OLIVES were collected under institutional ethics approval at their respective source institutions and are publicly available for research use.

Code is version-controlled on GitHub at `savita10/dr-dissertation`. Preprocessed artefacts (~24 GB total: 22 GB EyePACS shards + 1.8 GB OLIVES tensor) are stored on Google Drive. Documentation, including the preprocessing reconciliation report, is maintained in `docs/` alongside the source.

---

## Methodological design decisions made during preprocessing

A note on rigour: three preprocessing decisions worth flagging in the dissertation methods chapter as deliberate methodological choices rather than routine cleanup:

1. **Biomarker aggregation rule**: per-OCT-scan biomarker annotations are aggregated to visit-eye granularity using a max ("any-positive") rule rather than first-row deduplication. This matches how a clinician would summarise biomarker presence at the visit level (positive on any scan → visit positive) and recovers ~3× more positive labels than first-row dedup.

2. **`.tif`/`.png` format deduplication**: Prime_FULL stores the same fundus image in both lossless `.tif` and lossy `.png` formats. Preprocessing keeps the `.tif` version and discards the `.png` to avoid 2× duplication that would bias training toward Prime_FULL patterns and create patient leakage in patient-aware splits.

3. **Three-tier label structure preserved during preprocessing**: rather than discarding fundus images without full labels (the original 272-sample state), preprocessing retains all 3,142 unique fundus images with explicit per-tier mask tensors. This enables the SSL contrastive component to use unlabelled tier-0 samples while supervised components operate on their respective labelled subsets.

These decisions collectively reconcile the preprocessed dataset exactly with the cohort statistics reported in the OLIVES paper (Prabhushankar et al., NeurIPS 2022): 96 patients, 204 patient-eyes, ~15.4 visits per eye on average.

---

## Summary of changes from original methodology document

This version differs from the pre-preprocessing methodology in three substantive respects:

1. **DR severity classification scope**: clarified that the 5-class softmax classifier trains on EyePACS only, since OLIVES does not provide per-image DR severity grades. OLIVES branch inherits DR features through the shared encoder.

2. **OLIVES input modality**: clarified that this study uses fundus images as the sole input modality. OCT-derived biomarkers are supervision targets, not model inputs. The model predicts OCT findings from fundus alone.

3. **Three-tier label structure**: explicitly documented as a property of OLIVES discovered during preprocessing, with the mask-based loss formulation as the architectural mechanism for handling it.

The four phases, evaluation strategy, uncertainty quantification approach, and overall framing of the dissertation contribution are unchanged.
