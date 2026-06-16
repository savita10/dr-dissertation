# Phase 2 Kickoff — Model Development (v2, revised)

**Project**: Adaptive Multimodal ML for Diabetic Retinopathy Progression Prediction
**Institution**: University of Bath, MSc dissertation
**Submission deadline**: August 2026
**Phase 2 start**: May 2026
**Revision**: v2 — incorporates the agreed Phase 2a design review (multi-term loss with explicit CORAL alignment, augmentation pipeline specification, correctness-trap mitigations, `configs/model.yaml`, and the 2048-d diagnostic decision).

This document is the single self-contained reference for Phase 2. Anyone reading only this document — including a future you — has enough context to generate every Phase 2 implementation prompt without the preprocessing history.

> **Change log vs v1** (what this revision adds)
> - Loss design expanded from SupCon-only to a **four-active-term loss** (DR CE + biomarker BCE + BCVA/CST MSE + SupCon) **plus an explicit CORAL cross-dataset alignment term**, with an **optional NT-Xent two-view SSL term** built but defaulted off (weight 0).
> - Dropped the tier-0 "pseudo-class from biomarker presence" idea (would confound biomarker presence with DR severity on the very axis we are cleaning).
> - Added the **augmentation pipeline specification** (field-stop-targeting), applied identically to both datasets, with the **de-normalise → augment → re-normalise** wrapper because saved tensors are already ImageNet-normalised.
> - Added the **correctness traps** as defaults: empty-mask guards, regression target standardisation, `~isnan(bcva)` guard, pinned **9-of-16 biomarker index list**, class-weighted CE, CORAL single-dataset-batch guard.
> - Added **`configs/model.yaml`** to the architecture layer.
> - Fixed the **2048-d encoder output** as the representation for the Phase 2d diagnostics rerun (apples-to-apples with Phase 1).

---

## 1. Project context (the one-page version)

The dissertation builds a deep learning model that learns from two diabetic retinopathy fundus image datasets simultaneously and uses cross-dataset knowledge transfer to predict both DR severity and longitudinal disease progression.

**Two methodological contributions**:
1. **Longitudinal prognosis** — predict disease progression over time from sequential per-patient visits, using BCVA (visual acuity) and CST (retinal thickness) as continuous outcome measurements.
2. **Cross-modality knowledge transfer** — a single shared encoder processes both datasets, with explicit contrastive *and* distribution-level alignment in the latent space. The encoder learns features generalisable across acquisition equipment and patient populations.

**Architecture**: shared encoder + projection head + three output heads (DR softmax classifier + multi-label biomarker BCE classifier + FCN regressor for BCVA/CST).

**Distinctive framing**: "missing modalities as a first-class constraint" — different OLIVES samples have different label availability (the three-tier structure below), handled via mask-based loss formulation rather than synthetic missingness or sample discarding.

---

## 2. Datasets (final state, post-preprocessing)

### EyePACS
- **35,108 fundus images** from US screening clinics
- **5-class DR severity labels**: 0 (No DR, 73.5%) / 1 (Mild, 6.9%) / 2 (Moderate, 15.1%) / 3 (Severe, 2.5%) / 4 (Proliferative, 2.0%)
- Heavily class-imbalanced (canonical for EyePACS) → **class-weighted cross-entropy** on the DR head
- Stored as 8 shard `.pt` files at `/content/drive/MyDrive/dissertation/preprocessed/eyepacs/eyepacs_shard_*.pt` (7×5,000 + 1×108)

### OLIVES
- **3,142 unique fundus images** from 96 patients (204 patient-eyes) across two clinical trials (TREX DME + Prime)
- **Three-tier label structure**:
  - **Tier 0** (1,546 samples, 49.2%): fundus image only — no clinical, no biomarker labels. SSL contrastive pretraining material.
  - **Tier 1** (1,404 samples, 44.7%): + BCVA (continuous, ~0–100 ETDRS letters) + CST (continuous, microns) clinical regression targets.
  - **Tier 2** (190 samples, 6.0%): + 16 binary biomarker labels (of which **9 usable** — see Section 5.3).
  - **Anomalous** (2 samples): biomarker labels but no clinical labels (handled by masking).
  - Therefore `has_clinical.sum() = 1,594` (tier-1 + tier-2) and `has_biomarkers.sum() = 192` (tier-2 + anomalous).
- Stored as a single `.pt` file at `/content/drive/MyDrive/dissertation/preprocessed/olives/olives_fundus.pt` (1.8 GB)

### Cross-dataset compatibility
Both datasets share identical preprocessing: 3×224×224 float32, **ImageNet normalisation already applied** (means [0.485, 0.456, 0.406], stds [0.229, 0.224, 0.225]). The shared encoder can mechanically consume either. **Because the tensors are pre-normalised, photometric augmentation must de-normalise first** (Section 5.4).

---

## 3. OLIVES `.pt` file schema (what the Phase 2 dataset class consumes)

```python
torch.load("olives_fundus.pt") returns:
{
    "images":         torch.float32 [3142, 3, 224, 224],  # ImageNet-normalised
    "labels":         torch.int64   [3142],               # Binary DRIL flag — IGNORE (see note)
    "metadata":       list[dict]    of 3142 items,        # {trial, patient_id, visit, eye}
    "biomarkers":     torch.float32 [3142, 16],           # NaN where has_biomarkers=False
    "bcva":           torch.float32 [3142],               # NaN where has_clinical=False
    "cst":            torch.float32 [3142],               # NaN where has_clinical=False
    "has_clinical":   torch.bool    [3142],               # Mask for tier-1 loss
    "has_biomarkers": torch.bool    [3142],               # Mask for tier-2 loss
    "patient_ids":    torch.int64   [3142],               # -1 where missing (patient-aware splits)
    "eye_ids":        torch.int64   [3142],               # -1 where missing
    "keys":           list[tuple]   of 3142 items,        # (trial, patient_id, visit, eye)
}
```

**Critical notes for the Phase 2 dataset class**:
- The `labels` field stores a binary DRIL flag (one of the 16 biomarkers). It is **NOT** a 5-class DR severity label. Phase 2 must ignore it and use `biomarkers` + `has_biomarkers` instead.
- `has_clinical=True` does NOT guarantee BCVA/CST are valid. One sample (patient `0247GOD`, V3, OD) has `has_clinical=True` but `BCVA=NaN`. Regression loss must additionally check `~torch.isnan(bcva)` (and `cst`).

```python
# Correct masking pattern for regression loss
valid = has_clinical & ~torch.isnan(bcva) & ~torch.isnan(cst)
loss_bcva = mse(pred_bcva[valid], bcva_std[valid])   # bcva_std = standardised target
```

EyePACS shard `.pt` files have a simpler schema:
```python
{
    "images":    torch.float32 [N, 3, 224, 224],  # ImageNet-normalised
    "labels":    torch.int64   [N],               # DR severity 0-4
    "filenames": list[str]     of N items,
}
```

---

## 4. Phase 1 outcomes (what informs Phase 2 design)

Six diagnostics on raw ResNet-50 ImageNet features:

| Diagnostic | Result | Interpretation |
|---|---|---|
| Fréchet distance | 240.65 | Substantial domain shift (>200 = significant) |
| MMD² (RBF) | 0.153, p<0.0001 | Statistically significant distributional difference |
| k-NN cosine similarity | 0.55 (mean) | Moderate cross-dataset alignment |
| PCA on combined features | PC1 23.3% variance | Single axis dominates separation |
| PCA stratified by DR severity | EyePACS classes intermix on PC1 | PC1 does NOT track disease (spread 0.82 vs centroid gap 13.06 — 16× ratio) |
| PC1 vs image stats (Spearman) | ρ=-0.274 with black-pixel proportion, p=1.12×10⁻³⁵ | PC1 tracks acquisition (field-stop geometry), not pathology |

**Conclusion**: the domain shift is **acquisition-driven, not pathology-driven**. EyePACS images have ~20% black border (camera field-stop); OLIVES ~5%. This is exactly what cross-dataset alignment is designed to suppress — and is the direct motivation for both the **field-stop-targeting augmentation pipeline** (Section 5.4) and the **CORAL alignment term** (Section 5.2).

**Phase 2 success criteria** (re-running the same six diagnostics on the trained **2048-d encoder** embeddings):

| Metric | Phase 1 baseline | Phase 2 target |
|---|---|---|
| FID | 240.6 | < 100 (ideally < 50) |
| MMD p-value | < 0.0001 | > 0.05 (not significant) |
| Cosine similarity | 0.55 | > 0.70 |
| Black-border ↔ PC1 correlation | ρ = -0.27 | \|ρ\| < 0.1 |
| DR class spread on PC1 | Intermixed | Ordered by severity |

---

## 5. Architecture and loss design (confirmed)

### 5.1 Encoder backbone — ResNet-50, ImageNet1K-V2 pretrained
- `torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)`
- Strip the final FC layer (replace with `nn.Identity`) → **2048-dim encoder features**.
- Add a **projection head** (2 linear layers, 2048 → 512, BatchNorm + ReLU between) for the contrastive terms (standard SimCLR/SupCon pattern; applying contrastive loss directly on the 2048-d output over-constrains feature learning).
- Fine-tune the whole stack (do not freeze the backbone).
- **Forward returns both** `features` (2048-d) and `projection` (512-d). Heads consume `features`; contrastive/SupCon/NT-Xent consume `projection`; **CORAL consumes `features`** (see 5.2).
- **Rationale**: simpler than RETFound, easier comparison against literature baselines (Sükei et al., Kokilepersaud et al. both use ResNet-50).

### 5.2 Loss design — four active terms + explicit alignment (+ one optional)

The original SupCon-only design structures EyePACS by DR class but does **not** couple OLIVES to EyePACS (OLIVES has no DR labels). Since the headline success metric is the *cross-dataset* FID, an explicit alignment term is included.

| # | Term | Operates on | Subset it sees | Active by default | Config weight |
|---|---|---|---|---|---|
| 1 | DR classification | `dr_logits` (2048→head) | EyePACS only (`has_dr`) | ✅ | `w_dr` |
| 2 | Biomarker classification | `biomarker_logits` | OLIVES tier-2 (`has_biomarkers`) | ✅ | `w_biomarker` |
| 3 | BCVA/CST regression | `regression` | OLIVES tier-1 (`has_clinical & ~isnan`) | ✅ | `w_regression` |
| 4 | Supervised contrastive (SupCon) | `projection` (512) | EyePACS only (`has_dr`) | ✅ | `w_supcon` |
| 5 | **CORAL cross-dataset alignment** | **`features` (2048)** | **both datasets in batch** | ✅ | `w_coral` |
| 6 | NT-Xent two-view SSL | `projection` (512) | both datasets, two views | ⛔ (weight 0) | `w_ntxent` |

- **SupCon** (Khosla et al. 2020): positives = same DR class within the EyePACS subset of the batch; temperature τ=0.07. Requires ≥2 EyePACS samples and ≥1 positive pair, else returns 0.
- **CORAL** (Sun & Saenko 2016): minimises the Frobenius distance between the covariance of the EyePACS-subset features and the OLIVES-subset features:
  `L_coral = ‖Cov(F_eyepacs) − Cov(F_olives)‖²_F / (4·d²)`, with d=2048.
  - **Applied on the 2048-d `features`, not the 512-d projection**, because that is the representation the FID success metric evaluates and where alignment must land (the projection can otherwise absorb the invariance without it propagating to the backbone).
  - **Single-dataset-batch guard**: if either dataset subset has fewer than `coral_min_samples` (config, default 2) samples in the batch, CORAL returns a **zero scalar** — never NaN or error. (Covariance from a tiny per-domain subset in 2048-d is also poorly conditioned; the 2c sampler should guarantee a healthy OLIVES quota per joint batch, e.g. ≥16. If instability appears, a documented fallback is an EMA running covariance per domain.)
- **NT-Xent** (SimCLR): optional self-supervised term over two augmented views, letting tier-0 OLIVES (49% of OLIVES) contribute. Built with a config-exposed weight defaulting to **0**; enable later only if the four-term + CORAL configuration trains stably and time allows.
- **Dropped**: the tier-0 "pseudo-class from biomarker presence" pairing idea (confounds biomarker presence with DR severity on the alignment axis).

**Combined loss** returns the weighted total **and** a dict of per-term scalar values for logging:
`L = w_dr·L_dr + w_biomarker·L_bio + w_regression·L_reg + w_supcon·L_supcon + w_coral·L_coral + w_ntxent·L_ntxent`

### 5.3 Biomarker head targets — 9 usable biomarkers (pinned index list)

The 9 usable biomarkers (≥3 positives in the 192 tier-2 samples, excluding the saturated IR HRF) selected from the 16-wide `biomarkers` tensor. **Their 0-indexed positions in the 16-wide tensor and expected positive counts:**

| Head output | Biomarker | Tensor index (0-based) | Expected positives (of 192) |
|---|---|---|---|
| 0 | Disruption of EZ | 1 | 68 |
| 1 | IR hemorrhages | 3 | 67 |
| 2 | Partially attached vitreous face | 5 | 110 |
| 3 | Fully attached vitreous face | 6 | 134 |
| 4 | Preretinal tissue/hemorrhage | 7 | 48 |
| 5 | Vitreous debris | 8 | 164 |
| 6 | DRT/ME | 10 | 114 |
| 7 | Fluid (IRF) | 11 | 174 |
| 8 | Fluid (SRF) | 12 | 29 |

```python
USABLE_BIOMARKER_INDICES = [1, 3, 5, 6, 7, 8, 10, 11, 12]   # length 9
```

> **VERIFY in Phase 2b**: the index list above assumes the 16-wide tensor column order matches the Section 4.2 ordering of the preprocessing documentation. Before trusting it, the 2b data-loading checkpoint must select these 9 columns from the real tensor, sum positives over `has_biomarkers` rows, and assert the counts equal `[68, 67, 110, 134, 48, 164, 114, 174, 29]`. A mismatch means the tensor column order differs and the index list must be corrected — this is the off-by-one trap that would otherwise train heads on the wrong columns silently.

- Loss: **masked `BCEWithLogitsLoss`** with per-column `pos_weight = neg/pos` computed **from the training-split tier-2 samples** (not all 192, to avoid leakage).
- Caveat: only ~192 positive-bearing samples (some columns <30 positives). The head will overfit fast; run it as a genuinely **auxiliary** head at modest `w_biomarker`, and report per-biomarker AUC with confidence intervals at evaluation.

### 5.4 Augmentation pipeline (field-stop-targeting)

Phase 1 located the dataset separation on the field-stop/black-border axis. The augmentation pipeline attacks that axis directly and **is applied identically to both datasets** so their augmented input ranges overlap (domain randomisation — a heavily-cropped EyePACS image looks, in border proportion, like an OLIVES image).

Pipeline (in order), applied per image at train time:
1. **De-normalise** to [0,1] (undo ImageNet normalisation) — required before any photometric op.
2. `RandomResizedCrop(size=224, scale=(0.6, 1.0), ratio=(0.9, 1.1))` — varies visible field-stop area; tightened aspect ratio preserves the circular fundus disc.
3. `ColorJitter(brightness=0.2, contrast=0.2)` — mimics camera/exposure variation. *(Saturation/hue left off by default; fundus colour carries some signal.)*
4. `RandomRotation(degrees=15)` — fundus orientation carries no clinical signal; corner fill = 0 in [0,1] (consistent with existing dark border).
5. `RandomHorizontalFlip(p=0.5)` — same rationale.
6. **Re-normalise** with ImageNet mean/std.

Validation transform: deterministic resize/centre-crop to 224 + (no de-norm needed if reading already-normalised tensors and skipping photometric ops) — kept simple and label-preserving.

> **Why the de-norm wrapper is mandatory**: the saved tensors are already ImageNet-normalised (values centred near 0, some negative). `ColorJitter` brightness multiplies and contrast blends toward the mean; on normalised tensors this does not correspond to a real exposure/contrast change and silently degrades to noise. De-norm → augment in [0,1] → re-norm makes the photometric ops physically meaningful.

**Two-view note (affects 2a + 2b)**: NT-Xent (term 6) needs two independently-augmented views per image. The data layer must be able to emit a second view; the 2a `combined_loss` signature accepts an optional `projection_v2`. When `w_ntxent=0` (default), only one view is forwarded for efficiency.

### 5.5 Correctness traps folded in as defaults

These run without error but produce wrong numbers if missed:
1. **Empty-mask guards** — every masked loss (DR, biomarker, regression, SupCon, CORAL) returns a clean **0.0** (never NaN) when its relevant subset is empty in a batch.
2. **Regression target standardisation** — BCVA (~0–100) and CST (~200–600) are z-scored using mean/std computed on the **training-split tier-1** samples; stats persisted to `regression_stats.json` on Drive and inverted at evaluation. Without this, CST dominates MSE and BCVA barely trains.
3. **`~isnan` guard** on BCVA *and* CST on top of `has_clinical`.
4. **Pinned biomarker index list** (Section 5.3), validated in 2b.
5. **Class-weighted CE** for the DR head, weights ∝ inverse class frequency from the training split.
6. **CORAL single-dataset-batch guard** (Section 5.2) — returns 0 when either domain subset is too small.

### 5.6 `configs/model.yaml` (new)

Architecture/loss hyperparameters live here in OmegaConf style (consistent with `configs/preprocess.yaml`):

```yaml
# Architecture
backbone: resnet50
pretrained_weights: IMAGENET1K_V2
encoder_dim: 2048
projection_dim: 512
num_dr_classes: 5
num_biomarkers: 9
usable_biomarker_indices: [1, 3, 5, 6, 7, 8, 10, 11, 12]

# Loss weights
loss_weights:
  w_dr: 1.0
  w_biomarker: 0.5      # auxiliary — small label set
  w_regression: 1.0
  w_supcon: 0.5
  w_coral: 1.0
  w_ntxent: 0.0         # built but deferred

# Loss hyperparameters
supcon_temperature: 0.07
ntxent_temperature: 0.07
coral_min_samples: 2    # per-domain minimum before CORAL contributes
```

(Loss weights are starting points to tune in Phase 2c; they are deliberately exposed so the **"with vs without CORAL" ablation** is a one-line config change.)

---

## 6. Workflow context (how the project is built)

### Repository
- GitHub: `github.com/savita10/dr-dissertation`
- Local: VS Code workspace
- Colab: `/content/dr-dissertation` (cloned fresh each session)
- Drive (persistent): `/content/drive/MyDrive/dissertation/`

### Development cycle
1. **VS Code**: write/edit code in `src/`, configs in `configs/`.
2. **PowerShell**: `git add . && git commit -m "..." && git push`.
3. **Colab notebook**: `!git -C /content/dr-dissertation pull` then `!python -m src.module.name --config configs/whatever.yaml`.

### Repository structure (after Phase 2)
```
dr-dissertation/
├── src/
│   ├── data/
│   │   ├── preprocess.py              # Phase 0 (complete)
│   │   ├── transforms.py              # NEW 2b — augmentation + de-norm wrapper
│   │   ├── datasets.py                # NEW 2b — EyePACS + OLIVES datasets
│   │   ├── samplers.py                # NEW 2b — patient-aware splits + joint batching
│   │   └── dataloaders.py             # NEW 2b — DataLoader construction
│   ├── models/
│   │   ├── __init__.py                # NEW 2a
│   │   ├── encoder.py                 # NEW 2a — ResNet-50 + projection head
│   │   ├── heads.py                   # NEW 2a — DR / biomarker / regression heads
│   │   ├── model.py                   # NEW 2a — wrapper, single forward
│   │   ├── losses.py                  # NEW 2a — all loss terms + combined loss
│   │   └── smoke_test.py              # NEW 2a — CLI validation
│   ├── training/
│   │   ├── __init__.py                # NEW 2c
│   │   ├── trainer.py                 # NEW 2c — training loop
│   │   └── train.py                   # NEW 2c — entry point
│   ├── analysis/
│   │   ├── feature_extraction.py      # Phase 1 (reuse in 2d)
│   │   ├── distribution_diagnostics.py# Phase 1 (reuse in 2d)
│   │   └── phase2_diagnostics.py      # NEW 2d — thin wrapper, trained encoder
│   └── utils/
│       └── config.py                  # OmegaConf loader (existing)
├── configs/
│   ├── preprocess.yaml                # existing
│   ├── model.yaml                     # NEW 2a
│   └── train.yaml                     # NEW 2c
└── notebooks/                         # (Drive, not in repo)
    ├── phase2_training.ipynb          # NEW 2c
    └── phase2_evaluation.ipynb        # NEW 2d
```

### Code conventions to match
- **Config loading**: OmegaConf via the existing loader pattern.
- **CLI**: `argparse` with `--config configs/whatever.yaml`.
- **Logging**: numbered stages `[1/6 ...]`, tqdm bars with throughput.
- **Idempotent guards**: check `output_path.exists()` and skip with a message.
- **Persistent outputs to Drive**: never write to `/content/` (wiped).
- **Thin notebooks**: everything CLI-callable; the notebook just orchestrates.

### Dependencies (already in Colab)
`torch`, `torchvision`, `tqdm`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `omegaconf`.

---

## 7. Phase 2 implementation plan (summary)

Four sub-phases, each its own implementation prompt, each gated by a validation checkpoint. (Full decomposition with cross-boundary loss composition is in the companion `PHASE2_IMPLEMENTATION_PLAN.md`.)

### Phase 2a — Model architecture + losses
New files: `src/models/{__init__,encoder,heads,model,losses,smoke_test}.py`, `configs/model.yaml`.
Validation: `python -m src.models.smoke_test` — instantiate from config, forward dummy batch, check shapes, exercise mixed/single-dataset/all-tier-0 batches, assert masked + CORAL losses return 0 (not NaN) on empty subsets, assert combined loss is finite and backprops.

### Phase 2b — Data loading + augmentation + patient-aware sampling
New files: `src/data/{transforms,datasets,samplers,dataloaders}.py`, `regression_stats.json` (Drive).
Validation: load a batch from each loader; verify the **batch contract** (Section 5 of the plan doc); verify the three tiers' masks; **verify the 9 biomarker columns reproduce `[68,67,110,134,48,164,114,174,29]`**; verify augmented views de-normalise/re-normalise correctly; confirm patient-level split has no patient ID across train/val/test.

### Phase 2c — Training pipeline
New files: `src/training/{__init__,trainer,train}.py`, `configs/train.yaml`, `notebooks/phase2_training.ipynb`.
Validation: 2-epoch run on tiny subsets (1,000 EyePACS + 200 OLIVES) — pipeline runs end-to-end, per-term losses logged, joint batches carry a CORAL signal, best-model checkpoint written to Drive.

### Phase 2d — Post-training diagnostics rerun
New files: `src/analysis/phase2_diagnostics.py`, `notebooks/phase2_evaluation.ipynb`.
Re-run the six Phase 1 diagnostics on the trained **2048-d encoder** embeddings → `phase2_report.md` before/after table. Mostly reuses `distribution_diagnostics.py`.

---

## 8. Phase 2 deliverables

1. **Trained checkpoint** at `/content/drive/MyDrive/dissertation/checkpoints/phase2_best.pt` (~150 MB)
2. **Trained-encoder embeddings** at `/content/drive/MyDrive/dissertation/features/`
3. **`regression_stats.json`** (BCVA/CST mean+std for inversion)
4. **`phase2_report.md`** — before/after distributional comparison (2048-d)
5. **Per-head + per-loss-term training curves**
6. **`PHASE2_RESULTS.md`** — architecture, hyperparameters, final metrics, with/without-CORAL ablation

---

## 9. User preferences (for whoever generates prompts)

- Step-by-step prompts to paste into the coding assistant in VS Code.
- Push/pull workflow between PowerShell and Colab.
- Validation step after each sub-phase before moving on.
- Plain-language explanations alongside technical content (working professional, technically literate, not an ML researcher).
- **Share plans before code**; Phase 2 is too large to do speculatively.

---

## 10. Reference numbers (sanity checks)

| Quantity | Value |
|---|---|
| EyePACS samples | 35,108 |
| OLIVES total samples | 3,142 |
| OLIVES `has_clinical` (tier-1 + tier-2) | 1,594 |
| OLIVES `has_biomarkers` (tier-2 + anomalous) | 192 |
| OLIVES tier-2 (full labels) | 190 |
| OLIVES patients | 96 (56 TREX DME + 40 Prime) |
| OLIVES patient-eyes | 204 |
| Visits per eye (avg) | 15.4 |
| Encoder feature dim | 2048 |
| Projection dim | 512 |
| Usable biomarkers | 9 (indices `[1,3,5,6,7,8,10,11,12]`) |
| Image shape | 3 × 224 × 224 |
| EyePACS class 0 fraction | 73.5% |
| Biomarker positive counts (9 cols) | [68,67,110,134,48,164,114,174,29] |

If your tensors don't match these, something is wrong.
