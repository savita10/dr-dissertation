# Phase 2 Kickoff — Model Development

**Project**: Adaptive Multimodal ML for Diabetic Retinopathy Progression Prediction
**Institution**: University of Bath, MSc dissertation
**Submission deadline**: August 2026
**Phase 2 start**: May 2026

This document is the entry point for Phase 2 in a new chat session. It is self-contained: a new Claude reading this document will have enough context to begin generating Phase 2 implementation prompts without needing the full preprocessing history.

---

## 1. Project context (the one-page version)

The dissertation builds a deep learning model that learns from two diabetic retinopathy fundus image datasets simultaneously and uses cross-dataset knowledge transfer to predict both DR severity and longitudinal disease progression.

**Two methodological contributions**:
1. **Longitudinal prognosis** — predict disease progression over time from sequential per-patient visits, using BCVA (visual acuity) and CST (retinal thickness) as continuous outcome measurements.
2. **Cross-modality knowledge transfer** — a single shared encoder processes both datasets, with contrastive alignment via cosine similarity in the latent space. The encoder learns features generalisable across acquisition equipment and patient populations.

**Architecture**: shared encoder + three output heads (DR softmax classifier + multi-label biomarker BCE classifier + FCN regressor for BCVA/CST).

**Distinctive framing**: "missing modalities as a first-class constraint" — different OLIVES samples have different label availability (the three-tier structure described below), handled via mask-based loss formulation rather than synthetic missingness or sample discarding.

---

## 2. Datasets (final state, post-preprocessing)

### EyePACS
- **35,108 fundus images** from US screening clinics
- **5-class DR severity labels**: 0 (No DR, 73.5%) / 1 (Mild, 6.9%) / 2 (Moderate, 15.1%) / 3 (Severe, 2.5%) / 4 (Proliferative, 2.0%)
- Heavily class-imbalanced (canonical for EyePACS)
- Stored as 8 shard `.pt` files at `/content/drive/MyDrive/dissertation/preprocessed/eyepacs/eyepacs_shard_*.pt`

### OLIVES
- **3,142 unique fundus images** from 96 patients (204 patient-eyes) across two clinical trials (TREX DME + Prime)
- **Three-tier label structure**:
  - **Tier 0** (1,546 samples, 49.2%): fundus image only — no clinical, no biomarker labels. Used for SSL contrastive pretraining.
  - **Tier 1** (1,404 samples, 44.7%): + BCVA (continuous, ~0-100 letters) + CST (continuous, microns) clinical regression targets.
  - **Tier 2** (190 samples, 6.0%): + 16 binary biomarker labels (of which 9 are usable for classification — see Section 5).
  - **Anomalous** (2 samples): biomarker labels but no clinical labels (edge case handled by masking).
- Stored as a single `.pt` file at `/content/drive/MyDrive/dissertation/preprocessed/olives/olives_fundus.pt` (1.8 GB)

### Cross-dataset compatibility
Both datasets share identical preprocessing: 3×224×224 float32, ImageNet normalisation (means [0.485, 0.456, 0.406], stds [0.229, 0.224, 0.225]). The shared encoder can mechanically consume either.

---

## 3. OLIVES `.pt` file schema (what Phase 2 dataset class consumes)

```python
torch.load("olives_fundus.pt") returns:
{
    "images":         torch.float32 [3142, 3, 224, 224],  # ImageNet-normalised
    "labels":         torch.int64   [3142],               # Binary DRIL flag (mostly unused — see below)
    "metadata":       list[dict]    of 3142 items,        # {trial, patient_id, visit, eye}
    "biomarkers":     torch.float32 [3142, 16],           # NaN where has_biomarkers=False
    "bcva":           torch.float32 [3142],               # NaN where has_clinical=False
    "cst":            torch.float32 [3142],               # NaN where has_clinical=False
    "has_clinical":   torch.bool    [3142],               # Mask for tier-1 loss
    "has_biomarkers": torch.bool    [3142],               # Mask for tier-2 loss
    "patient_ids":    torch.int64   [3142],               # -1 where missing (used for patient-aware splits)
    "eye_ids":        torch.int64   [3142],               # -1 where missing
    "keys":           list[tuple]   of 3142 items,        # (trial, patient_id, visit, eye)
}
```

**Critical notes for the Phase 2 dataset class**:

- The `labels` field stores binary DRIL flag (one of the 16 biomarkers). It is NOT a 5-class DR severity label. Phase 2 must ignore this field and use the `biomarkers` tensor + `has_biomarkers` mask instead.
- `has_clinical=True` does NOT guarantee BCVA/CST are valid. One sample (patient 0247GOD, V3, OD) has `has_clinical=True` but BCVA=NaN. Regression loss must additionally check `~torch.isnan(bcva)`.

```python
# Correct masking pattern for regression loss
bcva_valid = data['has_clinical'] & ~torch.isnan(data['bcva'])
loss_bcva = mse(predictions[bcva_valid], targets[bcva_valid])
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

Phase 1 validated that the two datasets can be bridged in feature space. Six diagnostics on raw ResNet-50 ImageNet features:

| Diagnostic | Result | Interpretation |
|---|---|---|
| Fréchet distance | 240.65 | Substantial domain shift (>200 = significant) |
| MMD² (RBF) | 0.153, p<0.0001 | Statistically significant distributional difference |
| k-NN cosine similarity | 0.55 (mean) | Moderate cross-dataset alignment |
| PCA on combined features | PC1 23.3% variance | Single axis dominates separation |
| PCA stratified by DR severity | EyePACS classes intermix on PC1 | PC1 does NOT track disease (spread 0.82 vs centroid gap 13.06 — 16× ratio) |
| PC1 vs image stats (Spearman) | ρ=-0.274 with black-pixel proportion, p=1.12×10⁻³⁵ | PC1 tracks acquisition (field-stop geometry), not pathology |

**Conclusion**: the domain shift between EyePACS and OLIVES is **acquisition-driven, not pathology-driven**. EyePACS images have ~20% black border (camera field-stop); OLIVES has ~5%. This is exactly what cross-dataset contrastive alignment is designed to suppress.

**Phase 2 success criteria (re-running the same six diagnostics on trained encoder embeddings)**:

| Metric | Phase 1 baseline | Phase 2 target |
|---|---|---|
| FID | 240.6 | < 100 (ideally < 50) |
| MMD p-value | < 0.0001 | > 0.05 (not significant) |
| Cosine similarity | 0.55 | > 0.70 |
| Black-border ↔ PC1 correlation | ρ = -0.27 | \|ρ\| < 0.1 |
| DR class spread on PC1 | Intermixed | Ordered by severity |

This gives Phase 2 a quantitative before/after demonstration.

---

## 5. Architecture decisions already confirmed

### Q1 — Encoder backbone: **ResNet-50, ImageNet1K-V2 pretrained**
- Use `torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)`
- Strip final FC layer, replace with identity (returns 2048-dim features)
- Add a **projection head** (2 linear layers, 2048 → 512, with batch norm and ReLU between) for the contrastive alignment to operate on (standard SimCLR/SupCon pattern). This is best practice — applying contrastive loss directly on the 2048-dim encoder output tends to over-constrain feature learning.
- Fine-tune the whole stack from ImageNet pretraining (don't freeze the backbone).
- **Rationale**: simpler integration than RETFound, easier comparison against literature baselines (Sükei et al., Kokilepersaud et al. both use ResNet-50).

### Q2 — Contrastive loss: **Supervised contrastive (SupCon-style)**
- Use DR severity labels as the pairing signal
- Within a batch, positives = samples with same DR class (from either dataset); negatives = samples with different DR class
- For OLIVES tier-0/tier-1 samples without DR labels, treat them as "pseudo-class" derived from biomarker presence patterns (or alternatively, exclude them from the contrastive loss and only use them for SSL augmentation pairs)
- **Implementation reference**: SupCon paper by Khosla et al. 2020 (NeurIPS), formula in Section 3
- **Loss formula**: standard supervised contrastive: `L = -sum over positives of log[ exp(sim/τ) / sum over negatives of exp(sim/τ) ]` with temperature τ=0.07 as default
- **Rationale**: uses EyePACS DR labels meaningfully, simpler than mixed sup+unsup, well-established in literature

### Q3 — Biomarker head targets: **9 usable biomarkers (Option B)**
- Include all 9 biomarkers with ≥3 positives out of 192 tier-2 samples
- Exclude: IR HRF (saturated at 100% positive — useless), VMT (3 positives), PED serous (2), DRIL (5), RPE Disruption (6) — also Atrophy (17) and SHRM (17) which are borderline; include them since they have non-trivial positives.
- Final usable biomarker list (9 columns):
  1. EZ Disruption (68 positives, 35.4%)
  2. IR hemorrhages (67, 34.9%)
  3. Partially attached vitreous face (110, 57.3%)
  4. Fully attached vitreous face (134, 69.8%)
  5. Preretinal tissue/hemorrhage (48, 25.0%)
  6. Vitreous debris (164, 85.4%)
  7. DRT/ME (114, 59.4%)
  8. Fluid (IRF) (174, 90.6%)
  9. Fluid (SRF) (29, 15.1%)
- Use BCE loss per biomarker with positive class weighting (`pos_weight = neg/pos` per column) to handle imbalance
- **Rationale**: follows Sükei et al.'s approach, gives more learning signal than 4-biomarker focus

---

## 6. Workflow context (how the project is built)

### Repository
- GitHub: `github.com/savita10/dr-dissertation`
- Local: VS Code workspace
- Colab: `/content/dr-dissertation` (cloned fresh each session)
- Drive (persistent): `/content/drive/MyDrive/dissertation/`

### Development cycle
1. **VS Code (Claude Code)**: write/edit code in `src/`, modify `configs/preprocess.yaml`
2. **PowerShell**: `git add . && git commit -m "..." && git push`
3. **Colab notebook**: `!git -C /content/dr-dissertation pull` then run `!python -m src.module.name --config configs/preprocess.yaml`

### Existing repository structure
```
dr-dissertation/
├── src/
│   ├── data/
│   │   └── preprocess.py              # OLIVES + EyePACS preprocessing (complete)
│   ├── analysis/
│   │   ├── feature_extraction.py      # ResNet-50 feature extraction (Phase 1)
│   │   └── distribution_diagnostics.py # 6 diagnostics + report (Phase 1)
│   └── utils/
│       └── config.py                   # OmegaConf loader
├── configs/
│   └── preprocess.yaml                 # All paths, image size, normalisation
├── docs/
│   ├── METHODOLOGY.md                  # Project methodology
│   └── PREPROCESSING_DOCUMENTATION.md  # Data structure reference
└── notebooks/                          # (Drive, not in repo)
    ├── Preprocess.ipynb
    └── phase1_distributional_validation.ipynb
```

### Code conventions to match
- **Config loading**: OmegaConf via `_load_cfg(path)` pattern (see existing `preprocess.py`)
- **CLI**: `argparse` with `--config configs/whatever.yaml`
- **Logging**: numbered stages like `[1/6 ...]`, tqdm progress bars with throughput
- **Early-exit guards**: check `output_path.exists()` and skip with a print message (makes scripts idempotent)
- **Persistent outputs to Drive**: never write to `/content/` (gets wiped)
- **Devs**: write everything as CLI-callable modules so the Colab notebook is thin (just orchestration)

### Required dependencies (already in Colab)
- `torch`, `torchvision`, `tqdm`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `omegaconf`

---

## 7. Proposed Phase 2 implementation plan

I propose breaking Phase 2 into **three sub-phases**, each with its own Claude Code prompt:

### Phase 2a — Model architecture + losses (~400 lines of code)

**New files**:
- `src/models/__init__.py`
- `src/models/encoder.py` — ResNet-50 backbone with projection head (returns both encoder features and projection embedding)
- `src/models/heads.py` — three output heads (DR classifier, biomarker classifier, BCVA/CST regressor)
- `src/models/model.py` — wrapper combining encoder + heads, single forward pass returns all outputs
- `src/models/losses.py` — masked BCE for biomarkers, masked MSE for regression, supervised contrastive loss, combined loss with weighting

**Validation step**: a smoke test that instantiates the model, runs a forward pass on dummy data (batch of 8 random images), checks output shapes match expectations, and computes a dummy loss.

### Phase 2b — Data loading + patient-aware sampling (~250 lines of code)

**New files**:
- `src/data/datasets.py` — `EyePACSDataset` (reads shard files) + `OLIVESDataset` (reads the consolidated .pt). Both `__getitem__` return all relevant fields including masks.
- `src/data/samplers.py` — patient-aware train/val/test split logic (using `patient_ids` from OLIVES, while EyePACS just uses random splits since there's no patient information). Reproducible via seed.
- `src/data/dataloaders.py` — builds `DataLoader` objects for both datasets with appropriate batch sizes.

**Validation step**: load a small batch from each loader, print shapes, verify masks behave correctly across the three tiers.

### Phase 2c — Training pipeline (~400 lines of code)

**New files**:
- `src/training/__init__.py`
- `src/training/trainer.py` — training loop class: epoch loop, alternating EyePACS-only and joint OLIVES+EyePACS batches, per-head loss tracking, validation step, best-model checkpointing.
- `src/training/train.py` — entry-point script with `argparse`, reads `configs/train.yaml`, runs training.
- `configs/train.yaml` — hyperparameters: learning rate, batch size, num epochs, loss weights (4 terms), optimiser, scheduler, save frequency.

**Validation step**: train for 2 epochs on tiny subsets (1000 EyePACS + 200 OLIVES) to confirm the pipeline runs end-to-end without errors, then stop.

### Phase 2d — Post-training Phase 1 diagnostics rerun (small task)

After full training, re-run the Phase 1 diagnostics on the trained encoder embeddings. This produces the `phase2_report.md` with the before/after comparison. Mostly leverages existing `distribution_diagnostics.py`; only needs a thin wrapper to extract embeddings using the trained model.

**Outputs**:
- `notebooks/phase2_training.ipynb` — orchestrator in Colab
- `notebooks/phase2_evaluation.ipynb` — runs the post-training diagnostics

---

## 8. Phase 2 deliverables

Final outputs after Phase 2 completes:

1. **Trained model checkpoint** at `/content/drive/MyDrive/dissertation/checkpoints/phase2_best.pt` (~150 MB)
2. **New feature embeddings** at `/content/drive/MyDrive/dissertation/features/` using the trained encoder
3. **`phase2_report.md`** with the before/after distributional comparison
4. **Per-head training curves** showing how each loss component trended
5. **`PHASE2_RESULTS.md`** documenting architecture, hyperparameters, final metrics

---

## 9. What to do as the new Claude reading this

1. **Read `METHODOLOGY.md`** (will be uploaded alongside this document) for full project methodology context.
2. **Read `PREPROCESSING_DOCUMENTATION.md`** (will be uploaded) for the data structure details.
3. **Propose a refined plan for Phase 2a** — confirm the file structure I've outlined, suggest improvements if any, then propose the Claude Code prompt.
4. **Wait for user confirmation** before generating the actual Claude Code prompt. Phase 2 is too large to do speculatively.

The user prefers:
- Step-by-step prompts they can paste into Claude Code in VS Code
- Push/pull workflow between PowerShell and Colab
- Validation steps after each sub-phase before moving to the next
- Plain-language explanations alongside technical content (user is a working professional, technically literate but not an ML researcher)

---

## 10. Useful reference numbers (for sanity checks during Phase 2)

| Quantity | Value | Use case |
|---|---|---|
| EyePACS samples | 35,108 | Setting class weights, batch sizing |
| OLIVES total samples | 3,142 | Setting batch sizing |
| OLIVES tier-1 (has clinical) | 1,594 | Regression head training data |
| OLIVES tier-2 (has biomarker) | 192 | Biomarker head training data |
| OLIVES patients | 96 | Patient-aware split planning |
| OLIVES patient-eyes | 204 | Alternative split unit |
| OLIVES visits per eye (avg) | 15.4 | Longitudinal trajectory planning |
| Feature dim (raw ResNet) | 2048 | Encoder output before projection |
| Proposed projection dim | 512 | Embedding space for contrastive |
| Image dimensions | 3 × 224 × 224 | Model input shape |
| EyePACS class 0 fraction | 73.5% | Class imbalance baseline |
| Final EyePACS shard size | 5,000 | Last shard has 108 |

If your code produces tensors with dimensions that don't match these, something is wrong.
