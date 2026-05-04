# Dissertation Preprocessing Documentation

**Project**: Adaptive Multimodal Machine Learning for Diabetic Retinopathy Progression Prediction
**Institution**: University of Bath
**Submission deadline**: August 2026
**Document version**: Final preprocessing state, 27 April 2026

This document captures the complete preprocessing pipeline development for the OLIVES and EyePACS datasets, including data structure analysis, methodological decisions, bug fixes, and open methodological questions to resolve before downstream training. It is intended as a self-contained reference for understanding the preprocessing methodology and the rationale behind key design choices.

**Companion document**: see `docs/METHODOLOGY.md` for the four-phase project methodology (distributional validation, model development, uncertainty quantification, evaluation) that this preprocessing pipeline supports. The methodology document explains *what* the project does; this document explains *how the data was prepared* for it.

---

## 1. Project context and methodological framing

### 1.1 Two methodological pillars

The dissertation rests on two methodological pillars:

- **Longitudinal prognosis**: predict disease progression over time using sequential visits per patient. Requires per-visit continuous outcome measurements (e.g. visual acuity, central subfield thickness).
- **Cross-modality knowledge transfer**: use a shared encoder that projects two distinct fundus datasets (EyePACS and OLIVES) into a joint latent space, with contrastive alignment via cosine similarity. The encoder learns features generalisable across acquisition equipment and patient populations.

The architecture has dual output heads: a **softmax classifier** and an **FCN regressor**.

### 1.2 The "missing modalities" framing

Existing multimodal fusion papers in retinal imaging assume all modalities are available for every sample. This dissertation explicitly handles **missing modalities as a first-class constraint**, which is more representative of real-world clinical deployment where some patients may have only fundus, others only OCT, others both, etc.

This framing turns out to be more concrete than originally anticipated: the OLIVES dataset itself exhibits real missingness across labels (clinical, biomarker), making it an authentic testbed for the architectural choice rather than requiring synthetic missingness.

---

## 2. Datasets

### 2.1 EyePACS

**Source**: HuggingFace Hub (used for cross-sectional DR severity classification pretraining)
**Size**: 35,108 fundus images
**Labels**: 5-class DR severity grading (clinically standard)
**Role**: Pretrains the shared encoder for DR feature learning, then transfers to OLIVES via cross-dataset contrastive alignment.

#### 2.1.1 EyePACS class distribution

| DR Grade | Class label | Count | % |
|---|---|---|---|
| No DR | 0 | 25,802 | 73.5% |
| Mild | 1 | 2,438 | 6.9% |
| Moderate | 2 | 5,288 | 15.1% |
| Severe | 3 | 872 | 2.5% |
| Proliferative | 4 | 708 | 2.0% |

The distribution is heavily imbalanced (canonical for EyePACS — the dataset was collected from population-level screening rather than referral clinics). Training will require class weighting or focal loss to prevent collapse to the majority class.

#### 2.1.2 EyePACS preprocessed structure on Drive

```
/content/drive/MyDrive/dissertation/preprocessed/eyepacs/
├── eyepacs_shard_000.pt   (5,000 images)
├── eyepacs_shard_001.pt   (5,000 images)
├── eyepacs_shard_002.pt   (5,000 images)
├── eyepacs_shard_003.pt   (5,000 images)
├── eyepacs_shard_004.pt   (5,000 images)
├── eyepacs_shard_005.pt   (5,000 images)
├── eyepacs_shard_006.pt   (5,000 images)
├── eyepacs_shard_007.pt   (108 images — final partial shard)
└── metadata.json
```

Each `.pt` shard is a dict containing:
- `images`: `torch.float32 [N, 3, 224, 224]` — ImageNet-normalised
- `labels`: `torch.int64 [N]` — DR severity class (0-4)
- `filenames`: list of strings — `eyepacs_NNNNNN` indices

Sharding rationale: 35K images at 3×224×224×4 bytes = ~22 GB total. Sharded to fit in memory comfortably and to enable resumable preprocessing (existing shards skip on rerun).

### 2.2 OLIVES

**Source**: PhysioNet (Prabhushankar et al., NeurIPS 2022)
**Original size**: ~78K OCT B-scan images plus paired fundus images, organised by clinical trial
**Used in dissertation**: 3,142 unique fundus images (OCT not used directly; OCT-derived labels used as auxiliary supervision)
**Role**: Provides longitudinal trajectories for progression prediction and biomarker labels for auxiliary multi-label classification supervision.

#### 2.2.1 OLIVES dataset structure (raw)

The OLIVES distribution arrives as a triple-nested zip:

```
olive.zip                                  (outer container)
└── OLIVES.zip                             (inner)
    ├── OLIVES/TREX_DME.zip                (TREX DME trial)
    │   └── TREX DME/
    │       └── {ARM}/                     (e.g. GILA, TREX)
    │           └── {patient}/             (e.g. 0213TOS)
    │               └── V{visit}/          (e.g. V1, V14, V19)
    │                   └── {eye}/         (OD or OS)
    │                       ├── fundus_OS_V19.tif       (fundus image)
    │                       └── TREXJ_NNNNNN.tif        (OCT B-scans, ~49 per visit)
    └── OLIVES/Prime_FULL.zip              (Prime trial)
        └── Prime_FULL/
            └── {patient}/                 (e.g. 02-010)
                └── W{week}/               (e.g. W0, W36, W104)
                    └── {eye}/             (OD or OS)
                        ├── fundus_OD_W0.tif        (fundus, .tif version)
                        ├── fundus_W0.png           (fundus, .png version — DEDUPED)
                        └── TREXW_NNNNNN.tif        (OCT B-scans)
```

**Trial-specific path structure differs**:
- TREX DME path: `TREX DME/{arm}/{patient}/V{visit}/{eye}/{filename}` — 5 levels before filename
- Prime_FULL path: `Prime_FULL/{patient}/W{week}/{eye}/{filename}` — 4 levels before filename

This trial-specific path layout is the reason both `_csv_path_to_key()` and `_disk_path_to_key()` in `preprocess.py` have separate handling for each trial.

**Important Prime_FULL detail**: the same fundus image is stored in BOTH `.tif` (lossless) and `.png` formats. Without preprocessing-time deduplication, this would inflate Prime_FULL representation 2× relative to TREX DME. Our pipeline deduplicates by preferring `.tif` over `.png` (see Section 6.3).

#### 2.2.2 OLIVES label files

Located at `/content/drive/MyDrive/olives_labels/OLIVES_Dataset_Labels/`:

```
OLIVES_Dataset_Labels/
├── full_labels/
│   ├── Clinical_Data_Images.xlsx        ← TIER 1 source (78,185 rows, 1,594 unique visit-eyes)
│   ├── OCT-DR.xlsx                      ← Patient-level DR study summary (41 rows; not per-image)
│   └── OCT-DME.xlsx                     ← Patient-level DME study summary (42 rows; not per-image)
└── ml_centric_labels/
    ├── Biomarker_Clinical_Data_Images.csv  ← TIER 2 source (9,408 rows, 192 unique visit-eyes)
    └── Clinical_Data_Images.xlsx           ← Identical copy of full_labels/Clinical_Data_Images.xlsx
```

#### 2.2.3 Label file schemas

**Clinical_Data_Images.xlsx** (78,185 rows, 5 columns):
- `File_Path` — path to OCT B-scan (with leading slash)
- `BCVA` — Best Corrected Visual Acuity (continuous, ETDRS letters)
- `CST` — Central Subfield Thickness (continuous, microns)
- `Eye_ID` — integer eye identifier
- `Patient_ID` — integer patient identifier

**Biomarker_Clinical_Data_Images.csv** (9,408 rows):
- `Path (Trial/Arm/Folder/Visit/Eye/Image Name)` — path to OCT B-scan (with leading slash)
- `BCVA`, `CST`, `Eye_ID`, `Patient_ID` — same as clinical file
- `Scan (n/49)` — which OCT B-scan within the visit-eye sequence
- 16 binary biomarker columns (see section 4.2 below)

**Critical distinction**: each row in both files corresponds to a single OCT B-scan, **not** a fundus image. A typical visit-eye captures ~49 OCT B-scans, each represented as a separate row. BCVA and CST are visit-level measurements broadcast to every scan row of that visit. **Biomarkers vary across scans** (a biomarker is positive for a B-scan if the relevant pathology is visible in that specific cross-section — see Section 6.2 for the methodological implication).

### 2.3 Cohort reconciliation against the OLIVES paper

After full preprocessing and `.tif`/`.png` deduplication, our dataset reconciles exactly against the cohort statistics reported in the OLIVES paper (Prabhushankar et al., NeurIPS 2022):

| Metric | Paper (Table 2) | Our preprocessed dataset | Match |
|---|---|---|---|
| Total patients | 96 | 96 | ✓ exact |
| Average visits per patient | ~16 | 15.4 | ✓ within rounding |
| Patient-eyes (both eyes counted) | "96 Eyes" (paper terminology — actually means 96 patients) | 204 patient-eyes | Reconciled |
| Total unique visit-eyes with fundus | (not explicitly reported) | 3,142 | Consistent with 204 × 15.4 ≈ 3,142 |

**Per-trial breakdown (post-dedup)**:
- **TREX DME**: 1,849 fundus = 1,849 unique visit-eyes from 56 patients across 120 patient-eyes
- **Prime_FULL**: 1,293 fundus = 1,293 unique visit-eyes from 40 patients across 84 patient-eyes
- **Combined**: 96 unique patients (no overlap between trials)

This exact match with the paper's reported cohort provides strong validation of our preprocessing pipeline and gives the dissertation methods chapter a clean reference point.

---

## 3. Terminology reference

| Term | Meaning |
|---|---|
| **Visit-eye** | A single (patient, visit, eye) combination — the natural unit at which clinical and fundus data is captured. Format used as a 4-tuple key: `(trial, patient_id, visit_id, eye)`. |
| **Patient-eye** | A single (patient, eye) combination. There are 204 patient-eyes across our 96 patients, each contributing ~15.4 visit-eyes on average. The natural unit for patient-aware train/val/test splits. |
| **Visit** | A timepoint in the clinical trial schedule. Coded as `V{n}` in TREX DME (e.g. V1, V14, V19) and `W{n}` in Prime (e.g. W0, W36, W104). |
| **Eye** | OD (right eye) or OS (left eye). Each patient typically contributes both eyes. |
| **Fundus image** | Photograph of the back of the eye (retina). The primary imaging modality for this dissertation. |
| **OCT B-scan** | Optical Coherence Tomography cross-section. ~49 B-scans per visit-eye. Not used directly as model input but provides biomarker labels. |
| **BCVA** | Best Corrected Visual Acuity — letters read on an ETDRS chart (continuous, typically 0-100). |
| **CST** | Central Subfield Thickness — central retinal thickness in microns from OCT (continuous). |
| **DR** | Diabetic Retinopathy. Severity graded 0-4 (No DR / Mild / Moderate / Severe / Proliferative). |
| **DME** | Diabetic Macular Edema. Specific form of DR involving fluid accumulation. Targeted by TREX DME trial. |
| **DRSS** | Diabetic Retinopathy Severity Scale (also called ETDRS DR severity scale). |
| **DRT/ME** | Diabetic Retinal Thickening / Macular Edema. Biomarker label. Hallmark of DME. |
| **IRF** | Intra-Retinal Fluid. Biomarker label. Hallmark of DME. |
| **SRF** | Sub-Retinal Fluid. Biomarker label. |
| **IR HRF** | Intraretinal Hyperreflective Foci. Biomarker label. Common DR/DME imaging marker. |
| **EZ Disruption** | Disruption of the Ellipsoid Zone. Biomarker label. Photoreceptor layer integrity marker. |
| **DRIL** | Disorganisation of Retinal Inner Layers. Biomarker label. |
| **VMT** | Vitreomacular Traction. Biomarker label. |
| **PED** | Pigment Epithelial Detachment. Biomarker label. |
| **SHRM** | Subretinal Hyperreflective Material. Biomarker label. |
| **Tier 0 / 1 / 2** | Three levels of label availability adopted in this preprocessing — see section 4. |
| **Field stop** | Black perimeter of fundus images caused by the camera's circular field of view. Common artefact in EyePACS. |
| **Shared encoder** | Single encoder network consuming both EyePACS and OLIVES inputs and projecting to a joint latent space. The architectural foundation for cross-dataset knowledge transfer. |
| **Contrastive alignment** | Training objective that pulls together latent representations of paired/related samples (e.g. same DR severity from different datasets) and pushes apart unrelated samples. Used in the shared encoder's cross-dataset alignment loss. |
| **Cosine similarity** | Distance metric used for contrastive alignment. Compares angular similarity between latent vectors. |
| **Hypercube** | Term for the joint latent space into which both datasets project. |
| **EyePACS** | Public DR screening dataset (35K+ images). Used for shared encoder pretraining. |
| **OLIVES** | Multimodal retinal dataset combining fundus + OCT + clinical biomarkers from two clinical trials (TREX DME and Prime). Used for fusion/longitudinal modeling. |
| **Sükei et al. (2024)** | Baseline paper — SSL contrastive learning on OLIVES (Nature Scientific Reports). |
| **Kokilepersaud et al. (2023)** | Baseline paper — clinically-labelled contrastive learning. |
| **MICCAI 2025 OT paper** | Lan et al. baseline — Optimal Transport for missing modality fusion. |

---

## 4. The three-tier label structure (preprocessing logic)

### 4.1 Why three tiers

OLIVES image-level supervision is genuinely tiered. Different fundus images have different levels of label availability, and the preprocessing must preserve all images regardless of label coverage to maximise data utility for self-supervised contrastive pretraining.

**Final tier breakdown (3,142 total samples post-dedup):**

| Tier | Description | Count | % of total | Primary use |
|---|---|---|---|---|
| **Tier 0** | Fundus image only — no clinical, no biomarker labels | 1,546 | 49.2% | SSL contrastive pretraining for the shared encoder |
| **Tier 1** | Tier 0 + BCVA + CST clinical regression labels | 1,404 | 44.7% | + Longitudinal prognosis regression |
| **Tier 2** | Tier 1 + 16 biomarker labels | 190 | 6.0% | + Biomarker classification head |
| **Anomalous** | Biomarker labels but NOT clinical (edge case) | 2 | 0.06% | Masked appropriately by training |
| **Total** | | **3,142** | 100% | |

### 4.2 The 16 biomarker columns (full list, post-dedup)

**Final positive counts after both dedup fixes (192 tier-2 samples)**:

| # | Biomarker | Tier-2 positives | % | Status |
|---|---|---|---|---|
| 1 | Atrophy / thinning of retinal layers | 17 | 8.9% | Sparse |
| 2 | Disruption of EZ | 68 | 35.4% | Useful (partly OCT) |
| 3 | DRIL | 5 | 2.6% | Too sparse |
| 4 | IR hemorrhages | 67 | 34.9% | **Useful, fundus-visible** |
| 5 | IR HRF | 192 | 100.0% | **SATURATED — drop** |
| 6 | Partially attached vitreous face | 110 | 57.3% | High signal (OCT-derived) |
| 7 | Fully attached vitreous face | 134 | 69.8% | High signal (OCT-derived) |
| 8 | Preretinal tissue/hemorrhage | 48 | 25.0% | Useful |
| 9 | Vitreous debris | 164 | 85.4% | High signal (OCT-derived) |
| 10 | VMT | 3 | 1.6% | Too sparse |
| 11 | DRT/ME | 114 | 59.4% | **Useful, fundus-visible, clinically critical** |
| 12 | Fluid (IRF) | 174 | 90.6% | **High signal, fundus-visible, clinically critical** |
| 13 | Fluid (SRF) | 29 | 15.1% | **Borderline-useful, clinically critical** |
| 14 | Disruption of RPE | 6 | 3.1% | Too sparse |
| 15 | PED (serous) | 2 | 1.0% | Too sparse |
| 16 | SHRM | 17 | 8.9% | Marginal |

After excluding IR HRF (saturated/constant) and the six too-sparse columns (VMT, RPE, PED, DRIL, Atrophy, SHRM), there are **9 usable biomarkers**, of which **DRT/ME, IRF, IR hemorrhages, SRF** have the strongest direct fundus correlates and clinical relevance for diabetic macular edema.

### 4.3 DR severity classification — a methodological dead end on OLIVES

**Critical finding**: OLIVES does NOT provide per-image DR severity grades. The original idea (a 5-class softmax classifier head trained on OLIVES DR severity, mirroring the EyePACS approach) is not feasible.

Sources of DR severity in OLIVES:

| Source | Granularity | Coverage | Usable? |
|---|---|---|---|
| OCT-DR.xlsx — `DRSS` column | Patient-level (baseline only) | 41 patients in DR study | No — not per-image |
| OCT-DME.xlsx — visit-level summary | Patient-level study summary | 42 patients in DME study | No — not per-image |
| Biomarker `DRIL` column | Per-image binary | 5/192 positive (2.6%) | Effectively unusable |

Implication: any DR severity classifier head must be trained on EyePACS only. OLIVES provides BCVA/CST regression labels and biomarker classification labels instead.

---

## 5. Preprocessing pipeline implementation

### 5.1 Files in the repository

```
src/
└── data/
    └── preprocess.py                    ← Main preprocessing module

configs/
└── preprocess.yaml                      ← All paths, image size, normalisation

src/utils/
└── config.py                            ← OmegaConf config loader

docs/
└── PREPROCESSING_DOCUMENTATION.md       ← This document
```

Repository: `https://github.com/savita10/dr-dissertation`
Cloned in Colab to: `/content/dr-dissertation`

### 5.2 `configs/preprocess.yaml` schema

```yaml
image_size: 224
normalise_mean: [0.485, 0.456, 0.406]    # ImageNet
normalise_std: [0.229, 0.224, 0.225]     # ImageNet
eyepacs_shard_size: 5000

# EyePACS
eyepacs_source: /content/eyepacs_hf
eyepacs_output: /content/drive/MyDrive/dissertation/preprocessed/eyepacs

# OLIVES
olives_zip: /content/drive/MyDrive/olive.zip
olives_labels_csv: .../Biomarker_Clinical_Data_Images.csv      # tier-2
olives_clinical_xlsx: .../full_labels/Clinical_Data_Images.xlsx # tier-1
olives_output: /content/drive/MyDrive/dissertation/preprocessed/olives
scratch_dir: /content/scratch/olives_fundus
```

### 5.3 OLIVES preprocessing stages

The `preprocess_olives()` function runs in 6 stages:

1. **Extract fundus images** from triple-nested zip to `scratch_dir`. Resumable — files already present are skipped. Takes ~47 minutes from cold start due to slow zip-from-Drive read.
2. **Load biomarker labels** from CSV → `biomarker_lookup: dict[(trial,patient,visit,eye), 16-element float32 array]`. Uses **max-aggregation** across OCT scans within each visit-eye (see Section 6.2).
3. **Load clinical labels** from xlsx → `clinical_lookup: dict[(trial,patient,visit,eye), {BCVA, CST, Patient_ID, Eye_ID}]`. Uses first-row deduplication (safe because BCVA/CST are visit-level constants).
4. **Walk scratch dir** for fundus files (filter by `'fundus' in filename`). **Apply `.tif`/`.png` deduplication** to remove Prime_FULL duplicates (see Section 6.3). Result: 3,142 unique fundus paths from 4,435 raw files.
5. **Process all fundus images** with three-tier label assignment. For each fundus file:
   - Load and ImageNet-normalise to 3×224×224 float32
   - Look up clinical labels (NaN if missing, set `has_clinical=False`)
   - Look up biomarker labels (NaN if missing, set `has_biomarkers=False`)
   - Append to lists regardless of label availability
6. **Save** consolidated tensor file (~1.8 GB) + write metadata.json

### 5.4 OLIVES output schema (`olives_fundus.pt`)

```python
{
    "images":        torch.float32 [3142, 3, 224, 224],   # ImageNet-normalised
    "labels":        torch.int64   [3142],                # Binary DRIL flag (NOT multi-class DR)
    "metadata":      list[dict]    of 3142 items,         # {trial, patient_id, visit, eye}
    "biomarkers":    torch.float32 [3142, 16],            # NaN where has_biomarkers=False
    "bcva":          torch.float32 [3142],                # NaN where has_clinical=False
    "cst":           torch.float32 [3142],                # NaN where has_clinical=False
    "has_clinical":  torch.bool    [3142],                # Mask for tier-1 supervision
    "has_biomarkers":torch.bool    [3142],                # Mask for tier-2 supervision
    "patient_ids":   torch.int64   [3142],                # -1 where missing
    "eye_ids":       torch.int64   [3142],                # -1 where missing
    "keys":          list[tuple]   of 3142 items,         # (trial, patient, visit, eye)
}
```

**CRITICAL note on the `labels` field**: this is a binary DRIL flag (one of the 16 biomarker columns), not a multi-class DR severity label. It is only meaningful where `has_biomarkers=True`; elsewhere it is 0 by convention. Downstream code MUST mask by `has_biomarkers` before computing classification loss to avoid trivially predicting 0 with 99%+ accuracy.

**CRITICAL note on the `has_clinical` mask**: `has_clinical=True` means "row exists in clinical xlsx", not "BCVA/CST values are valid". One sample (patient 0247GOD, visit V3, OD) has `has_clinical=True` but BCVA=NaN and CST=NaN due to OLIVES upstream data quality. Regression loss must additionally check `~torch.isnan(bcva)`:

```python
bcva_valid_mask = data['has_clinical'] & ~torch.isnan(data['bcva'])
bcva_loss = mse(predictions[bcva_valid_mask], data['bcva'][bcva_valid_mask])
```

### 5.5 OLIVES `metadata.json` structure

```json
{
  "total_images": 3142,
  "num_shards": 1,
  "image_size": [224, 224],
  "normalisation": {
    "mean": [0.485, 0.456, 0.406],
    "std":  [0.229, 0.224, 0.225]
  },
  "class_distribution": {
    "0": 190,
    "unlabelled": 2952
  },
  "olives_label_coverage": {
    "total_fundus": 3142,
    "with_clinical_labels": 1594,
    "with_biomarker_labels": 192,
    "clinical_keys_in_lookup": 1594,
    "biomarker_keys_in_lookup": 192
  },
  "preprocessing_date": "2026-04-27T..."
}
```

### 5.6 EyePACS preprocessing

The `preprocess_eyepacs()` function runs in 4 stages:
1. Load HuggingFace dataset from local SSD (`/content/eyepacs_hf`)
2. Plan sharding (8 shards of up to 5,000 images each)
3. Write shards (skipping any that already exist — resumable)
4. Write metadata.json

EyePACS image transform pipeline is **identical** to OLIVES — same 224×224 resize, same ImageNet normalisation. This is the structural prerequisite for the shared encoder.

---

## 6. Bugs found and fixed during development

### 6.1 Bug 1: Wrong column name for clinical xlsx file

**Symptom**: `KeyError: "Expected column 'Path (Trial/Arm/Folder/Visit/Eye/Image Name)' in Clinical_Data_Images.xlsx. Found: ['File_Path', 'BCVA', 'CST', 'Eye_ID', 'Patient_ID']"`

**Root cause**: Both the biomarker CSV and clinical xlsx were assumed to share the same path column name. They don't:
- Biomarker CSV: `Path (Trial/Arm/Folder/Visit/Eye/Image Name)`
- Clinical xlsx: `File_Path`

**Fix**: Added module constant `CLINICAL_PATH_COLUMN = "File_Path"` and used it in `_build_clinical_lookup()` while leaving `_build_biomarker_lookup()` to use the original `PATH_COLUMN`.

**Status**: Fixed and committed.

### 6.2 Bug 2: Biomarker dedup using `keep='first'` instead of `max` aggregation

**Symptom**: After the first end-to-end successful run, six biomarker columns showed zero positives in the saved tensor (DRIL, VMT, Fluid SRF, Disruption of RPE, PED serous, SHRM), despite having 10-233 positives in the raw CSV.

**Root cause**: Biomarkers are annotated per OCT B-scan (~49 scans per visit-eye). A biomarker may be positive only on certain B-scans (those cutting through the affected retinal region). `drop_duplicates(keep='first')` kept only the first row's values per visit-eye. If the first scan in the schedule didn't contain the affected region, the biomarker was recorded as 0 at the visit-eye level even when later scans had positives.

This was missed initially because we incorrectly assumed biomarker values were broadcast across all OCT scans within a visit-eye (the way BCVA and CST are). They are not — biomarkers are genuinely per-scan annotations.

**Diagnostic confirming the bug** (192 visit-eye keys total):

| Biomarker | Varies (scans differ) | Max=1 (any scan +ve) | First=1 (first scan +ve) |
|---|---|---|---|
| DRT/ME | 104 | 114 | 31 |
| Fluid (IRF) | 162 | 174 | 44 |
| Fluid (SRF) | 29 | 29 | 0 |
| IR HRF | 158 | 192 | 90 |

Massive gaps between Max=1 and First=1 confirmed widespread under-counting.

**Fix**: Replace `drop_duplicates(subset='_key')` with `groupby('_key')[biomarker_cols].max()`. The `max` aggregation is the correct semantic: "any positive scan → visit-eye is positive", which matches how a clinician would summarise biomarker presence at the visit level.

**Status**: Fixed and committed. Post-fix biomarker positive counts confirmed accurate (e.g. DRT/ME jumped from 31 → 114, IRF from 44 → 174, SRF from 0 → 29 at the 192-key level).

This bug fix is methodologically defensible and worth mentioning in the dissertation methods chapter as evidence of preprocessing rigour.

### 6.3 Bug 3: Prime_FULL `.tif`/`.png` duplication

**Symptom**: After the three-tier preprocessing was working, the dataset reported 4,435 fundus images, but a per-trial diagnostic showed only 3,142 unique visit-eyes. The discrepancy was concentrated in Prime_FULL (2,586 files for 1,293 unique visit-eyes — exactly 2× duplication).

**Root cause**: Prime_FULL stores the same fundus image in both `.tif` (lossless) and `.png` formats for every visit-eye. TREX DME has only `.tif`. Without preprocessing-time deduplication, every Prime_FULL fundus image was treated as two separate dataset entries, biasing training toward Prime patterns and creating patient leakage risk for any patient-aware split.

**Validated against the OLIVES paper**: After applying `.tif`/`.png` dedup, our dataset matches the paper's reported cohort exactly (96 patients, ~16 visits per eye), confirming this was an artefact of file storage rather than legitimate dataset duplication.

**Fix**: Added a deduplication step at the end of stage 4 (walking scratch dir) that groups fundus paths by `(trial, patient, visit, eye)` key and keeps `.tif` over `.png` when both exist. Reports number of duplicates removed.

**Status**: Fixed and committed. Post-fix counts: 1,849 TREX DME + 1,293 Prime_FULL = 3,142 total fundus, matching paper cohort exactly.

This is the third preprocessing fix and brings the dataset to its final reconciled state.

### 6.4 Data quality finding (not a bug): one visit-eye with NaN clinical values

Patient `0247GOD`, visit V3, OD has a row in `Clinical_Data_Images.xlsx` where BCVA and CST are both NaN. This is genuine OLIVES data quality (the visit was recorded but measurements were missing) and propagates to one sample in the saved tensor where `has_clinical=True` but `bcva=NaN`.

**Implication for downstream code**: see CRITICAL note on `has_clinical` in Section 5.4. Regression loss must additionally check `~torch.isnan(bcva)`.

---

## 7. Cross-dataset compatibility analysis

### 7.1 Structural compatibility (confirmed)

| Property | EyePACS | OLIVES |
|---|---|---|
| Image shape | (3, 224, 224) | (3, 224, 224) ✓ |
| Dtype | torch.float32 | torch.float32 ✓ |
| Normalisation | ImageNet means/stds | ImageNet means/stds ✓ |

The shared encoder is mechanically able to consume either dataset.

### 7.2 Distribution shift between datasets (characterised)

Per-channel statistics across 200-image samples after ImageNet normalisation:

| Dataset | R mean | G mean | B mean | Aggregate mean | Aggregate std |
|---|---|---|---|---|---|
| EyePACS | -0.250 | -0.697 | -0.869 | **-0.605** | 1.023 |
| OLIVES  | -0.093 | +0.034 | +0.256 | **+0.065** | 0.769 |

Black-pixel proportion (raw R < ~0.06):
- EyePACS: **20.15%** of pixels
- OLIVES: **4.62%** of pixels

**Interpretation**: EyePACS images contain substantial field-stop black border (the dark perimeter of the camera's circular field of view), which drags per-channel means strongly negative and inflates contrast. OLIVES images have minimal border (different acquisition equipment) and are mostly retinal content, with distribution closer to ImageNet zero-mean.

**Methodological framing**: This domain shift is not a bug — it is exactly the situation that motivates cross-dataset contrastive alignment in the shared encoder architecture. The shared encoder must learn representations invariant to acquisition geometry (field-stop, framing) while preserving retinal pathology features. The distribution gap becomes part of the methodological narrative rather than a problem to solve before training.

**Phase 1 confirmation** (post-preprocessing): downstream feature-level analysis using ResNet-50 ImageNet features confirmed this pixel-level shift propagates to feature space (FID=240.65, MMD² p<0.0001) but is dominantly **acquisition-driven rather than pathology-driven** — DR severity classes intermix on the dominant separation axis (PC1) while black-pixel proportion correlates strongly with PC1 (ρ=-0.274, p=1.12×10⁻³⁵). See `METHODOLOGY.md` Section "Phase 1: Distributional validation" for the full analysis. This mechanism-level evidence supports the bridgeability of the shift via cross-dataset contrastive alignment.

---

## 8. Summary of preprocessing state

| Component | Status |
|---|---|
| EyePACS preprocessing | Complete: 35,108 images, 8 shards, 5-class DR labels |
| OLIVES preprocessing | Complete: 3,142 unique fundus images, three-tier labels |
| Cross-dataset structural compatibility | Confirmed |
| Domain shift characterisation | Documented (motivates cross-dataset contrastive alignment) |
| Phase 1 distributional validation | Complete (FID=240.65, MMD² p<0.0001, mechanism-confirmed acquisition-driven shift) |
| Cohort reconciliation with OLIVES paper | Confirmed (96 patients exact match) |
| Bug fix #1 — Clinical xlsx column name | Fixed (`File_Path`, not biomarker CSV path) |
| Bug fix #2 — Biomarker dedup | Fixed (max-aggregation across OCT scans) |
| Bug fix #3 — Prime_FULL .tif/.png duplication | Fixed (`.tif` preferred over `.png`) |
| Repository | `dr-dissertation` on GitHub, cloned to `/content/dr-dissertation` |
| Output dataset (OLIVES) | `/content/drive/MyDrive/dissertation/preprocessed/olives/olives_fundus.pt` (1.8 GB) |
| Output dataset (EyePACS) | `/content/drive/MyDrive/dissertation/preprocessed/eyepacs/eyepacs_shard_*.pt` (~22 GB) |

---

## 9. Open methodological questions

These are the methodological decisions to resolve before further code is written. Each question is grounded in concrete preprocessing evidence rather than abstract architectural choice.

### Q1. Classifier head target on OLIVES

**Context**: OLIVES does not provide per-image DR severity grades (Section 4.3). The original dual-head plan (softmax DR severity classifier + FCN regressor) needs reformulation. After preprocessing, 9 biomarker columns have ≥3 positives in the 192 tier-2 samples, with 4 having strong fundus correlates and clinical relevance: DRT/ME, IRF, IR hemorrhages, SRF.

**Options**:

- **Option A — Clinically-focused 4-biomarker head**: train BCE multi-label classifier on DRT/ME (114), IRF (174), IR hemorrhages (67), SRF (29). All four have strong fundus correlates and clinical relevance. Smallest label space, cleanest narrative.
- **Option B — Pragmatic 9-biomarker head**: train on all 9 biomarkers with ≥3 positives (excluding saturated IR HRF). Larger label space, includes some OCT-derived biomarkers the fundus model can only weakly predict.
- **Option C — Two-head split**: primary head on the 4 fundus-grounded biomarkers, auxiliary head on the 5 OCT-correlated biomarkers at lower loss weight, framed as cross-modal correlation supervision.

**Recommendation**: Option A is the cleanest dissertation story and avoids training on biomarkers the fundus model fundamentally cannot predict. Option C is the most elegant fit for the "cross-modality knowledge transfer" framing but adds a head requiring justification.

### Q2. DR severity classification target

**Context**: With OLIVES DR severity unavailable, the 5-class DR classifier head must train on EyePACS only. The shared encoder then transfers learned DR features to OLIVES via cross-dataset contrastive alignment.

**Decision**: EyePACS-only DR classification is the proposed approach, with the encoder's DR features inherited by the OLIVES branch through cross-modal alignment rather than direct supervision.

### Q3. Pre-encoder preprocessing for domain shift

**Context**: Documented domain shift between EyePACS and OLIVES driven by field-stop geometry (20% vs 5% black pixels, mean shift of ~0.7 in normalised pixel space — Section 7.2).

**Options**:

- Rely entirely on the shared encoder's contrastive alignment to bridge the shift (no extra preprocessing).
- Add **field-stop crop** (detect circular fundus region, crop tight to it before resize) — standard in fundus literature.
- Add **CLAHE / contrast normalisation** for cross-dataset harmonisation — used in several recent DR papers.

Each option is defensible. The choice depends on how much of the methodology should rest on architectural learning vs preprocessing harmonisation.

### Q4. Anomalous tier-2-only samples (2 samples)

Two fundus images have biomarker labels but no clinical labels (key exists in biomarker CSV but not in clinical xlsx).

**Proposed handling**: these samples participate in biomarker classification loss (where `has_biomarkers=True`) but skip BCVA/CST regression loss (where `~has_clinical`). This is automatic given the mask-based loss formulation.

### Q5. Patient-aware train/val/test split ratio

**Context**: 96 unique patients across both trials (56 TREX DME + 40 Prime_FULL). Standard 70/20/10 split would give roughly 67/19/10 patients, with downstream sample counts proportional to ~204 patient-eyes × 15.4 visits per eye.

**Decision**: Choose split ratio. Given the small cohort (96 patients), there's a defensible argument for 80/10/10, 70/15/15, or 70/20/10. Splits MUST be at the patient level (not visit-eye or sample) to prevent leakage of the same eye's trajectory across train/val/test.

### Q6. Methodological framing of preprocessing bug fixes

**Context**: The shift from `drop_duplicates(keep='first')` to `groupby().max()` aggregation across OCT scans is a methodologically meaningful "any-positive" decision rule for visit-eye level biomarker assignment. The `.tif`/`.png` dedup is similarly a meaningful design choice rather than routine cleanup.

**Note**: Both of these design choices are worth mentioning in the dissertation methods chapter as deliberate, defensible methodological decisions (rather than glossing over them as routine preprocessing). They demonstrate preprocessing rigour and direct engagement with the OLIVES dataset's annotation structure.

---

## 10. Files modified during this development cycle

| File | Change type | Notes |
|---|---|---|
| `src/data/preprocess.py` | Major refactor | New three-tier matching logic, masks, dual lookups |
| `configs/preprocess.yaml` | Added `olives_clinical_xlsx` key | Required for OLIVES preprocessing |
| `src/data/preprocess.py` | Bug fix #1 | Added `CLINICAL_PATH_COLUMN = "File_Path"` constant |
| `src/data/preprocess.py` | Bug fix #2 | Replaced first-row dedup with max-aggregation in `_build_biomarker_lookup` |
| `src/data/preprocess.py` | Bug fix #3 | Added `.tif`/`.png` dedup in stage 4 (walk scratch dir) |
| `docs/PREPROCESSING_DOCUMENTATION.md` | New file | This document |

---

## 11. Next steps

1. **Resolve open methodological questions** in section 9.
2. **Update OLIVES dataset class** (`src/data/datasets.py` or equivalent) to consume the new schema with `has_clinical` / `has_biomarkers` masks.
3. **Update training loss functions** to mask BCE biomarker loss by `has_biomarkers`, MSE BCVA/CST loss by `has_clinical & ~isnan(bcva)`.
4. **Implement t-SNE distribution check** (already identified as prerequisite) on encoder outputs from EyePACS and OLIVES samples to validate that contrastive alignment is doing its job.
5. **Write the cross-dataset contrastive head** with cosine similarity loss between EyePACS and OLIVES samples within batch.
6. **Begin longitudinal prognosis training** using the patient_ids and visit information from the saved tensor to construct per-patient trajectory sequences.
