# Diabetic Retinopathy Progression Prediction

A deep learning framework for predicting diabetic retinopathy progression by jointly encoding the EyePACS and OLIVES fundus image datasets in a shared latent space. The project combines cross-dataset contrastive alignment with multi-task supervision (DR severity classification, biomarker detection, and continuous clinical outcome regression) to enable knowledge transfer between two datasets with structurally different label availability.

**Institution**: University of Bath, MSc dissertation
**Status**: Phase 1 (distributional validation) complete; Phase 2 (model development) in progress

## Project Structure

```
dr-dissertation/
├── src/
│   ├── data/         # Preprocessing, dataset classes, dataloaders, samplers, transforms
│   ├── analysis/     # ResNet-50 feature extraction and distributional diagnostics
│   ├── models/       # Shared encoder, output heads, model wrapper, loss functions
│   ├── training/     # Training loop and entry-point scripts
│   └── utils/        # Config loading, Colab setup, zip inspection helpers
├── configs/          # YAML configuration files (paths, preprocessing, model, training)
├── notebooks/        # Colab notebook orchestrators (numbered by execution order)
├── docs/             # Methodology, phase documentation, results
└── requirements.txt
```

## Documentation

The `docs/` folder contains the full methodological documentation:

| Document | Description |
|---|---|
| `METHODOLOGY.md` | Full project methodology across all four phases, with Phase 1 findings integrated |
| `PREPROCESSING_DOCUMENTATION.md` | Data structure reference: three-tier label structure, schema of saved `.pt` files, preprocessing bug fixes |
| `PHASE1_SUMMARY.md` | Retrospective summary of Phase 1 — objectives, diagnostics, findings, design decisions |
| `PHASE2_KICKOFF.md` | Phase 2 implementation plan (model architecture, losses, training pipeline) |
| `POST_TRAINING_REFERENCE.md` | Reference guide for Phase 3 (uncertainty) and Phase 4 (evaluation) |

## Phases

The project proceeds in four phases:

**Phase 1 — Distributional validation** (complete)
Quantify and mechanistically diagnose the cross-dataset gap between EyePACS and OLIVES in ResNet-50 feature space. Six diagnostics (FID, MMD, k-NN cosine similarity, PCA, PCA stratified by DR severity, image stats correlation) characterise the gap as substantial (FID 240.65) but acquisition-driven rather than pathology-driven (PC1 tracks field-stop geometry with Spearman ρ = -0.27, p < 10⁻³⁵; DR severity classes intermix on PC1 with 16× ratio relative to dataset centroid distance). This mechanism-level finding supports cross-dataset contrastive alignment as a principled architectural choice.

**Phase 2 — Model development** (in progress)
Shared encoder (ResNet-50, ImageNet-pretrained, fine-tuned) with three output heads (DR severity softmax, biomarker BCE, BCVA/CST regression) and four training loss terms (cross-entropy, masked BCE, masked MSE, supervised contrastive). Mask-based loss formulation handles the three-tier label structure (OLIVES tier-0: 49.2% unlabelled, tier-1: 44.7% clinical labels only, tier-2: 6.0% full biomarker labels).

**Phase 3 — Uncertainty quantification** (pending)
Monte Carlo Dropout-based confidence estimation, with calibration analysis (reliability diagrams, ECE) and risk-stratification curves.

**Phase 4 — Evaluation** (pending)
Standard performance metrics on held-out test set, longitudinal trajectory prediction (the dissertation's key scientific claim: distinguishing same-baseline patients with different future BCVA/CST trajectories), and cross-dataset transfer validation.

## Data

### EyePACS

- **Source**: HuggingFace dataset [`bumbledeep/eyepacs`](https://huggingface.co/datasets/bumbledeep/eyepacs) (MIT licence).
- **Format**: HuggingFace Arrow format, saved locally to Google Drive at `configs/paths.yaml :: eyepacs_hf_dir`.
- **Size**: 35,108 rows, single `train` split.
- **Columns**: `image` (PIL), `label_code` (int, 0–4), `label` (string).
- **Classes**: `no_diabetic_retinopathy`, `mild`, `moderate`, `severe`, `proliferative_diabetic_retinopathy`.
- The legacy `EYEPACS.zip` path is retained in `paths.yaml` (`eyepacs_zip`) for reference only; all loaders use the HuggingFace version via `load_from_disk`.

### OLIVES

- **Source**: `olive.zip` on Google Drive — a wrapper containing `OLIVES.zip`, which in turn contains the two trial archives `TREX_DME.zip` and `Prime_FULL.zip`. Labels are already extracted to `configs/paths.yaml :: olives_labels_dir`.
- **Modalities**: fundus (`.tif` / `.png`), OCT (`.tif` / `.png`), and clinical/biomarker metadata. This project uses fundus images only as model input; OCT-derived biomarkers serve as supervision targets rather than additional input modalities.
- **Trial archives**:
  - `TREX_DME.zip` — 98,897 files. Folder pattern: `TREX DME/GILA/{patient_id}/V{visit}/{eye}/`. Contains 1,849 fundus images (`fundus_OD/OS_V{visit}.tif`), 90,706 OCT scans (`TREXW_000000.tif` …), and 1,045 sequence files (`.npy`).
  - `Prime_FULL.zip` — 68,807 files. Folder pattern: `Prime_FULL/{patient_id}/W{week}/{eye}/`. Contains 2,586 fundus images (`fundus_OD/OS_W{week}`), 65,375 OCT scans (`0.png`, `1.png` …), and 738 sequence files (`.npy`).
  - **Total fundus images**: ~4,435 across both trials. Fundus images are identifiable by `fundus` in the filename.
- **Labels CSV**: `OLIVES_Dataset_Labels/ml_centric_labels/Biomarker_Clinical_Data_Images.csv` — 9,408 rows × 22 columns. Key columns: `Path` (links to image, matches the zip folder structure), `Patient_ID`, `Eye_ID`, `BCVA`, `CST`, plus 16 binary biomarker columns.
- **Final cohort after preprocessing**: 3,142 unique fundus images from 96 patients (204 patient-eyes) across the two trials, with mean 15.4 visits per eye. Matches the cohort reported in Prabhushankar et al., NeurIPS 2022. See `docs/PREPROCESSING_DOCUMENTATION.md` for the bug fixes that recovered this match.

## Setup and Reproduction

### Prerequisites

- Google Colab with GPU runtime (T4 minimum, A100 preferred for Phase 2)
- Google Drive with at least 25 GB free space for preprocessed tensors
- GitHub personal access token with repository read access, stored in Colab Secrets as `GITHUB_TOKEN`
- Both source datasets (EyePACS HuggingFace dataset and `olive.zip`) downloaded to Google Drive

### Local development

```bash
pip install -r requirements.txt
```

### Reproduction sequence

Notebooks are numbered by execution order and located in `notebooks/`. Each notebook is self-contained — it clones the repo, mounts Drive, and runs the relevant code modules from `src/`.

1. **`00_setup_test.ipynb`** — verifies the Colab environment, clones the repo, mounts Drive, prints loaded paths. Run once at the start of any new Colab session.
2. **`01_inspect_data.ipynb`** — non-destructive inspection of the three source zip archives. Reports sizes, file counts, and per-archive layout. No data is extracted.
3. **`01b_inspect_olives.ipynb`** — deeper OLIVES-specific inspection. Loads the biomarker labels CSV, scans the nested zip structure, and (optionally, after explicit confirmation) extracts ~4,435 fundus images.
4. **`02_preprocess.ipynb`** — preprocesses both datasets into normalised 224×224 PyTorch tensors saved to Drive. Resumable: existing shards are skipped per-shard, and the OLIVES output is skipped entirely if its `.pt` file already exists.
5. **`03_phase1_distributional_validation.ipynb`** — runs the six Phase 1 diagnostics on ResNet-50 features extracted from both datasets. Produces `phase1_report.md` and three plots saved to Drive at `dissertation/results/phase1/`.
6. **`04_phase2_training.ipynb`** *(in progress)* — Phase 2 model training pipeline.

### Configuration

All paths and parameters live in YAML files under `configs/`. The primary config files are:

- `configs/paths.yaml` — dataset and output locations on Drive
- `configs/preprocess.yaml` — preprocessing parameters (image size, normalisation, shard size)
- `configs/model.yaml` — model architecture hyperparameters *(Phase 2)*
- `configs/train.yaml` — training hyperparameters *(Phase 2)*

## Citation

If you reference this work, please cite:

```
Nair, Savita. (2026). Cross-Dataset Contrastive Alignment and Adaptive Multimodal Fusion for Diabetic Retinopathy Progression Prediction with Missing Modality Handling.
MSc dissertation, University of Bath.
```

## Acknowledgements

I am grateful to Dr Deblina Bhattacharjee for their guidance throughout this dissertation.

Datasets used in this project are publicly available under their respective licences:

- **EyePACS** via the [`bumbledeep/eyepacs`](https://huggingface.co/datasets/bumbledeep/eyepacs) HuggingFace dataset (MIT licence), originally from the [Kaggle Diabetic Retinopathy Detection competition](https://www.kaggle.com/c/diabetic-retinopathy-detection/)
- **OLIVES** from Prabhushankar et al. ([NeurIPS 2022](https://alregib.ece.gatech.edu/olives-dataset/)), DOI: [10.5281/zenodo.7105232](https://doi.org/10.5281/zenodo.7105232)

This work is conducted as part of the MSc programme at the University of Bath.

## Licence

Code: MIT Licence (see `LICENSE` file if present in repository, otherwise default to MIT)
Documentation and analysis: CC-BY 4.0
