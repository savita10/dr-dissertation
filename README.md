# Diabetic Retinopathy Progression Prediction

This project develops a deep learning framework for predicting diabetic retinopathy progression using EyePACS and OLIVES datasets. The current phase focuses on validating cross-dataset distribution compatibility. Model architecture, training pipelines, and evaluation will follow once validation confirms the datasets can be jointly encoded in a shared latent space.

## Project Structure

```
dissertation/
├── src/
│   ├── data/         # Preprocessing and dataset loaders
│   ├── analysis/     # Feature extraction, distribution metrics, plotting
│   └── utils/        # Helpers, config loading, Colab setup
├── configs/          # YAML configuration files
├── notebooks/        # Colab notebook wrappers
├── docs/             # Validation reports
└── requirements.txt
```

## Phase 1: Distribution Validation

Before any model training, this phase validates that EyePACS and OLIVES share a compatible feature-space distribution. Validation includes:

- Dataset inspection and preprocessing
- Feature extraction via pretrained encoders
- Distribution metrics (MMD, FID, KL divergence)
- UMAP/t-SNE visualisation of joint embedding space

## Data

### EyePACS

- **Source:** HuggingFace dataset [`bumbledeep/eyepacs`](https://huggingface.co/datasets/bumbledeep/eyepacs) (MIT licence).
- **Format:** HuggingFace Arrow format, saved locally to Google Drive at `configs/paths.yaml :: eyepacs_hf_dir`.
- **Size:** 35,108 rows, single `train` split.
- **Columns:** `image` (PIL), `label_code` (int, 0–4), `label` (string).
- **Classes:** `no_diabetic_retinopathy`, `mild`, `moderate`, `severe`, `proliferative_diabetic_retinopathy`.
- The legacy `EYEPACS.zip` path is retained in `paths.yaml` (`eyepacs_zip`) for reference only; all loaders use the HuggingFace version via `load_from_disk`.

### OLIVES

- **Source:** `olive.zip` on Google Drive — a wrapper containing `OLIVES.zip`, which in turn contains the two trial archives `TREX_DME.zip` and `Prime_FULL.zip`. Labels are already extracted to `configs/paths.yaml :: olives_labels_dir`.
- **Modalities:** fundus (`.tif` / `.png`), OCT (`.tif` / `.png`), and clinical/biomarker metadata.
- **Trial archives:**
  - `TREX_DME.zip` — 98,897 files. Folder pattern: `TREX DME/GILA/{patient_id}/V{visit}/{eye}/`. Contains 1,849 fundus images (`fundus_OD/OS_V{visit}.tif`), 90,706 OCT scans (`TREXW_000000.tif` …), and 1,045 sequence files (`.npy`).
  - `Prime_FULL.zip` — 68,807 files. Folder pattern: `Prime_FULL/{patient_id}/W{week}/{eye}/`. Contains 2,586 fundus images (`fundus_OD/OS_W{week}`), 65,375 OCT scans (`0.png`, `1.png` …), and 738 sequence files (`.npy`).
  - **Total fundus images:** ~4,435 across both trials. Fundus images are identifiable by `fundus` in the filename.
- **Labels CSV:** `OLIVES_Dataset_Labels/ml_centric_labels/Biomarker_Clinical_Data_Images.csv` — 9,408 rows × 22 columns. Key columns: `Path` (links to image, matches the zip folder structure), `Patient_ID`, `Eye_ID`, `BCVA`, `CST`, plus 16 binary biomarker columns.
- **Loader:** `src/data/olives_loader.py` exposes `load_olives_labels`, `get_fundus_paths_from_zip`, `extract_fundus_images` (fundus only, resumable), and `join_fundus_to_labels`. The Colab wrapper is `notebooks/01b_inspect_olives.ipynb`.

## Setup

```bash
pip install -r requirements.txt
```
