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

- Multimodal dataset (fundus + OCT + clinical metadata). Loader to be added.

## Setup

```bash
pip install -r requirements.txt
```
