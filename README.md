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

## Setup

```bash
pip install -r requirements.txt
```
