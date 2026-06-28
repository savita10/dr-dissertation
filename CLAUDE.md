# CLAUDE.md

Project conventions for the diabetic retinopathy progression prediction dissertation.

## Framework & Environment

- **Framework:** PyTorch 2.x. Do **not** use TensorFlow.
- **Training environment:** Google Colab Pro, with all data and artefacts stored on Google Drive.

## Code Style

- Type hints are required on all function signatures (parameters and return types).
- Format all Python code with `black`.
- No unused imports — strip them before committing.

## Configuration

- All configuration lives in YAML files under `configs/`, loaded via `omegaconf`.
- **No hardcoded hyperparameters** in source files. Every tunable value (learning rate, batch size, image size, paths, seeds, etc.) belongs in a config.
- All paths that reference Google Drive or the Colab filesystem go in `configs/paths.yaml`.

## Notebooks

- Notebooks are **only** for Colab wrappers and exploratory analysis.
- Reusable logic lives in `src/`, not in notebooks.

## Version Control

- Never commit raw dataset files, model checkpoints, or `.pt` / `.npy` tensor files. The `.gitignore` already excludes these — do not bypass it.

## Progress Reporting

- Use `tqdm` for all progress bars (data loading, feature extraction, training loops).

## Datasets

- **EyePACS** — fundus images only (single modality). Source: HuggingFace dataset `bumbledeep/eyepacs`, saved to Drive in Arrow format and loaded via `datasets.load_from_disk`. Do **not** use `EYEPACS.zip` — the zip path is retained in `configs/paths.yaml` for reference only.
- **OLIVES** — multimodal (fundus + OCT + clinical).

## Current Phase: Phase 2 (Model Development)

The project is in **Phase 2 — Model Development**. Phases 2a (model architecture + multi-term loss), 2b (data loading, augmentation, patient-aware sampling), and 2c (training pipeline) are complete; **Phase 2d (post-training diagnostics)** is in progress.

> Phase 1 distribution validation passed (FID 240.65); project advanced to Phase 2 model development.

## Roadmap

1. Encoder pretraining
2. Contrastive alignment
3. Multimodal fusion
4. Longitudinal prediction
5. Uncertainty quantification
