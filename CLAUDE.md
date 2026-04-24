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

## Current Phase: Distribution Validation

The project is currently in the **DISTRIBUTION VALIDATION** phase. The goal is to validate whether EyePACS and OLIVES feature distributions are compatible in a shared latent space.

The dissertation supervisor requires this validation to pass before any model training or architectural development begins. **Do not** scaffold or implement model architecture, training loops, or evaluation code during this phase.

## Roadmap (post-validation)

Once distribution validation is satisfactory, the project will proceed with:

1. Encoder pretraining
2. Contrastive alignment
3. Multimodal fusion
4. Longitudinal prediction
5. Uncertainty quantification

None of the above should be implemented until validation results are confirmed.
