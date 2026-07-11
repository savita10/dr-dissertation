# CLAUDE.md

Project conventions for the diabetic retinopathy progression prediction dissertation.

This project studies **cross-modality** DR knowledge transfer (colour fundus → near-IR fundus): EyePACS is colour fundus photography and OLIVES is near-infrared (near-IR, scanning-laser) fundus imaging. The gap between them is a **modality gap**, of which camera field-stop geometry is one measurable facet (not the whole story) — earlier "cross-dataset" / "acquisition-driven" framing has been superseded by this cross-modality framing.

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

- **EyePACS** — colour fundus photography only (single modality). Source: HuggingFace dataset `bumbledeep/eyepacs`, saved to Drive in Arrow format and loaded via `datasets.load_from_disk`. Do **not** use `EYEPACS.zip` — the zip path is retained in `configs/paths.yaml` for reference only.
- **OLIVES** — multimodal (near-IR scanning-laser fundus + OCT + clinical).
- **Modality gap** — EyePACS (colour fundus) and OLIVES (near-IR fundus) are different imaging modalities; the distributional gap between them is a modality gap, with camera field-stop geometry as one measurable facet rather than the whole story.

## Current Phase: Phase 3/4 (Uncertainty & Cross-Modality Transfer)

The project is in **Phase 3/4 — uncertainty quantification (test-time augmentation on the frozen model) and uncertainty-aware zero-shot cross-modality DR grade transfer (EyePACS colour → OLIVES near-IR), validated indirectly.** Phases 1 and 2 are complete.

> Phase 1 distribution validation passed (FID 240.65) and Phase 2 model development is complete; the project has advanced to Phase 3/4.

## Roadmap

1. Encoder pretraining
2. Contrastive alignment
3. Multimodal fusion
4. Longitudinal prediction
5. Uncertainty quantification
