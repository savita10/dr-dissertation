# Cross-Modality Diabetic Retinopathy Assessment

**Transfer Failure, In-Domain Structural Prediction, and Recovery by Input Harmonisation**

MSc Dissertation, University of Bath.

**GitHub repository:** https://github.com/savita10/dr-dissertation

This repository contains the complete code to reproduce the results of a study on cross-modality transfer of deep learning for diabetic retinopathy (DR) assessment, from colour fundus photography (EyePACS) to near-infrared reflectance imaging acquired by confocal scanning laser ophthalmoscopy (OLIVES).

---

## 1. Study overview

A multi-task model is trained on colour fundus images and evaluated across the modality gap to near-infrared images. The study establishes three findings:

1. **Zero-shot cross-modality DR grading fails, and fails confidently.** Applied to near-infrared OLIVES images, the colour-trained grader collapses to the majority class (96.1% predicted no-DR), and its predictive uncertainty does not flag the failure.
2. **In-domain structural prediction succeeds; functional prediction does not.** From near-infrared images the model recovers retinal fluid biomarkers (AUROC 0.90–0.96) and, more weakly, central subfield thickness, but not visual acuity.
3. **The grading collapse is partly recoverable by input harmonisation.** Green-channel extraction with histogram matching to the target intensity distribution recovers substantial, clinically-validated grading signal, at a cost of 0.02 in in-domain validation quadratic weighted kappa.

## 2. Datasets

Both datasets are publicly available.

| Dataset | Modality | Source / link |
|---|---|---|
| EyePACS | Colour fundus, DR-graded 0–4 | Hugging Face: https://huggingface.co/datasets/bumbledeep/eyepacs (originally the Kaggle Diabetic Retinopathy Detection competition) |
| OLIVES | Near-infrared fundus + OCT-derived clinical labels and biomarkers | Zenodo DOI 10.5281/zenodo.7105232 — https://doi.org/10.5281/zenodo.7105232 (see also https://github.com/olivesgatech/OLIVES_Dataset) |

## 3. Shared data (checkpoints, tensors, labels)

The trained checkpoints and preprocessed data are too large for GitHub and are hosted on Google Drive.

**Shared Drive folder:** `https://drive.google.com/drive/folders/1xZH8GxuSqSETpxs91cM6OhSytJvSh9MT?usp=drive_link`

The shared folder mirrors the `dissertation/` working directory. The files that matter for reproduction are:

- `checkpoints/phase2_coral_on_best.pt` — **the main model** (colour). Used for all Phase 1–4 results. (Ignore the `coral_off`, `smoke`, and `last` variants — they are not needed.)
- `greyscale_experiment/phase2_greenhistmatch_best.pt` — **the harmonisation model**. Used for the Section 6.5 recovery results.
- `preprocessed/eyepacs/` — EyePACS preprocessed tensors (8 shards + metadata).
- `preprocessed/olives/` — OLIVES preprocessed tensor + metadata.
- `preprocessed/regression_stats.json` — BCVA/CST normalisation statistics.
- `greyscale_experiment/olives_reference_hist.npy` — the harmonisation reference histogram.

## 4. Two ways to reproduce the results

**Path A — Use the preprocessed tensors (recommended, faster).**
Download the checkpoints and the `preprocessed/` tensors from the shared folder, place them on your own Drive, and run the evaluation directly. This skips the multi-step raw-data download and preprocessing.

**Path B — Regenerate everything from the public raw data (full pipeline).**
Download the public EyePACS and OLIVES datasets, run the preprocessing step to regenerate the tensors, then run the evaluation. Choose this if you wish to verify the preprocessing as well. Note the raw-data setup is more involved (see Section 7).

Both paths run the **same evaluation commands** (Section 8). The only difference is whether you download the tensors (Path A) or regenerate them (Path B).

## 5. Required Google Drive folder structure

The configuration files use absolute Drive paths of the form `/content/drive/MyDrive/dissertation/...`. When you mount your own Google Drive in Colab, `MyDrive` refers to **your** Drive, so recreate this structure on your own Drive and the paths will resolve without editing any config. If you place files elsewhere, edit the corresponding paths in `configs/*.yaml`.

```
MyDrive/
└── dissertation/
    ├── checkpoints/
    │   └── phase2_coral_on_best.pt
    ├── greyscale_experiment/
    │   ├── phase2_greenhistmatch_best.pt
    │   └── olives_reference_hist.npy
    ├── preprocessed/
    │   ├── eyepacs/                     (8 tensor shards + metadata.json)
    │   ├── olives/                      (olives_fundus.pt + metadata.json)
    │   └── regression_stats.json
    ├── features/                        (created by the pipeline)
    ├── reports/                         (created by the pipeline)
    ├── results/                         (created by the pipeline)
    └── figures/                         (created by the pipeline)
```

For **Path B** you additionally need the raw data in place (see Section 7).

## 6. Setup (both paths)

1. Open the project in Google Colab with a **GPU runtime** (Runtime, then Change runtime type, then GPU).
2. **Clone the repository and mount Drive.** The repository is public, so no credentials are required:
   ```python
   from google.colab import drive
   import os
   drive.mount('/content/drive')
   REPO_DIR = "/content/dr-dissertation"
   if not os.path.exists(REPO_DIR):
       !git clone https://github.com/savita10/dr-dissertation.git {REPO_DIR}
   else:
       !git -C {REPO_DIR} fetch origin && git -C {REPO_DIR} reset --hard origin/main
   !cd /content/dr-dissertation && pip install -r requirements.txt -q
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   if torch.cuda.is_available():
       print(f"GPU: {torch.cuda.get_device_name(0)}")
   ```
   Every command below is prefixed with `cd /content/dr-dissertation &&` so it runs from the repository root; this is required for the `src` and `scripts` modules to be found.
3. Confirm the Drive folder structure of Section 5 is in place on your Drive.

## 7. Path B only — raw data download and preprocessing

Skip this section entirely if you are using the preprocessed tensors (Path A).

**Download the raw data to these locations** (paths are set in `configs/preprocess.yaml`; edit them if you use different locations):

- **EyePACS:** download from Hugging Face into `/content/eyepacs_hf` (a runtime-local path; re-downloaded each session).
- **OLIVES images:** download the archive to `MyDrive/olive.zip`.
- **OLIVES labels:** download the label files into `MyDrive/olives_labels/OLIVES_Dataset_Labels/`:
  - `ml_centric_labels/Biomarker_Clinical_Data_Images.csv` (tier-2 biomarker labels)
  - `full_labels/Clinical_Data_Images.xlsx` (tier-1 clinical labels — **required**; preprocessing fails without it)

**Run preprocessing** (regenerates the tensors into `preprocessed/`):
```python
!cd /content/dr-dissertation && python -m src.data.preprocess --config configs/preprocess.yaml
```

## 8. Evaluation pipeline

Run these in order. Later phases consume the feature outputs of earlier phases, so the sequence matters. Each command writes a JSON report, a markdown summary, and figures under `reports/`, `results/`, and `figures/`.

```python
# Phase 1 — distributional diagnostics between the two modalities
!cd /content/dr-dissertation && python -m src.analysis.feature_extraction --dataset eyepacs --config configs/preprocess.yaml
!cd /content/dr-dissertation && python -m src.analysis.feature_extraction --dataset olives --config configs/preprocess.yaml
!cd /content/dr-dissertation && python -m src.analysis.distribution_diagnostics --config configs/preprocess.yaml

# Phase 2 — alignment diagnostics (post-training gap, CORAL ablation)
!cd /content/dr-dissertation && python -m src.analysis.phase2_diagnostics --config configs/diagnostics.yaml

# Phase 3a — EyePACS calibration and test-time-augmentation uncertainty
!cd /content/dr-dissertation && python -m src.analysis.calibration_eyepacs --config configs/uncertainty.yaml

# Phase 4 — cross-modality transfer and in-domain evaluation
!cd /content/dr-dissertation && python -m src.analysis.zero_shot_grading --config configs/zero_shot.yaml
!cd /content/dr-dissertation && python -m src.analysis.dr_failure_characterisation --config configs/zero_shot.yaml
!cd /content/dr-dissertation && python -m src.analysis.indomain_evaluation --config configs/indomain_eval.yaml
!cd /content/dr-dissertation && python -m src.analysis.phase4_report --config configs/zero_shot.yaml

# Greyscale harmonisation experiment (evaluation only — uses the supplied harmonised checkpoint)
!cd /content/dr-dissertation && python -m scripts.compute_olives_histogram --config configs/greyscale_train.yaml
!cd /content/dr-dissertation && python -m src.analysis.zero_shot_grading --config configs/greyscale_eval.yaml
!cd /content/dr-dissertation && python -m src.analysis.greyscale_concordance --config configs/greyscale_eval.yaml
!cd /content/dr-dissertation && python -m src.analysis.greyscale_categorical_concordance --config configs/greyscale_eval.yaml
```

**Training is not required** to reproduce the results, because the trained checkpoints are supplied. See the optional training section below if you wish to train the models yourself.

### Optional — train the models from scratch

The supplied checkpoints reproduce the reported results exactly. If you instead wish to train the models yourself, first complete the Path B raw-data setup and preprocessing (Section 7), then run:

```python
# Main model (colour) — writes a checkpoint to checkpoints/
!cd /content/dr-dissertation && python -m src.training.train --config configs/train.yaml

# Harmonised model (green-channel histogram-matched) — writes to greyscale_experiment/
!cd /content/dr-dissertation && python -m scripts.train_greyscale --config configs/greyscale_train.yaml
```

Each run reports its best validation DR quadratic weighted kappa at the end (the main model is approximately 0.615). Training the main model takes several hours on GPU.

**Important:** a retrained model will not be bit-identical to the supplied checkpoint, because training involves randomness (weight initialisation, data ordering, and GPU-level non-determinism). Results from a retrained model will therefore be close to, but may differ slightly from, the values in the expected-results table. To reproduce the reported numbers exactly, use the supplied checkpoints (Paths A or B).

## 9. Expected results (verification table)

If your run is correct, the values below should match. The **source file** column gives the output file (written under `reports/` or `greyscale_experiment/`) in which each value appears, so you can locate and check it directly. Small variation in the last decimal place is expected on different hardware.

**Phase 1 — cross-modality distributional gap**

| Metric | Expected | Source file |
|---|---|---|
| Pre-alignment gap (FID) | 240.65 | reports/phase2_results.json |
| Post-alignment gap (FID) | 74.64 | reports/phase2_results.json |

**Phase 2 — in-domain grading**

| Metric | Expected | Source file |
|---|---|---|
| In-domain validation QWK | ≈ 0.615 | reports/phase2_results.json |

**Phase 3a — calibration and uncertainty (EyePACS test set)**

| Metric | Expected | Source file |
|---|---|---|
| Test accuracy | 0.736 | reports/phase3a_calibration.json |
| Test QWK | 0.642 | reports/phase3a_calibration.json |
| Expected calibration error (ECE) | 0.12 | reports/phase3a_calibration.json |
| Selective accuracy at low coverage | up to ~0.90 | reports/phase3a_calibration.json |

**Phase 4a/4b — zero-shot cross-modality grading (OLIVES) and failure**

| Metric | Expected | Source file |
|---|---|---|
| Predicted grade-0 share (collapse) | 96.1% | reports/phase4b_failure.json |
| Grade-1 / 2 / 3 / 4 shares | 0.5% / 0.4% / 0.8% / 2.2% | reports/phase4b_failure.json |
| Grade vs central subfield thickness (wrong-direction) | rho = -0.492 | reports/phase4b_failure.json |

**Phase 4c — in-domain evaluation, held-out OLIVES test split (28 biomarker-labelled images)**

| Metric | Expected | Source file |
|---|---|---|
| IRF AUROC | 0.958 | reports/phase4c_indomain.json |
| SRF AUROC | 0.904 | reports/phase4c_indomain.json |
| Vitreous debris AUROC | 0.875 | reports/phase4c_indomain.json |
| CST R-squared | 0.198 | reports/phase4c_indomain.json |
| BCVA R-squared | -0.194 | reports/phase4c_indomain.json |

**Greyscale harmonisation — harmonised model (OLIVES)**

| Metric | Expected | Source file |
|---|---|---|
| Predicted grade-0 share (recovered) | 53.2% | greyscale_experiment/greyscale_concordance.json |
| Grade vs central subfield thickness (correct direction) | rho = +0.332 | greyscale_experiment/greyscale_concordance.json |
| Primary biomarker AUROC range | 0.685 - 0.764 | greyscale_experiment/greyscale_categorical_concordance.json |
| Patient-level consistency test | p < 0.001 | greyscale_experiment/greyscale_categorical_concordance.json |
| In-domain validation QWK cost | 0.615 -> 0.595 (0.02) | greyscale training log |

The OLIVES split is patient-aware with a fixed seed (42), so the held-out test split is deterministic and should match exactly. Values are reported to the precision at which they appear in the source files; differences beyond the last decimal place may indicate a configuration or data-placement issue.

## 10. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` (or `scripts`) | The command is not running from the repository root. Ensure every command is prefixed with `cd /content/dr-dissertation &&` as shown in Section 8. |
| `FileNotFoundError` for a checkpoint or tensor | A file is missing from the expected Drive path. Confirm the folder structure in Section 5 is recreated on your Drive and the shared files are placed correctly. |
| A later phase fails on a missing feature file (e.g. `tta_eyepacs_test.pt`) | The phases were run out of order. Run them in the sequence given in Section 8; later phases consume earlier phases' outputs. |
| A script logs "cuda unavailable, falling back to cpu" | No GPU attached. Set Runtime, then Change runtime type, then GPU, and re-run. |
| The clone fails or asks for credentials | The repository is public, so no token is needed. Use the plain clone URL in Section 6; if you cloned earlier while it was private, delete the runtime and start a fresh session. |
| Paths point to a different Drive layout | The configuration files use absolute `MyDrive/dissertation/...` paths. Recreate the Section 5 structure on your own Drive, or edit the paths in `configs/*.yaml`. |

## 11. Repository structure

```
src/
  models/        DRModel (ResNet-50 encoder + DR / biomarker / regression heads)
  training/      training entry point
  data/          preprocessing
  analysis/      per-phase evaluation modules
configs/         YAML configuration (paths are set here)
scripts/         greyscale reference-histogram computation
notebooks/       Colab notebooks that orchestrate each phase
```

## 12. Requirements

Python 3, PyTorch 2.x with torchvision (no TensorFlow), plus numpy, scikit-image, scikit-learn, pandas, matplotlib, tqdm, omegaconf, and the Hugging Face `datasets` library. Install with `pip install -r requirements.txt`. A GPU runtime is required for training and for the evaluation forward passes.

## 13. Notes for evaluators

- All results are produced by the Section 8 commands from the supplied frozen checkpoints; no retraining is required.
- The configuration files use absolute `MyDrive/dissertation/...` paths. Recreating the folder structure of Section 5 on your own Drive is the simplest way to make them resolve; otherwise edit the paths in `configs/*.yaml`.
- Run the evaluation phases in the given order, as later phases consume earlier phases' outputs.
- This is a research study on frozen models. It is not a clinical tool and makes no diagnostic claims about individuals.
