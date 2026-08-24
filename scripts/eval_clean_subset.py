"""Leakage-free (patient-clean) QWK evaluation on the EyePACS validation split.

Both frozen checkpoints evaluated on the full EyePACS validation split and on
the patient-clean subset whose fellow eye is absent from training.

Expected output (acceptance criteria):
    baseline    full-val QWK 0.6148   clean-val QWK 0.6173
    harmonised  full-val QWK 0.5951   clean-val QWK 0.5917

Usage (from the repo root):
    python -m scripts.eval_clean_subset \
        --baseline-ckpt /content/drive/MyDrive/dissertation/checkpoints/phase2_coral_on_best.pt \
        --harmonised-ckpt /content/drive/MyDrive/dissertation/greyscale_experiment/phase2_greenhistmatch_best.pt \
        --reference-hist /content/drive/MyDrive/dissertation/greyscale_experiment/olives_reference_hist.npy
"""
import argparse
import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score

from src.analysis.uncertainty import load_full_model, read_merged_config
from src.preprocessing.harmonise import harmonise_eyepacs_in_bundle
from src.data.dataloaders import build_dataloaders

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

EXPECTED = {
    "baseline_full": 0.6148,
    "baseline_clean": 0.6173,
    "harmonised_full": 0.5951,
    "harmonised_clean": 0.5917,
    "clean_n": 1638,
    "full_n": 5266,
}


def build_clean_mask(train_indices, val_indices):
    train_set = set(int(i) for i in train_indices)
    train_patients = {i // 2 for i in train_set}
    patient_level = np.array([(int(i) // 2) not in train_patients for i in val_indices])
    pair_level = np.array([(int(i) ^ 1) not in train_set for i in val_indices])
    assert (patient_level == pair_level).all(), (
        "Dual-definition equivalence check failed: patient-level and "
        "pair-level clean masks disagree."
    )
    return patient_level


@torch.no_grad()
def eyepacs_val_predictions(model, val_loader, device):
    preds, labels = [], []
    model.eval()
    for batch in val_loader:
        is_eyepacs = batch["dataset"] == 0
        if not is_eyepacs.any():
            continue
        images = batch["images"][is_eyepacs].to(device)
        out = model(images)
        preds.append(out["dr_logits"].argmax(dim=1).cpu())
        labels.append(batch["dr_labels"][is_eyepacs].cpu())
    return torch.cat(preds).numpy(), torch.cat(labels).numpy()


def qwk(labels, preds):
    return cohen_kappa_score(labels, preds, weights="quadratic")


def evaluate_checkpoint(ckpt_path, device, clean_mask, harmonise=None):
    config = read_merged_config(ckpt_path)
    bundle = build_dataloaders(config)
    if harmonise is not None:
        harmonise_eyepacs_in_bundle(
            bundle, harmonise, IMAGENET_MEAN, IMAGENET_STD, source_channel=1
        )
    model = load_full_model(ckpt_path, device)
    preds, labels = eyepacs_val_predictions(model, bundle["val_loader"], device)
    assert len(preds) == len(clean_mask), (
        f"EyePACS val prediction count {len(preds)} != mask length "
        f"{len(clean_mask)}; loader order vs split-index alignment is broken."
    )
    full = qwk(labels, preds)
    clean = qwk(labels[clean_mask], preds[clean_mask])
    return full, clean, len(preds), int(clean_mask.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-ckpt", required=True)
    ap.add_argument("--harmonised-ckpt", required=True)
    ap.add_argument("--reference-hist", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = read_merged_config(args.baseline_ckpt)
    bundle = build_dataloaders(config)
    splits = bundle["splits"]["eyepacs"]
    clean_mask = build_clean_mask(splits["train"], splits["val"])

    b_full, b_clean, n_full, n_clean = evaluate_checkpoint(
        args.baseline_ckpt, device, clean_mask
    )
    h_full, h_clean, _, _ = evaluate_checkpoint(
        args.harmonised_ckpt, device, clean_mask, harmonise=args.reference_hist
    )

    print(f"EyePACS validation: full n={n_full}, patient-clean n={n_clean}")
    print(f"{'model':<12}{'full-val QWK':>14}{'clean-val QWK':>15}")
    print(f"{'baseline':<12}{b_full:>14.4f}{b_clean:>15.4f}")
    print(f"{'harmonised':<12}{h_full:>14.4f}{h_clean:>15.4f}")
    print(f"delta (harmonised - baseline): full {h_full - b_full:+.4f}, "
          f"clean {h_clean - b_clean:+.4f}")

    ok = (
        round(b_full, 4) == EXPECTED["baseline_full"]
        and round(h_full, 4) == EXPECTED["harmonised_full"]
        and n_clean == EXPECTED["clean_n"]
        and n_full == EXPECTED["full_n"]
    )
    print("ACCEPTANCE:", "PASS" if ok else "FAIL -- do not commit; "
          "diff against the original Colab cell before proceeding.")


if __name__ == "__main__":
    main()
