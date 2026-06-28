"""Phase 2d post-training diagnostics: does the trained encoder close the gap?

Runs the *unchanged* Phase 1 distribution diagnostics (FID, MMD, k-NN cosine,
PCA / PC1 variance, PC1↔black-border Spearman, DR-class separation on PC1) on
features from the trained encoder, for each arm (``coral_on`` / ``coral_off``)
and split (``full`` / ``test``). Emits ``phase2_report.md`` with three tables:

* Table 1 — before/after vs the Phase 1 baseline (full data, ``coral_on``).
* Table 2 — the with/without-CORAL ablation (full data).
* Table 3 — generalisation on the held-out test split.

plus a provenance block and ``phase2_results.json`` for plotting / the appendix.

Run:
    python -m src.analysis.phase2_diagnostics --config configs/diagnostics.yaml
"""

import argparse
import json
import math
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from scipy.stats import spearmanr

from src.analysis.distribution_diagnostics import (
    N_DR_CLASSES,
    compute_cosine_similarity_diagnostics,
    compute_frechet_distance,
    compute_mmd,
    compute_pca_by_severity,
    compute_pca_projection,
)
from src.analysis.trained_feature_extraction import (
    extract_features_for_checkpoint,
    load_saved_features,
)
from src.utils.config import load_config

REPORT_FILENAME = "phase2_report.md"
RESULTS_JSON_FILENAME = "phase2_results.json"
SEVERITY_PLOT_TEMPLATE = "phase2_pca_severity_{arm}_{split}.png"


# --------------------------------------------------------------------- helpers
def _features_np(blob: dict[str, Any]) -> np.ndarray:
    feats = blob["features"]
    if isinstance(feats, torch.Tensor):
        feats = feats.detach().cpu().numpy()
    return np.asarray(feats, dtype=np.float32)


def _json_safe(value: object) -> object:
    """Convert tensors/ndarrays/non-finite floats so json.dump emits valid JSON."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return _json_safe(value.item())
    if torch.is_tensor(value):
        return _json_safe(value.item() if value.ndim == 0 else value.tolist())
    return value


def _f(value: float, nd: int = 4) -> str:
    return "nan" if value is None or not math.isfinite(value) else f"{value:.{nd}f}"


def _p(value: float) -> str:
    if value is None or not math.isfinite(value):
        return "nan"
    return "<0.0001" if value < 1e-4 else f"{value:.4f}"


def _verdict(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def _dr_ordering(severity_means: dict[str, float]) -> dict[str, Any]:
    """Whether EyePACS DR-class mean PC1 is monotonically ordered by severity."""
    pairs = [
        (cls, severity_means[f"class_{cls}"])
        for cls in range(N_DR_CLASSES)
        if math.isfinite(severity_means.get(f"class_{cls}", float("nan")))
    ]
    class_means = [m for _, m in pairs]
    if len(pairs) < 2:
        return {"ordered": False, "rank_rho": float("nan"), "class_means": class_means}
    rank_rho, _ = spearmanr([c for c, _ in pairs], class_means)
    increasing = all(b > a for a, b in zip(class_means, class_means[1:]))
    decreasing = all(b < a for a, b in zip(class_means, class_means[1:]))
    return {
        "ordered": bool(increasing or decreasing),
        "rank_rho": float(rank_rho),
        "class_means": [float(m) for m in class_means],
    }


def compute_arm_split_metrics(
    eye_blob: dict[str, Any],
    oli_blob: dict[str, Any],
    report_dir: Path,
    arm: str,
    split: str,
) -> dict[str, Any]:
    """Run the reused Phase 1 diagnostics on one arm+split's trained features."""
    eye = _features_np(eye_blob)
    oli = _features_np(oli_blob)
    eye_stats = eye_blob["image_stats"]
    oli_stats = oli_blob["image_stats"]
    eye_labels = eye_blob["labels"]
    if isinstance(eye_labels, torch.Tensor):
        eye_labels = eye_labels.cpu().numpy()
    eye_labels = np.asarray(eye_labels).astype(np.int64)

    if eye.shape[1] != oli.shape[1]:
        raise ValueError(
            f"Feature dim mismatch EyePACS={eye.shape[1]} vs OLIVES={oli.shape[1]}."
        )

    # PCA (reused) — one joint fit drives PC1 variance and the black-border rho.
    coords_eye, coords_oli, pca = compute_pca_projection(eye, oli)
    pca_var = [float(v) for v in pca.explained_variance_ratio_[:2]]
    eye_pc1 = coords_eye[:, 0]
    oli_pc1 = coords_oli[:, 0]

    fid = compute_frechet_distance(eye, oli)
    mmd = compute_mmd(eye, oli)
    cosine = compute_cosine_similarity_diagnostics(eye, oli)
    cosine_mean = 0.5 * (cosine["a_to_b_mean"] + cosine["b_to_a_mean"])

    # PC1 ↔ black-border Spearman (same metric/definition as Phase 1).
    rho_eye, p_eye = spearmanr(eye_stats["black_pixel"], eye_pc1)
    rho_oli, p_oli = spearmanr(oli_stats["black_pixel"], oli_pc1)

    # DR-class separation on PC1 (reused; writes the severity scatter plot).
    plot_path = report_dir / SEVERITY_PLOT_TEMPLATE.format(arm=arm, split=split)
    severity_means = compute_pca_by_severity(eye, eye_labels, oli, plot_path)
    ordering = _dr_ordering(severity_means)
    centroid_distance = abs(
        severity_means["eyepacs_all"] - severity_means["olives"]
    )

    return {
        "arm": arm,
        "split": split,
        "n_eyepacs": int(eye.shape[0]),
        "n_olives": int(oli.shape[0]),
        "fid": float(fid),
        "mmd": mmd,
        "cosine": cosine,
        "cosine_mean": float(cosine_mean),
        "pca_var": pca_var,
        "black_border_rho": {
            "eyepacs": {"rho": float(rho_eye), "p_value": float(p_eye)},
            "olives": {"rho": float(rho_oli), "p_value": float(p_oli)},
        },
        "severity": {
            "means": severity_means,
            "ordered": ordering["ordered"],
            "rank_rho": ordering["rank_rho"],
            "centroid_distance": float(centroid_distance),
            "plot": plot_path.name,
        },
    }


# ---------------------------------------------------------------------- report
def _ordering_cell(metrics: dict[str, Any]) -> str:
    sev = metrics["severity"]
    state = "ordered" if sev["ordered"] else "intermixed"
    return f"{state} (ρ_rank={_f(sev['rank_rho'], 2)})"


def _table1(results: dict[str, dict[str, Any]], cfg: DictConfig) -> list[str]:
    m = results["coral_on"]["full"]
    base = cfg.phase1_baseline
    tgt = cfg.targets
    rho_eye = m["black_border_rho"]["eyepacs"]["rho"]

    rows = [
        (
            "FID (Fréchet)",
            _f(float(base.fid), 2),
            _f(m["fid"], 2),
            f"< {float(tgt.fid_max):.0f}",
            m["fid"] < float(tgt.fid_max),
        ),
        (
            "MMD p-value",
            "<0.0001",
            _p(m["mmd"]["p_value"]),
            f"> {float(tgt.mmd_p_min):.2f}",
            m["mmd"]["p_value"] > float(tgt.mmd_p_min),
        ),
        (
            "k-NN cosine (mean)",
            _f(float(base.cosine), 3),
            _f(m["cosine_mean"], 3),
            f"> {float(tgt.cosine_min):.2f}",
            m["cosine_mean"] > float(tgt.cosine_min),
        ),
        (
            "PC1 ↔ black-border ρ (EyePACS)",
            _f(float(base.pc1_black_border_rho), 3),
            _f(rho_eye, 3),
            f"|ρ| < {float(tgt.rho_abs_max):.2f}",
            abs(rho_eye) < float(tgt.rho_abs_max),
        ),
        (
            "DR classes on PC1",
            "intermixed",
            "ordered" if m["severity"]["ordered"] else "intermixed",
            "ordered",
            m["severity"]["ordered"],
        ),
    ]

    lines = ["## Table 1 — Before / after (full data, coral_on)", ""]
    lines.append("| Metric | Phase 1 baseline | Phase 2 (trained) | Target | Pass? |")
    lines.append("|--------|-----------------:|------------------:|:------:|:-----:|")
    n_pass = 0
    for name, baseline, trained, target, passed in rows:
        n_pass += int(passed)
        lines.append(
            f"| {name} | {baseline} | {trained} | {target} | {_verdict(passed)} |"
        )
    lines.append("")
    lines.append(
        f"**{n_pass}/{len(rows)} targets met.** "
        + _prose_before_after(m, cfg)
    )
    lines.append("")
    lines.append(f"![PCA by severity — coral_on/full]({m['severity']['plot']})")
    lines.append("")
    return lines


def _prose_before_after(m: dict[str, Any], cfg: DictConfig) -> str:
    base = cfg.phase1_baseline
    tgt = cfg.targets
    rho_eye = m["black_border_rho"]["eyepacs"]["rho"]
    fid_drop = float(base.fid) - m["fid"]
    fid_dir = "down" if fid_drop >= 0 else "up"
    parts = [
        f"After training with cross-dataset alignment, the Fréchet distance moves "
        f"from the Phase 1 baseline of {float(base.fid):.2f} to {m['fid']:.2f} "
        f"({fid_dir} {abs(fid_drop):.2f}; target < {float(tgt.fid_max):.0f})."
    ]
    if m["mmd"]["p_value"] > float(tgt.mmd_p_min):
        parts.append(
            f"The MMD permutation test no longer separates the datasets "
            f"(p = {_p(m['mmd']['p_value'])} > {float(tgt.mmd_p_min):.2f}), i.e. they "
            "have become statistically indistinguishable in feature space."
        )
    else:
        parts.append(
            f"The MMD permutation test still distinguishes the datasets "
            f"(p = {_p(m['mmd']['p_value'])})."
        )
    rho_collapsed = abs(rho_eye) < float(tgt.rho_abs_max)
    rho_phrase = "collapsed toward zero" if rho_collapsed else "not yet collapsed"
    encoder_phrase = (
        "no longer organising its dominant axis around camera geometry"
        if rho_collapsed
        else "still partly organised around camera geometry"
    )
    parts.append(
        f"The PC1↔black-border correlation moves from "
        f"ρ = {float(base.pc1_black_border_rho):+.3f} to ρ = {rho_eye:+.3f}: the "
        f"acquisition (field-stop) axis that dominated Phase 1 has {rho_phrase}, "
        f"indicating the encoder is {encoder_phrase}."
    )
    return " ".join(parts)


def _ablation_table(
    results: dict[str, dict[str, Any]], split: str, title: str
) -> list[str]:
    on = results["coral_on"][split]
    off = results["coral_off"][split]
    rows = [
        ("FID (Fréchet)", _f(on["fid"], 2), _f(off["fid"], 2)),
        ("MMD p-value", _p(on["mmd"]["p_value"]), _p(off["mmd"]["p_value"])),
        ("k-NN cosine (mean)", _f(on["cosine_mean"], 3), _f(off["cosine_mean"], 3)),
        (
            "PC1 ↔ black-border ρ (EyePACS)",
            _f(on["black_border_rho"]["eyepacs"]["rho"], 3),
            _f(off["black_border_rho"]["eyepacs"]["rho"], 3),
        ),
        ("DR classes on PC1", _ordering_cell(on), _ordering_cell(off)),
    ]
    lines = [title, ""]
    lines.append("| Metric | coral_on | coral_off |")
    lines.append("|--------|---------:|----------:|")
    for name, on_val, off_val in rows:
        lines.append(f"| {name} | {on_val} | {off_val} |")
    lines.append("")
    fid_delta = off["fid"] - on["fid"]
    lines.append(
        f"**FID delta (off − on): {fid_delta:+.2f}.** "
        + _prose_ablation(on, off, fid_delta)
    )
    lines.append("")
    return lines


def _prose_ablation(
    on: dict[str, Any], off: dict[str, Any], fid_delta: float
) -> str:
    if fid_delta > 0:
        head = (
            f"Alignment-on shows a markedly lower cross-dataset FID than "
            f"alignment-off ({on['fid']:.2f} vs {off['fid']:.2f}), so the CORAL "
            f"term — and not DR/biomarker supervision alone — is what closes the gap."
        )
    elif fid_delta < 0:
        head = (
            f"Unexpectedly, alignment-off has the lower FID ({off['fid']:.2f} vs "
            f"{on['fid']:.2f}); the alignment term did not reduce the cross-dataset "
            "distance on this data and warrants investigation."
        )
    else:
        head = "The two arms show effectively identical cross-dataset FID."
    tail = (
        f"On the acquisition axis, ρ(black-border) is "
        f"{on['black_border_rho']['eyepacs']['rho']:+.3f} (on) vs "
        f"{off['black_border_rho']['eyepacs']['rho']:+.3f} (off)."
    )
    return f"{head} {tail}"


def _provenance_block(results: dict[str, dict[str, Any]], arms: list[str]) -> list[str]:
    lines = ["## Provenance", ""]
    lines.append("| Arm | Checkpoint | Epoch | Best val dr_qwk | w_coral | w_supcon | w_dr |")
    lines.append("|-----|------------|------:|----------------:|--------:|---------:|-----:|")
    for arm in arms:
        prov = None
        for split in ("full", "test"):
            if arm in results and split in results[arm]:
                prov = results[arm][split].get("provenance", {})
                break
        if not prov:
            continue
        lw = prov.get("loss_weights", {})
        ckpt_name = Path(str(prov.get("checkpoint", "?"))).name
        lines.append(
            f"| {arm} | `{ckpt_name}` | {prov.get('epoch', '?')} | "
            f"{_f(float(prov.get('best_metric', float('nan'))), 4)} | "
            f"{lw.get('w_coral', '?')} | {lw.get('w_supcon', '?')} | "
            f"{lw.get('w_dr', '?')} |"
        )
    lines.append("")
    lines.append(
        "> `coral_off` is the ablation arm trained with `w_coral = 0`; all other "
        "loss weights match `coral_on`."
    )
    lines.append("")
    return lines


def build_report(results: dict[str, dict[str, Any]], cfg: DictConfig) -> str:
    arms = [str(a) for a in cfg.arms]
    splits = [str(s) for s in cfg.splits]

    def has(arm: str, split: str) -> bool:
        return arm in results and split in results[arm]

    lines = ["# Phase 2d — Post-training Diagnostics", ""]
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append(
        "All metrics are produced by the unchanged Phase 1 diagnostics "
        "(`src/analysis/distribution_diagnostics.py`) run on trained-encoder "
        "features, so they are directly comparable to the Phase 1 baseline."
    )
    lines.append("")

    if has("coral_on", "full"):
        lines += _table1(results, cfg)
    else:
        lines += [
            "## Table 1 — Before / after",
            "",
            "_coral_on/full not available._",
            "",
        ]

    if has("coral_on", "full") and has("coral_off", "full"):
        lines += _ablation_table(
            results, "full", "## Table 2 — Ablation (coral_on vs coral_off, full data)"
        )
    else:
        lines += [
            "## Table 2 — Ablation",
            "",
            "_both arms on full data not available._",
            "",
        ]

    if has("coral_on", "test") and has("coral_off", "test"):
        lines += _ablation_table(
            results, "test", "## Table 3 — Generalisation (held-out test split)"
        )
        lines.append(
            "> Same metrics on unseen test samples; if the alignment holds here it "
            "is not an artefact of fitting the full distribution."
        )
        lines.append("")
    else:
        lines += [
            "## Table 3 — Generalisation (test split)",
            "",
            "_both arms on test split not available._",
            "",
        ]

    lines += _provenance_block(results, arms)

    lines.append("## Coverage")
    lines.append("")
    lines.append(f"Arms: {', '.join(arms)} · Splits: {', '.join(splits)}")
    lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------- orchestrate
def run_diagnostics(cfg: DictConfig, device: str, batch_size: int) -> dict[str, Any]:
    arms = [str(a) for a in cfg.arms]
    splits = [str(s) for s in cfg.splits]
    checkpoint_dir = Path(str(cfg.checkpoint_dir))
    report_dir = Path(str(cfg.report_dir))
    report_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, Any]] = {}
    total = len(arms) * len(splits)
    step = 0
    for arm in arms:
        checkpoint_path = checkpoint_dir / f"phase2_{arm}_best.pt"
        results[arm] = {}
        for split in splits:
            step += 1
            print(f"\n[{step}/{total}] {arm}/{split}: extracting trained features")
            t0 = time.time()
            bundle = extract_features_for_checkpoint(
                checkpoint_path, cfg, split, device, batch_size=batch_size
            )
            eye_blob = load_saved_features(bundle["eyepacs_path"])
            oli_blob = load_saved_features(bundle["olives_path"])

            print(f"[{step}/{total}] {arm}/{split}: running Phase 1 diagnostics")
            metrics = compute_arm_split_metrics(
                eye_blob, oli_blob, report_dir, arm, split
            )
            metrics["provenance"] = bundle["provenance"]
            results[arm][split] = metrics
            print(
                f"[{step}/{total}] {arm}/{split}: FID={metrics['fid']:.2f} "
                f"MMD_p={_p(metrics['mmd']['p_value'])} "
                f"cosine={metrics['cosine_mean']:.3f} "
                f"rho={metrics['black_border_rho']['eyepacs']['rho']:+.3f} "
                f"({time.time() - t0:.1f}s)"
            )

    report = build_report(results, cfg)
    report_path = report_dir / REPORT_FILENAME
    report_path.write_text(report, encoding="utf-8")
    print(f"\n[report] written to {report_path}")

    json_path = report_dir / RESULTS_JSON_FILENAME
    payload = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "phase1_baseline": OmegaConf.to_container(cfg.phase1_baseline, resolve=True),
        "targets": OmegaConf.to_container(cfg.targets, resolve=True),
        "results": _json_safe(results),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[report] raw numbers written to {json_path}")
    return results


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2d post-training distribution diagnostics."
    )
    parser.add_argument("--config", default="configs/diagnostics.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        cfg = load_config(args.config)
    except Exception as exc:
        print(f"[ERROR] failed to load config from {args.config}: {exc}")
        traceback.print_exc()
        return 2

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[phase2d] cuda requested but unavailable — falling back to cpu")
        device = "cpu"

    try:
        run_diagnostics(cfg, device=device, batch_size=args.batch_size)
    except Exception as exc:
        print(f"\n[ERROR] phase 2d diagnostics failed: {exc}")
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
