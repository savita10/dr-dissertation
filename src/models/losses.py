"""Loss terms for the multi-term, cross-dataset DR objective.

Each term is a function taking the model output dict plus the relevant batch
fields and returning a scalar tensor. Every masked term returns a graph-safe
``0.0`` scalar (connected to the graph, never NaN, never an error) when its
relevant subset is empty in the batch, so ``total.backward()`` always works.

Batch field contract (see PHASE2_KICKOFF.md §3, §5):
    dataset:        LongTensor  [B]      0 = EyePACS, 1 = OLIVES
    dr_labels:      LongTensor  [B]      valid where dataset == 0
    biomarkers:     FloatTensor [B, 9]   valid where has_biomarkers
    has_biomarkers: BoolTensor  [B]
    bcva, cst:      FloatTensor [B]      STANDARDISED; NaN where invalid
    has_clinical:   BoolTensor  [B]

Derived masks:
    has_dr     = dataset == 0
    is_olives  = dataset == 1
    bcva_valid = has_clinical & ~isnan(bcva) & ~isnan(cst)
"""

import torch
import torch.nn.functional as F
from omegaconf import DictConfig


def _zero_like(anchor: torch.Tensor) -> torch.Tensor:
    """A scalar 0.0 still connected to the graph through ``anchor``.

    Multiplying the sum by 0 keeps the autograd edge alive (the term
    contributes a real, zero-valued gradient) without ever producing NaN.
    """
    return anchor.sum() * 0.0


def dr_loss(
    out: dict[str, torch.Tensor],
    dr_labels: torch.Tensor,
    dataset: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Class-weighted cross-entropy over the EyePACS (has_dr) subset only."""
    has_dr = dataset == 0
    if not bool(has_dr.any()):
        return _zero_like(out["dr_logits"])
    logits = out["dr_logits"][has_dr]
    targets = dr_labels[has_dr]
    return F.cross_entropy(logits, targets, weight=class_weights)


def biomarker_loss(
    out: dict[str, torch.Tensor],
    biomarkers: torch.Tensor,
    has_biomarkers: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Masked multi-label BCE over the tier-2 (has_biomarkers) subset.

    ``pos_weight`` (length = num_biomarkers) re-weights positives per column.
    """
    if not bool(has_biomarkers.any()):
        return _zero_like(out["biomarker_logits"])
    logits = out["biomarker_logits"][has_biomarkers]
    targets = biomarkers[has_biomarkers]
    return F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight
    )


def regression_loss(
    out: dict[str, torch.Tensor],
    bcva: torch.Tensor,
    cst: torch.Tensor,
    has_clinical: torch.Tensor,
) -> torch.Tensor:
    """MSE on standardised (bcva, cst) over the valid tier-1 subset.

    Valid requires ``has_clinical`` AND both targets non-NaN (one sample has
    ``has_clinical=True`` but ``BCVA=NaN``).
    """
    valid = has_clinical & ~torch.isnan(bcva) & ~torch.isnan(cst)
    if not bool(valid.any()):
        return _zero_like(out["regression"])
    targets = torch.stack([bcva, cst], dim=1)[valid]
    preds = out["regression"][valid]
    return F.mse_loss(preds, targets)


def supcon_loss(
    out: dict[str, torch.Tensor],
    dr_labels: torch.Tensor,
    dataset: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Supervised contrastive loss (Khosla et al. 2020) on the EyePACS subset.

    Positives are same-DR-class samples within the has_dr subset. Requires
    >=2 samples and >=1 positive pair, else returns a graph-safe 0.
    """
    has_dr = dataset == 0
    if int(has_dr.sum()) < 2:
        return _zero_like(out["projection"])

    z = F.normalize(out["projection"][has_dr], dim=1)
    labels = dr_labels[has_dr].view(-1, 1)
    n = z.shape[0]

    # Cosine-similarity logits, scaled by temperature.
    logits = (z @ z.T) / temperature
    # Numerical stability: subtract the per-row max (detached -> no grad).
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    # Mask out self-comparisons.
    self_mask = ~torch.eye(n, dtype=torch.bool, device=z.device)
    pos_mask = (labels == labels.T) & self_mask
    if not bool(pos_mask.any()):
        return _zero_like(out["projection"])

    # log-prob via stable log-sum-exp over non-self entries.
    exp_logits = torch.exp(logits) * self_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    pos_count = pos_mask.sum(dim=1)
    # Mean log-likelihood over each anchor's positives (anchors with >=1).
    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / pos_count.clamp(min=1)
    has_pos = pos_count > 0
    return -mean_log_prob_pos[has_pos].mean()


def coral_loss(
    out: dict[str, torch.Tensor],
    dataset: torch.Tensor,
    min_samples: int = 2,
) -> torch.Tensor:
    """CORAL (Sun & Saenko 2016) on the 2048-d features.

    Frobenius distance between the per-domain feature covariances, normalised
    by ``4 * d**2``. Returns a graph-safe 0 if either domain subset has fewer
    than ``min_samples`` samples (single-dataset-batch guard).
    """
    features = out["features"]
    eyepacs = features[dataset == 0]
    olives = features[dataset == 1]
    if eyepacs.shape[0] < min_samples or olives.shape[0] < min_samples:
        return _zero_like(features)

    cov_e = _covariance(eyepacs)
    cov_o = _covariance(olives)
    d = features.shape[1]
    diff = cov_e - cov_o
    return (diff * diff).sum() / (4.0 * d * d)


def _covariance(x: torch.Tensor) -> torch.Tensor:
    """Differentiable sample covariance: centre, then CᵀC / (n-1)."""
    n = x.shape[0]
    centred = x - x.mean(dim=0, keepdim=True)
    return (centred.T @ centred) / (n - 1)


def ntxent_loss(
    projection_v1: torch.Tensor,
    projection_v2: torch.Tensor | None,
    temperature: float = 0.07,
) -> torch.Tensor:
    """SimCLR NT-Xent over two augmented views.

    Returns a graph-safe 0 when ``projection_v2`` is None (no second view).
    """
    if projection_v2 is None:
        return _zero_like(projection_v1)

    z1 = F.normalize(projection_v1, dim=1)
    z2 = F.normalize(projection_v2, dim=1)
    n = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # [2n, d]

    logits = (z @ z.T) / temperature
    # Exclude self-similarity from the softmax denominator.
    self_mask = torch.eye(2 * n, dtype=torch.bool, device=z.device)
    logits = logits.masked_fill(self_mask, float("-inf"))

    # Positive of row i<n is i+n, and of row i>=n is i-n.
    targets = torch.cat(
        [
            torch.arange(n, device=z.device) + n,
            torch.arange(n, device=z.device),
        ]
    )
    return F.cross_entropy(logits, targets)


def combined_loss(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    cfg: DictConfig,
    class_weights: torch.Tensor | None = None,
    pos_weight: torch.Tensor | None = None,
    projection_v2: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Weighted sum of all loss terms.

    Returns ``(total, components)`` where ``components`` maps each active term
    name to its detached scalar value for logging. NT-Xent is included only
    when ``w_ntxent > 0`` and a second view is supplied.
    """
    lw = cfg.loss_weights

    dataset = batch["dataset"]
    dr = dr_loss(out, batch["dr_labels"], dataset, class_weights=class_weights)
    bio = biomarker_loss(
        out, batch["biomarkers"], batch["has_biomarkers"], pos_weight=pos_weight
    )
    reg = regression_loss(
        out, batch["bcva"], batch["cst"], batch["has_clinical"]
    )
    supcon = supcon_loss(
        out, batch["dr_labels"], dataset, temperature=float(cfg.supcon_temperature)
    )
    coral = coral_loss(out, dataset, min_samples=int(cfg.coral_min_samples))

    total = (
        float(lw.w_dr) * dr
        + float(lw.w_biomarker) * bio
        + float(lw.w_regression) * reg
        + float(lw.w_supcon) * supcon
        + float(lw.w_coral) * coral
    )

    components: dict[str, float] = {
        "dr": dr.detach().item(),
        "biomarker": bio.detach().item(),
        "regression": reg.detach().item(),
        "supcon": supcon.detach().item(),
        "coral": coral.detach().item(),
    }

    if float(lw.w_ntxent) > 0.0 and projection_v2 is not None:
        ntxent = ntxent_loss(
            out["projection"],
            projection_v2,
            temperature=float(cfg.ntxent_temperature),
        )
        total = total + float(lw.w_ntxent) * ntxent
        components["ntxent"] = ntxent.detach().item()

    components["total"] = total.detach().item()
    return total, components
