"""Reproducible splits and the joint EyePACS+OLIVES batch sampler.

The OLIVES split is patient-aware (no patient crosses train/val/test); EyePACS
is a plain per-sample split. The ``JointBatchSampler`` guarantees every train
batch carries both datasets so the CORAL alignment term always has a signal.
"""

import math
import random
from collections import defaultdict
from typing import Iterator, Sequence

import torch


def _sample_tier(has_clinical: bool, has_biomarkers: bool) -> int:
    """0 = fundus only, 1 = + clinical, 2 = + biomarkers (highest available)."""
    if has_biomarkers:
        return 2
    if has_clinical:
        return 1
    return 0


def _print_tier_counts(
    name: str,
    indices: Sequence[int],
    has_clinical: torch.Tensor,
    has_biomarkers: torch.Tensor,
) -> None:
    counts = [0, 0, 0]
    for idx in indices:
        counts[_sample_tier(bool(has_clinical[idx]), bool(has_biomarkers[idx]))] += 1
    print(
        f"  {name:<5} n={len(indices):>5}  tier0={counts[0]:>5} "
        f"tier1={counts[1]:>5} tier2={counts[2]:>5}"
    )


def patient_aware_split(
    patient_ids: torch.Tensor,
    has_clinical: torch.Tensor,
    has_biomarkers: torch.Tensor,
    ratios: Sequence[float],
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    """Split OLIVES at the patient level (all of a patient's samples together).

    Samples with ``patient_id == -1`` go to train. Patients are bucketed by
    their highest available tier and each bucket is split by ``ratios`` so each
    split receives some tier-1/tier-2 samples when patient counts allow.
    """
    train_ratio, val_ratio, _ = ratios
    patient_ids = patient_ids.long()

    # Group sample indices by patient; route missing-id samples straight to train.
    by_patient: dict[int, list[int]] = defaultdict(list)
    train: list[int] = []
    for idx in range(patient_ids.shape[0]):
        pid = int(patient_ids[idx])
        if pid == -1:
            train.append(idx)
        else:
            by_patient[pid].append(idx)

    # Each patient's highest tier, for stratified bucketing.
    patient_tier: dict[int, int] = {}
    for pid, idxs in by_patient.items():
        patient_tier[pid] = max(
            _sample_tier(bool(has_clinical[i]), bool(has_biomarkers[i])) for i in idxs
        )

    rng = random.Random(seed)
    val: list[int] = []
    test: list[int] = []
    for tier in (2, 1, 0):  # spread the scarcest tier first
        bucket = sorted(pid for pid, t in patient_tier.items() if t == tier)
        rng.shuffle(bucket)
        n = len(bucket)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        for pid in bucket[:n_train]:
            train.extend(by_patient[pid])
        for pid in bucket[n_train : n_train + n_val]:
            val.extend(by_patient[pid])
        for pid in bucket[n_train + n_val :]:
            test.extend(by_patient[pid])

    print("[split] OLIVES patient-aware split (tier coverage):")
    _print_tier_counts("train", train, has_clinical, has_biomarkers)
    _print_tier_counts("val", val, has_clinical, has_biomarkers)
    _print_tier_counts("test", test, has_clinical, has_biomarkers)
    return sorted(train), sorted(val), sorted(test)


def random_split(
    n: int, ratios: Sequence[float], seed: int
) -> tuple[list[int], list[int], list[int]]:
    """Per-sample train/val/test split of ``range(n)``, seeded."""
    train_ratio, val_ratio, _ = ratios
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    train = indices[:n_train]
    val = indices[n_train : n_train + n_val]
    test = indices[n_train + n_val :]
    print(
        f"[split] EyePACS random split: train={len(train)} "
        f"val={len(val)} test={len(test)}"
    )
    return train, val, test


class JointBatchSampler:
    """Yields batches mixing a fixed EyePACS and OLIVES quota.

    Over a ConcatDataset([eyepacs_subset, olives_subset]) where OLIVES indices
    are offset by ``n_eyepacs``. EyePACS is drawn shuffled without replacement
    within an epoch; OLIVES is reshuffled and wraps (with replacement across the
    epoch) so its quota is always full. Epoch length = ceil(n_eyepacs / quota).
    """

    def __init__(
        self,
        n_eyepacs: int,
        n_olives: int,
        eyepacs_per_batch: int,
        olives_per_batch: int,
        seed: int,
    ) -> None:
        self.n_eyepacs = n_eyepacs
        self.n_olives = n_olives
        self.eyepacs_per_batch = eyepacs_per_batch
        self.olives_per_batch = olives_per_batch
        self.seed = seed
        self.num_batches = math.ceil(n_eyepacs / eyepacs_per_batch)
        self._epoch = 0

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1

        eyepacs = list(range(self.n_eyepacs))
        rng.shuffle(eyepacs)
        olives_pool: list[int] = []

        def draw_olives(k: int) -> list[int]:
            drawn: list[int] = []
            while len(drawn) < k:
                if not olives_pool:
                    olives_pool.extend(range(self.n_olives))
                    rng.shuffle(olives_pool)
                drawn.append(self.n_eyepacs + olives_pool.pop())
            return drawn

        for start in range(0, self.n_eyepacs, self.eyepacs_per_batch):
            ep_chunk = eyepacs[start : start + self.eyepacs_per_batch]
            yield ep_chunk + draw_olives(self.olives_per_batch)
