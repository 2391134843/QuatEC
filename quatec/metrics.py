"""Hierarchical EC accuracy."""

from __future__ import annotations

import numpy as np
import torch


def _matching_depth(predicted: str, target: str) -> int:
    depth = 0
    for predicted_part, target_part in zip(predicted.split("."), target.split(".")):
        if predicted_part != target_part:
            break
        depth += 1
    return depth


def hierarchical_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    index_to_ec: dict[int, str],
) -> dict[str, float]:
    scores = logits.detach().cpu().numpy()
    targets = labels.detach().cpu().numpy()
    totals = np.zeros(4, dtype=np.float64)

    for score, target in zip(scores, targets):
        true_indices = np.flatnonzero(target > 0)
        if not len(true_indices):
            continue
        predicted_indices = np.argpartition(-score, len(true_indices) - 1)[
            : len(true_indices)
        ]
        predicted_ecs = [index_to_ec[int(index)] for index in predicted_indices]
        true_ecs = [index_to_ec[int(index)] for index in true_indices]
        depths = [
            max(_matching_depth(predicted, true) for predicted in predicted_ecs)
            for true in true_ecs
        ]
        for level in range(1, 5):
            totals[level - 1] += np.mean(np.asarray(depths) >= level)

    denominator = max(1, len(targets))
    return {
        f"level_{level}": float(totals[level - 1] / denominator)
        for level in range(1, 5)
    }

