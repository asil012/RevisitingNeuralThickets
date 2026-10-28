"""First-seen plurality voting from anonymized answer groups."""

from __future__ import annotations

import numpy as np

from neural_thickets.data import Bundle


def plurality_vote_scores(
    groups: np.ndarray,
    correct: np.ndarray,
    orders: np.ndarray,
    ks: tuple[int, ...],
) -> dict[int, np.ndarray]:
    """Return per-question correctness averaged across saved voter orderings."""
    assert groups.shape == correct.shape
    assert orders.ndim == 2 and max(ks) <= orders.shape[1]
    scores = {k: np.zeros((len(orders), groups.shape[1]), dtype=bool) for k in ks}
    slots = set(ks)
    for repeat, order in enumerate(orders):
        for q in range(groups.shape[1]):
            counts: dict[int, int] = {}
            first_position: dict[int, int] = {}
            first_row: dict[int, int] = {}
            winner = -1
            for position, row in enumerate(order[: max(ks)], 1):
                group = int(groups[row, q])
                if group >= 0:
                    first_position.setdefault(group, position)
                    first_row.setdefault(group, int(row))
                    counts[group] = counts.get(group, 0) + 1
                    if (
                        winner < 0
                        or counts[group] > counts[winner]
                        or (
                            counts[group] == counts[winner]
                            and first_position[group] < first_position[winner]
                        )
                    ):
                        winner = group
                if position in slots and winner >= 0:
                    scores[position][repeat, q] = correct[first_row[winner], q]
    return {k: value.mean(axis=0) for k, value in scores.items()}


def olmo_countdown_curves(bundle: Bundle) -> dict:
    data = bundle.npz("score_data.npz")
    prefix = "cells.olmo_countdown"
    groups = data[f"{prefix}.top50_answer_group"]
    correct = data[f"{prefix}.top50_correct"]
    anchor_groups = data[f"{prefix}.anchor_answer_group"]
    anchor_correct = data[f"{prefix}.anchor_correct"]
    ks = (1, 2, 3, 5, 10, 20, 30, 40, 50)
    randopt = {1: float(100 * correct[0].mean())}
    anchor = {1: float(100 * anchor_correct.mean())}
    for k in ks[1:]:
        randopt[k] = float(
            100 * plurality_vote_scores(groups, correct, np.arange(50)[None, :], (k,))[k].mean()
        )
        orders = data[f"{prefix}.anchor_orders_k{k}"]
        anchor[k] = float(
            100 * plurality_vote_scores(anchor_groups, anchor_correct, orders, (k,))[k].mean()
        )
    return {"randopt": randopt, "anchor": anchor}
