"""Recompute OLMo selection, noise-null, and transfer curves from score matrices."""

from __future__ import annotations

import numpy as np

from neural_thickets.data import Bundle

KS = (1, 5, 10, 20, 50)


def _score_subset(reward: np.ndarray, columns: np.ndarray) -> np.ndarray:
    return reward[:, np.sort(columns)].mean(axis=1)


def _null_lifts(
    reward: np.ndarray,
    subsets: dict[int, np.ndarray],
    seeds: np.ndarray,
) -> dict[int, np.ndarray]:
    n_candidates, n_questions = reward.shape
    columns = np.arange(n_questions)[None, :]
    lifts = {m: np.empty(len(seeds)) for m in subsets}
    for repeat, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, n_candidates, size=(n_candidates, n_questions), dtype=np.int32)
        simulated = reward[indices, columns]
        for m, selections in subsets.items():
            scores = _score_subset(simulated, selections[repeat])
            lifts[m][repeat] = 100 * (scores.max() - scores.mean())
    return lifts


def recompute_selection_transfer(bundle: Bundle) -> dict:
    data = bundle.npz("selection_transfer_data.npz")
    expected = bundle.json("selection.json")["cells"]
    cells = {}
    for task in ("gsm8k", "mbpp"):
        prefix = f"selection_transfer.{task}"
        reward = data[f"{prefix}.reward"]
        holdout = data[f"{prefix}.holdout_accuracy"]
        seeds = data[f"{prefix}.pooled_mc_seeds"]
        transfer_ks = tuple(data[f"{prefix}.transfer_ks"].tolist())
        assert transfer_ks == KS
        subsets_by_m = {
            reference["m"]: data[f"{prefix}.null_subsets_m{reference['m']}"]
            for reference in expected[f"olmo_{task}"]["curve"]
        }
        null_by_m = _null_lifts(reward, subsets_by_m, seeds)
        curve = []
        for reference in expected[f"olmo_{task}"]["curve"]:
            m = reference["m"]
            subsets = data[f"{prefix}.observed_subsets_m{m}"]
            observed = np.empty(len(subsets))
            transfer = np.empty((len(subsets), len(KS)))
            top50 = np.empty((len(subsets), 50), dtype=np.int32)
            for repeat, columns in enumerate(subsets):
                scores = _score_subset(reward, columns)
                observed[repeat] = 100 * (scores.max() - scores.mean())
                order = np.argsort(-scores, kind="stable")[:50]
                top50[repeat] = order
                transfer[repeat] = [100 * (holdout[order[:k]].mean() - holdout.mean()) for k in KS]
            np.testing.assert_allclose(
                observed, data[f"{prefix}.observed_lift_m{m}"], rtol=0, atol=1e-14
            )
            np.testing.assert_array_equal(top50, data[f"{prefix}.observed_top50_row_indices_m{m}"])
            np.testing.assert_allclose(
                transfer, data[f"{prefix}.observed_transfer_m{m}"], rtol=0, atol=1e-14
            )
            null = null_by_m[m]
            np.testing.assert_allclose(
                null, data[f"{prefix}.pooled_null_lift_m{m}"], rtol=0, atol=1e-13
            )
            row = {
                "m": m,
                "observed_mean_points": float(observed.mean()),
                "observed_sd_points": float(observed.std(ddof=1)) if len(observed) > 1 else 0.0,
                "pooled_null_mean_points": float(null.mean()),
                "pooled_null_sd_points": float(null.std(ddof=1)),
                "transfer": {},
            }
            for index, k in enumerate(KS):
                values = transfer[:, index]
                interval = np.percentile(values, (2.5, 97.5)) if len(values) > 1 else (None, None)
                row["transfer"][str(k)] = {
                    "mean_points": float(values.mean()),
                    "p2_5_points": None if interval[0] is None else float(interval[0]),
                    "p97_5_points": None if interval[1] is None else float(interval[1]),
                }
            for name in (
                "observed_mean_points",
                "observed_sd_points",
                "pooled_null_mean_points",
                "pooled_null_sd_points",
            ):
                np.testing.assert_allclose(row[name], reference[name], rtol=0, atol=1e-13)
            for k in KS:
                for name in ("mean_points", "p2_5_points", "p97_5_points"):
                    actual = row["transfer"][str(k)][name]
                    target = reference["transfer"][str(k)][name]
                    if target is None:
                        assert actual is None
                        continue
                    np.testing.assert_allclose(
                        actual,
                        target,
                        rtol=0,
                        atol=1e-13,
                    )
            curve.append(row)
        cells[f"olmo_{task}"] = {"curve": curve}
    return {"cells": cells}


def recompute_headline_decay(bundle: Bundle) -> dict:
    data = bundle.npz("headline_data.npz")
    reference = bundle.json("headline.json")["decay"]["curve"]
    reward = data["decay_reward"]
    subsets = {row["m"]: data[f"decay.null_subsets_m{row['m']}"] for row in reference}
    nulls = _null_lifts(reward, subsets, data["decay.pooled_mc_seeds"])
    curve = []
    for expected in reference:
        m = expected["m"]
        observed = np.asarray(
            [
                100 * ((scores := _score_subset(reward, columns)).max() - scores.mean())
                for columns in data[f"decay.observed_subsets_m{m}"]
            ]
        )
        np.testing.assert_allclose(observed, data[f"decay.observed_lift_m{m}"], rtol=0, atol=1e-13)
        null = nulls[m]
        np.testing.assert_allclose(null, data[f"decay.pooled_null_lift_m{m}"], rtol=0, atol=5e-13)
        row = {
            "m": m,
            "observed_mean_points": float(observed.mean()),
            "observed_sd_points": float(observed.std(ddof=1)) if len(observed) > 1 else 0.0,
            "pooled_null_mean_points": float(null.mean()),
            "pooled_null_sd_points": float(null.std(ddof=1)),
        }
        for name in row.keys() - {"m"}:
            np.testing.assert_allclose(row[name], expected[name], rtol=0, atol=5e-13)
        curve.append(row)
    return {"curve": curve}
