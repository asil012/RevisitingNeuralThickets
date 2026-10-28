"""Recompute the paper's accuracy- and reward-density curves."""

from __future__ import annotations

import numpy as np

from neural_thickets.data import Bundle

MODELS = ("olmo", "q05", "q15", "q3", "q7", "q32")
TASKS = ("countdown", "gsm8k", "mbpp")
FAMILIES = ("weights", "sampling")
THRESHOLDS = np.asarray((0.9, 0.95, 1.0, 1.05, 1.1))
QWEN_MODELS = ("q05", "q15", "q3", "q7", "q32")
SIZE_VARIANTS = (
    ("accuracy", "population"),
    ("reward", "population"),
    ("accuracy", "greedy_anchor"),
    ("reward", "greedy_anchor"),
)


def _typed_rows(bundle: Bundle) -> list[dict]:
    rows = bundle.csv("density_curves.csv")
    for row in rows:
        for key in ("m", "n_runs", "zero_reference_subsets"):
            row[key] = int(row[key])
        for key in (
            "threshold",
            "mean_density",
            "mean_reference_score",
            "subset_sd",
            "subset_p025",
            "subset_p975",
        ):
            row[key] = float(row[key])
    return rows


def _cell_rows(
    model: str,
    task: str,
    family: str,
    scores: np.ndarray,
    sizes: np.ndarray,
    orders: np.ndarray,
) -> list[dict]:
    assert scores.ndim == 2 and scores.dtype == bool
    assert np.array_equal(np.sort(orders, axis=1), np.tile(np.arange(scores.shape[1]), (200, 1)))
    densities = np.empty((len(THRESHOLDS), len(orders), len(sizes)))
    references = np.empty((len(orders), len(sizes)))
    for repeat, order in enumerate(orders):
        cumulative = np.cumsum(scores[:, order], axis=1, dtype=np.uint16)
        means = cumulative[:, sizes - 1] / sizes
        reference = means.mean(axis=0)
        delta = means[None, :, :] - THRESHOLDS[:, None, None] * reference
        densities[:, repeat] = (delta > 1e-12).mean(axis=1)
        densities[:, repeat] += 0.5 * (np.abs(delta) <= 1e-12).mean(axis=1)
        references[repeat] = reference
    rows = []
    for threshold_index, threshold in enumerate(THRESHOLDS):
        for size_index, size in enumerate(sizes):
            values = densities[threshold_index, :, size_index]
            low, high = np.quantile(values, (0.025, 0.975))
            rows.append(
                {
                    "model": model,
                    "task": task,
                    "family": family,
                    "metric": "accuracy",
                    "threshold": float(threshold),
                    "m": int(size),
                    "n_runs": len(scores),
                    "mean_density": float(values.mean()),
                    "mean_reference_score": float(references[:, size_index].mean()),
                    "subset_sd": float(values.std(ddof=0)),
                    "subset_p025": float(low),
                    "subset_p975": float(high),
                    "zero_reference_subsets": int((references[:, size_index] == 0).sum()),
                }
            )
    return rows


def recompute_density_rows(bundle: Bundle, *, verify: bool = True) -> list[dict]:
    data = bundle.npz("density_data.npz")
    computed = {}
    for task in TASKS:
        sizes = data[f"density_plan.{task}.sizes"]
        orders = data[f"density_plan.{task}.orders"]
        qids = data[f"density_plan.{task}.qids"]
        for model in MODELS:
            for family in FAMILIES:
                prefix = f"density.{model}_{task}.{family}"
                np.testing.assert_array_equal(data[f"{prefix}.qids"], qids)
                rows = _cell_rows(model, task, family, data[f"{prefix}.correct"], sizes, orders)
                for row in rows:
                    key = tuple(row[name] for name in ("model", "task", "family", "threshold", "m"))
                    assert key not in computed
                    computed[key] = row
    expected = _typed_rows(bundle)
    assert len(computed) == len(expected) == 1560
    output = []
    for reference in expected:
        key = tuple(reference[name] for name in ("model", "task", "family", "threshold", "m"))
        row = computed[key]
        if verify:
            assert row["n_runs"] == reference["n_runs"]
            for name in (
                "mean_density",
                "mean_reference_score",
                "subset_sd",
                "subset_p025",
                "subset_p975",
            ):
                np.testing.assert_allclose(row[name], reference[name], rtol=0, atol=1e-14)
            assert row["zero_reference_subsets"] == reference["zero_reference_subsets"]
        output.append(row)
    return output


def _density_size_draws(
    scores: np.ndarray,
    reference_scores: np.ndarray,
    orders: np.ndarray,
    *,
    size: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64)
    reference_scores = np.asarray(reference_scores, dtype=np.float64)
    assert scores.ndim == reference_scores.ndim == 2
    assert scores.shape[1] == reference_scores.shape[1] == orders.shape[1]
    assert len(orders) == 200 and 0 < size <= scores.shape[1]
    assert np.isfinite(scores).all() and np.isfinite(reference_scores).all()
    draws = np.empty((len(THRESHOLDS), len(orders)), dtype=np.float64)
    references = np.empty(len(orders), dtype=np.float64)
    for repeat, order in enumerate(orders):
        columns = order[:size]
        means = scores[:, columns].mean(axis=1)
        reference = float(reference_scores[:, columns].mean())
        delta = means[None, :] - THRESHOLDS[:, None] * reference
        draws[:, repeat] = (delta > 1e-12).mean(axis=1)
        draws[:, repeat] += 0.5 * (np.abs(delta) <= 1e-12).mean(axis=1)
        references[repeat] = reference
    assert np.all(np.diff(draws, axis=0) <= 0)
    return draws, references


def recompute_density_size_variants(bundle: Bundle, *, verify: bool = True) -> list[dict]:
    """Recompute Qwen size curves under population and greedy-anchor references."""
    data = bundle.npz("density_data.npz")
    expected = {
        tuple(row[name] for name in ("model", "task", "family", "threshold", "m")): row
        for row in _typed_rows(bundle)
        if row["model"] in QWEN_MODELS and row["metric"] == "accuracy" and row["m"] == 200
    }
    assert len(expected) == len(QWEN_MODELS) * len(TASKS) * len(FAMILIES) * len(THRESHOLDS)
    rows = []
    for task in TASKS:
        orders = data[f"density_plan.{task}.orders"]
        qids = data[f"density_plan.{task}.qids"]
        for model in QWEN_MODELS:
            for metric, reference_name in SIZE_VARIANTS:
                field = "correct" if metric == "accuracy" else "reward"
                greedy_prefix = f"density.{model}_{task}.greedy"
                greedy_scores = data[f"{greedy_prefix}.{field}"]
                np.testing.assert_array_equal(data[f"{greedy_prefix}.qids"], qids)
                np.testing.assert_array_equal(data[f"{greedy_prefix}.seeds"], [42])
                for family in FAMILIES:
                    prefix = f"density.{model}_{task}.{family}"
                    np.testing.assert_array_equal(data[f"{prefix}.qids"], qids)
                    scores = data[f"{prefix}.{field}"]
                    reference_scores = (
                        scores if reference_name == "population" else greedy_scores
                    )
                    draws, references = _density_size_draws(scores, reference_scores, orders)
                    for threshold_index, threshold in enumerate(THRESHOLDS):
                        values = draws[threshold_index]
                        low, high = np.quantile(values, (0.025, 0.975))
                        row = {
                            "model": model,
                            "task": task,
                            "family": family,
                            "metric": metric,
                            "reference": reference_name,
                            "threshold": float(threshold),
                            "m": 200,
                            "n_runs": len(scores),
                            "mean_density": float(values.mean()),
                            "mean_reference_score": float(references.mean()),
                            "subset_sd": float(values.std(ddof=0)),
                            "subset_p025": float(low),
                            "subset_p975": float(high),
                            "zero_reference_subsets": int((references == 0).sum()),
                        }
                        if verify and (metric, reference_name) == ("accuracy", "population"):
                            key = (model, task, family, float(threshold), 200)
                            reference = expected[key]
                            assert row["n_runs"] == reference["n_runs"]
                            for name in (
                                "mean_density",
                                "mean_reference_score",
                                "subset_sd",
                                "subset_p025",
                                "subset_p975",
                            ):
                                np.testing.assert_allclose(
                                    row[name], reference[name], rtol=0, atol=1e-14
                                )
                            assert (
                                row["zero_reference_subsets"]
                                == reference["zero_reference_subsets"]
                            )
                        rows.append(row)
    assert len(rows) == len(SIZE_VARIANTS) * len(expected)
    return rows
