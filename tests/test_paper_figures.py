import matplotlib.pyplot as plt
import numpy as np

from neural_thickets.analysis.density import recompute_density_size_variants
from neural_thickets.data import Bundle
from neural_thickets.figures import write_figures
from neural_thickets.figures.paper import _density_style, _single_style


def test_headline_selection_and_holdout_contract():
    scatter = Bundle.default().json("headline.json")["scatter"]
    selection = np.asarray(scatter["selection_accuracy"])
    holdout = np.asarray(scatter["holdout_accuracy"])
    reward = np.asarray(scatter["selection_reward"])
    assert selection.shape == holdout.shape == reward.shape == (1000,)
    winner = int(np.argmax(reward))
    assert scatter["candidate_ids"][winner] == 2141
    assert np.isclose(selection[winner], 69.85714285714286)
    assert np.isclose(holdout[winner], 69.0)


def test_all_figures_render_on_cpu(tmp_path):
    outputs = write_figures(Bundle.default(), tmp_path)
    assert {path.name for path in outputs} == {
        "headline_random_split_seed42.pdf",
        "selection_transfer.pdf",
        "single_experts_by_model.pdf",
        "accuracy_density_by_m.pdf",
        "accuracy_density_by_size.pdf",
        "density_by_size_variants.pdf",
    }
    assert all(path.stat().st_size > 10_000 for path in outputs)
    plt.close("all")


def test_density_size_variants_use_matched_metric_and_reference_contracts():
    rows = recompute_density_size_variants(Bundle.default())
    assert len(rows) == 600
    lookup = {
        tuple(
            row[name]
            for name in ("model", "task", "family", "metric", "reference", "threshold")
        ): row
        for row in rows
    }
    assert len(lookup) == len(rows)
    changed_sampling_curve = False
    for key, row in lookup.items():
        model, task, family, metric, reference, threshold = key
        if reference == "greedy_anchor":
            other_family = "sampling" if family == "weights" else "weights"
            other = lookup[(model, task, other_family, metric, reference, threshold)]
            assert row["mean_reference_score"] == other["mean_reference_score"]
        if family == "sampling" and reference == "greedy_anchor":
            population = lookup[(model, task, family, metric, "population", threshold)]
            changed_sampling_curve |= row["mean_density"] != population["mean_density"]
        if task == "mbpp" and reference == "greedy_anchor" and metric == "reward":
            accuracy = lookup[(model, task, family, "accuracy", reference, threshold)]
            assert row["mean_density"] == accuracy["mean_density"]
    assert changed_sampling_curve


def test_specialized_figure_styles_preserve_paper_axes():
    fig, (single, density) = plt.subplots(1, 2)
    single.set_xticks([0, 1])
    single.set_xticks([0.5], minor=True)
    _single_style(single)
    _density_style(density)

    assert {label.get_fontsize() for label in single.get_xticklabels()} == {9.5}
    assert single.get_xaxis().get_major_ticks()[0].tick1line.get_markersize() == 3
    assert single.get_xaxis().get_minor_ticks()[0].tick1line.get_markersize() == 1.5
    assert density.get_ylim() == (-2, 102)
    np.testing.assert_array_equal(density.get_yticks(), [0, 50, 100])
    assert {label.get_fontsize() for label in density.get_yticklabels()} == {8}
    plt.close(fig)
