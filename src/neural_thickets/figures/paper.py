"""Regenerate every active paper figure from the compact bundle."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, NullLocator

from neural_thickets.analysis.density import (
    recompute_density_rows,
    recompute_density_size_variants,
)
from neural_thickets.analysis.selection import (
    recompute_headline_decay,
    recompute_selection_transfer,
)
from neural_thickets.analysis.voting import olmo_countdown_curves
from neural_thickets.data import Bundle

BLUE = "#356FA8"
ORANGE = "#D97932"
GREEN = "#3E8A63"
GRAY = "#68737D"
GRID = "#D9E2E8"
MODEL_COLORS = {"olmo": BLUE, "q3": "#8CBD98", "q7": "#2F7D4E", "q32": "#173F2B"}


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor("#FBFCFE")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.55, alpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color("#9BAAB6")
    ax.spines["bottom"].set_color("#9BAAB6")
    ax.tick_params(labelsize=8.2, length=2.5, color="#7C8993")


def _single_style(ax: plt.Axes) -> None:
    ax.set_facecolor("#FBFCFE")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color("#9BAAB6")
    ax.spines["bottom"].set_color("#9BAAB6")
    ax.tick_params(labelsize=9.5, length=3, color="#7C8993")
    ax.tick_params(axis="x", which="minor", length=1.5, color="#A3ABB2")


def _density_style(ax: plt.Axes) -> None:
    ax.set_facecolor("#FBFCFE")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color("#A3AFB9")
    ax.spines["bottom"].set_color("#A3AFB9")
    ax.set_ylim(-2, 102)
    ax.set_yticks([0, 50, 100])
    ax.tick_params(labelsize=8, length=2, width=0.6, color="#7C8993")


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _headline(bundle: Bundle, output_dir: Path) -> None:
    data = bundle.json("headline.json")
    scatter, voting = data["scatter"], data["voting"]
    decay = recompute_headline_decay(bundle)
    raw = bundle.npz("headline_data.npz")
    fig = plt.figure(figsize=(7.25, 3.25))
    fig.patch.set_facecolor("white")
    grid = fig.add_gridspec(2, 2, width_ratios=[0.95, 1.05])
    left = fig.add_subplot(grid[:, 0])
    top = fig.add_subplot(grid[0, 1])
    bottom = fig.add_subplot(grid[1, 1])

    selection_columns = np.searchsorted(raw["source_qids"], raw["selection_qids"])
    holdout_columns = np.searchsorted(raw["source_qids"], raw["holdout_qids"])
    assert np.array_equal(raw["source_qids"][selection_columns], raw["selection_qids"])
    assert np.array_equal(raw["source_qids"][holdout_columns], raw["holdout_qids"])
    x = 100 * raw["correct"][:, selection_columns].mean(axis=1)
    y = 100 * raw["correct"][:, holdout_columns].mean(axis=1)
    reward = 100 * raw["reward"][:, selection_columns].mean(axis=1)
    np.testing.assert_allclose(x, scatter["selection_accuracy"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(y, scatter["holdout_accuracy"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(reward, scatter["selection_reward"], rtol=0, atol=2e-6)
    order = np.argsort(-reward, kind="stable")
    left.scatter(
        x, y, s=12, alpha=0.35, color=BLUE, linewidths=0, label="Perturbations", rasterized=True
    )
    left.scatter(
        x[order[0]],
        y[order[0]],
        marker="*",
        s=130,
        color=ORANGE,
        edgecolor="black",
        linewidth=0.55,
        zorder=5,
        label="Rank 1",
    )
    left.scatter(
        x[order[:20]].mean(),
        y[order[:20]].mean(),
        marker="D",
        s=48,
        color="#F3BE87",
        edgecolor="black",
        linewidth=0.55,
        zorder=5,
        label="Top-20 mean",
    )
    anchors = 100 * raw["anchor_holdout_correct"].mean(axis=1)
    np.testing.assert_allclose(anchors, scatter["anchor_holdout_accuracy"], rtol=0, atol=1e-12)
    left.axhspan(anchors.min(), anchors.max(), color=ORANGE, alpha=0.10, linewidth=0)
    left.axhline(anchors.mean(), color=ORANGE, linestyle=(0, (3, 2)), linewidth=1.0)
    left.text(63.25, anchors.mean() + 0.22, "anchor", color=ORANGE, fontsize=8.8)
    left.set(xlim=(63, 72), ylim=(63, 72), xticks=[63, 66, 69, 72], yticks=[63, 66, 69, 72])
    left.set_xlabel("Selection accuracy (%)", fontsize=10)
    left.set_ylabel("Holdout accuracy (%)", fontsize=10)
    left.set_title("Selection and held-out performance", fontsize=10, pad=8)
    left.set_aspect("equal", adjustable="box")
    left.legend(
        loc="lower right",
        fontsize=8.4,
        frameon=True,
        framealpha=0.92,
        borderpad=0.35,
        handletextpad=0.35,
        columnspacing=0.65,
    )
    _style(left)
    left.tick_params(labelsize=9)

    rows = decay["curve"]
    m = np.asarray([row["m"] for row in rows])
    observed = np.asarray([row["observed_mean_points"] for row in rows])
    null = np.asarray([row["pooled_null_mean_points"] for row in rows])
    null_sd = np.asarray([row["pooled_null_sd_points"] for row in rows])
    top.plot(
        m, observed, marker="o", color=BLUE, linewidth=1.4, markersize=3.5, label="Observed lift"
    )
    top.plot(m, null, color=GRAY, linestyle="--", linewidth=1.25, label="Noise null")
    top.fill_between(m, null - 2 * null_sd, null + 2 * null_sd, color=GRAY, alpha=0.15, linewidth=0)
    top.set_xscale("log", base=2)
    top.set(ylim=(0, 25), yticks=[0, 10, 20])
    top.set_xticks(m, [str(value) for value in m])
    top.set_xlabel("Selection questions $m$", fontsize=9.5)
    top.set_ylabel("Top-1 reward lift", fontsize=9.5)
    top.set_title("Selection lift and noise null", fontsize=10, pad=8)
    top.legend(loc="upper right", fontsize=7.5, frameon=False, handlelength=1.7, labelspacing=0.25)
    _style(top)

    curves = olmo_countdown_curves(bundle)
    ks = np.asarray([int(key) for key in voting["ttmv"]])
    anchor = [curves["anchor"][int(k)] for k in ks]
    randopt = [curves["randopt"][int(k)] for k in ks]
    np.testing.assert_allclose(
        anchor, [voting["ttmv"][str(k)]["mean"] for k in ks], rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(randopt, [voting["randopt"][str(k)] for k in ks], rtol=0, atol=1e-12)
    bottom.plot(
        ks,
        randopt,
        color=BLUE,
        marker="s",
        linewidth=1.3,
        markersize=3,
        label=r"RandOpt(5000, $K$)",
    )
    bottom.plot(
        ks,
        anchor,
        color=ORANGE,
        marker="o",
        linewidth=1.6,
        markersize=3.4,
        label=r"Anchor Majority-of-$K$",
    )
    bottom.set_xscale("log")
    bottom.set(xlim=(0.9, 58), ylim=(64, 95), yticks=[65, 75, 85, 95])
    bottom.set_xticks([1, 3, 10, 20, 50], ["1", "3", "10", "20", "50"])
    bottom.set_xlabel("Number of voters $K$", fontsize=9.5)
    bottom.set_ylabel("Test accuracy (%)", fontsize=9.5)
    bottom.set_title(r"Aggregating anchor model vs. top-$K$ experts", fontsize=10, pad=6)
    handles, labels = bottom.get_legend_handles_labels()
    bottom.legend(
        handles[::-1],
        labels[::-1],
        loc="lower right",
        fontsize=8.4,
        frameon=False,
        handlelength=1.6,
        labelspacing=0.35,
    )
    _style(bottom)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.16, top=0.91, hspace=0.76, wspace=0.38)
    _save(fig, output_dir, "headline_random_split_seed42")


def _selection_transfer(bundle: Bundle, output_dir: Path) -> None:
    cells = recompute_selection_transfer(bundle)["cells"]
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.1))
    fig.patch.set_facecolor("white")
    for ax, cell, title in zip(
        axes, (cells["olmo_gsm8k"], cells["olmo_mbpp"]), ("OLMo GSM8K", "OLMo MBPP")
    ):
        rows = cell["curve"]
        m = np.asarray([row["m"] for row in rows])
        observed = np.asarray([row["observed_mean_points"] for row in rows])
        observed_sd = np.asarray([row["observed_sd_points"] for row in rows])
        null = np.asarray([row["pooled_null_mean_points"] for row in rows])
        null_sd = np.asarray([row["pooled_null_sd_points"] for row in rows])
        held = np.asarray([row["transfer"]["1"]["mean_points"] for row in rows])
        low = np.asarray([row["transfer"]["1"]["p2_5_points"] for row in rows], dtype=float)
        high = np.asarray([row["transfer"]["1"]["p97_5_points"] for row in rows], dtype=float)
        ax.errorbar(
            m,
            observed,
            yerr=observed_sd,
            marker="o",
            markersize=4,
            linewidth=1.3,
            elinewidth=0.7,
            capsize=1.8,
            color=BLUE,
            label="Apparent rank-1 lift",
        )
        ax.plot(m, null, color=GRAY, linestyle="--", linewidth=1.25, label="Noise null")
        ax.fill_between(
            m, null - 2 * null_sd, null + 2 * null_sd, color=GRAY, alpha=0.10, linewidth=0
        )
        ax.plot(
            m,
            held,
            marker="s",
            markersize=3.5,
            color=ORANGE,
            linewidth=1.2,
            label="Held-out rank-1 lift",
        )
        ax.fill_between(m, low, high, color=ORANGE, alpha=0.11, linewidth=0)
        ax.axhline(0, color="#59636B", linewidth=0.65)
        ax.set_xscale("log", base=2)
        ax.set_xticks(m, [str(value) for value in m])
        ax.set_xlabel("Selection questions $m$", fontsize=10)
        ax.set_title(title, fontsize=10.2, pad=6)
        _style(ax)
        ax.tick_params(labelsize=9.2)
    axes[0].set_ylabel("Lift over population (points)", fontsize=10)
    handles, labels = axes[1].get_legend_handles_labels()
    order = [
        labels.index(name)
        for name in ("Apparent rank-1 lift", "Noise null", "Held-out rank-1 lift")
    ]
    axes[1].legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="upper right",
        fontsize=8.2,
        frameon=False,
        labelspacing=0.3,
    )
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.19, top=0.86, wspace=0.25)
    _save(fig, output_dir, "selection_transfer")


def _single_experts(bundle: Bundle, output_dir: Path) -> None:
    reference = bundle.json("rank_profiles.json")["cells"]
    scores = bundle.npz("score_data.npz")
    models = (
        ("olmo", "OLMo-3-7B-Instruct", BLUE, "o"),
        ("q3", "Qwen2.5-3B-Instruct", "#8CBD98", "^"),
        ("q7", "Qwen2.5-7B-Instruct", "#2F7D4E", "s"),
        ("q32", "Qwen2.5-32B-Instruct", "#173F2B", "D"),
    )
    datasets = (("gsm8k", "GSM8K"), ("mbpp", "MBPP"), ("countdown", "Countdown"))
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 4.65), sharex=True)
    fig.patch.set_facecolor("white")
    for ax, (model, title, _, _) in zip(axes.flat, models):
        ax.axhline(0, color=GRAY, linewidth=0.9, linestyle=(0, (4, 3)), zorder=2)
        values = [0.0]
        for dataset, label in datasets:
            tag = f"{model}_{dataset}"
            prefix = f"cells.{tag}"
            ranking = scores[f"{prefix}.top50_candidate_ids"][:20]
            candidate_ids = scores[f"{prefix}.test_candidate_ids"]
            positions = {int(candidate): row for row, candidate in enumerate(candidate_ids)}
            rows = np.asarray([positions[int(candidate)] for candidate in ranking])
            accuracy = 100 * scores[f"{prefix}.test_correct"][rows].mean(axis=1)
            anchor = 100 * scores[f"{prefix}.fresh_anchor_correct"].mean()
            gaps = accuracy - anchor
            np.testing.assert_allclose(gaps, reference[tag]["gap_points"], rtol=0, atol=1e-12)
            color = {"gsm8k": BLUE, "mbpp": GREEN, "countdown": ORANGE}[dataset]
            marker = {"gsm8k": "o", "mbpp": "s", "countdown": "^"}[dataset]
            ax.plot(
                np.arange(1, 21),
                gaps,
                label=label,
                color=color,
                marker=marker,
                markersize=3.6,
                markerfacecolor="white",
                markeredgewidth=0.95,
                linewidth=1.35,
                zorder=3,
            )
            values.extend(gaps)
        low, high = min(values), max(values)
        pad = 0.10 * (high - low)
        ax.set_ylim(low - pad, high + pad)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 4, 5, 10]))
        ax.set_title(title, fontsize=11.5, pad=8)
        ax.set_xlim(0.5, 20.5)
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.set_xticks(range(1, 21), minor=True)
        _single_style(ax)
    fig.supylabel("Expert − anchor accuracy", fontsize=11, x=0.018)
    fig.supxlabel("Expert rank on the selection set", fontsize=11, y=0.09)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.55, 0.005),
        ncol=3,
        frameon=False,
        fontsize=10.5,
        handlelength=2.2,
        columnspacing=2,
    )
    fig.subplots_adjust(left=0.11, right=0.99, bottom=0.21, top=0.92, hspace=0.36, wspace=0.23)
    _save(fig, output_dir, "single_experts_by_model")


def _density(bundle: Bundle, output_dir: Path) -> None:
    rows = recompute_density_rows(bundle)
    tasks = ("countdown", "gsm8k", "mbpp")
    main_models = ("olmo", "q3", "q7", "q32")
    styles = {"weights": "-", "sampling": (0, (3, 1.6))}
    names = {"olmo": "OLMo 7B", "q3": "Qwen 3B", "q7": "Qwen 7B", "q32": "Qwen 32B"}
    fig, axes = plt.subplots(2, 3, figsize=(6.6, 3.8), sharey="row", sharex="col")
    fig.patch.set_facecolor("white")
    for r, cutoff in enumerate((0.95, 1.05)):
        limits = (0.01, 0.999) if cutoff < 1 else (0.001, 0.99)
        ticks = [
            value
            for value in (0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999)
            if limits[0] <= value <= limits[1]
        ]
        for c, task in enumerate(tasks):
            ax = axes[r, c]
            _density_style(ax)
            for model in main_models:
                for family, linestyle in styles.items():
                    data = sorted(
                        (
                            row
                            for row in rows
                            if (row["model"], row["task"], row["family"], row["threshold"])
                            == (model, task, family, cutoff)
                        ),
                        key=lambda row: row["m"],
                    )
                    ax.plot(
                        [row["m"] for row in data],
                        np.clip([row["mean_density"] for row in data], *limits),
                        color=MODEL_COLORS[model],
                        linestyle=linestyle,
                        linewidth=1.1,
                        marker="o" if family == "weights" else None,
                        markersize=1.7,
                    )
            ax.set_yscale("logit")
            ax.set_ylim(*limits)
            ax.set_yticks(ticks, [f"{100 * value:g}" for value in ticks])
            ax.yaxis.set_minor_locator(NullLocator())
            end = {"countdown": 1400, "gsm8k": 1319, "mbpp": 500}[task]
            ax.set_xlim(0, end * 1.02)
            ax.set_xticks(
                {"countdown": [0, 700, 1400], "gsm8k": [0, 600, 1200], "mbpp": [0, 250, 500]}[task]
            )
            if r == 0:
                ax.set_title(
                    "Countdown" if task == "countdown" else task.upper(), fontsize=10.5, pad=6
                )
    fig.subplots_adjust(left=0.15, right=0.99, top=0.91, bottom=0.14, hspace=0.18, wspace=0.13)
    for r, cutoff in enumerate((0.95, 1.05)):
        box = axes[r, 0].get_position()
        fig.text(
            0.065,
            (box.y0 + box.y1) / 2,
            f"{cutoff:.0%}",
            ha="center",
            va="center",
            fontsize=9.5,
        )
    fig.supylabel("Accuracy density (%)", fontsize=10, x=0.004)
    fig.supxlabel("Number of evaluation questions, $m$", fontsize=10, y=0.015)
    model_handles = [
        Line2D([], [], color=MODEL_COLORS[m], linewidth=1.6, label=names[m]) for m in main_models
    ]
    family_handles = [
        Line2D([], [], color="#525C65", linewidth=1.4, linestyle=styles[f], label=label)
        for f, label in (("weights", "Perturbations"), ("sampling", "Stochastic anchor"))
    ]
    handles = [
        model_handles[0],
        model_handles[2],
        family_handles[0],
        model_handles[1],
        model_handles[3],
        family_handles[1],
    ]
    legend = fig.legend(
        handles=handles,
        loc="lower right",
        bbox_to_anchor=(0.98, axes[0, 2].get_position().y0 - 0.025),
        ncol=2,
        frameon=True,
        fancybox=True,
        facecolor="white",
        edgecolor="#BCC6CF",
        framealpha=1,
        fontsize=8.3,
        handlelength=1.8,
        handletextpad=0.6,
        columnspacing=1,
        labelspacing=0.6,
        borderpad=0.65,
        borderaxespad=0,
    )
    legend.get_frame().set_linewidth(0.7)
    _save(fig, output_dir, "accuracy_density_by_m")

    scales = {"q05": 0.5, "q15": 1.5, "q3": 3.0, "q7": 7.0, "q32": 32.0}
    colors = ("#86ADC9", "#4C83B1", "#1F4F7D", "#397B65", "#B36636")
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.65), sharey=True)
    fig.patch.set_facecolor("white")
    for ax, task in zip(axes, tasks):
        _density_style(ax)
        for cutoff, color in zip((0.9, 0.95, 1.0, 1.05, 1.1), colors):
            for family, linestyle in styles.items():
                data = sorted(
                    (
                        row
                        for row in rows
                        if row["model"] in scales
                        and (row["task"], row["family"], row["threshold"], row["m"])
                        == (task, family, cutoff, 200)
                    ),
                    key=lambda row: scales[row["model"]],
                )
                ax.plot(
                    [scales[row["model"]] for row in data],
                    [100 * row["mean_density"] for row in data],
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.15,
                    marker="o",
                    markersize=3,
                    markerfacecolor=color if family == "weights" else "white",
                    markeredgewidth=0.7,
                )
        ax.set_xscale("log")
        ax.set_xlim(0.4, 40)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticks(list(scales.values()), [f"{size:g}B" for size in scales.values()])
        ax.set_title(
            "Countdown" if task == "countdown" else task.upper(), fontsize=10.5, pad=7
        )
    axes[0].set_ylabel("Accuracy density (%)", fontsize=10)
    fig.subplots_adjust(left=0.095, right=0.99, top=0.92, bottom=0.32, wspace=0.16)
    fig.supxlabel("Qwen model size", fontsize=9.5, y=0.15)
    cutoff_handles = [
        Line2D([], [], color=color, linewidth=1.6, label=f"{cutoff:.0%}")
        for cutoff, color in zip((0.9, 0.95, 1, 1.05, 1.1), colors)
    ]
    fig.legend(
        handles=cutoff_handles + family_handles,
        loc="lower center",
        bbox_to_anchor=(0.54, 0.005),
        ncol=7,
        frameon=False,
        fontsize=8.3,
        handlelength=1.5,
        handletextpad=0.4,
        columnspacing=0.8,
    )
    _save(fig, output_dir, "accuracy_density_by_size")


_SIZE_TASKS = ("countdown", "gsm8k", "mbpp")
_SIZE_SCALES = {"q05": 0.5, "q15": 1.5, "q3": 3.0, "q7": 7.0, "q32": 32.0}
_SIZE_THRESHOLDS = (0.9, 0.95, 1.0, 1.05, 1.1)
_SIZE_COLORS = ("#86ADC9", "#4C83B1", "#1F4F7D", "#397B65", "#B36636")
_SIZE_STYLES = {"weights": "-", "sampling": (0, (3, 1.6))}
_SIZE_VARIANTS = (
    ("reward", "population", "Reward density (%)\nPopulation reference"),
    ("accuracy", "greedy_anchor", "Accuracy density (%)\nGreedy-anchor reference"),
    ("reward", "greedy_anchor", "Reward density (%)\nGreedy-anchor reference"),
)


def _density_size_handles() -> list[Line2D]:
    thresholds = [
        Line2D([], [], color=color, linewidth=1.6, label=f"{threshold:.0%}")
        for threshold, color in zip(_SIZE_THRESHOLDS, _SIZE_COLORS)
    ]
    families = [
        Line2D([], [], color="#525C65", linewidth=1.4, linestyle=_SIZE_STYLES[family], label=label)
        for family, label in (("weights", "Perturbations"), ("sampling", "Stochastic anchor"))
    ]
    return thresholds + families


def _draw_density_size_row(
    axes: np.ndarray,
    rows: list[dict],
    *,
    metric: str,
    reference: str,
    show_titles: bool,
) -> None:
    for ax, task in zip(axes, _SIZE_TASKS):
        _density_style(ax)
        for threshold, color in zip(_SIZE_THRESHOLDS, _SIZE_COLORS):
            for family, linestyle in _SIZE_STYLES.items():
                selected = sorted(
                    (
                        row
                        for row in rows
                        if (
                            row["task"],
                            row["family"],
                            row["metric"],
                            row["reference"],
                            row["threshold"],
                            row["m"],
                        )
                        == (task, family, metric, reference, threshold, 200)
                    ),
                    key=lambda row: _SIZE_SCALES[row["model"]],
                )
                assert [row["model"] for row in selected] == list(_SIZE_SCALES)
                ax.plot(
                    [_SIZE_SCALES[row["model"]] for row in selected],
                    [100 * row["mean_density"] for row in selected],
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.15,
                    marker="o",
                    markersize=3,
                    markerfacecolor=color if family == "weights" else "white",
                    markeredgewidth=0.7,
                )
        ax.set_xscale("log")
        ax.set_xlim(0.4, 40)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticks(
            list(_SIZE_SCALES.values()),
            [f"{size:g}B" for size in _SIZE_SCALES.values()],
        )
        if show_titles:
            ax.set_title(
                "Countdown" if task == "countdown" else task.upper(),
                fontsize=10.5,
                pad=7,
            )


def _density_size_variant_figure(
    rows: list[dict],
    *,
    metric: str,
    reference: str,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.65), sharey=True)
    fig.patch.set_facecolor("white")
    _draw_density_size_row(
        axes,
        rows,
        metric=metric,
        reference=reference,
        show_titles=True,
    )
    axes[0].set_ylabel(f"{metric.capitalize()} density (%)", fontsize=10)
    fig.subplots_adjust(left=0.095, right=0.99, top=0.92, bottom=0.32, wspace=0.16)
    fig.supxlabel("Qwen model size", fontsize=9.5, y=0.15)
    fig.legend(
        handles=_density_size_handles(),
        loc="lower center",
        bbox_to_anchor=(0.54, 0.005),
        ncol=7,
        frameon=False,
        fontsize=8.3,
        handlelength=1.5,
        handletextpad=0.4,
        columnspacing=0.8,
    )
    return fig


def _density_size_variants(bundle: Bundle, output_dir: Path) -> None:
    rows = recompute_density_size_variants(bundle)
    fig, axes = plt.subplots(3, 3, figsize=(6.9, 6.55), sharex="col", sharey=True)
    fig.patch.set_facecolor("white")
    for row_index, (metric, reference, label) in enumerate(_SIZE_VARIANTS):
        _draw_density_size_row(
            axes[row_index],
            rows,
            metric=metric,
            reference=reference,
            show_titles=row_index == 0,
        )
        axes[row_index, 0].set_ylabel(label, fontsize=9)
    fig.subplots_adjust(left=0.125, right=0.99, top=0.965, bottom=0.14, hspace=0.22, wspace=0.16)
    fig.supxlabel("Qwen model size", fontsize=9.5, y=0.06)
    fig.legend(
        handles=_density_size_handles(),
        loc="lower center",
        bbox_to_anchor=(0.55, 0.005),
        ncol=7,
        frameon=False,
        fontsize=8.3,
        handlelength=1.5,
        handletextpad=0.4,
        columnspacing=0.8,
    )
    _save(fig, output_dir, "density_by_size_variants")


def write_density_size_variant_figures(bundle: Bundle, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    rows = recompute_density_size_variants(bundle)
    stems = {
        ("reward", "population"): "reward_density_by_size_population",
        ("accuracy", "greedy_anchor"): "accuracy_density_by_size_anchor",
        ("reward", "greedy_anchor"): "reward_density_by_size_anchor",
    }
    for (metric, reference), stem in stems.items():
        fig = _density_size_variant_figure(
            rows,
            metric=metric,
            reference=reference,
        )
        _save(fig, output_dir, stem)
    return [output_dir / f"{stem}.pdf" for stem in stems.values()]


def write_figures(bundle: Bundle, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    _headline(bundle, output_dir)
    _selection_transfer(bundle, output_dir)
    _single_experts(bundle, output_dir)
    _density(bundle, output_dir)
    _density_size_variants(bundle, output_dir)
    return sorted(output_dir.glob("*.pdf"))
