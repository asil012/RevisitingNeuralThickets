"""Regenerate the paper's active numerical tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from neural_thickets.analysis.voting import plurality_vote_scores
from neural_thickets.data import Bundle

MODELS = {"olmo": "OLMo 7B", "q3": "Qwen 3B", "q7": "Qwen 7B", "q32": "Qwen 32B"}
TASKS = {"countdown": "Countdown", "gsm8k": "GSM8K", "mbpp": "MBPP"}


def _table(columns, rows):
    assert all(len(row) == len(columns) for row in rows)
    return {"columns": columns, "rows": rows}


def _signed(value: float) -> str:
    return f"{value:+.2f}" if not np.isclose(value, 0) else "0.00"


def _voting(bundle: Bundle):
    outcomes = bundle.npz("outcomes.npz")
    scores = bundle.npz("score_data.npz")
    rows = []
    for model in ("olmo", "q3", "q7", "q32"):
        for task in ("gsm8k", "mbpp", "countdown"):
            tag = f"{model}_{task}"
            prefix = f"cells.{tag}"
            expert_groups = scores[f"{prefix}.top50_answer_group"]
            expert_correct = scores[f"{prefix}.top50_correct"]
            randopt = plurality_vote_scores(
                expert_groups, expert_correct, np.arange(50)[None, :], (10, 20, 50)
            )
            anchor_groups = scores[f"{prefix}.anchor_answer_group"]
            anchor_correct = scores[f"{prefix}.anchor_correct"]
            if tag == "olmo_countdown":
                anchor = {
                    k: plurality_vote_scores(
                        anchor_groups,
                        anchor_correct,
                        scores[f"{prefix}.anchor_orders_k{k}"],
                        (k,),
                    )[k]
                    for k in (10, 20, 50)
                }
            else:
                anchor = plurality_vote_scores(
                    anchor_groups,
                    anchor_correct,
                    scores[f"{prefix}.anchor_orders"],
                    (10, 20, 50),
                )
            row = [MODELS[model], TASKS[task]]
            for k in (10, 20, 50):
                question = randopt[k] - anchor[k]
                stored = outcomes[f"{tag}.question.gap.k{k}"]
                np.testing.assert_allclose(question, stored, rtol=0, atol=1e-15)
                bootstrap = outcomes[f"{tag}.bootstrap.gap.k{k}"]
                mean = 100 * float(question.mean())
                low, high = np.quantile(bootstrap, (0.025, 0.975))
                row.extend((_signed(mean), "Yes" if low > 0 or high < 0 else "No"))
            rows.append(row)
    return _table(
        [
            "Model",
            "Task",
            "K=10 difference",
            "K=10 significant",
            "K=20 difference",
            "K=20 significant",
            "K=50 difference",
            "K=50 significant",
        ],
        rows,
    )


def _density_populations(bundle: Bundle):
    rows = bundle.csv("density_curves.csv")
    names = {
        "olmo": "OLMo-3-7B-Instruct",
        "q05": "Qwen2.5-0.5B-Instruct",
        "q15": "Qwen2.5-1.5B-Instruct",
        "q3": "Qwen2.5-3B-Instruct",
        "q7": "Qwen2.5-7B-Instruct",
        "q32": "Qwen2.5-32B-Instruct",
    }
    output = []
    for model in ("olmo", "q05", "q15", "q3", "q7", "q32"):
        counts = []
        for task in ("countdown", "gsm8k", "mbpp"):
            values = {
                int(row["n_runs"])
                for row in rows
                if (row["model"], row["task"], row["family"]) == (model, task, "weights")
            }
            assert len(values) == 1
            counts.append(values.pop())
        output.append([names[model], *counts])
    return _table(["Model", "Countdown", "GSM8K", "MBPP"], output)


def _null_variance(support: dict):
    labels = (
        ("olmo_countdown_common1600", "OLMo 7B", "Countdown, headline sweep"),
        ("olmo_countdown", "OLMo 7B", "Countdown, transfer diagnostic"),
        ("olmo_gsm8k", "OLMo 7B", "GSM8K"),
        ("olmo_mbpp", "OLMo 7B", "MBPP"),
        ("q3_countdown", "Qwen 3B", "Countdown"),
        ("q3_gsm8k", "Qwen 3B", "GSM8K"),
        ("q3_mbpp", "Qwen 3B", "MBPP"),
        ("q7_countdown", "Qwen 7B", "Countdown"),
        ("q7_gsm8k", "Qwen 7B", "GSM8K"),
        ("q7_mbpp", "Qwen 7B", "MBPP"),
        ("q32_countdown", "Qwen 32B", "Countdown"),
        ("q32_gsm8k", "Qwen 32B", "GSM8K"),
        ("q32_mbpp", "Qwen 32B", "MBPP"),
    )
    rows = []
    for key, model, task in labels:
        record = support["variances"][key]
        rows.append([model, task, record["n"], record["q"], f"{record['v']:.4f}"])
    return _table(["Model", "Task / analysis", "Candidates N", "Questions Q", "v_hat"], rows)


def _decay(selection: dict):
    rows = []
    for key, task in (
        ("olmo_countdown", "Countdown"),
        ("olmo_gsm8k", "GSM8K"),
        ("olmo_mbpp", "MBPP"),
    ):
        for record in selection["cells"][key]["curve"]:
            transfer = record["transfer"]["1"]
            low = transfer["p2_5_points"]
            high = transfer["p97_5_points"]
            assert (low is None) == (high is None)
            half_width = "" if low is None else f"{(high - low) / 2:.2f}"
            mean = transfer["mean_points"]
            mean = 0.0 if round(mean, 2) == 0 else mean
            rows.append(
                [
                    task,
                    record["m"],
                    f"{record['observed_mean_points']:.2f}",
                    f"{record['pooled_null_mean_points']:.2f}",
                    f"{record['pooled_null_sd_points']:.2f}",
                    f"{mean:.2f}",
                    half_width,
                ]
            )
    return _table(
        [
            "Task",
            "m",
            "Apparent lift",
            "Noise null mean",
            "Noise null SD",
            "Held-out rank-1",
            "Held-out interval half-width",
        ],
        rows,
    )


def _qwen_decay(selection: dict, support: dict):
    rows = []
    for model in ("q3", "q7", "q32"):
        tasks = ("gsm8k", "mbpp")
        if model in ("q3", "q7"):
            tasks += ("countdown",)
        for task in tasks:
            records = (
                support["q3_countdown_curve"]
                if (model, task) == ("q3", "countdown")
                else selection["cells"][f"{model}_{task}"]["curve"]
            )
            curve = {
                row["m"]: row["observed_mean_points"]
                for row in records
            }
            rows.append(
                [
                    MODELS[model],
                    TASKS[task],
                    *["" if m not in curve else f"{curve[m]:.2f}" for m in (25, 200, 400, 800)],
                ]
            )
    return _table(["Model", "Task", "m=25", "m=200", "m=400", "m=800"], rows)


def _selection_robustness(selection: dict, support: dict):
    rows = []
    for model in ("olmo", "q3", "q7", "q32"):
        for task in ("countdown", "gsm8k", "mbpp"):
            key = f"{model}_{task}"
            if key == "q3_countdown":
                record = next(row for row in support["q3_countdown_curve"] if row["m"] == 200)
            elif key == "q32_countdown":
                record = support["q32_countdown_fixed200"]
            else:
                record = next(row for row in selection["cells"][key]["curve"] if row["m"] == 200)
            lift = record["observed_mean_points"]
            null = record["pooled_null_mean_points"]
            transfer = ""
            if "transfer" in record:
                value = record["transfer"]["50"]["mean_points"]
                transfer = "0.00" if np.isclose(value, 0) else f"{value:.2f}"
            rows.append(
                [
                    MODELS[model],
                    TASKS[task],
                    f"{lift:.2f}",
                    f"{null:.2f}",
                    f"{100 * null / lift:.1f}%",
                    transfer,
                ]
            )
    return _table(["Model", "Task", "Lift", "Noise null", "Null / lift", "Top-50 transfer"], rows)


def _fresh_anchors(bundle: Bundle):
    scores = bundle.npz("score_data.npz")
    rows = []
    for model in ("olmo", "q3", "q7", "q32"):
        for task in ("countdown", "gsm8k", "mbpp"):
            correct = scores[f"cells.{model}_{task}.fresh_anchor_correct"]
            values = 100 * correct.mean(axis=1)
            summary = (values.mean(), values.min(), values.max(), values.std(ddof=1))
            rows.append(
                [
                    MODELS[model],
                    TASKS[task],
                    *[f"{value:.2f}" for value in summary],
                ]
            )
    return _table(["Model", "Task", "Mean", "Minimum", "Maximum", "SD"], rows)


def _uspto(bundle: Bundle):
    data = bundle.npz("uspto_data.npz")
    rows = []
    for condition in ("original", "shuffled"):
        expert = 100 * data[f"{condition}.expert_correct"].mean(axis=1)
        anchor = 100 * data[f"{condition}.anchor_correct"].mean()
        rows.append(
            [
                condition.title(),
                *[f"{(expert[:k].mean() - anchor):+.2f}" for k in (1, 5, 10, 20)],
            ]
        )
    return _table(["Answer choices", "Top-1", "Top-5", "Top-10", "Top-20"], rows)


def build_tables(bundle: Bundle) -> dict:
    selection = bundle.json("selection.json")
    support = bundle.json("source_summaries.json")
    tables = bundle.json("table_inputs.json")
    tables.update(
        {
            "tab:voting-main": _voting(bundle),
            "tab:uspto-shuffle": _uspto(bundle),
            "tab:density-populations": _density_populations(bundle),
            "tab:null-variance": _null_variance(support),
            "tab:decay-full": _decay(selection),
            "tab:qwen-selection-decay": _qwen_decay(selection, support),
            "tab:selection-robustness": _selection_robustness(selection, support),
            "tab:fresh-anchors": _fresh_anchors(bundle),
        }
    )
    assert len(tables) == 15
    return tables


def write_tables(bundle: Bundle, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = build_tables(bundle)
    outputs = []
    for label, table in tables.items():
        assert table["columns"] and table["rows"]
        path = output_dir / f"{label.removeprefix('tab:').replace('-', '_')}.csv"
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(table["columns"])
            writer.writerows(table["rows"])
        outputs.append(path)
    with (output_dir / "tables.json").open("w") as stream:
        json.dump(tables, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return outputs
