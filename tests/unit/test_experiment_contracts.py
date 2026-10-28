import dataclasses
from pathlib import Path

import pytest

from neural_thickets.experiments.contracts import contract_names, load_contract
from neural_thickets.experiments.data import MBPP_USER, Example
from neural_thickets.experiments.runner import _score_rows
from neural_thickets.perturbations.population import contract_population
from neural_thickets.types import BatchingStrategy, ExperimentKind


def test_every_contract_loads_and_required_families_exist() -> None:
    names = contract_names()
    assert all(load_contract(name) for name in names)
    assert len(names) == 55
    assert "headline_olmo_countdown_fixed_sigma" in names
    assert "candidate930_repeats" in names
    assert "density_greedy_anchor_qwen05b_suite" in names
    assert "uspto_shuffled_answers" in names


def test_mbpp_prompt_template_is_exact() -> None:
    assert MBPP_USER == (
        "{text}\n\nYour code should pass these tests:\n{tests}\n\n"
        "Think through your solution in <think> </think> tags.\n"
        "Return your final Python code in <answer> </answer> tags, e.g.:\n"
        "<answer>\ndef solution(x):\n return x + 1\n</answer>"
    )


def test_qwen32_uses_prefix_of_full_5000_draw() -> None:
    full = contract_population(load_contract("search_olmo_countdown"))
    qwen32 = contract_population(load_contract("search_qwen32b_countdown"))
    assert qwen32 == full[:1000]
    assert [(row.seed, row.sigma) for row in qwen32[:3]] == [
        (424486224, 0.001),
        (2037198350, 0.001),
        (481929201, 0.0005),
    ]
    contract = load_contract("search_qwen32b_countdown")
    assert contract.canonical_selected_indices[:7] == (958, 76, 318, 32, 731, 30, 71)
    assert len(contract.canonical_selected_indices) == 50


def test_qwen32_density_filters_the_same_prefix() -> None:
    qwen32 = contract_population(load_contract("search_qwen32b_countdown"))
    expected = tuple(candidate for candidate in qwen32 if candidate.sigma == 0.001)
    actual = contract_population(load_contract("density_qwen32b_countdown"))
    assert len(expected) == 338
    assert actual == expected[:100]
    cohorts = dict(load_contract("density_qwen32b_countdown").candidate_cohorts)
    assert cohorts["reused_wave4"] == (36, 119, 197, 250)
    assert len(cohorts["density_fill_20260922"]) == 96


def test_density_evaluation_pools_match_production_order() -> None:
    countdown = load_contract("density_qwen05b_countdown").dataset.evaluation_qids
    gsm8k = load_contract("density_qwen05b_gsm8k").dataset.evaluation_qids
    mbpp = load_contract("density_qwen05b_mbpp").dataset.evaluation_qids
    assert countdown == tuple(range(200, 1600))
    assert gsm8k == tuple(range(1_000_000, 1_001_319))
    assert mbpp == (*range(200, 400), *range(464, 964))


def test_headline_and_chemistry_execution_contracts() -> None:
    headline = load_contract("headline_olmo_countdown_fixed_sigma")
    assert headline.engine.max_model_len == 8192
    assert headline.engine.max_num_seqs == 32
    assert headline.engine.prompt_batch_size == 200
    chemistry = load_contract("search_olmo_uspto50k")
    assert chemistry.engine.workers == 2
    assert chemistry.top_k == 500
    assert chemistry.runtime.vllm == "0.16.0"
    assert chemistry.dataset.source_revision == "8296db0b791758743cb8efb5d62cc10d03c2b988"
    assert dict(chemistry.dataset.file_sha256) == {
        "train.parquet": "5050f5598583b2d0c6033a01c61fa14702455ccbe908a2d9fcfe00f7415e86bc",
        "validation.parquet": (
            "48eb1bd375edbd291704bb127114fc0df4dd654c388a241aa755f887a0e9b617"
        ),
        "source_train.parquet": (
            "53542c1620fee15f6b89d1958639b518e72499479a29411c64d0cb400dbfbade"
        ),
        "source_validation.parquet": (
            "9cb4381b2235a7c5815c0e881593c324f244a6533a0d770d4598669f669a22d9"
        ),
    }


def test_historical_batching_and_fresh_anchor_process_contracts() -> None:
    search = load_contract("search_qwen7b_gsm8k")
    assert search.batching is BatchingStrategy.FULL_POOL
    assert search.engine.workers == 4
    voting = load_contract("ttmv_qwen32b_countdown")
    assert voting.batching is BatchingStrategy.CONTIGUOUS_WORKER_SHARD
    assert voting.engine.workers == 2
    fresh = load_contract("fresh_anchor_olmo_suite")
    assert fresh.batching is BatchingStrategy.FIXED_CHUNKS
    assert [fresh.dataset.key, *[item.key for item in fresh.additional_datasets]] == [
        "countdown",
        "gsm8k",
        "mbpp",
    ]


def test_packaged_contract_matches_reviewable_root_contract() -> None:
    root = Path(__file__).resolve().parents[2]
    source = root / "configs" / "experiments.yaml"
    packaged = root / "src" / "neural_thickets" / "configs" / "experiments.yaml"
    assert source.read_bytes() == packaged.read_bytes()


def test_fresh_anchor_contract_is_not_a_search_contract() -> None:
    contract = load_contract("fresh_anchor_qwen32b_suite")
    assert contract.kind is ExperimentKind.ANCHOR
    assert contract.repeat_seeds == (42, 43, 44, 45, 46)
    assert contract.samples_per_question == 1
    assert contract.population_size is None
    assert contract.perturbation_semantics is None
    assert [dataset.key for dataset in (contract.dataset, *contract.additional_datasets)] == [
        "countdown",
        "gsm8k",
        "mbpp",
    ]


def test_density_greedy_anchor_contract_matches_figure_reference() -> None:
    contract = load_contract("density_greedy_anchor_qwen05b_suite")
    assert contract.kind is ExperimentKind.ANCHOR
    assert contract.decode.temperature == 0
    assert contract.repeat_seeds == (42,)
    assert contract.samples_per_question == 1
    assert contract.batching is BatchingStrategy.CONTIGUOUS_WORKER_SHARD
    assert contract.dataset.evaluation_qids == tuple(range(200, 1600))
    assert contract.additional_datasets[0].evaluation_qids == tuple(
        range(1_000_000, 1_001_319)
    )
    assert contract.additional_datasets[1].test_qids == tuple(range(464, 964))


def test_countdown_selection_metric_is_reference_536_reward() -> None:
    contract = load_contract("search_olmo_countdown")
    example = Example(
        qid=0,
        messages=(),
        gold={"numbers": [1, 2, 3], "target": 9},
    )
    generated = [
        {
            "qid": 0,
            "generated_text": "reasoning</think>\n<answer>(1 + 2) * 3 = 9</answer>",
        }
    ]
    row = _score_rows(contract, (example,), generated)[0]
    assert contract.selection_metric == "reference_reward"
    assert row["reference_reward"] == row["reward"] == 1.1
    assert row["historical_d409_reward"] == 0.1


def test_iid_exact_rejects_tensor_parallel_models() -> None:
    contract = load_contract("audit_iid_countdown_full_grid")
    model = dataclasses.replace(contract.model, tensor_parallel_size=2)
    with pytest.raises(AssertionError, match="TP=1"):
        dataclasses.replace(contract, model=model)

