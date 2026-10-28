import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import pytest

import neural_thickets.experiments.runner as runner
from neural_thickets.experiments.contracts import load_contract
from neural_thickets.experiments.runner import (
    _atomic_jsonl,
    _paper_test_qids_for_task,
    _plan,
    _prepare,
    _reported_selection_ranking,
    _score,
    _select,
    _serialized_contract,
    _version_matches,
    _vote_curve,
)


def test_prepare_rejects_config_changes(tmp_path) -> None:
    contract = load_contract("fresh_anchor_olmo_suite", output_dir=tmp_path)
    _prepare(contract, SimpleNamespace())
    _prepare(contract, SimpleNamespace())
    changed = dataclasses.replace(contract, repeat_seeds=(42,))
    with pytest.raises(AssertionError, match="contract drift"):
        _prepare(changed, SimpleNamespace())


def test_contract_identity_excludes_host_run_location(tmp_path) -> None:
    host = load_contract("search_olmo_mbpp", output_dir=tmp_path / "host-runs")
    container = dataclasses.replace(host, output_dir=tmp_path / "container-runs")

    assert _serialized_contract(host) == _serialized_contract(container)
    assert "output_dir" not in _serialized_contract(host)


def test_vote_order_matrix_uses_seed_sequence_once_for_every_budget() -> None:
    answers = [[str(row) for _ in range(4)] for row in range(5)]
    correct = np.array([[row % 2 == 0] * 4 for row in range(5)], dtype=np.bool_)
    _, orderings = _vote_curve(
        answers,
        correct,
        budgets=(2, 3),
        seed=(20260910, 4, 1),
        orderings=3,
        strategy="seed_sequence",
    )
    rng = np.random.default_rng(np.random.SeedSequence((20260910, 4, 1)))
    expected = [rng.permutation(5).tolist() for _ in range(3)]
    assert orderings == expected


def test_legacy_countdown_k1_averages_every_single_voter() -> None:
    answers = [[str(row)] * 3 for row in range(5)]
    correct = np.array(
        [
            [True, True, True],
            [True, True, False],
            [True, False, False],
            [False, False, False],
            [False, False, False],
        ],
        dtype=np.bool_,
    )
    curve, orderings = _vote_curve(
        answers,
        correct,
        budgets=(1, 3, 5),
        seed=(20260908,),
        orderings=20,
        strategy="legacy_olmo_countdown",
    )
    assert curve["1"]["mean"] == pytest.approx(float(correct.mean()))
    assert curve["1"]["orderings"] == 5
    assert len(orderings) == 20


def test_paper_evaluation_pools() -> None:
    assert _paper_test_qids_for_task("countdown") == set(range(200, 1600))
    assert _paper_test_qids_for_task("gsm8k") == set(range(1_000_000, 1_001_319))
    assert _paper_test_qids_for_task("mbpp") == set(range(464, 964))


def test_runtime_allows_cuda_build_variation_but_not_torch_version_drift() -> None:
    assert _version_matches("torch", "2.9.1+cu126", "2.9.1+cu128")
    assert not _version_matches("torch", "2.9.1+cu126", "2.9.2+cu128")
    assert _version_matches("vllm", "0.16.0", "0.16.0")


def test_shuffle_checks_runtime_before_loading_models(tmp_path, monkeypatch) -> None:
    contract = load_contract("uspto_shuffled_answers", output_dir=tmp_path)
    _prepare(contract, SimpleNamespace())
    (contract.output_dir / "preflight.json").write_text(
        json.dumps({"contract": contract.name})
    )

    def fail_runtime(*_args):
        raise RuntimeError("runtime checked")

    monkeypatch.setattr(runner, "_verify_runtime", fail_runtime)
    args = SimpleNamespace(
        repeat_seed=42,
        allow_runtime_drift=False,
        output_dir=tmp_path,
    )
    with pytest.raises(RuntimeError, match="runtime checked"):
        runner._shuffle(contract, args)


def test_plan_is_cpu_only_and_expands_workers(capsys) -> None:
    contract = load_contract("search_qwen32b_countdown")
    _plan(contract, SimpleNamespace())
    plan = json.loads(capsys.readouterr().out)
    assert plan["concurrent_gpus"] == 4
    assert plan["population_candidates"] == 1000
    test = next(row for row in plan["phases"] if row["phase"] == "test")
    assert test["invocations"] == [{"worker_index": 0}, {"worker_index": 1}]

    chemistry = load_contract("uspto_shuffled_answers")
    _plan(chemistry, SimpleNamespace())
    chemistry_plan = json.loads(capsys.readouterr().out)
    assert [row["contract"] for row in chemistry_plan["requires"]] == [
        "search_olmo_uspto50k",
        "uspto_anchor_original",
    ]


def test_select_preserves_stable_order_inside_reward_tie(tmp_path) -> None:
    contract = dataclasses.replace(
        load_contract("search_qwen32b_countdown", output_dir=tmp_path),
        population_size=3,
        top_k=2,
        canonical_selected_indices=(2, 0),
    )
    _prepare(contract, SimpleNamespace())
    search = contract.output_dir / "search"
    for index, reward in enumerate((0.5, 0.4, 0.5)):
        _atomic_jsonl(
            search / f"candidate_{index:05d}.jsonl.gz",
            [
                {
                    "candidate": {"index": index},
                    "reference_reward": reward,
                }
                for _ in contract.dataset.selection_qids
            ],
        )
    _select(contract, SimpleNamespace())
    selected = json.loads((contract.output_dir / "selected.json").read_text())
    assert [row["index"] for row in selected] == [2, 0]
    assert [row["score"] for row in selected] == [0.5, 0.5]


def test_reported_ranking_uses_canonical_prefix_and_preserves_population() -> None:
    recomputed = np.asarray([9, 7, 3, 5, 1], dtype=np.int32)

    reported = _reported_selection_ranking(recomputed, (7, 9, 5))

    assert reported.tolist() == [7, 9, 5, 3, 1]


def test_mbpp_score_phase_requires_fixed_hash_seed(tmp_path, monkeypatch) -> None:
    contract = load_contract("search_olmo_mbpp", output_dir=tmp_path)
    _prepare(contract, SimpleNamespace())
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    with pytest.raises(AssertionError, match="PYTHONHASHSEED=0"):
        _score(contract, SimpleNamespace(score_workers=1))

