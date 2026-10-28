from __future__ import annotations

import argparse
import dataclasses
import gzip
import hashlib
import importlib
import json
import os
import platform
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from neural_thickets.experiments.contracts import contract_names, load_contract
from neural_thickets.experiments.data import Example, load_examples
from neural_thickets.perturbations.population import contract_population, paper_population
from neural_thickets.scoring.countdown import score_countdown
from neural_thickets.scoring.gsm8k import score_gsm8k
from neural_thickets.scoring.mbpp import extract_mbpp, score_mbpp_batch_isolated
from neural_thickets.scoring.types import Outcome
from neural_thickets.scoring.uspto50k import score_uspto50k
from neural_thickets.types import (
    BatchingStrategy,
    DatasetSpec,
    ExperimentContract,
    ExperimentKind,
    PerturbationSemantics,
    SearchScope,
    SummaryScope,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Neural Thickets experiment")
    parser.add_argument("--contract", choices=contract_names(), required=True)
    parser.add_argument(
        "--phase",
        choices=(
            "plan",
            "prepare",
            "preflight",
            "search",
            "score",
            "select",
            "test",
            "anchor",
            "candidate-repeat",
            "shuffle",
            "summarize",
        ),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/datasets"))
    parser.add_argument("--candidate-start", type=int, default=0)
    parser.add_argument("--candidate-stop", type=int)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--repeat-seed", type=int)
    parser.add_argument("--score-workers", type=int, default=4)
    parser.add_argument("--unsafe-mbpp-scoring", action="store_true")
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--allow-runtime-drift", action="store_true")
    parser.add_argument("--allow-no-gpu-preflight", action="store_true")
    args = parser.parse_args()
    contract = load_contract(args.contract, output_dir=args.output_dir)
    dispatch = {
        "plan": _plan,
        "prepare": _prepare,
        "preflight": _preflight,
        "search": _search,
        "score": _score,
        "select": _select,
        "test": _test,
        "anchor": _anchor,
        "candidate-repeat": _candidate_repeat,
        "shuffle": _shuffle,
        "summarize": _summarize,
    }
    dispatch[args.phase](contract, args)


def _plan(contract: ExperimentContract, args: argparse.Namespace) -> None:
    del args
    phases: list[dict[str, object]] = [
        {"phase": "prepare", "invocations": [{}]},
        {"phase": "preflight", "invocations": [{}]},
    ]
    mbpp = any(
        dataset.key == "mbpp"
        for dataset in (contract.dataset, *contract.additional_datasets)
    )
    if contract.kind is ExperimentKind.SEARCH:
        phases.append(
            {
                "phase": "search",
                "invocations": [
                    {"worker_index": worker}
                    for worker in range(contract.engine.workers)
                ],
            }
        )
        if mbpp:
            phases.append({"phase": "score", "after": "search", "invocations": [{}]})
        if contract.search_scope is SearchScope.SELECT_THEN_TEST:
            phases.append({"phase": "select", "invocations": [{}]})
            phases.append(
                {
                    "phase": "test",
                    "invocations": [
                        {"worker_index": worker}
                        for worker in range(contract.engine.workers)
                    ],
                }
            )
            if mbpp:
                phases.append({"phase": "score", "after": "test", "invocations": [{}]})
    elif contract.kind is ExperimentKind.ANCHOR:
        phases.append(
            {
                "phase": "anchor",
                "invocations": [
                    {"repeat_seed": seed, "worker_index": worker}
                    for seed in contract.repeat_seeds
                    for worker in range(contract.engine.workers)
                ],
            }
        )
        if mbpp:
            phases.append({"phase": "score", "invocations": [{}]})
    elif contract.kind is ExperimentKind.CANDIDATE_REPEAT:
        phases.append(
            {
                "phase": "candidate-repeat",
                "invocations": [
                    {"repeat_seed": seed, "worker_index": 0}
                    for seed in contract.repeat_seeds
                ],
            }
        )
        if mbpp:
            phases.append({"phase": "score", "invocations": [{}]})
    else:
        assert contract.kind is ExperimentKind.LABEL_SHUFFLE
        phases.append(
            {
                "phase": "shuffle",
                "invocations": [
                    {"repeat_seed": contract.repeat_seeds[0], "worker_index": worker}
                    for worker in range(contract.engine.workers)
                ],
            }
        )
    phases.append({"phase": "summarize", "invocations": [{}]})
    requires = []
    if contract.kind is ExperimentKind.LABEL_SHUFFLE:
        requires = [
            {
                "contract": contract.source_search,
                "artifacts": ["selected.json", "FINAL_MANIFEST.json"],
            },
            {
                "contract": "uspto_anchor_original",
                "artifacts": ["FINAL_MANIFEST.json"],
            },
        ]
    print(
        json.dumps(
            {
                "contract": contract.name,
                "kind": contract.kind.value,
                "model": dataclasses.asdict(contract.model),
                "workers": contract.engine.workers,
                "tensor_parallel_size": contract.model.tensor_parallel_size,
                "concurrent_gpus": (
                    contract.engine.workers * contract.model.tensor_parallel_size
                ),
                "batching": contract.batching.value,
                "repeat_seeds": list(contract.repeat_seeds),
                "population_candidates": (
                    len(contract_population(contract))
                    if contract.kind in {
                        ExperimentKind.SEARCH,
                        ExperimentKind.CANDIDATE_REPEAT,
                    }
                    else 0
                ),
                "expected_outputs": _planned_outputs(contract),
                "requires": requires,
                "phases": phases,
            },
            indent=2,
        )
    )


def _prepare(contract: ExperimentContract, args: argparse.Namespace) -> None:
    contract.output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = contract.output_dir / "contract.json"
    serialized = _serialized_contract(contract)
    if contract_path.exists():
        assert json.loads(contract_path.read_text(encoding="utf-8")) == serialized, (
            "contract drift in an existing run directory",
            contract_path,
        )
    else:
        assert not any(contract.output_dir.iterdir()), (
            "refusing to initialize a nonempty run directory without contract.json",
            contract.output_dir,
        )
        _atomic_json(contract_path, serialized)
    if contract.kind in {ExperimentKind.SEARCH, ExperimentKind.CANDIDATE_REPEAT}:
        population = paper_population(
            contract.population_size,
            population_size=contract.population_draw_size,
            seed=contract.population_seed,
            sigmas=contract.sigma_ladder,
        )
        population_path = contract.output_dir / "population.json"
        serialized_population = _jsonable(population)
        if population_path.exists():
            assert json.loads(population_path.read_text(encoding="utf-8")) == serialized_population
        else:
            _atomic_json(population_path, serialized_population)


def _search(contract: ExperimentContract, args: argparse.Namespace) -> None:
    assert contract.kind is ExperimentKind.SEARCH
    _require_prepared(contract)
    _require_preflight(contract)
    _verify_runtime(contract, args.allow_runtime_drift)
    examples = load_examples(contract.dataset, args.cache_dir)
    rows = examples["selection"]
    if contract.search_scope is SearchScope.FULL_GRID:
        rows = examples["evaluation"] or rows + examples["test"]
    candidates = contract_population(contract)
    assert 0 <= args.worker_index < contract.engine.workers
    candidates = tuple(
        candidate
        for candidate in candidates
        if candidate.index % contract.engine.workers == args.worker_index
    )
    stop = len(candidates) if args.candidate_stop is None else args.candidate_stop
    assert 0 <= args.candidate_start < stop <= len(candidates)
    engine, tokenizer = _engine(contract, perturb=True)
    initialization = engine.collective_rpc(
        "initialize_perturbations",
        args=(contract.perturbation_semantics.value, contract.anchor_storage.value),
    )
    _atomic_json(
        contract.output_dir / "worker_state" / f"search_worker_{args.worker_index}.json",
        initialization,
    )
    prompts, input_manifest = _prepare_inputs(
        contract,
        contract.dataset,
        rows,
        tokenizer,
        label="search",
    )
    output = contract.output_dir / "search"
    output.mkdir(parents=True, exist_ok=True)
    completed = []
    for cycle_index, candidate in enumerate(
        candidates[args.candidate_start:stop],
        start=args.candidate_start,
    ):
        path = output / f"candidate_{candidate.index:05d}.jsonl.gz"
        if path.exists():
            _validate_shard(path, rows, samples=1, candidate_id=candidate.index)
            completed.append(path)
            continue
        engine.collective_rpc("perturb", args=(candidate.index, candidate.seed, candidate.sigma))
        generated = _generate(
            engine,
            rows,
            contract,
            prompts=prompts,
            input_manifest=input_manifest,
            samples=1,
            metadata={
                "phase": "search",
                "split": (
                    "evaluation"
                    if contract.search_scope is SearchScope.FULL_GRID
                    else "selection"
                ),
                "worker_index": args.worker_index,
                "cycle_index": cycle_index,
            },
        )
        engine.collective_rpc("restore")
        _atomic_jsonl(
            path,
            _score_rows(contract, rows, generated, candidate=_jsonable(candidate)),
        )
        _validate_shard(path, rows, samples=1, candidate_id=candidate.index)
        completed.append(path)
    _verify_exact_anchor(engine, initialization, contract)
    completion_id = (
        f"search_worker_{args.worker_index}_candidates_"
        f"{args.candidate_start}_{stop}"
    )
    _completion(
        contract,
        "search",
        args.worker_index,
        completed,
        completion_id=completion_id,
    )


def _preflight(contract: ExperimentContract, args: argparse.Namespace) -> None:
    _require_prepared(contract)
    if contract.kind is ExperimentKind.LABEL_SHUFFLE:
        _external_dependencies(contract)
    _verify_runtime(contract, args.allow_runtime_drift)
    from transformers import AutoConfig, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        contract.model.model_id,
        revision=contract.model.revision,
    )
    config = AutoConfig.from_pretrained(
        contract.model.model_id,
        revision=contract.model.revision,
    )
    native_limit = int(getattr(config, "max_position_embeddings"))
    model_limit = contract.engine.max_model_len or native_limit
    datasets = []
    total_questions = 0
    for dataset in (contract.dataset, *contract.additional_datasets):
        loaded = load_examples(dataset, args.cache_dir)
        if contract.kind is ExperimentKind.SEARCH:
            examples = loaded["selection"]
            if contract.search_scope is SearchScope.FULL_GRID:
                examples = loaded["evaluation"] or examples + loaded["test"]
        else:
            examples = loaded["evaluation"] or loaded["test"]
        if contract.kind is ExperimentKind.LABEL_SHUFFLE:
            examples, _ = _shuffle_uspto(examples, contract.shuffle_seed)
        prompt_rows = []
        for example in examples:
            prompt = tokenizer.apply_chat_template(
                list(example.messages),
                tokenize=False,
                add_generation_prompt=True,
            )
            token_count = len(tokenizer.encode(prompt, add_special_tokens=False))
            assert token_count + dataset.max_tokens <= model_limit, (
                dataset.name,
                example.qid,
                token_count,
                dataset.max_tokens,
                model_limit,
            )
            prompt_rows.append(
                {
                    "qid": example.qid,
                    "tokens": token_count,
                    "sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                }
            )
        total_questions += len(examples)
        datasets.append(
            {
                "name": dataset.name,
                "task": dataset.key,
                "questions": len(examples),
                "max_new_tokens": dataset.max_tokens,
                "prompts": prompt_rows,
            }
        )
    required_gpus = contract.engine.workers * contract.model.tensor_parallel_size
    try:
        import torch

        available_gpus = torch.cuda.device_count()
    except ImportError:
        available_gpus = 0
    if not args.allow_no_gpu_preflight:
        assert available_gpus >= required_gpus, (available_gpus, required_gpus)
    if contract.kind is ExperimentKind.SEARCH:
        population = len(contract_population(contract))
        files = {"search": population}
        expected_rows = {"search": population * total_questions}
        if contract.search_scope is SearchScope.SELECT_THEN_TEST:
            test_questions = len(load_examples(contract.dataset, args.cache_dir)["test"])
            files["test"] = contract.top_k
            expected_rows["test"] = contract.top_k * test_questions
    elif contract.kind is ExperimentKind.ANCHOR:
        files = {
            "anchor": len(contract.repeat_seeds) * len(datasets) * contract.engine.workers
        }
        expected_rows = {
            "anchor": len(contract.repeat_seeds)
            * contract.samples_per_question
            * total_questions
        }
    elif contract.kind is ExperimentKind.CANDIDATE_REPEAT:
        files = {"candidate-repeat": len(contract.repeat_seeds)}
        expected_rows = {"candidate-repeat": len(contract.repeat_seeds) * total_questions}
    else:
        files = {"shuffle": (contract.top_k + 1) * contract.engine.workers}
        expected_rows = {"shuffle": (contract.top_k + 1) * total_questions}
    mbpp_questions = next(
        (
            len(load_examples(dataset, args.cache_dir)["evaluation"])
            or len(load_examples(dataset, args.cache_dir)["test"])
            for dataset in (contract.dataset, *contract.additional_datasets)
            if dataset.key == "mbpp"
        ),
        0,
    )
    if mbpp_questions:
        if contract.kind is ExperimentKind.SEARCH:
            scored_files = files["search"] + files.get("test", 0)
            scored_rows = expected_rows["search"] + expected_rows.get("test", 0)
        elif contract.kind is ExperimentKind.ANCHOR:
            scored_files = len(contract.repeat_seeds) * contract.engine.workers
            scored_rows = (
                len(contract.repeat_seeds) * contract.samples_per_question * mbpp_questions
            )
        else:
            assert contract.kind is ExperimentKind.CANDIDATE_REPEAT
            scored_files = files["candidate-repeat"]
            scored_rows = expected_rows["candidate-repeat"]
        files["score"] = scored_files
        expected_rows["score"] = scored_rows
    planned = _planned_outputs(contract)
    assert files == {phase: values["files"] for phase, values in planned.items()}
    assert expected_rows == {phase: values["rows"] for phase, values in planned.items()}
    _atomic_json(
        contract.output_dir / "preflight.json",
        {
            "contract": contract.name,
            "model": dataclasses.asdict(contract.model),
            "native_model_limit": native_limit,
            "effective_model_limit": model_limit,
            "datasets": datasets,
            "expected_output_files": files,
            "expected_rows": expected_rows,
            "required_gpus": required_gpus,
            "available_gpus": available_gpus,
            "environment": _environment_info(),
        },
    )


def _planned_outputs(contract: ExperimentContract) -> dict[str, dict[str, int]]:
    generated = {
        dataset.key: len(dataset.evaluation_qids or dataset.test_qids)
        for dataset in (contract.dataset, *contract.additional_datasets)
    }
    outputs: dict[str, dict[str, int]] = {}
    if contract.kind is ExperimentKind.SEARCH:
        candidates = len(contract_population(contract))
        search_questions = len(contract.dataset.selection_qids)
        if contract.search_scope is SearchScope.FULL_GRID:
            search_questions = len(contract.dataset.evaluation_qids) or (
                len(contract.dataset.selection_qids) + len(contract.dataset.test_qids)
            )
        outputs["search"] = {
            "files": candidates,
            "rows": candidates * search_questions,
        }
        if contract.search_scope is SearchScope.SELECT_THEN_TEST:
            outputs["test"] = {
                "files": contract.top_k,
                "rows": contract.top_k * len(contract.dataset.test_qids),
            }
    elif contract.kind is ExperimentKind.ANCHOR:
        outputs["anchor"] = {
            "files": (
                len(contract.repeat_seeds)
                * len(generated)
                * contract.engine.workers
            ),
            "rows": (
                len(contract.repeat_seeds)
                * contract.samples_per_question
                * sum(generated.values())
            ),
        }
    elif contract.kind is ExperimentKind.CANDIDATE_REPEAT:
        outputs["candidate-repeat"] = {
            "files": len(contract.repeat_seeds),
            "rows": len(contract.repeat_seeds) * generated[contract.dataset.key],
        }
    else:
        assert contract.kind is ExperimentKind.LABEL_SHUFFLE
        outputs["shuffle"] = {
            "files": (contract.top_k + 1) * contract.engine.workers,
            "rows": (contract.top_k + 1) * generated[contract.dataset.key],
        }
    if "mbpp" in generated:
        if contract.kind is ExperimentKind.SEARCH:
            files = outputs["search"]["files"] + outputs.get("test", {}).get("files", 0)
            rows = outputs["search"]["rows"] + outputs.get("test", {}).get("rows", 0)
        elif contract.kind is ExperimentKind.ANCHOR:
            files = len(contract.repeat_seeds) * contract.engine.workers
            rows = (
                len(contract.repeat_seeds)
                * contract.samples_per_question
                * generated["mbpp"]
            )
        else:
            assert contract.kind is ExperimentKind.CANDIDATE_REPEAT
            files = outputs["candidate-repeat"]["files"]
            rows = outputs["candidate-repeat"]["rows"]
        outputs["score"] = {"files": files, "rows": rows}
    return outputs


def _score(contract: ExperimentContract, args: argparse.Namespace) -> None:
    _require_prepared(contract)
    assert args.score_workers > 0
    assert os.environ.get("PYTHONHASHSEED") == "0", (
        "deterministic MBPP rescoring requires PYTHONHASHSEED=0 before Python starts"
    )
    raw_paths = []
    for path in contract.output_dir.rglob("*.jsonl.gz"):
        if "scored" in path.relative_to(contract.output_dir).parts:
            continue
        first = _first_jsonl(path)
        if first and first.get("scored") is False and first["dataset"] == "mbpp":
            raw_paths.append(path)
    assert raw_paths, "contract has no unscored MBPP shards"
    todo = [
        path
        for path in sorted(raw_paths)
        if not (contract.output_dir / "scored" / path.relative_to(contract.output_dir)).exists()
    ]
    manifest_paths = {
        str(_first_jsonl(path)["input_manifest"])
        for path in todo
    }
    input_cache = {
        relative: _input_lookup(contract, relative)
        for relative in manifest_paths
    }
    with ThreadPoolExecutor(max_workers=args.score_workers) as executor:
        score_file = partial(
            _score_mbpp_file,
            contract,
            input_cache=input_cache,
            unsafe_allow_unisolated=args.unsafe_mbpp_scoring,
        )
        list(executor.map(score_file, todo))
    scored_paths = [
        contract.output_dir / "scored" / path.relative_to(contract.output_dir)
        for path in raw_paths
    ]
    assert all(path.is_file() for path in scored_paths)
    _completion(contract, "score", 0, scored_paths)


def _score_mbpp_file(
    contract: ExperimentContract,
    path: Path,
    *,
    input_cache: dict[str, dict[int, dict[str, object]]],
    unsafe_allow_unisolated: bool,
) -> None:
    rows = _read_jsonl(path)
    assert rows and all(row["scored"] is False for row in rows)
    inputs = input_cache[str(rows[0]["input_manifest"])]
    cases = [
        (
            str(row["generated_text"]),
            inputs[int(row["qid"])]["gold"]["test_list"],
            inputs[int(row["qid"])]["gold"]["test_setup_code"],
        )
        for row in rows
    ]
    outcomes = score_mbpp_batch_isolated(
        cases,
        unsafe_allow_unisolated=unsafe_allow_unisolated,
    )
    scored = [
        _scored_row(contract, row, outcome, row.get("candidate"), {})
        for row, outcome in zip(rows, outcomes)
    ]
    destination = contract.output_dir / "scored" / path.relative_to(contract.output_dir)
    _atomic_jsonl(destination, scored)


def _select(contract: ExperimentContract, args: argparse.Namespace) -> None:
    _require_prepared(contract)
    assert contract.kind is ExperimentKind.SEARCH
    assert contract.search_scope is SearchScope.SELECT_THEN_TEST
    files = sorted(_result_dir(contract, "search").glob("candidate_*.jsonl.gz"))
    expected = len(contract_population(contract))
    assert len(files) == expected, (len(files), expected)
    scores = []
    metric = contract.selection_metric
    assert metric is not None
    for path in files:
        rows = _read_jsonl(path)
        assert len(rows) == len(contract.dataset.selection_qids)
        index = int(rows[0]["candidate"]["index"])
        scores.append((sum(float(row[metric]) for row in rows) / len(rows), index))
    scores.sort(key=lambda item: (-item[0], item[1]))
    if contract.canonical_selected_indices:
        score_by_index = {index: score for score, index in scores}
        computed = {index for _, index in scores[: contract.top_k]}
        assert computed == set(contract.canonical_selected_indices), (
            "canonical selection set changed",
            computed ^ set(contract.canonical_selected_indices),
        )
        scores = [
            (score_by_index[index], index)
            for index in contract.canonical_selected_indices
        ]
    selected = [
        {"rank": rank, "score": score, "index": index}
        for rank, (score, index) in enumerate(scores[: contract.top_k], 1)
    ]
    _atomic_json(contract.output_dir / "selected.json", selected)


def _test(contract: ExperimentContract, args: argparse.Namespace) -> None:
    _require_prepared(contract)
    _require_preflight(contract)
    assert contract.kind is ExperimentKind.SEARCH
    assert contract.search_scope is SearchScope.SELECT_THEN_TEST
    _verify_runtime(contract, args.allow_runtime_drift)
    selected = json.loads((contract.output_dir / "selected.json").read_text(encoding="utf-8"))
    assert len(selected) == contract.top_k
    lookup = {candidate.index: candidate for candidate in contract_population(contract)}
    examples = load_examples(contract.dataset, args.cache_dir)["test"]
    engine, tokenizer = _engine(contract, perturb=True)
    initialization = engine.collective_rpc(
        "initialize_perturbations",
        args=(contract.perturbation_semantics.value, contract.anchor_storage.value),
    )
    _atomic_json(
        contract.output_dir / "worker_state" / f"test_worker_{args.worker_index}.json",
        initialization,
    )
    prompts, input_manifest = _prepare_inputs(
        contract,
        contract.dataset,
        examples,
        tokenizer,
        label="test",
    )
    output = contract.output_dir / "test"
    output.mkdir(parents=True, exist_ok=True)
    assert 0 <= args.worker_index < contract.engine.workers
    assigned = [
        row
        for row in selected
        if (int(row["rank"]) - 1) % contract.engine.workers == args.worker_index
    ]
    completed = []
    for selected_row in assigned:
        candidate = lookup[int(selected_row["index"])]
        filename = (
            f"rank_{int(selected_row['rank']):03d}_candidate_{candidate.index:05d}.jsonl.gz"
        )
        path = output / filename
        if path.exists():
            _validate_shard(path, examples, samples=1, candidate_id=candidate.index)
            completed.append(path)
            continue
        engine.collective_rpc("perturb", args=(candidate.index, candidate.seed, candidate.sigma))
        generated = _generate(
            engine,
            examples,
            contract,
            prompts=prompts,
            input_manifest=input_manifest,
            samples=1,
            metadata={
                "phase": "test",
                "split": "test",
                "worker_index": args.worker_index,
                "cycle_index": (int(selected_row["rank"]) - 1) // contract.engine.workers,
            },
        )
        engine.collective_rpc("restore")
        _atomic_jsonl(
            path,
            _score_rows(contract, examples, generated, candidate=_jsonable(candidate)),
        )
        _validate_shard(path, examples, samples=1, candidate_id=candidate.index)
        completed.append(path)
    _verify_exact_anchor(engine, initialization, contract)
    _completion(contract, "test", args.worker_index, completed)


def _anchor(contract: ExperimentContract, args: argparse.Namespace) -> None:
    _require_prepared(contract)
    _require_preflight(contract)
    assert contract.kind is ExperimentKind.ANCHOR
    assert args.repeat_seed in contract.repeat_seeds
    _verify_runtime(contract, args.allow_runtime_drift)
    engine, tokenizer = _engine(contract, perturb=False, seed=args.repeat_seed)
    initialization = engine.collective_rpc("initialize_anchor_audit")
    _atomic_json(
        contract.output_dir
        / "worker_state"
        / f"anchor_seed_{args.repeat_seed}_worker_{args.worker_index}.json",
        initialization,
    )
    suffix = f"_worker_{args.worker_index}" if contract.engine.workers > 1 else ""
    completed = []
    for dataset in (contract.dataset, *contract.additional_datasets):
        loaded = load_examples(dataset, args.cache_dir)
        examples = loaded["evaluation"] or loaded["test"]
        examples = _question_shard(contract, examples, args.worker_index)
        path = contract.output_dir / (
            f"anchor_{dataset.key}_seed_{args.repeat_seed}{suffix}.jsonl.gz"
        )
        if path.exists():
            _validate_shard(path, examples, samples=contract.samples_per_question)
            completed.append(path)
            continue
        prompts, input_manifest = _prepare_inputs(
            contract,
            dataset,
            examples,
            tokenizer,
            label=f"anchor_w{args.worker_index}",
        )
        generated = _generate(
            engine,
            examples,
            contract,
            dataset=dataset,
            prompts=prompts,
            input_manifest=input_manifest,
            samples=contract.samples_per_question,
            seed=args.repeat_seed,
            metadata={
                "phase": "anchor",
                "split": "test",
                "worker_index": args.worker_index,
                "cycle_index": 0,
            },
        )
        _atomic_jsonl(path, _score_rows(contract, examples, generated, dataset=dataset))
        _validate_shard(path, examples, samples=contract.samples_per_question)
        completed.append(path)
    _completion(
        contract,
        "anchor",
        args.worker_index,
        completed,
        completion_id=f"anchor_seed_{args.repeat_seed}_worker_{args.worker_index}",
    )


def _candidate_repeat(contract: ExperimentContract, args: argparse.Namespace) -> None:
    _require_prepared(contract)
    _require_preflight(contract)
    assert contract.kind is ExperimentKind.CANDIDATE_REPEAT
    assert args.repeat_seed in contract.repeat_seeds
    assert len(contract.candidate_indices) == 1
    _verify_runtime(contract, args.allow_runtime_drift)
    candidate = paper_population(
        contract.population_size,
        population_size=contract.population_draw_size,
        seed=contract.population_seed,
        sigmas=contract.sigma_ladder,
    )[contract.candidate_indices[0]]
    examples = load_examples(contract.dataset, args.cache_dir)["test"]
    path = contract.output_dir / f"candidate_{candidate.index}_seed_{args.repeat_seed}.jsonl.gz"
    if path.exists():
        _validate_shard(path, examples, samples=1, candidate_id=candidate.index)
        _completion(
            contract,
            "candidate-repeat",
            args.worker_index,
            [path],
            completion_id=(
                f"candidate_repeat_seed_{args.repeat_seed}_worker_{args.worker_index}"
            ),
        )
        return
    engine, tokenizer = _engine(contract, perturb=True, seed=args.repeat_seed)
    initialization = engine.collective_rpc(
        "initialize_perturbations",
        args=(contract.perturbation_semantics.value, contract.anchor_storage.value),
    )
    _atomic_json(contract.output_dir / "worker_state" / "candidate_repeat.json", initialization)
    prompts, input_manifest = _prepare_inputs(
        contract,
        contract.dataset,
        examples,
        tokenizer,
        label="repeat",
    )
    engine.collective_rpc("perturb", args=(candidate.index, candidate.seed, candidate.sigma))
    generated = _generate(
        engine,
        examples,
        contract,
        prompts=prompts,
        input_manifest=input_manifest,
        samples=1,
        seed=args.repeat_seed,
        metadata={
            "phase": "candidate_repeat",
            "split": "test",
            "worker_index": 0,
            "cycle_index": 0,
        },
    )
    engine.collective_rpc("restore", args=(True,))
    _atomic_jsonl(path, _score_rows(contract, examples, generated, candidate=_jsonable(candidate)))
    _validate_shard(path, examples, samples=1, candidate_id=candidate.index)
    _verify_exact_anchor(engine, initialization, contract)
    _completion(
        contract,
        "candidate-repeat",
        args.worker_index,
        [path],
        completion_id=f"candidate_repeat_seed_{args.repeat_seed}_worker_{args.worker_index}",
    )


def _shuffle(contract: ExperimentContract, args: argparse.Namespace) -> None:
    _require_prepared(contract)
    _require_preflight(contract)
    assert contract.kind is ExperimentKind.LABEL_SHUFFLE
    assert args.repeat_seed in contract.repeat_seeds
    _verify_runtime(contract, args.allow_runtime_drift)
    source = load_contract(contract.source_search, output_dir=args.output_dir)
    selected = json.loads((source.output_dir / "selected.json").read_text(encoding="utf-8"))
    selected = selected[: contract.top_k]
    lookup = {candidate.index: candidate for candidate in contract_population(source)}
    examples = load_examples(contract.dataset, args.cache_dir)["test"]
    shuffled, permutations = _shuffle_uspto(examples, contract.shuffle_seed)
    permutation_path = contract.output_dir / "permutations.json"
    if permutation_path.exists():
        assert json.loads(permutation_path.read_text(encoding="utf-8")) == permutations
    else:
        _atomic_json(permutation_path, permutations)
    shuffled = _question_shard(contract, shuffled, args.worker_index)
    engine, tokenizer = _engine(contract, perturb=True, seed=args.repeat_seed)
    initialization = engine.collective_rpc(
        "initialize_perturbations",
        args=(contract.perturbation_semantics.value, contract.anchor_storage.value),
    )
    _atomic_json(
        contract.output_dir / "worker_state" / f"shuffle_worker_{args.worker_index}.json",
        initialization,
    )
    output = contract.output_dir
    prompts, input_manifest = _prepare_inputs(
        contract,
        contract.dataset,
        shuffled,
        tokenizer,
        label=f"shuffle_w{args.worker_index}",
    )
    anchor_path = output / f"anchor_worker_{args.worker_index}.jsonl.gz"
    if anchor_path.exists():
        _validate_shard(anchor_path, shuffled, samples=1)
    else:
        generated = _generate(
            engine,
            shuffled,
            contract,
            prompts=prompts,
            input_manifest=input_manifest,
            samples=1,
            seed=args.repeat_seed,
            metadata={
                "phase": "shuffle_anchor",
                "split": "test",
                "worker_index": args.worker_index,
                "cycle_index": 0,
            },
        )
        _atomic_jsonl(anchor_path, _score_rows(contract, shuffled, generated))
        _validate_shard(anchor_path, shuffled, samples=1)
    completed = [anchor_path]
    for row in selected:
        candidate = lookup[int(row["index"])]
        path = output / (
            f"rank_{int(row['rank']):03d}_candidate_{candidate.index:05d}"
            f"_worker_{args.worker_index}.jsonl.gz"
        )
        if path.exists():
            _validate_shard(path, shuffled, samples=1, candidate_id=candidate.index)
            completed.append(path)
            continue
        engine.collective_rpc("perturb", args=(candidate.index, candidate.seed, candidate.sigma))
        generated = _generate(
            engine,
            shuffled,
            contract,
            prompts=prompts,
            input_manifest=input_manifest,
            samples=1,
            seed=args.repeat_seed,
            metadata={
                "phase": "shuffle_expert",
                "split": "test",
                "worker_index": args.worker_index,
                "cycle_index": int(row["rank"]),
            },
        )
        engine.collective_rpc("restore")
        _atomic_jsonl(
            path,
            _score_rows(contract, shuffled, generated, candidate=_jsonable(candidate)),
        )
        _validate_shard(path, shuffled, samples=1, candidate_id=candidate.index)
        completed.append(path)
    _verify_exact_anchor(engine, initialization, contract)
    _completion(contract, "shuffle", args.worker_index, completed)


def _engine(
    contract: ExperimentContract,
    *,
    perturb: bool,
    seed: int | None = None,
) -> tuple[Any, Any]:
    from transformers import AutoTokenizer
    from vllm import LLM

    engine = contract.engine
    kwargs = {
        "model": contract.model.model_id,
        "tokenizer": contract.model.model_id,
        "revision": contract.model.revision,
        "tokenizer_revision": contract.model.revision,
        "dtype": contract.dtype,
        "tensor_parallel_size": contract.model.tensor_parallel_size,
        "seed": contract.decode.seed if seed is None else seed,
        "enforce_eager": engine.enforce_eager,
        "enable_prefix_caching": engine.enable_prefix_caching,
        "gpu_memory_utilization": engine.gpu_memory_utilization,
    }
    for key in ("max_model_len", "max_num_seqs", "max_num_batched_tokens"):
        value = getattr(engine, key)
        if value is not None:
            kwargs[key] = value
    kwargs["worker_extension_cls"] = "neural_thickets.experiments.worker.PerturbationWorker"
    tokenizer = AutoTokenizer.from_pretrained(
        contract.model.model_id,
        revision=contract.model.revision,
    )
    return LLM(**kwargs), tokenizer


def _generate(
    engine: Any,
    examples: tuple[Example, ...],
    contract: ExperimentContract,
    *,
    prompts: tuple[str, ...],
    input_manifest: str,
    metadata: dict[str, object],
    dataset: DatasetSpec | None = None,
    samples: int,
    seed: int | None = None,
) -> list[dict[str, object]]:
    from vllm import SamplingParams

    dataset = contract.dataset if dataset is None else dataset
    assert len(prompts) == len(examples)
    assert len(prompts) == len(set(prompts))
    decode = contract.decode
    params = SamplingParams(
        n=samples,
        temperature=decode.temperature,
        top_p=decode.top_p,
        top_k=decode.top_k,
        repetition_penalty=decode.repetition_penalty,
        presence_penalty=decode.presence_penalty,
        frequency_penalty=decode.frequency_penalty,
        seed=decode.seed if seed is None else seed,
        max_tokens=dataset.max_tokens,
    )
    rows = []
    batch_size = (
        contract.engine.prompt_batch_size
        if contract.batching is BatchingStrategy.FIXED_CHUNKS
        else len(examples)
    )
    for start in range(0, len(examples), batch_size):
        batch_examples = examples[start : start + batch_size]
        batch_prompts = prompts[start : start + batch_size]
        outputs = engine.generate(batch_prompts, params, use_tqdm=False)
        for example, prompt, result in zip(batch_examples, batch_prompts, outputs):
            assert result.prompt == prompt
            for sample_index, completion in enumerate(result.outputs):
                rows.append(
                    {
                        "qid": example.qid,
                        "dataset": dataset.key,
                        "sample_index": sample_index,
                        "generation_seed": decode.seed if seed is None else seed,
                        "input_manifest": input_manifest,
                        "generated_text": completion.text,
                        "generated_token_ids": list(completion.token_ids),
                        "finish_reason": completion.finish_reason,
                        "stop_reason": completion.stop_reason,
                        **metadata,
                    }
                )
    assert len(rows) == len(examples) * samples
    return rows


def _score_rows(
    contract: ExperimentContract,
    examples: tuple[Example, ...],
    generated: list[dict[str, object]],
    *,
    dataset: DatasetSpec | None = None,
    candidate: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    dataset = contract.dataset if dataset is None else dataset
    by_qid = {example.qid: example for example in examples}
    if dataset.key == "mbpp":
        return [
            {
                **raw,
                "answer": extract_mbpp(str(raw["generated_text"])),
                "valid": bool(extract_mbpp(str(raw["generated_text"]))),
                "correct": None,
                "reward": None,
                "answer_reward": None,
                "format_reward": None,
                "reference_reward": None,
                "scored": False,
                "scoring_status": "deferred",
                "scoring_reason": "requires isolated offline MBPP execution",
                "contract": contract.name,
                "model": contract.model.model_id,
                "model_revision": contract.model.revision,
                "candidate": candidate,
            }
            for raw in generated
        ]
    rows = []
    for raw in generated:
        example = by_qid[int(raw["qid"])]
        response = str(raw["generated_text"])
        task = dataset.key
        if task == "countdown":
            historical = score_countdown(
                response,
                **example.gold,
                version="historical_d409",
            )
            outcome = score_countdown(response, **example.gold, version="reference_536")
            extra = {
                "historical_d409_reward": historical.reward,
                "historical_d409_answer_reward": historical.answer_reward,
                "historical_d409_format_reward": historical.format_reward,
                "reference_536_answer_reward": outcome.answer_reward,
                "reference_536_format_reward": outcome.format_reward,
                "reference_reward": outcome.reward,
            }
        elif task == "gsm8k":
            outcome = score_gsm8k(response, str(example.gold))
            extra = {}
        elif task == "uspto50k":
            outcome = score_uspto50k(response, str(example.gold))
            extra = {}
        else:
            raise AssertionError(task)
        rows.append(_scored_row(contract, raw, outcome, candidate, extra))
    return rows


def _scored_row(
    contract: ExperimentContract,
    raw: dict[str, object],
    outcome: Outcome,
    candidate: dict[str, object] | None,
    extra: dict[str, object],
) -> dict[str, object]:
    return {
        **raw,
        **dataclasses.asdict(outcome),
        "reference_reward": outcome.reward,
        **extra,
        "contract": contract.name,
        "model": contract.model.model_id,
        "model_revision": contract.model.revision,
        "candidate": candidate,
        "scored": True,
    }


def _shuffle_uspto(
    examples: tuple[Example, ...],
    seed: int,
) -> tuple[tuple[Example, ...], dict[str, object]]:
    rng = np.random.default_rng(seed)
    output = []
    records = []
    for example in examples:
        permutation = (rng.permutation(10) + 1).tolist()
        system = example.messages[0]["content"]
        head, body = system.split("## Reaction Classes:\n")
        descriptions = [line.split(": ", 1)[1] for line in body.splitlines()]
        assert len(descriptions) == 10
        lines = sorted(
            (permutation[index], description)
            for index, description in enumerate(descriptions)
        )
        permuted_system = head + "## Reaction Classes:\n" + "\n".join(
            f"{number}: {description}" for number, description in lines
        )
        gold = str(permutation[int(example.gold) - 1])
        records.append(
            {
                "qid": example.qid,
                "permutation": permutation,
                "original_class": int(example.gold),
                "shuffled_class": int(gold),
            }
        )
        output.append(
            Example(
                qid=example.qid,
                messages=(
                    {"role": "system", "content": permuted_system},
                    dict(example.messages[1]),
                ),
                gold=gold,
            )
        )
    return tuple(output), {
        "seed": seed,
        "convention": "permutation[c-1] is the number shown for original class c",
        "rows": records,
    }


def _summarize(contract: ExperimentContract, args: argparse.Namespace) -> None:
    _require_prepared(contract)
    _require_preflight(contract)
    output = contract.output_dir / "summary"
    output.mkdir(parents=True, exist_ok=True)
    if contract.kind is ExperimentKind.SEARCH:
        _summarize_search(contract, output, args.bootstrap_seed)
    elif contract.kind is ExperimentKind.ANCHOR:
        _summarize_anchor(contract, output)
    elif contract.kind is ExperimentKind.LABEL_SHUFFLE:
        _summarize_uspto_shuffle(contract, output)
    else:
        root = (
            contract.output_dir / "scored"
            if contract.dataset.key == "mbpp"
            else contract.output_dir
        )
        files = sorted(path for path in root.glob("*.jsonl.gz") if path.is_file())
        assert files
        summaries = []
        for path in files:
            rows = _read_jsonl(path)
            summaries.append(
                {
                    "file": path.name,
                    "rows": len(rows),
                    "accuracy": float(np.mean([row["correct"] for row in rows])),
                    "valid_rate": float(np.mean([row["valid"] for row in rows])),
                }
            )
        _atomic_json(output / "summary.json", summaries)
    _finalize_manifest(contract)


def _summarize_uspto_shuffle(contract: ExperimentContract, output: Path) -> None:
    assert contract.dataset.key == "uspto50k"
    run_root = contract.output_dir.parent
    source = load_contract(contract.source_search, output_dir=run_root)
    normal_files = sorted(_result_dir(source, "test").glob("rank_*.jsonl.gz"))
    assert len(normal_files) == source.top_k
    normal_experts = [_read_jsonl(path) for path in normal_files[:20]]
    normal_anchor_root = run_root / "uspto_anchor_original"
    normal_anchor_files = sorted(normal_anchor_root.glob("anchor_uspto50k_seed_42_*.jsonl.gz"))
    assert len(normal_anchor_files) == 2
    normal_anchor = [row for path in normal_anchor_files for row in _read_jsonl(path)]

    shuffled_anchor_files = sorted(contract.output_dir.glob("anchor_worker_*.jsonl.gz"))
    assert len(shuffled_anchor_files) == contract.engine.workers
    shuffled_anchor = [row for path in shuffled_anchor_files for row in _read_jsonl(path)]
    shuffled_experts = []
    for rank in range(1, 21):
        files = sorted(contract.output_dir.glob(f"rank_{rank:03d}_*_worker_*.jsonl.gz"))
        assert len(files) == contract.engine.workers
        shuffled_experts.append([row for path in files for row in _read_jsonl(path)])

    def row(experts, anchor):
        qids = [int(item["qid"]) for item in anchor]
        assert len(qids) == len(set(qids)) == 1001
        for expert in experts:
            assert [int(item["qid"]) for item in expert] == qids
        anchor_accuracy = float(np.mean([item["correct"] for item in anchor]))
        values = {}
        for k in (1, 5, 10, 20):
            accuracy = float(
                np.mean([item["correct"] for expert in experts[:k] for item in expert])
            )
            values[str(k)] = 100 * (accuracy - anchor_accuracy)
        return {"anchor_accuracy": anchor_accuracy, "topk_minus_anchor_points": values}

    _atomic_json(
        output / "summary.json",
        {
            "contract": contract.name,
            "dependencies": _external_dependencies(contract),
            "normal": row(normal_experts, normal_anchor),
            "shuffled": row(shuffled_experts, shuffled_anchor),
        },
    )


def _summarize_search(contract: ExperimentContract, output: Path, bootstrap_seed: int) -> None:
    files = sorted(_result_dir(contract, "search").glob("candidate_*.jsonl.gz"))
    candidates = contract_population(contract)
    assert len(files) == len(candidates), (len(files), len(candidates))
    qids = np.array([int(row["qid"]) for row in _iter_jsonl(files[0])], dtype=np.int64)
    assert len(qids) == len(set(qids.tolist()))
    shape = (len(files), len(qids))
    rewards = np.empty(shape, dtype=np.float64)
    correct = np.empty(shape, dtype=np.bool_)
    valid = np.empty(shape, dtype=np.bool_)
    candidate_ids = np.empty(len(files), dtype=np.int32)
    for candidate_row, path in enumerate(files):
        candidate_id = None
        count = 0
        for question, row in enumerate(_iter_jsonl(path)):
            assert question < len(qids)
            assert int(row["qid"]) == int(qids[question])
            row_candidate = int(row["candidate"]["index"])
            candidate_id = row_candidate if candidate_id is None else candidate_id
            assert candidate_id == row_candidate
            rewards[candidate_row, question] = float(row["reference_reward"])
            correct[candidate_row, question] = bool(row["correct"])
            valid[candidate_row, question] = bool(row["valid"])
            count += 1
        assert count == len(qids)
        assert candidate_id is not None
        candidate_ids[candidate_row] = candidate_id
    lookup = {candidate.index: candidate for candidate in candidates}
    seeds = np.array([lookup[int(index)].seed for index in candidate_ids], dtype=np.int64)
    sigmas = np.array([lookup[int(index)].sigma for index in candidate_ids], dtype=np.float64)
    _atomic_npz(
        output / "candidate_matrix.npz",
        candidate_ids=candidate_ids,
        candidate_seeds=seeds,
        candidate_sigmas=sigmas,
        qids=qids,
        reference_reward=rewards,
        correct=correct,
        valid=valid,
    )

    summary_qids = _summary_qids(contract, contract.dataset)
    summary_columns = np.array([qid in summary_qids for qid in qids])
    assert summary_columns.sum() == len(summary_qids)
    selection_qids = set(contract.dataset.selection_qids[:200])
    selection_columns = np.array([qid in selection_qids for qid in qids])
    if selection_columns.sum() != 200:
        _atomic_json(
            output / "summary.json",
            {
                "contract": contract.name,
                "candidate_count": len(candidate_ids),
                "generated_questions": len(qids),
                "summary_questions": len(summary_qids),
                "mean_accuracy_by_candidate": correct[:, summary_columns].mean(axis=1).tolist(),
                "mean_reference_reward_by_candidate": (
                    rewards[:, summary_columns].mean(axis=1).tolist()
                ),
                "selection_ranking": None,
                "note": "Evaluation-only fixed-population contract; no selection ranking applied.",
            },
        )
        return
    ranking_rows = np.argsort(-rewards[:, selection_columns].mean(axis=1), kind="stable")
    recomputed_ranking_ids = candidate_ids[ranking_rows]
    ranking_ids = _reported_selection_ranking(
        recomputed_ranking_ids,
        contract.canonical_selected_indices,
    )
    test_qids = summary_qids
    if contract.search_scope is SearchScope.FULL_GRID:
        test_columns = np.array([qid in test_qids for qid in qids])
        assert test_columns.sum() == len(test_qids)
        selected_rows = ranking_rows[: min(50, len(ranking_rows))]
        test_correct = correct[selected_rows][:, test_columns]
        test_answers = []
        for index in selected_rows:
            selected = _filter_rows(files[index], test_qids)
            assert len(selected) == len(test_qids)
            test_answers.append([str(row["answer"]) for row in selected])
    else:
        selected_rows_data = _selected_test_rows(contract, limit=50)
        assert selected_rows_data
        selected_qids = [int(row["qid"]) for row in selected_rows_data[0]]
        keep = [qid in test_qids for qid in selected_qids]
        assert sum(keep) == len(test_qids)
        test_correct = np.array(
            [
                [bool(row["correct"]) for row, use in zip(rows, keep) if use]
                for rows in selected_rows_data
            ],
            dtype=np.bool_,
        )
        test_answers = [
            [str(row["answer"]) for row, use in zip(rows, keep) if use]
            for rows in selected_rows_data
        ]
    vote_curve, vote_orderings = _vote_curve(
        test_answers,
        test_correct,
        budgets=contract.aggregation_budgets,
        strategy=None,
    )
    rng = np.random.default_rng(bootstrap_seed)
    top50_correct = test_correct[:50]
    contrast = top50_correct[0].astype(float) - top50_correct.mean(axis=0)
    draws = contrast[rng.integers(0, contrast.size, size=(2000, contrast.size))].mean(axis=1)
    summary = {
        "contract": contract.name,
        "candidate_count": len(candidate_ids),
        "selection_questions": int(selection_columns.sum()),
        "test_questions": int(test_correct.shape[1]),
        "selection_ranking_candidate_ids": ranking_ids.tolist(),
        "recomputed_selection_ranking_candidate_ids": recomputed_ranking_ids.tolist(),
        "selection_order_source": (
            "canonical_production_tie_order"
            if contract.canonical_selected_indices
            else "recomputed_stable_sort"
        ),
        "top_candidate_test_accuracy": float(top50_correct[0].mean()),
        "top50_mean_test_accuracy": float(top50_correct.mean()),
        "vote_curve": vote_curve,
        "vote_orderings": vote_orderings,
        "bootstrap_top1_minus_selected_bank_mean": {
            "estimate": float(contrast.mean()),
            "interval_95": [float(value) for value in np.quantile(draws, [0.025, 0.975])],
            "resamples": 2000,
            "seed": bootstrap_seed,
        },
    }
    _atomic_json(output / "summary.json", summary)


def _summarize_anchor(contract: ExperimentContract, output: Path) -> None:
    summary: dict[str, object] = {"contract": contract.name, "datasets": {}}
    for dataset in (contract.dataset, *contract.additional_datasets):
        summary_qids = _summary_qids(contract, dataset)
        root = contract.output_dir / "scored" if dataset.key == "mbpp" else contract.output_dir
        files = sorted(root.glob(f"anchor_{dataset.key}_seed_*.jsonl.gz"))
        expected_files = len(contract.repeat_seeds) * contract.engine.workers
        assert len(files) == expected_files, (dataset.key, len(files), expected_files)
        seed_totals: dict[int, list[int]] = {}
        for path in files:
            for row in _iter_jsonl(path):
                if int(row["qid"]) not in summary_qids:
                    continue
                total = seed_totals.setdefault(int(row["generation_seed"]), [0, 0])
                total[0] += 1
                total[1] += int(bool(row["correct"]))
        per_seed = [
            {"seed": seed, "rows": total, "accuracy": correct / total}
            for seed, (total, correct) in sorted(seed_totals.items())
        ]
        assert len(per_seed) == len(contract.repeat_seeds)
        assert all(
            row["rows"] == len(summary_qids) * contract.samples_per_question
            for row in per_seed
        )
        dataset_summary: dict[str, object] = {"per_seed": per_seed}
        if contract.samples_per_question > 1 and contract.aggregation_orderings:
            compact_rows = [
                {
                    "qid": int(row["qid"]),
                    "sample_index": int(row["sample_index"]),
                    "answer": str(row["answer"]),
                    "correct": bool(row["correct"]),
                }
                for path in files
                for row in _iter_jsonl(path)
                if int(row["qid"]) in summary_qids
            ]
            qids = sorted({row["qid"] for row in compact_rows})
            answers = []
            correct = []
            for sample in range(contract.samples_per_question):
                sample_rows = {
                    int(row["qid"]): row
                    for row in compact_rows
                    if int(row["sample_index"]) == sample
                }
                assert set(sample_rows) == set(qids)
                answers.append([str(sample_rows[qid]["answer"]) for qid in qids])
                correct.append([bool(sample_rows[qid]["correct"]) for qid in qids])
            vote_curve, vote_orderings = _vote_curve(
                answers,
                np.array(correct, dtype=np.bool_),
                budgets=contract.aggregation_budgets,
                seed=contract.aggregation_seed,
                orderings=contract.aggregation_orderings,
                strategy=contract.aggregation_strategy,
            )
            dataset_summary["vote_curve"] = vote_curve
            dataset_summary["vote_orderings"] = vote_orderings
        elif contract.samples_per_question > 1:
            dataset_summary["voting_aggregation"] = (
                "Delegated to the paper analysis, which stores the exact "
                "per-cell voter order matrices and paired bootstrap plan."
            )
        summary["datasets"][dataset.key] = dataset_summary
    _atomic_json(output / "summary.json", summary)


def _selected_test_rows(
    contract: ExperimentContract,
    *,
    limit: int,
) -> list[list[dict[str, object]]]:
    files = sorted(_result_dir(contract, "test").glob("rank_*.jsonl.gz"))
    assert len(files) == contract.top_k, (len(files), contract.top_k)
    return [_read_jsonl(path) for path in files[:limit]]


def _reported_selection_ranking(
    recomputed: np.ndarray,
    canonical_prefix: tuple[int, ...],
) -> np.ndarray:
    if not canonical_prefix:
        return recomputed
    selected = set(canonical_prefix)
    remainder = [value for value in recomputed if int(value) not in selected]
    reported = np.asarray([*canonical_prefix, *remainder], dtype=recomputed.dtype)
    assert set(reported.tolist()) == set(recomputed.tolist())
    assert len(reported) == len(recomputed)
    return reported


def _filter_rows(path: Path, qids: set[int]) -> list[dict[str, object]]:
    return [row for row in _iter_jsonl(path) if int(row["qid"]) in qids]


def _paper_test_qids(contract: ExperimentContract) -> set[int]:
    return _paper_test_qids_for_task(contract.dataset.key)


def _summary_qids(contract: ExperimentContract, dataset: DatasetSpec) -> set[int]:
    if contract.summary_scope is SummaryScope.PAPER:
        return _paper_test_qids_for_task(dataset.key)
    assert contract.summary_scope is SummaryScope.GENERATED
    return set(dataset.evaluation_qids or dataset.test_qids)


def _paper_test_qids_for_task(task: str) -> set[int]:
    if task == "countdown":
        return set(range(200, 1600))
    if task == "gsm8k":
        return set(range(1_000_000, 1_001_319))
    if task == "mbpp":
        return set(range(464, 964))
    if task == "uspto50k":
        return set(range(1_000_000, 1_001_001))
    raise AssertionError(task)


def _vote_curve(
    answers: list[list[str]],
    correct: np.ndarray,
    *,
    budgets: tuple[int, ...],
    seed: tuple[int, ...] | None = None,
    orderings: int = 0,
    strategy: str | None,
) -> tuple[dict[str, dict[str, float | int]], list[list[int]]]:
    assert len(answers) == correct.shape[0]
    budgets = tuple(value for value in budgets if value <= len(answers))
    assert budgets
    identity = np.arange(len(answers))
    if strategy is None:
        assert seed is None and orderings == 0
        random_orders = np.stack([identity])
    elif strategy == "seed_sequence":
        assert seed and orderings > 0
        rng = np.random.default_rng(np.random.SeedSequence(seed))
        random_orders = np.stack([rng.permutation(len(answers)) for _ in range(orderings)])
    else:
        assert strategy == "legacy_olmo_countdown" and seed and len(seed) == 1
        rng = np.random.default_rng(seed[0])
        random_orders = np.stack([rng.permutation(len(answers)) for _ in range(orderings)])
    curve = {}
    for budget in budgets:
        if strategy == "legacy_olmo_countdown" and budget == 1:
            accuracies = [float(row.mean()) for row in correct]
            curve[str(budget)] = {
                "mean": float(np.mean(accuracies)),
                "sd": float(np.std(accuracies)),
                "orderings": len(accuracies),
            }
            continue
        voter_orders = np.stack([identity]) if budget == len(answers) else random_orders
        accuracies = []
        for voter_order in voter_orders:
            outcomes = []
            for column in range(correct.shape[1]):
                chosen = voter_order[:budget]
                voters = [answers[row][column] for row in chosen if answers[row][column]]
                if not voters:
                    outcomes.append(False)
                    continue
                counts: dict[str, int] = {}
                for answer in voters:
                    counts[answer] = counts.get(answer, 0) + 1
                winner = max(counts, key=counts.get)
                winner_row = next(row for row in chosen if answers[row][column] == winner)
                outcomes.append(bool(correct[winner_row, column]))
            accuracies.append(float(np.mean(outcomes)))
        curve[str(budget)] = {
            "mean": float(np.mean(accuracies)),
            "sd": float(np.std(accuracies)),
            "orderings": len(voter_orders),
        }
    return curve, random_orders.tolist()


def _question_shard(
    contract: ExperimentContract,
    examples: tuple[Example, ...],
    worker_index: int,
) -> tuple[Example, ...]:
    assert 0 <= worker_index < contract.engine.workers
    if contract.batching is not BatchingStrategy.CONTIGUOUS_WORKER_SHARD:
        assert contract.engine.workers == 1
        return examples
    per_worker = (len(examples) + contract.engine.workers - 1) // contract.engine.workers
    start = worker_index * per_worker
    return examples[start : start + per_worker]


def _validate_shard(
    path: Path,
    examples: tuple[Example, ...],
    *,
    samples: int,
    candidate_id: int | None = None,
) -> None:
    expected_rows = len(examples) * samples
    count = 0
    input_manifest = None
    for count, row in enumerate(_iter_jsonl(path), start=1):
        offset = count - 1
        example = examples[offset // samples]
        assert int(row["qid"]) == example.qid
        assert int(row["sample_index"]) == offset % samples
        if candidate_id is None:
            assert row.get("candidate") is None
        else:
            assert int(row["candidate"]["index"]) == candidate_id
        current_manifest = str(row["input_manifest"])
        input_manifest = current_manifest if input_manifest is None else input_manifest
        assert current_manifest == input_manifest
    assert count == expected_rows, (path, count, expected_rows)
    assert input_manifest is not None
    manifest_path = (
        path.parents[1] / input_manifest
        if path.parent.name in {"search", "test"}
        else None
    )
    if manifest_path is None or not manifest_path.exists():
        run_root = next(
            parent
            for parent in path.parents
            if (parent / "contract.json").is_file()
        )
        manifest_path = run_root / input_manifest
    assert manifest_path.is_file(), manifest_path


def _completion(
    contract: ExperimentContract,
    phase: str,
    worker_index: int,
    files: list[Path],
    *,
    completion_id: str | None = None,
) -> None:
    assert all(path.is_file() for path in files)
    records = [
        {
            "path": str(path.relative_to(contract.output_dir)),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "rows": sum(1 for _ in _iter_jsonl(path)),
        }
        for path in sorted(files)
    ]
    contract_path = contract.output_dir / "contract.json"
    population_path = contract.output_dir / "population.json"
    supporting_paths = sorted(
        [
            *contract.output_dir.glob("inputs/*.manifest.json"),
            *contract.output_dir.glob("worker_state/*.json"),
        ]
    )
    _atomic_json(
        contract.output_dir
        / "completion"
        / f"{completion_id or f'{phase}_worker_{worker_index}'}.json",
        {
            "contract": contract.name,
            "phase": phase,
            "worker_index": worker_index,
            "workers": contract.engine.workers,
            "files": records,
            "contract_sha256": _sha256(contract_path),
            "population_sha256": _sha256(population_path) if population_path.exists() else None,
            "supporting_files": [
                {
                    "path": str(path.relative_to(contract.output_dir)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in supporting_paths
            ],
            "model": dataclasses.asdict(contract.model),
            "datasets": [
                dataclasses.asdict(dataset)
                for dataset in (contract.dataset, *contract.additional_datasets)
            ],
            "environment": _environment_info(),
        },
    )


def _expected_phase_paths(contract: ExperimentContract) -> dict[str, set[str]]:
    paths: dict[str, set[str]] = {}
    mbpp_paths: set[str] = set()
    if contract.kind is ExperimentKind.SEARCH:
        paths["search"] = {
            f"search/candidate_{candidate.index:05d}.jsonl.gz"
            for candidate in contract_population(contract)
        }
        if contract.dataset.key == "mbpp":
            mbpp_paths.update(paths["search"])
        if contract.search_scope is SearchScope.SELECT_THEN_TEST:
            selected = json.loads(
                (contract.output_dir / "selected.json").read_text(encoding="utf-8")
            )
            assert [int(row["rank"]) for row in selected] == list(
                range(1, contract.top_k + 1)
            )
            paths["test"] = {
                f"test/rank_{int(row['rank']):03d}_candidate_{int(row['index']):05d}.jsonl.gz"
                for row in selected
            }
            if contract.dataset.key == "mbpp":
                mbpp_paths.update(paths["test"])
    elif contract.kind is ExperimentKind.ANCHOR:
        paths["anchor"] = set()
        for dataset in (contract.dataset, *contract.additional_datasets):
            for seed in contract.repeat_seeds:
                for worker in range(contract.engine.workers):
                    suffix = f"_worker_{worker}" if contract.engine.workers > 1 else ""
                    relative = f"anchor_{dataset.key}_seed_{seed}{suffix}.jsonl.gz"
                    paths["anchor"].add(relative)
                    if dataset.key == "mbpp":
                        mbpp_paths.add(relative)
    elif contract.kind is ExperimentKind.CANDIDATE_REPEAT:
        (candidate,) = contract_population(contract)
        paths["candidate-repeat"] = {
            f"candidate_{candidate.index}_seed_{seed}.jsonl.gz"
            for seed in contract.repeat_seeds
        }
        if contract.dataset.key == "mbpp":
            mbpp_paths.update(paths["candidate-repeat"])
    else:
        assert contract.kind is ExperimentKind.LABEL_SHUFFLE
        assert contract.source_search
        source = load_contract(contract.source_search, output_dir=contract.output_dir.parent)
        selected = json.loads(
            (source.output_dir / "selected.json").read_text(encoding="utf-8")
        )[: contract.top_k]
        paths["shuffle"] = {
            f"anchor_worker_{worker}.jsonl.gz"
            for worker in range(contract.engine.workers)
        }
        paths["shuffle"].update(
            f"rank_{int(row['rank']):03d}_candidate_{int(row['index']):05d}_worker_{worker}.jsonl.gz"
            for row in selected
            for worker in range(contract.engine.workers)
        )
    if mbpp_paths:
        paths["score"] = {f"scored/{relative}" for relative in mbpp_paths}
    return paths


def _finalize_manifest(contract: ExperimentContract) -> None:
    completion_paths = sorted(contract.output_dir.glob("completion/*.json"))
    assert completion_paths
    preflight_path = contract.output_dir / "preflight.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    artifacts: dict[str, dict[str, object]] = {}
    phase_artifacts: dict[str, dict[str, dict[str, object]]] = {}
    for completion_path in completion_paths:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        assert completion["contract"] == contract.name
        phase = str(completion["phase"])
        phase_files = phase_artifacts.setdefault(phase, {})
        for record in completion["files"]:
            path = contract.output_dir / record["path"]
            assert path.is_file()
            assert path.stat().st_size == record["bytes"]
            assert _sha256(path) == record["sha256"]
            prior = artifacts.setdefault(record["path"], record)
            assert prior == record
            phase_prior = phase_files.setdefault(record["path"], record)
            assert phase_prior == record
    expected_paths = _expected_phase_paths(contract)
    assert set(expected_paths) == set(preflight["expected_output_files"])
    for phase, expected_files in preflight["expected_output_files"].items():
        records = phase_artifacts.get(phase, {})
        assert set(records) == expected_paths[phase], phase
        assert len(records) == expected_files, (phase, len(records), expected_files)
        rows = sum(int(record["rows"]) for record in records.values())
        assert rows == preflight["expected_rows"][phase], (
            phase,
            rows,
            preflight["expected_rows"][phase],
        )
    summary_paths = sorted(contract.output_dir.glob("summary/*"))
    files = {}
    for path in sorted(contract.output_dir.rglob("*")):
        if not path.is_file() or path.name == "FINAL_MANIFEST.json" or ".tmp" in path.name:
            continue
        relative = str(path.relative_to(contract.output_dir))
        if relative in artifacts:
            files[relative] = {
                "bytes": artifacts[relative]["bytes"],
                "sha256": artifacts[relative]["sha256"],
            }
        else:
            files[relative] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
    _atomic_json(
        contract.output_dir / "FINAL_MANIFEST.json",
        {
            "contract": contract.name,
            "contract_sha256": _sha256(contract.output_dir / "contract.json"),
            "preflight_sha256": _sha256(contract.output_dir / "preflight.json"),
            "completion_manifests": {
                str(path.relative_to(contract.output_dir)): _sha256(path)
                for path in completion_paths
            },
            "artifacts": artifacts,
            "files": files,
            "summaries": {
                str(path.relative_to(contract.output_dir)): {
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in summary_paths
                if path.is_file()
            },
            "external_dependencies": _external_dependencies(contract),
            "environment": _environment_info(),
        },
    )


def _external_dependencies(contract: ExperimentContract) -> dict[str, dict[str, str]]:
    if contract.kind is not ExperimentKind.LABEL_SHUFFLE:
        return {}
    assert contract.source_search
    run_root = contract.output_dir.parent
    dependencies = {}
    for name in (contract.source_search, "uspto_anchor_original"):
        manifest = run_root / name / "FINAL_MANIFEST.json"
        assert manifest.is_file(), f"complete dependency first: {name}"
        document = json.loads(manifest.read_text(encoding="utf-8"))
        assert document["contract"] == name
        dependencies[name] = {
            "path": str(manifest.relative_to(run_root)),
            "sha256": _sha256(manifest),
        }
    selected = run_root / contract.source_search / "selected.json"
    assert selected.is_file()
    dependencies[contract.source_search]["selected_sha256"] = _sha256(selected)
    return dependencies


def _environment_info() -> dict[str, object]:
    packages = {}
    for name in ("vllm", "torch", "ray", "transformers", "tokenizers", "numpy"):
        try:
            packages[name] = importlib.import_module(name).__version__
        except ImportError:
            packages[name] = None
    gpu = []
    cuda = None
    try:
        import torch

        cuda = torch.version.cuda
        gpu = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    except ImportError:
        pass
    return {
        "python": platform.python_version(),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "platform": platform.platform(),
        "packages": packages,
        "cuda": cuda,
        "gpus": gpu,
    }


def _verify_exact_anchor(
    engine: Any,
    initialization: list[dict[str, object]],
    contract: ExperimentContract,
) -> None:
    if contract.perturbation_semantics is PerturbationSemantics.PAPER_TIED_DRIFTING:
        return
    actual = engine.collective_rpc("weight_fingerprint", args=("sampled",))
    expected = [worker["anchor_fingerprint"] for worker in initialization]
    assert actual == expected, "exact restoration failed"


def _result_dir(contract: ExperimentContract, relative: str) -> Path:
    root = contract.output_dir / "scored" if contract.dataset.key == "mbpp" else contract.output_dir
    path = root / relative
    assert path.exists(), f"missing completed results: {path}"
    return path


def _verify_runtime(contract: ExperimentContract, allow_drift: bool) -> None:
    if allow_drift:
        return
    expected = dataclasses.asdict(contract.runtime)
    actual = {"python": platform.python_version()}
    for package in ("vllm", "torch", "ray", "transformers", "tokenizers", "numpy"):
        actual[package] = importlib.import_module(package).__version__
    mismatches = {
        name: (value, actual[name])
        for name, value in expected.items()
        if value != "not-recorded"
        and not _version_matches(name, value, actual[name])
    }
    assert not mismatches, mismatches


def _version_matches(package: str, expected: str, actual: str) -> bool:
    if package == "torch":
        return actual.split("+", 1)[0] == expected.split("+", 1)[0]
    return actual.startswith(expected)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _require_prepared(contract: ExperimentContract) -> None:
    path = contract.output_dir / "contract.json"
    assert path.is_file(), f"run `--phase prepare` first: {path}"
    assert json.loads(path.read_text(encoding="utf-8")) == _serialized_contract(contract), (
        "contract drift in existing run directory",
        path,
    )


def _require_preflight(contract: ExperimentContract) -> None:
    path = contract.output_dir / "preflight.json"
    assert path.is_file(), f"run `--phase preflight` first: {path}"
    preflight = json.loads(path.read_text(encoding="utf-8"))
    assert preflight["contract"] == contract.name


def _prepare_inputs(
    contract: ExperimentContract,
    dataset: DatasetSpec,
    examples: tuple[Example, ...],
    tokenizer: Any,
    *,
    label: str,
) -> tuple[tuple[str, ...], str]:
    prompts = tuple(
        tokenizer.apply_chat_template(
            list(example.messages),
            tokenize=False,
            add_generation_prompt=True,
        )
        for example in examples
    )
    records = [
        {
            "qid": example.qid,
            "dataset": dataset.key,
            "dataset_spec": dataset.name,
            "messages": list(example.messages),
            "prompt": prompt,
            "gold": example.gold,
        }
        for example, prompt in zip(examples, prompts)
    ]
    relative = Path("inputs") / f"{dataset.name}_{label}.jsonl.gz"
    path = contract.output_dir / relative
    if path.exists():
        assert _read_jsonl(path) == records
    else:
        _atomic_jsonl(path, records)
    _atomic_json(
        path.with_suffix(path.suffix + ".manifest.json"),
        {
            "rows": len(records),
            "sha256": _sha256(path),
            "qids": [example.qid for example in examples],
            "dataset": dataclasses.asdict(dataset),
            "model": dataclasses.asdict(contract.model),
        },
    )
    return prompts, str(relative)


def _input_lookup(
    contract: ExperimentContract,
    relative: str,
) -> dict[int, dict[str, object]]:
    path = contract.output_dir / relative
    rows = _read_jsonl(path)
    lookup = {int(row["qid"]): row for row in rows}
    assert len(lookup) == len(rows)
    return lookup


def _atomic_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(temporary, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _temporary_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return list(_iter_jsonl(path))


def _iter_jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def _first_jsonl(path: Path) -> dict[str, object] | None:
    return next(_iter_jsonl(path), None)


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _jsonable(value: object) -> object:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value"):
        return value.value
    return value


def _serialized_contract(contract: ExperimentContract) -> dict[str, object]:
    serialized = dataclasses.asdict(contract)
    serialized.pop("output_dir")
    value = _jsonable(serialized)
    assert isinstance(value, dict)
    return value


if __name__ == "__main__":
    main()

