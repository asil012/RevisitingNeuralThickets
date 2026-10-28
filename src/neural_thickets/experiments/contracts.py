from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

from neural_thickets.types import (
    AnchorStorage,
    BatchingStrategy,
    DatasetSpec,
    DecodeSpec,
    EngineSpec,
    ExperimentContract,
    ExperimentKind,
    ModelSpec,
    PerturbationSemantics,
    RuntimeSpec,
    SearchScope,
    SummaryScope,
)

DEFAULT_CONTRACTS = Path(str(files("neural_thickets.configs").joinpath("experiments.yaml")))


def contract_names(path: Path = DEFAULT_CONTRACTS) -> tuple[str, ...]:
    document = _document(path)
    return tuple(sorted(document["experiments"]))


def load_contract(
    name: str,
    *,
    path: Path = DEFAULT_CONTRACTS,
    output_dir: Path = Path("runs"),
) -> ExperimentContract:
    document = _document(path)
    raw = _mapping(document["experiments"], name)
    model_raw = _mapping(document["models"], str(raw["model"]))
    dataset_raw = _mapping(document["datasets"], str(raw["dataset"]))
    runtime_raw = _mapping(document["runtimes"], str(raw["runtime"]))
    engine_raw = _mapping(document["engines"], str(raw["engine"]))
    decode_raw = _mapping(document["decodes"], str(raw["decode"]))
    _exact_keys(
        raw,
        {
            "kind",
            "model",
            "dataset",
            "additional_datasets",
            "decode",
            "batching",
            "runtime",
            "engine",
            "semantics",
            "search_scope",
            "anchor_storage",
            "population_size",
            "population_draw_size",
            "population_seed",
            "sigma_ladder",
            "candidate_filter_sigma",
            "candidate_limit",
            "top_k",
            "repeat_seeds",
            "samples_per_question",
            "candidate_indices",
            "candidate_cohorts",
            "canonical_selected_indices",
            "source_search",
            "shuffle_seed",
            "selection_metric",
            "aggregation_seed",
            "aggregation_orderings",
            "aggregation_budgets",
            "aggregation_strategy",
            "summary_scope",
        },
    )
    _exact_keys(model_raw, {"model_id", "revision", "tensor_parallel_size"})
    _exact_keys(
        dataset_raw,
        {
            "task",
            "source",
            "source_revision",
            "file_sha256",
            "selection_split",
            "test_split",
            "selection_qids",
            "test_qids",
            "evaluation_qid_ranges",
            "max_tokens",
        },
    )
    _exact_keys(
        decode_raw,
        {
            "temperature",
            "top_p",
            "top_k",
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
            "seed",
        },
    )
    _exact_keys(
        runtime_raw,
        {"python", "vllm", "torch", "ray", "transformers", "tokenizers", "numpy"},
    )
    _exact_keys(
        engine_raw,
        {
            "workers",
            "max_model_len",
            "max_num_seqs",
            "max_num_batched_tokens",
            "prompt_batch_size",
            "gpu_memory_utilization",
            "enforce_eager",
            "enable_prefix_caching",
        },
    )
    return ExperimentContract(
        name=name,
        kind=ExperimentKind(str(raw["kind"])),
        model=ModelSpec(key=str(raw["model"]), **model_raw),
        dataset=DatasetSpec(
            name=str(raw["dataset"]),
            key=str(dataset_raw["task"]),
            source=str(dataset_raw["source"]),
            source_revision=_optional_string(dataset_raw["source_revision"]),
            file_sha256=tuple(
                (str(filename), str(digest))
                for filename, digest in dataset_raw["file_sha256"].items()
            ),
            selection_split=str(dataset_raw["selection_split"]),
            test_split=str(dataset_raw["test_split"]),
            selection_qids=_qid_tuple(dataset_raw["selection_qids"]),
            test_qids=_qid_tuple(dataset_raw["test_qids"]),
            evaluation_qids=_qid_ranges(dataset_raw["evaluation_qid_ranges"]),
            max_tokens=int(dataset_raw["max_tokens"]),
        ),
        additional_datasets=tuple(
            _dataset_spec(str(dataset_name), document)
            for dataset_name in raw["additional_datasets"]
        ),
        decode=DecodeSpec(**decode_raw),
        batching=BatchingStrategy(str(raw["batching"])),
        perturbation_semantics=(
            None if raw["semantics"] is None else PerturbationSemantics(str(raw["semantics"]))
        ),
        search_scope=(
            None if raw["search_scope"] is None else SearchScope(str(raw["search_scope"]))
        ),
        anchor_storage=AnchorStorage(str(raw["anchor_storage"])),
        population_size=None if raw["population_size"] is None else int(raw["population_size"]),
        population_draw_size=(
            None if raw["population_draw_size"] is None else int(raw["population_draw_size"])
        ),
        population_seed=None if raw["population_seed"] is None else int(raw["population_seed"]),
        sigma_ladder=tuple(float(value) for value in raw["sigma_ladder"]),
        candidate_filter_sigma=(
            None if raw["candidate_filter_sigma"] is None else float(raw["candidate_filter_sigma"])
        ),
        candidate_limit=None if raw["candidate_limit"] is None else int(raw["candidate_limit"]),
        top_k=int(raw["top_k"]),
        repeat_seeds=tuple(int(seed) for seed in raw["repeat_seeds"]),
        samples_per_question=int(raw["samples_per_question"]),
        candidate_indices=tuple(int(index) for index in raw["candidate_indices"]),
        candidate_cohorts=tuple(
            (str(label), tuple(int(index) for index in indices))
            for label, indices in raw["candidate_cohorts"].items()
        ),
        canonical_selected_indices=tuple(
            int(index) for index in raw["canonical_selected_indices"]
        ),
        source_search=_optional_string(raw["source_search"]),
        shuffle_seed=None if raw["shuffle_seed"] is None else int(raw["shuffle_seed"]),
        selection_metric=_optional_string(raw["selection_metric"]),
        aggregation_seed=(
            None
            if raw["aggregation_seed"] is None
            else tuple(int(value) for value in raw["aggregation_seed"])
        ),
        aggregation_orderings=int(raw["aggregation_orderings"]),
        aggregation_budgets=tuple(int(value) for value in raw["aggregation_budgets"]),
        aggregation_strategy=_optional_string(raw["aggregation_strategy"]),
        summary_scope=SummaryScope(str(raw["summary_scope"])),
        dtype="bfloat16",
        runtime=RuntimeSpec(**runtime_raw),
        engine=EngineSpec(**engine_raw),
        reference_commit=str(document["reference_commit"]),
        output_dir=output_dir / name,
    )


def _document(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    _exact_keys(
        data,
        {
            "schema_version",
            "reference_commit",
            "models",
            "datasets",
            "decodes",
            "runtimes",
            "engines",
            "experiments",
        },
    )
    assert data["schema_version"] == 1
    assert len(str(data["reference_commit"])) == 40
    for key in ("models", "datasets", "decodes", "runtimes", "engines", "experiments"):
        assert isinstance(data[key], dict) and data[key]
    return data


def _mapping(values: dict[str, Any], key: str) -> dict[str, Any]:
    assert key in values, key
    value = values[key]
    assert isinstance(value, dict), key
    return value


def _dataset_spec(name: str, document: dict[str, Any]) -> DatasetSpec:
    raw = _mapping(document["datasets"], name)
    _exact_keys(
        raw,
        {
            "task",
            "source",
            "source_revision",
            "file_sha256",
            "selection_split",
            "test_split",
            "selection_qids",
            "test_qids",
            "evaluation_qid_ranges",
            "max_tokens",
        },
    )
    return DatasetSpec(
        name=name,
        key=str(raw["task"]),
        source=str(raw["source"]),
        source_revision=_optional_string(raw["source_revision"]),
        file_sha256=tuple(
            (str(filename), str(digest))
            for filename, digest in raw["file_sha256"].items()
        ),
        selection_split=str(raw["selection_split"]),
        test_split=str(raw["test_split"]),
        selection_qids=_qid_tuple(raw["selection_qids"]),
        test_qids=_qid_tuple(raw["test_qids"]),
        evaluation_qids=_qid_ranges(raw["evaluation_qid_ranges"]),
        max_tokens=int(raw["max_tokens"]),
    )


def _exact_keys(values: dict[str, Any], expected: set[str]) -> None:
    assert set(values) == expected, (set(values) - expected, expected - set(values))


def _optional_string(value: object) -> str | None:
    return None if value is None else str(value)


def _qid_tuple(value: object) -> tuple[int, ...]:
    assert isinstance(value, list) and value
    if len(value) == 2 and all(isinstance(item, int) for item in value):
        start, stop = value
        assert start < stop
        return tuple(range(start, stop))
    assert all(isinstance(item, int) for item in value)
    return tuple(value)


def _qid_ranges(value: object) -> tuple[int, ...]:
    assert isinstance(value, list)
    qids = []
    for pair in value:
        assert isinstance(pair, list) and len(pair) == 2
        start, stop = pair
        assert isinstance(start, int) and isinstance(stop, int) and start < stop
        qids.extend(range(start, stop))
    assert len(qids) == len(set(qids))
    return tuple(qids)

