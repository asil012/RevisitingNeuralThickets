from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class PerturbationSemantics(StrEnum):
    """Weight-noise and restoration behavior used in the paper and audit."""

    PAPER_TIED_DRIFTING = "paper_tied_drifting"
    PAPER_TIED_EXACT = "paper_tied_exact"
    IID_EXACT = "iid_exact"


class ExperimentKind(StrEnum):
    SEARCH = "search"
    ANCHOR = "anchor"
    CANDIDATE_REPEAT = "candidate_repeat"
    LABEL_SHUFFLE = "label_shuffle"


class SearchScope(StrEnum):
    SELECT_THEN_TEST = "select_then_test"
    FULL_GRID = "full_grid"


class AnchorStorage(StrEnum):
    NONE = "none"
    SAME_DEVICE = "same_device"
    CPU = "cpu"


class BatchingStrategy(StrEnum):
    FULL_POOL = "full_pool"
    FIXED_CHUNKS = "fixed_chunks"
    CONTIGUOUS_WORKER_SHARD = "contiguous_worker_shard"


class SummaryScope(StrEnum):
    PAPER = "paper"
    GENERATED = "generated"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    revision: str
    tensor_parallel_size: int

    def __post_init__(self) -> None:
        assert self.key
        assert "/" in self.model_id
        assert len(self.revision) == 40
        assert self.tensor_parallel_size > 0


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    key: str
    source: str
    source_revision: str | None
    file_sha256: tuple[tuple[str, str], ...]
    selection_split: str
    test_split: str
    selection_qids: tuple[int, ...]
    test_qids: tuple[int, ...]
    evaluation_qids: tuple[int, ...]
    max_tokens: int

    def __post_init__(self) -> None:
        assert self.name
        assert self.key in {"countdown", "gsm8k", "mbpp", "uspto50k"}
        assert self.source
        assert all(name and len(digest) == 64 for name, digest in self.file_sha256)
        assert self.selection_split
        assert self.test_split
        assert self.selection_qids
        assert self.test_qids
        assert not set(self.selection_qids) & set(self.test_qids)
        assert self.max_tokens > 0


@dataclass(frozen=True)
class DecodeSpec:
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    presence_penalty: float
    frequency_penalty: float
    seed: int

    def __post_init__(self) -> None:
        assert self.temperature >= 0
        assert 0 < self.top_p <= 1
        assert self.top_k == -1 or self.top_k > 0
        assert self.repetition_penalty > 0


@dataclass(frozen=True)
class PerturbationSpec:
    index: int
    seed: int
    sigma: float

    def __post_init__(self) -> None:
        assert self.index >= 0
        assert 0 <= self.seed < 2**31
        assert self.sigma > 0


@dataclass(frozen=True)
class RuntimeSpec:
    python: str
    vllm: str
    torch: str
    ray: str
    transformers: str
    tokenizers: str
    numpy: str


@dataclass(frozen=True)
class EngineSpec:
    workers: int
    max_model_len: int | None
    max_num_seqs: int | None
    max_num_batched_tokens: int | None
    prompt_batch_size: int
    gpu_memory_utilization: float
    enforce_eager: bool
    enable_prefix_caching: bool

    def __post_init__(self) -> None:
        assert self.workers > 0
        assert self.max_model_len is None or self.max_model_len > 0
        assert self.max_num_seqs is None or self.max_num_seqs > 0
        assert self.max_num_batched_tokens is None or self.max_num_batched_tokens > 0
        assert self.prompt_batch_size > 0
        assert 0 < self.gpu_memory_utilization < 1
        assert self.enforce_eager
        assert not self.enable_prefix_caching


@dataclass(frozen=True)
class ExperimentContract:
    name: str
    kind: ExperimentKind
    model: ModelSpec
    dataset: DatasetSpec
    additional_datasets: tuple[DatasetSpec, ...]
    decode: DecodeSpec
    batching: BatchingStrategy
    perturbation_semantics: PerturbationSemantics | None
    search_scope: SearchScope | None
    anchor_storage: AnchorStorage
    population_size: int | None
    population_draw_size: int | None
    population_seed: int | None
    sigma_ladder: tuple[float, ...]
    candidate_filter_sigma: float | None
    candidate_limit: int | None
    top_k: int
    repeat_seeds: tuple[int, ...]
    samples_per_question: int
    candidate_indices: tuple[int, ...]
    candidate_cohorts: tuple[tuple[str, tuple[int, ...]], ...]
    canonical_selected_indices: tuple[int, ...]
    source_search: str | None
    shuffle_seed: int | None
    selection_metric: str | None
    aggregation_seed: tuple[int, ...] | None
    aggregation_orderings: int
    aggregation_budgets: tuple[int, ...]
    aggregation_strategy: str | None
    summary_scope: SummaryScope
    dtype: str
    runtime: RuntimeSpec
    engine: EngineSpec
    reference_commit: str
    output_dir: Path

    def __post_init__(self) -> None:
        assert self.name
        assert self.dtype == "bfloat16"
        assert len(self.reference_commit) == 40
        assert self.aggregation_orderings >= 0
        assert all(budget > 0 for budget in self.aggregation_budgets)
        if self.perturbation_semantics is PerturbationSemantics.IID_EXACT:
            assert self.model.tensor_parallel_size == 1, (
                "iid_exact currently requires TP=1; TP>1 needs rank-aware RNG streams"
            )
        if self.aggregation_orderings:
            assert self.aggregation_seed
            assert self.aggregation_budgets
            assert self.aggregation_strategy in {"seed_sequence", "legacy_olmo_countdown"}
        else:
            assert self.aggregation_seed is None
            assert self.aggregation_strategy is None
        if self.batching is BatchingStrategy.CONTIGUOUS_WORKER_SHARD:
            assert self.engine.workers > 1
        if self.kind is ExperimentKind.SEARCH:
            assert not self.additional_datasets
            assert self.perturbation_semantics is not None
            assert self.search_scope is not None
            assert self.anchor_storage is not AnchorStorage.NONE
            assert self.population_size is not None and self.population_size > 0
            assert self.population_draw_size is not None
            assert self.population_size <= self.population_draw_size
            assert self.population_seed is not None and self.population_seed >= 0
            assert self.sigma_ladder and all(sigma > 0 for sigma in self.sigma_ladder)
            assert self.candidate_filter_sigma is None or (
                self.candidate_filter_sigma in self.sigma_ladder
            )
            assert self.candidate_limit is None or 0 < self.candidate_limit <= self.population_size
            if self.search_scope is SearchScope.SELECT_THEN_TEST:
                assert 0 < self.top_k <= self.population_size
                assert not self.canonical_selected_indices or (
                    len(self.canonical_selected_indices) == self.top_k
                    and len(set(self.canonical_selected_indices)) == self.top_k
                )
            else:
                assert self.top_k == 0
                assert not self.canonical_selected_indices
            assert self.selection_metric == "reference_reward"
            assert not self.repeat_seeds
            assert self.samples_per_question == 1
            if self.candidate_indices:
                assert self.candidate_filter_sigma is None
                assert self.candidate_limit is None
                assert all(0 <= index < self.population_size for index in self.candidate_indices)
            if self.candidate_cohorts:
                flattened = [index for _, indices in self.candidate_cohorts for index in indices]
                assert sorted(flattened) == sorted(self.candidate_indices)
            assert self.source_search is None and self.shuffle_seed is None
            return
        assert self.search_scope is None
        assert self.top_k >= 0
        assert self.repeat_seeds and all(seed >= 0 for seed in self.repeat_seeds)
        assert self.samples_per_question > 0
        if self.kind is ExperimentKind.ANCHOR:
            assert self.perturbation_semantics is None
            assert self.anchor_storage is AnchorStorage.NONE
            assert self.population_size is None and self.population_draw_size is None
            assert self.population_seed is None
            assert not self.sigma_ladder and not self.candidate_indices
            assert not self.candidate_cohorts
            assert not self.canonical_selected_indices
            assert self.source_search is None and self.shuffle_seed is None
            assert self.selection_metric is None
            return
        if self.kind is ExperimentKind.CANDIDATE_REPEAT:
            assert not self.additional_datasets
            assert self.perturbation_semantics is not None
            assert self.population_size and self.population_draw_size
            assert self.population_size <= self.population_draw_size
            assert self.population_seed is not None
            assert self.sigma_ladder and self.candidate_indices
            assert not self.candidate_cohorts
            assert not self.canonical_selected_indices
            assert self.source_search is None and self.shuffle_seed is None
            assert self.selection_metric is None
            return
        assert self.kind is ExperimentKind.LABEL_SHUFFLE
        assert not self.additional_datasets
        assert self.perturbation_semantics is not None
        assert self.source_search and self.shuffle_seed is not None
        assert self.selection_metric is None
        assert self.population_size is None and self.population_draw_size is None
        assert self.population_seed is None
        assert not self.sigma_ladder and not self.candidate_indices
        assert not self.candidate_cohorts
        assert not self.canonical_selected_indices

