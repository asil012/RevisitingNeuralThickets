from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from neural_thickets.types import DatasetSpec

COUNTDOWN_SYSTEM = (
    "You are a helpful assistant. You first think about the reasoning process "
    "in your mind and then provide the user with the answer."
)
COUNTDOWN_USER = (
    "Using the numbers {numbers}, create an equation that equals {target}. "
    "You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
    "Show your work in <think> </think> tags. "
    "And return the final answer in <answer> </answer> tags, for example "
    "<answer> (1 + 2) / 3 </answer>."
)
GSM8K_INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'
MBPP_SYSTEM = (
    "You are a Python programming assistant. Write clean, correct Python code to solve the given "
    "problem."
)
MBPP_USER = (
    "{text}\n\nYour code should pass these tests:\n{tests}\n\n"
    "Think through your solution in <think> </think> tags.\n"
    "Return your final Python code in <answer> </answer> tags, e.g.:\n"
    "<answer>\ndef solution(x):\n return x + 1\n</answer>"
)
USPTO_SYSTEM = (
    "You are an expert organic chemist. Your task is to classify chemical reactions into one of "
    "10 standard reaction categories based on the transformation type.\n\n"
    "## Reaction Classes:\n"
    "1: Heteroatom alkylation/arylation - N, O, S attacking C (e.g., SN2, ether formation)\n"
    "2: Acylation - Forming C=O bonds with N, O, S (e.g., amide, ester formation)\n"
    "3: C-C bond formation - New C-C bonds (e.g., Suzuki, Heck, Grignard)\n"
    "4: Heterocycle formation - Creating rings with N, O, S\n"
    "5: Protections - Adding protecting groups (Boc, Bn, TBS, etc.)\n"
    "6: Deprotections - Removing protecting groups\n"
    "7: Reductions - Adding H, removing O (e.g., ketone→alcohol, nitro→amine)\n"
    "8: Oxidations - Adding O, removing H (e.g., alcohol→ketone)\n"
    "9: Functional group interconversion - Changing one FG to another\n"
    "10: Functional group addition - Adding new FG to molecule (e.g., halogenation)"
)
USPTO_USER = (
    "Classify this reaction:\n\nReactants >> Product:\n{rxn_smiles}\n\n"
    "Analyze the key transformation and output the class number (1-10) in <answer>X</answer> tags."
)


@dataclass(frozen=True)
class Example:
    qid: int
    messages: tuple[dict[str, str], ...]
    gold: object


def load_examples(spec: DatasetSpec, cache_dir: Path) -> dict[str, tuple[Example, ...]]:
    loaders = {
        "countdown": _load_countdown,
        "gsm8k": _load_gsm8k,
        "mbpp": _load_mbpp,
        "uspto50k": _load_uspto50k,
    }
    examples = loaders[spec.key](spec, cache_dir)
    selection = tuple(examples[qid] for qid in spec.selection_qids)
    test = tuple(examples[qid] for qid in spec.test_qids)
    evaluation = tuple(examples[qid] for qid in spec.evaluation_qids)
    assert [row.qid for row in selection] == list(spec.selection_qids)
    assert [row.qid for row in test] == list(spec.test_qids)
    assert [row.qid for row in evaluation] == list(spec.evaluation_qids)
    return {"selection": selection, "test": test, "evaluation": evaluation}


def _load_countdown(spec: DatasetSpec, cache_dir: Path) -> dict[int, Example]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "countdown.json"
    if not path.exists():
        with urllib.request.urlopen(spec.source, timeout=120) as response:
            payload = response.read()
        path.write_bytes(payload)
    if spec.source_revision:
        assert _sha256(path) == spec.source_revision
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {
        index: Example(
            qid=index,
            messages=(
                {"role": "system", "content": COUNTDOWN_SYSTEM},
                {
                    "role": "user",
                    "content": COUNTDOWN_USER.format(
                        numbers=row["numbers"],
                        target=row["target"],
                    ),
                },
            ),
            gold={"numbers": row["numbers"], "target": row["target"]},
        )
        for index, row in enumerate(rows)
    }


def _load_gsm8k(spec: DatasetSpec, cache_dir: Path) -> dict[int, Example]:
    from datasets import load_dataset

    dataset = load_dataset(
        spec.source,
        "main",
        revision=spec.source_revision,
        cache_dir=str(cache_dir),
    )
    examples = {}
    for index, row in enumerate(dataset["train"]):
        examples[index] = _gsm8k_example(index, row)
    for index, row in enumerate(dataset["test"]):
        qid = 1_000_000 + index
        examples[qid] = _gsm8k_example(qid, row)
    return examples


def _gsm8k_example(qid: int, row: dict[str, str]) -> Example:
    gold = row["answer"].rsplit("####", 1)[-1].strip().replace(",", "")
    content = f'{row["question"]} {GSM8K_INSTRUCTION}'
    return Example(qid=qid, messages=({"role": "user", "content": content},), gold=gold)


def _load_mbpp(spec: DatasetSpec, cache_dir: Path) -> dict[int, Example]:
    from datasets import load_dataset

    dataset = load_dataset(
        spec.source,
        "full",
        revision=spec.source_revision,
        cache_dir=str(cache_dir),
    )
    rows = [row for split in ("train", "validation", "test") for row in dataset[split]]
    return {
        index: Example(
            qid=index,
            messages=(
                {"role": "system", "content": MBPP_SYSTEM},
                {
                    "role": "user",
                    "content": MBPP_USER.format(
                        text=row["text"],
                        tests="\n".join(row["test_list"][:3]),
                    ),
                },
            ),
            gold={
                "code": row["code"],
                "test_list": list(row["test_list"]),
                "test_setup_code": row.get("test_setup_code", ""),
            },
        )
        for index, row in enumerate(rows)
    }


def _load_uspto50k(spec: DatasetSpec, cache_dir: Path) -> dict[int, Example]:
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    hashes = dict(spec.file_sha256)
    assert spec.source_revision and hashes
    data_root = cache_dir / "uspto_50k"
    local_paths = {split: data_root / f"{split}.parquet" for split in ("train", "validation")}
    if all(path.is_file() for path in local_paths.values()):
        paths = local_paths
        for split, path in paths.items():
            assert _sha256(path) == hashes[path.name], path
    else:
        paths = {
            split: Path(
                hf_hub_download(
                    repo_id=spec.source,
                    repo_type="dataset",
                    filename=f"data/{split}-00000-of-00001.parquet",
                    revision=spec.source_revision,
                    cache_dir=cache_dir,
                )
            )
            for split in ("train", "validation")
        }
        for split, path in paths.items():
            assert _sha256(path) == hashes[f"source_{split}.parquet"], path
    dataset = load_dataset(
        "parquet",
        data_files={split: str(path) for split, path in paths.items()},
        cache_dir=str(cache_dir),
    )
    examples = {}
    for split, offset in (("train", 0), ("validation", 1_000_000)):
        for index, row in enumerate(dataset[split]):
            qid = offset + index
            smiles = re.sub(r":(\d+)\]", "]", row["rxn_smiles"])
            examples[qid] = Example(
                qid=qid,
                messages=(
                    {"role": "system", "content": USPTO_SYSTEM},
                    {"role": "user", "content": USPTO_USER.format(rxn_smiles=smiles)},
                ),
                gold=str(row["class"]),
            )
    return examples


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

