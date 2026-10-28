import ast
import inspect
import textwrap

import pytest

from neural_thickets.scoring import sandbox
from neural_thickets.scoring.countdown import score_countdown
from neural_thickets.scoring.gsm8k import score_gsm8k, tolerant_gsm8k_correct
from neural_thickets.scoring.mbpp import (
    extract_mbpp,
    score_mbpp_batch_isolated,
    score_mbpp_isolated,
)
from neural_thickets.scoring.sandbox import isolation_available
from neural_thickets.scoring.uspto50k import score_uspto50k
from neural_thickets.scoring.voting import majority_vote


def test_countdown_versions_preserve_equals_sign_difference() -> None:
    response = "reasoning</think>\n<answer>(1 + 2) * 3 = 9</answer>"
    historical = score_countdown(response, [1, 2, 3], 9, version="historical_d409")
    reference = score_countdown(response, [1, 2, 3], 9, version="reference_536")
    assert historical.answer_reward == 0
    assert reference.answer_reward == 1
    assert reference.correct


def test_countdown_reference_reward_keeps_raw_tag_whitespace() -> None:
    response = "reasoning</think>\n<answer>\n(1 + 2) * 3\n</answer>"
    outcome = score_countdown(response, [1, 2, 3], 9, version="reference_536")
    assert outcome.answer_reward == 0
    assert outcome.correct
    assert outcome.scoring_status == "scored"


def test_countdown_reward_accepts_leading_space_like_randopt() -> None:
    response = "reasoning</think>\n<answer> (1 + 2) / 3 </answer>"
    for version in ("historical_d409", "reference_536"):
        outcome = score_countdown(response, [1, 2, 3], 1, version=version)
        assert outcome.answer_reward == 1
        assert outcome.correct


def test_countdown_floor_division_matches_randopt_eval() -> None:
    response = "reasoning</think>\n<answer>(8 // 3) * 1</answer>"
    outcome = score_countdown(response, [8, 3, 1], 2)
    assert outcome.answer_reward == 1
    assert outcome.correct


def test_countdown_exponent_tower_fails_fast_with_guard_reason() -> None:
    outcome = score_countdown(
        "reasoning</think>\n<answer>2 ** 3 ** 4</answer>",
        [2, 3, 4],
        0,
        version="reference_536",
    )
    assert not outcome.correct
    assert outcome.scoring_status == "guarded"
    assert "exponent" in outcome.scoring_reason


def test_gsm8k_keeps_reference_trailing_period_quirk() -> None:
    outcome = score_gsm8k("reasoning\n#### 42.", "42")
    assert outcome.answer == "42."
    assert not outcome.correct
    assert tolerant_gsm8k_correct(outcome.answer, "42")


def test_gsm8k_reward_and_vote_extraction_are_separate() -> None:
    outcome = score_gsm8k("#### 41\nthen 42", "42")
    assert outcome.reward == 1
    assert outcome.answer == "41"
    assert not outcome.correct


def test_mbpp_isolated_grader_and_extraction() -> None:
    response = "<answer>\ndef add1(x):\n    return x + 1\n</answer>"
    assert extract_mbpp(response).startswith("def add1")
    outcome = score_mbpp_isolated(
        response,
        ["assert add1(1) == 2", "assert add1(-1) == 0"],
        timeout_seconds=10,
        memory_gib=0,
        unsafe_allow_unisolated=True,
    )
    assert outcome.correct
    assert outcome.scoring_status == "passed"
    batch = score_mbpp_batch_isolated(
        [
            (response, ["assert add1(2) == 3"], ""),
            (response, ["assert add1(2) == 4"], ""),
        ],
        memory_gib=0,
        unsafe_allow_unisolated=True,
    )
    assert [row.correct for row in batch] == [True, False]
    assert [row.scoring_status for row in batch] == ["passed", "failed"]


def test_mbpp_requires_secure_isolation() -> None:
    available, _ = isolation_available()
    if available:
        pytest.skip("secure Linux isolation is available")
    with pytest.raises(RuntimeError, match="Secure MBPP isolation is unavailable"):
        score_mbpp_isolated("<answer>\nprint(1)\n</answer>", [])


def test_security_enforcement_does_not_depend_on_python_assertions() -> None:
    source = textwrap.dedent(inspect.getsource(sandbox._restrict))
    tree = ast.parse(source)

    assert not any(isinstance(node, ast.Assert) for node in ast.walk(tree))


def test_uspto_fallback_and_vote_tie_break() -> None:
    assert score_uspto50k("I choose class 2", 2).correct
    outside = score_uspto50k("<answer>11</answer>", 2)
    assert outside.answer == "11" and outside.valid and not outside.correct
    assert majority_vote(["A", "B"]) == "A"
    assert majority_vote(["", "B", "B"]) == "B"

