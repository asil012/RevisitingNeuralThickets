# Copyright 2024 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified in 2026 for this repository. See LICENSE-APACHE.

from __future__ import annotations

import ast
import re
from fractions import Fraction
from typing import Literal

from neural_thickets.scoring.types import Outcome

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_PAPER_CHARS = re.compile(r"^[0-9+\-*/() ]+$")
_CORRECTED_CHARS = re.compile(r"^[0-9+\-*/()= ]+$")
_FORMULA_CHARS = re.compile(r"^[0-9+\-*/() ]+$")


def score_countdown(
    response: str,
    numbers: list[int],
    target: int | float,
    *,
    version: Literal["historical_d409", "reference_536"] = "reference_536",
) -> Outcome:
    """Score one Countdown response with the pinned RandOpt rules."""

    assert version in {"historical_d409", "reference_536"}
    format_reward = _format_reward("<think>" + response)
    reward_formula = _extract_raw_formula(response)
    expression = _expression(reward_formula, numbers, version)
    answer_correct, reward_guard = _correct_expression(expression, numbers, target)
    answer_reward = float(answer_correct)
    vote_answer, valid, vote_guard = _vote_answer(extract_formula(response), numbers)
    correct = valid and _matches_target(vote_answer, target)
    guard = reward_guard or vote_guard
    return Outcome(
        answer=vote_answer,
        valid=valid,
        correct=correct,
        reward=answer_reward + 0.1 * format_reward,
        answer_reward=answer_reward,
        format_reward=format_reward,
        scoring_status="guarded" if guard else "scored",
        scoring_reason=guard,
    )


def extract_formula(response: str) -> str:
    return _extract_raw_formula(response).strip()


def _extract_raw_formula(response: str) -> str:
    matches = _ANSWER_RE.findall(response)
    return matches[-1] if matches else ""


def _format_reward(response: str) -> float:
    if re.match(r"^<think>.*?</think>\n<answer>.*?</answer>$", response, re.DOTALL):
        return 1.0
    reward = 0.0
    if re.search(r"<think>.*?</think>", response, re.DOTALL):
        reward += 0.1
    if re.search(r"<answer>.*?</answer>", response, re.DOTALL):
        reward += 0.5
    return reward


def _expression(formula: str, numbers: list[int], version: str) -> str:
    allowed = _PAPER_CHARS if version == "historical_d409" else _CORRECTED_CHARS
    if not formula or not allowed.match(formula):
        return ""
    if version == "historical_d409" or "=" not in formula:
        return formula
    expected = sorted(numbers)
    matching = [
        side
        for side in (part.strip() for part in formula.split("="))
        if side
        and _FORMULA_CHARS.match(side)
        and sorted(int(value) for value in re.findall(r"\d+", side)) == expected
    ]
    return matching[0] if len(matching) == 1 else ""


def _correct_expression(
    expression: str,
    numbers: list[int],
    target: int | float,
) -> tuple[bool, str | None]:
    if not expression:
        return False, None
    used = [int(value) for value in re.findall(r"\d+", expression)]
    if sorted(used) != sorted(numbers):
        return False, None
    try:
        result = _safe_arithmetic(expression)
    except ArithmeticGuard as error:
        return False, str(error)
    except (SyntaxError, TypeError, ValueError, ZeroDivisionError):
        return False, None
    return _matches_target(result, target), None


def _vote_answer(formula: str, numbers: list[int]) -> tuple[str, bool, str | None]:
    expression = _expression(formula, numbers, "reference_536")
    if not expression:
        return "", False, None
    used = [int(value) for value in re.findall(r"\d+", expression)]
    if any(value > 1_000 for value in used) or sorted(used) != sorted(numbers):
        return "", False, None
    try:
        result = _safe_arithmetic(expression)
        value = float(result)
    except ArithmeticGuard as error:
        return "", False, str(error)
    except (SyntaxError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return "", False, None
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value))), True, None
    return str(float(result)), True, None


def _matches_target(value: object, target: int | float) -> bool:
    try:
        return abs(float(value) - float(target)) < 1e-5
    except (TypeError, ValueError, OverflowError):
        return False


class ArithmeticGuard(Exception):
    pass


def _safe_arithmetic(expression: str) -> Fraction:
    if len(expression) > 4096:
        raise ArithmeticGuard("expression exceeds 4096 characters")
    tree = ast.parse(expression.lstrip(" \t"), mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 128:
        raise ArithmeticGuard("expression exceeds 128 syntax nodes")
    return _evaluate_node(tree.body, depth=0)


def _evaluate_node(node: ast.AST, *, depth: int) -> Fraction:
    if depth > 32:
        raise ArithmeticGuard("expression depth exceeds 32")
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
    ):
        result = Fraction(node.value)
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate_node(node.operand, depth=depth + 1)
        result = value if isinstance(node.op, ast.UAdd) else -value
    elif isinstance(node, ast.BinOp):
        left = _evaluate_node(node.left, depth=depth + 1)
        right = _evaluate_node(node.right, depth=depth + 1)
        if isinstance(node.op, ast.Add):
            result = left + right
        elif isinstance(node.op, ast.Sub):
            result = left - right
        elif isinstance(node.op, ast.Mult):
            result = left * right
        elif isinstance(node.op, ast.Div):
            result = left / right
        elif isinstance(node.op, ast.FloorDiv):
            result = Fraction(left // right)
        elif isinstance(node.op, ast.Pow):
            if right.denominator != 1 or abs(right.numerator) > 32:
                raise ArithmeticGuard("exponent must be an integer in [-32, 32]")
            exponent = int(right.numerator)
            if exponent < 0 and left == 0:
                raise ZeroDivisionError
            result = left**exponent
        else:
            raise ValueError("unsupported arithmetic operator")
    else:
        raise ValueError("unsupported arithmetic syntax")
    if result.numerator.bit_length() > 4096 or result.denominator.bit_length() > 4096:
        raise ArithmeticGuard("arithmetic result exceeds 4096 bits")
    return result

