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

import re

from neural_thickets.scoring.types import Outcome

_CLIP_CHARS = 300


def score_gsm8k(response: str, gold: str) -> Outcome:
    """Match RandOpt's strict-then-flexible GSM8K scorer, punctuation quirks included."""

    strict = extract_gsm8k(response, method="strict")
    flexible = extract_gsm8k(response, method="flexible") if strict != str(gold) else strict
    reward = strict == str(gold) or flexible == str(gold)
    answer = strict if strict is not None else flexible
    answer = answer or ""
    correct = answer == str(gold)
    return Outcome(
        answer=answer,
        valid=bool(answer),
        correct=correct,
        reward=float(reward),
        answer_reward=float(reward),
        format_reward=None,
    )


def extract_gsm8k(response: str, *, method: str) -> str | None:
    assert method in {"strict", "flexible"}
    text = response[-_CLIP_CHARS:]
    if method == "strict":
        matches = re.findall(r"#### (\-?[0-9\.\,]+)", text)
        return matches[-1].replace(",", "").replace("$", "") if matches else None
    matches = re.findall(r"(\-?[0-9\.\,]+)", text)
    for answer in reversed(matches):
        if answer not in {"", "."}:
            return answer
    return None


def tolerant_gsm8k_correct(answer: str | None, gold: str | None) -> bool:
    left = _normalize_tolerant(answer)
    right = _normalize_tolerant(gold)
    if not left or not right:
        return False
    try:
        return float(left) == float(right)
    except (ValueError, OverflowError):
        return left == right


def _normalize_tolerant(value: str | None) -> str:
    text = "" if value is None else str(value).strip()
    if text.endswith((".", ",")):
        text = text[:-1]
    if text.startswith("$"):
        text = text[1:]
    return text.replace(",", "").strip()

