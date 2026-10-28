import re

from neural_thickets.scoring.types import Outcome


def score_uspto50k(response: str, gold: str | int) -> Outcome:
    answer = extract_uspto50k(response)
    correct = bool(answer) and int(answer) == int(gold)
    return Outcome(
        answer=answer,
        valid=bool(answer),
        correct=correct,
        reward=float(correct),
        answer_reward=float(correct),
        format_reward=None,
    )


def extract_uspto50k(response: str) -> str:
    matches = re.findall(r"<answer>\s*(\d+)\s*</answer>", response, re.IGNORECASE)
    if matches:
        return matches[-1]
    matches = re.findall(r"\b([1-9]|10)\b", response)
    return matches[-1] if matches else ""

