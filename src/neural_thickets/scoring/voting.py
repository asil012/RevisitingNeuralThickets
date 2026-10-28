from collections import Counter
from collections.abc import Sequence


def majority_vote(answers: Sequence[str]) -> str:
    """Vote over valid answers; ties favor the earliest voter, as in RandOpt."""

    valid = [answer for answer in answers if answer]
    return Counter(valid).most_common(1)[0][0] if valid else ""

