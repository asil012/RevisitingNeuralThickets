from dataclasses import dataclass


@dataclass(frozen=True)
class Outcome:
    answer: str
    valid: bool
    correct: bool
    reward: float
    answer_reward: float
    format_reward: float | None
    scoring_status: str = "scored"
    scoring_reason: str | None = None

    def __post_init__(self) -> None:
        assert 0 <= self.reward <= 1.1
        assert 0 <= self.answer_reward <= 1
        assert self.format_reward is None or 0 <= self.format_reward <= 1
        assert self.scoring_status

