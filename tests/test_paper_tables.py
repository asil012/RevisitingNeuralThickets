import numpy as np

from neural_thickets.analysis.tables import build_tables
from neural_thickets.data import Bundle


def test_all_active_tables_regenerate():
    tables = build_tables(Bundle.default())
    assert len(tables) == 15
    assert all(table["rows"] for table in tables.values())


def test_main_voting_table_matches_paper_values():
    rows = build_tables(Bundle.default())["tab:voting-main"]["rows"]
    assert rows[0] == ["OLMo 7B", "GSM8K", "+0.41", "No", "+0.30", "No", "+0.56", "No"]
    assert rows[2] == ["OLMo 7B", "Countdown", "-1.57", "Yes", "-1.24", "Yes", "-0.71", "No"]
    assert rows[-1] == ["Qwen 32B", "Countdown", "+1.50", "Yes", "+2.27", "Yes", "+1.86", "Yes"]


def test_voting_significance_is_recomputed_from_bootstraps():
    bundle = Bundle.default()
    arrays = bundle.npz("outcomes.npz")
    for tag, expected in (
        ("olmo_countdown", (True, True, False)),
        ("q3_countdown", (True, True, False)),
    ):
        observed = []
        for k in (10, 20, 50):
            low, high = np.quantile(arrays[f"{tag}.bootstrap.gap.k{k}"], (0.025, 0.975))
            observed.append(bool(low > 0 or high < 0))
        assert tuple(observed) == expected


def test_uspto_and_null_tables_match_paper_precision():
    tables = build_tables(Bundle.default())
    assert tables["tab:uspto-shuffle"]["rows"] == [
        ["Original", "+7.79", "+5.21", "+5.42", "+5.06"],
        ["Shuffled", "+3.20", "+1.48", "+1.22", "+1.12"],
    ]
    variance = tables["tab:null-variance"]["rows"]
    assert variance[0][-1] == "0.0915"
    assert variance[-1][-1] == "0.0409"


def test_selection_transfer_table_uses_interval_half_widths():
    table = build_tables(Bundle.default())["tab:decay-full"]
    assert table["columns"][-2:] == ["Held-out rank-1", "Held-out interval half-width"]
    assert table["rows"][0][-2:] == ["0.67", "2.51"]
    assert table["rows"][8][-2:] == ["0.00", "1.21"]
    assert table["rows"][-1][-2:] == ["1.72", ""]


def test_qwen_decay_table_includes_countdown_sweeps():
    rows = build_tables(Bundle.default())["tab:qwen-selection-decay"]["rows"]
    assert rows[2] == ["Qwen 3B", "Countdown", "31.42", "24.08", "23.89", "23.83"]
    assert rows[5] == ["Qwen 7B", "Countdown", "31.66", "18.16", "16.76", "16.14"]
