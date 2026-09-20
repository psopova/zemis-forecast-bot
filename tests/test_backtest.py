"""The fitting harness has to actually respond to the data.

A sweep that returns the same answer whatever it is fed is worse than no sweep,
because it looks like evidence. These tests feed it two synthetic worlds with
opposite lessons and check it draws opposite conclusions.
"""

import json
import sys
from datetime import datetime, timezone

import pytest


from backtest.run import _score_binary, _score_mc, _score_numeric, fit


def binary_row(prob_pct: int, resolution: str, runs: int = 3) -> dict:
    return {
        "question_id": 1,
        "type": "binary",
        "title": "t",
        "resolution": resolution,
        "options": [],
        "raw_outputs": [f"reasoning\nPROBABILITY: {prob_pct}%" for _ in range(runs)],
    }


def numeric_row(values, resolution, open_lower=False, open_upper=True) -> dict:
    block = "PERCENTILES:\n" + "\n".join(
        f"{int(p * 100)}: {v}" for p, v in values
    )
    return {
        "question_id": 2,
        "type": "numeric",
        "title": "t",
        "resolution": str(resolution),
        "options": [],
        "scaling": {"range_min": 0, "range_max": 1000, "zero_point": None},
        "open_lower_bound": open_lower,
        "open_upper_bound": open_upper,
        "inbound_outcome_count": 200,
        "raw_outputs": [block, block, block],
    }


def mc_row(probs: dict, resolution: str) -> dict:
    block = "PROBABILITIES:\n" + "\n".join(f"{k}: {int(v * 100)}%" for k, v in probs.items())
    return {
        "question_id": 3,
        "type": "multiple_choice",
        "title": "t",
        "resolution": resolution,
        "options": list(probs),
        "raw_outputs": [block, block],
    }


def test_binary_scorer_prefers_being_right():
    right = _score_binary(binary_row(90, "yes"), shrink=1.0, floor=0.01)
    wrong = _score_binary(binary_row(90, "no"), shrink=1.0, floor=0.01)
    assert right > 0 > wrong


def test_binary_scorer_returns_none_on_unparseable_output():
    row = binary_row(90, "yes")
    row["raw_outputs"] = ["no number at all"]
    assert _score_binary(row, 0.9, 0.05) is None


def test_shrinkage_helps_an_overconfident_and_wrong_bot():
    """When confident forecasts keep missing, the sweep should want more shrinkage."""
    rows = [binary_row(95, "no") for _ in range(8)] + [binary_row(95, "yes") for _ in range(2)]
    heavy = sum(_score_binary(r, shrink=0.5, floor=0.05) for r in rows)
    none = sum(_score_binary(r, shrink=1.0, floor=0.01) for r in rows)
    assert heavy > none


def test_shrinkage_hurts_a_confident_and_right_bot():
    """And the reverse, so the sweep is measuring rather than always shrinking."""
    rows = [binary_row(95, "yes") for _ in range(9)] + [binary_row(95, "no")]
    heavy = sum(_score_binary(r, shrink=0.5, floor=0.05) for r in rows)
    none = sum(_score_binary(r, shrink=1.0, floor=0.01) for r in rows)
    assert none > heavy


def test_numeric_scorer_rewards_a_distribution_over_the_truth():
    close = numeric_row([(0.1, 480), (0.5, 500), (0.9, 520)], 500)
    far = numeric_row([(0.1, 80), (0.5, 100), (0.9, 120)], 500)
    assert _score_numeric(close, 1.0) > _score_numeric(far, 1.0)


def test_widening_helps_when_the_truth_sits_in_the_tail():
    row = numeric_row([(0.1, 480), (0.5, 500), (0.9, 520)], 545)
    assert _score_numeric(row, 1.4) > _score_numeric(row, 1.0)


def test_the_distribution_has_tails_rather_than_cliff_edges():
    """Regression guard on a real bug: the density used to be a box.

    Interpolating outward from the outermost elicited percentile extrapolated
    the steep middle slope, so the distribution stopped dead just past the 90th
    percentile and an outcome one bucket outside scored identically to one at
    the far end of the scale. Under a log rule that is the difference between
    losing a little and losing everything on a question, so the score must fall
    off gradually as the outcome moves away from the median.
    """
    elicited = [(0.1, 480), (0.5, 500), (0.9, 520)]
    scores = [
        _score_numeric(numeric_row(elicited, truth), 1.0)
        for truth in (500, 520, 530, 545, 560)
    ]
    assert all(a > b for a, b in zip(scores, scores[1:])), scores
    # And the decay is gradual, not a single step to the floor.
    steps = [a - b for a, b in zip(scores, scores[1:])]
    assert max(steps) < 120, steps


def test_widening_costs_when_the_truth_is_dead_centre():
    row = numeric_row([(0.1, 480), (0.5, 500), (0.9, 520)], 500)
    assert _score_numeric(row, 1.0) > _score_numeric(row, 1.6)


def test_mc_scorer_and_mixing_behaviour():
    right = mc_row({"A": 0.8, "B": 0.1, "C": 0.1}, "A")
    wrong = mc_row({"A": 0.8, "B": 0.1, "C": 0.1}, "C")
    assert _score_mc(right, 0.05) > 0 > _score_mc(wrong, 0.05)
    assert _score_mc(wrong, 0.2) > _score_mc(wrong, 0.0)
    assert _score_mc(right, 0.0) > _score_mc(right, 0.2)


def test_mc_scorer_skips_rows_whose_resolution_is_not_an_option():
    row = mc_row({"A": 0.5, "B": 0.5}, "Z")
    assert _score_mc(row, 0.05) is None


def test_fit_runs_over_a_mixed_cache(tmp_path, capsys):
    rows = (
        [binary_row(80, "yes") for _ in range(5)]
        + [binary_row(80, "no") for _ in range(3)]
        + [numeric_row([(0.1, 300), (0.5, 400), (0.9, 500)], 420)]
        + [numeric_row([(0.1, 100), (0.5, 200), (0.9, 300)], 700)]
        + [mc_row({"A": 0.6, "B": 0.4}, "A")]
    )
    cache = tmp_path / "runs.json"
    cache.write_text(json.dumps({"collected_at": datetime.now(timezone.utc).isoformat(), "rows": rows}))

    class Args:
        pass

    args = Args()
    args.cache = str(cache)
    assert fit(args) == 0

    out = capsys.readouterr().out
    parsed = json.loads(out[: out.index("\nRead these")])
    assert set(parsed) == {"binary", "numeric", "multiple_choice"}
    for section in parsed.values():
        assert section["best"] is not None
        assert section["best"]["n"] > 0
    assert "MiniBench" in out, "the leakage caveat must be printed with the numbers"
