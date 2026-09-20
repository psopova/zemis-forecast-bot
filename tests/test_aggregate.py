import math

import pytest


from bot.aggregate import (
    BINARY_FLOOR,
    MC_FLOOR,
    aggregate_binary,
    aggregate_multiple_choice,
    aggregate_percentiles,
    break_even_error_rate,
    calibrate_binary,
    calibrate_multiple_choice,
    logit,
    sigmoid,
    widen_percentiles,
)


def test_median_ignores_a_hallucinating_member():
    assert aggregate_binary([0.30, 0.32, 0.31, 0.999, 0.29]) == pytest.approx(0.31)


def test_aggregate_binary_handles_empty_and_invalid():
    assert aggregate_binary([]) is None
    assert aggregate_binary([None, 0.0, 1.0, "x" and None]) is None


def test_logit_sigmoid_roundtrip():
    for p in (0.001, 0.05, 0.5, 0.95, 0.999):
        assert sigmoid(logit(p)) == pytest.approx(p, abs=1e-12)


def test_calibration_shrinks_extremes_and_leaves_the_middle_alone():
    assert calibrate_binary(0.5) == pytest.approx(0.5)
    assert calibrate_binary(0.02) > 0.02
    assert calibrate_binary(0.98) < 0.98
    assert 0.3 < calibrate_binary(0.3) < 0.36


def test_calibration_respects_the_hard_cap():
    for p in (1e-6, 0.0001, 0.001):
        assert calibrate_binary(p) == pytest.approx(BINARY_FLOOR)
    for p in (0.9999, 0.99999):
        assert calibrate_binary(p) == pytest.approx(1 - BINARY_FLOOR)


def test_calibration_is_monotone():
    ps = [i / 1000 for i in range(1, 1000)]
    out = [calibrate_binary(p) for p in ps]
    assert all(b >= a - 1e-12 for a, b in zip(out, out[1:]))


def test_the_chosen_floor_is_justified_by_the_scoring_rule():
    """Capping at 5% pays if a stated 0.2% is wrong more than about 3% of the time.

    This is the arithmetic the floor is chosen on, kept as a test so the
    constant cannot drift away from its reason.
    """
    rate = break_even_error_rate(stated=0.002, cap=BINARY_FLOOR)
    assert rate < 0.05, rate
    # And capping a merely-confident forecast is a much closer call, which is
    # why the floor is not set higher still.
    assert break_even_error_rate(stated=0.04, cap=0.10) > rate


def test_percentile_aggregation_is_monotone_and_median_based():
    runs = [
        [(0.1, 10), (0.5, 20), (0.9, 30)],
        [(0.1, 12), (0.5, 22), (0.9, 35)],
        [(0.1, 11), (0.5, 21), (0.9, 900)],
    ]
    got = aggregate_percentiles(runs)
    assert dict(got)[0.5] == pytest.approx(21)
    assert dict(got)[0.9] == pytest.approx(35)
    values = [v for _, v in got]
    assert values == sorted(values)


def test_percentile_aggregation_forces_monotonicity():
    got = aggregate_percentiles([[(0.1, 50), (0.5, 40), (0.9, 30)]])
    values = [v for _, v in got]
    assert values == sorted(values)


def test_percentile_aggregation_handles_partial_runs():
    runs = [[(0.1, 10), (0.5, 20)], [(0.5, 24), (0.9, 40)], []]
    got = dict(aggregate_percentiles(runs))
    assert got[0.1] == pytest.approx(10)
    assert got[0.9] == pytest.approx(40)


def test_widening_pushes_out_and_keeps_the_median():
    points = [(0.1, 10), (0.5, 20), (0.9, 30)]
    wide = dict(widen_percentiles(points, 1.5))
    assert wide[0.5] == pytest.approx(20)
    assert wide[0.1] < 10
    assert wide[0.9] > 30


def test_multiple_choice_aggregation_and_calibration_sum_to_one():
    options = ["A", "B", "C"]
    runs = [
        {"A": 0.7, "B": 0.2, "C": 0.1},
        {"A": 0.8, "B": 0.15, "C": 0.05},
        {"A": 0.6, "B": 0.3, "C": 0.1},
    ]
    merged = aggregate_multiple_choice(runs, options)
    out = calibrate_multiple_choice(merged, options)
    assert sum(out.values()) == pytest.approx(1.0, abs=1e-9)
    assert all(0.001 <= v <= 0.999 for v in out.values())
    assert out["A"] > out["B"] > out["C"]


@pytest.mark.parametrize("k", [2, 3, 5, 12, 40])
def test_multiple_choice_calibration_always_submittable(k):
    options = [f"opt{i}" for i in range(k)]
    nasty = {options[0]: 1.0}
    out = calibrate_multiple_choice(nasty, options)
    assert sum(out.values()) == pytest.approx(1.0, abs=1e-9)
    assert all(0.001 <= v <= 0.999 for v in out.values()), out
    assert len(out) == k


def test_multiple_choice_never_zeroes_an_option():
    options = ["A", "B", "C"]
    out = calibrate_multiple_choice({"A": 1.0, "B": 0.0, "C": 0.0}, options)
    assert min(out.values()) >= MC_FLOOR * 0.5
    assert sum(out.values()) == pytest.approx(1.0, abs=1e-9)


def test_multiple_choice_aggregation_survives_no_usable_runs():
    assert aggregate_multiple_choice([None, None], ["A", "B"]) is None
