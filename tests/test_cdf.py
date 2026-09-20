"""Every CDF this bot can produce must pass the server's validator.

The fuzz tests here are the point of the file: a rejected CDF scores zero, and
under a squared-score prize rule zeros are the most expensive thing that can
happen. So we hammer the builder with malformed model output and assert it
always emits something legal.
"""

import math
import random

import numpy as np
import pytest


from bot.cdf import (
    DEFAULT_INBOUND_OUTCOME_COUNT,
    build_cdf,
    max_bucket_mass,
    min_bucket_mass,
    percentiles_from_cdf,
    safe_cdf,
    validate_cdf,
)
from bot.scaling import Scaling

BOUNDS = [(False, False), (True, False), (False, True), (True, True)]
COUNTS = [2, 9, 11, 100, 200, 201]


def assert_valid(cdf, count, lo_open, hi_open):
    errors = validate_cdf(cdf, count, lo_open, hi_open)
    assert not errors, f"{errors} (count={count}, bounds={lo_open},{hi_open})"


# --------------------------------------------------------------------------
# The fallback must itself be legal, or the safety net has a hole in it.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("count", COUNTS)
@pytest.mark.parametrize("lo_open,hi_open", BOUNDS)
def test_safe_cdf_is_always_valid(count, lo_open, hi_open):
    assert_valid(safe_cdf(count, lo_open, hi_open), count, lo_open, hi_open)


# --------------------------------------------------------------------------
# Normal use.
# --------------------------------------------------------------------------
def test_ordinary_percentiles_linear():
    s = Scaling(0, 100)
    pts = [(0.05, 20), (0.25, 35), (0.5, 50), (0.75, 65), (0.95, 80)]
    cdf, notes = build_cdf(pts, s, False, False)
    assert_valid(cdf, 200, False, False)
    assert notes == []
    read = percentiles_from_cdf(cdf, s, [0.5])
    assert abs(read[0.5] - 50) < 3


def test_ordinary_percentiles_log_scaled():
    s = Scaling(1, 1_000_000, zero_point=0)
    pts = [(0.1, 100), (0.5, 5_000), (0.9, 200_000)]
    cdf, notes = build_cdf(pts, s, True, True)
    assert_valid(cdf, 200, True, True)
    read = percentiles_from_cdf(cdf, s, [0.5])
    assert 1_000 < read[0.5] < 25_000, read


def test_median_tracks_input_when_the_spread_fits_inside_the_range():
    s = Scaling(0, 1000)
    for median in (200, 500, 800):
        half = min(median, 1000 - median) * 0.5
        pts = [(0.1, median - half), (0.5, median), (0.9, median + half)]
        cdf, notes = build_cdf(pts, s, False, False)
        assert_valid(cdf, 200, False, False)
        assert notes == []
        read = percentiles_from_cdf(cdf, s, [0.5])[0.5]
        assert abs(read - median) < 25, (median, read)


def test_mass_outside_a_closed_bound_is_conditioned_and_reported():
    """The model claimed the impossible; we condition on the range and say so.

    The alternative, piling the excess onto the edge bucket, would assert a
    spike at the cap that the model never stated. Conditioning keeps the shape
    the model did state, at the cost of pulling the median inward, and the note
    makes the prompt bug visible in the run log.
    """
    s = Scaling(0, 1000)
    pts = [(0.1, 480), (0.5, 800), (0.9, 1130)]
    cdf, notes = build_cdf(pts, s, False, False)
    assert_valid(cdf, 200, False, False)
    assert any("closed upper bound" in n for n in notes), notes
    read = percentiles_from_cdf(cdf, s, [0.5])[0.5]
    assert 600 < read < 800, read


def test_same_input_against_an_open_upper_bound_keeps_the_median():
    """With the bound open, the overflow is legal mass and the median holds."""
    s = Scaling(0, 1000)
    pts = [(0.1, 480), (0.5, 800), (0.9, 1130)]
    cdf, notes = build_cdf(pts, s, False, True)
    assert_valid(cdf, 200, False, True)
    assert notes == []
    read = percentiles_from_cdf(cdf, s, [0.5])[0.5]
    assert abs(read - 800) < 60, read


# --------------------------------------------------------------------------
# Malformed model output. None of these may raise or produce an invalid CDF.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "pts",
    [
        [],
        [(0.5, 50)],
        [(0.5, 50), (0.5, 50)],
        [(0.9, 10), (0.5, 50), (0.1, 90)],            # probabilities reversed
        [(0.1, 50), (0.5, 50), (0.9, 50)],            # a point mass
        [(0.1, -1e9), (0.5, 0), (0.9, 1e9)],          # values far outside the range
        [(0.0, 10), (1.0, 90)],                       # degenerate probabilities
        [(-0.5, 10), (1.5, 90), (0.5, 50)],
        [(float("nan"), 10), (0.5, 50), (0.9, 70)],
        [(0.5, float("inf")), (0.9, 70)],
        [(0.1, None), (0.5, 50), (0.9, 70)],
        [("x", "y"), (0.5, 50), (0.9, 70)],
        [(0.5, 50), (0.500000001, 50.0000001), (0.9, 70)],
        [(p / 101, p) for p in range(1, 101)],        # a hundred points
    ],
)
@pytest.mark.parametrize("lo_open,hi_open", BOUNDS)
def test_malformed_input_still_yields_a_valid_cdf(pts, lo_open, hi_open):
    cdf, _ = build_cdf(pts, Scaling(0, 100), lo_open, hi_open)
    assert_valid(cdf, 200, lo_open, hi_open)


def test_spike_is_capped_not_rejected():
    s = Scaling(0, 100)
    pts = [(0.01, 49.9), (0.5, 50.0), (0.99, 50.1)]
    cdf, _ = build_cdf(pts, s, False, False)
    assert_valid(cdf, 200, False, False)
    pmf = np.diff(cdf)
    assert pmf.max() <= max_bucket_mass(200)


def test_closed_bounds_are_exact():
    cdf, _ = build_cdf([(0.1, 10), (0.9, 90)], Scaling(0, 100), False, False)
    assert cdf[0] == 0.0
    assert cdf[-1] == 1.0


def test_open_bounds_keep_mass_outside():
    cdf, _ = build_cdf([(0.1, 10), (0.9, 90)], Scaling(0, 100), True, True)
    assert cdf[0] >= 0.001
    assert cdf[-1] <= 0.999


def test_uniform_mix_bounds_the_worst_case():
    """A confidently wrong distribution still assigns real mass everywhere."""
    s = Scaling(0, 100)
    pts = [(0.05, 1.0), (0.5, 2.0), (0.95, 3.0)]
    cdf, _ = build_cdf(pts, s, False, False)
    assert_valid(cdf, 200, False, False)
    pmf = np.diff(cdf)
    # Truth landing at the far end still gets a scoreable, not catastrophic, mass.
    assert pmf[-1] >= min_bucket_mass(200) * 2


# --------------------------------------------------------------------------
# Fuzz.
# --------------------------------------------------------------------------
def test_fuzz_random_inputs_are_always_valid():
    rng = random.Random(20260920)
    for _ in range(3000):
        count = rng.choice(COUNTS)
        lo_open, hi_open = rng.choice(BOUNDS)
        rmin = rng.uniform(-1e6, 1e6)
        rmax = rmin + rng.choice([1e-3, 1.0, 100.0, 1e6, 1e9])
        zero = None
        if rng.random() < 0.3:
            zero = rmin - rng.uniform(0.01, 1000.0)
        s = Scaling(rmin, rmax, zero)

        k = rng.randint(0, 9)
        pts = []
        for _ in range(k):
            p = rng.choice([rng.random(), rng.gauss(0.5, 0.4), 0.0, 1.0, -0.2, 1.2])
            v = rng.choice(
                [
                    rng.uniform(rmin, rmax),
                    rng.uniform(rmin - 10 * (rmax - rmin), rmax + 10 * (rmax - rmin)),
                    float("nan"),
                    float("inf"),
                    rmin,
                    rmax,
                ]
            )
            pts.append((p, v))

        cdf, _ = build_cdf(pts, s, lo_open, hi_open, count)
        assert len(cdf) == count + 1
        assert all(math.isfinite(x) for x in cdf)
        assert_valid(cdf, count, lo_open, hi_open)


def test_fuzz_well_formed_inputs_never_hit_the_fallback():
    """Sane model output should be used, not silently replaced by the uniform."""
    rng = random.Random(7)
    fallbacks = 0
    trials = 500
    for _ in range(trials):
        rmin = rng.uniform(-1000, 1000)
        rmax = rmin + rng.uniform(1, 10_000)
        s = Scaling(rmin, rmax)
        lo_open, hi_open = rng.choice(BOUNDS)
        centre = rng.uniform(rmin, rmax)
        width = (rmax - rmin) * rng.uniform(0.02, 0.4)
        pts = [
            (0.1, centre - width),
            (0.25, centre - width / 2),
            (0.5, centre),
            (0.75, centre + width / 2),
            (0.9, centre + width),
        ]
        cdf, notes = build_cdf(pts, s, lo_open, hi_open)
        assert_valid(cdf, 200, lo_open, hi_open)
        if any("fall" in n or "failed" in n for n in notes):
            fallbacks += 1
    assert fallbacks == 0, f"{fallbacks}/{trials} well-formed inputs fell back"


# --------------------------------------------------------------------------
# The full numeric path: aggregate -> widen -> extend tails -> build.
# Tail extension can push anchors a long way outside the range, so the
# validator has to hold across the whole chain, not just build_cdf alone.
# --------------------------------------------------------------------------
def test_fuzz_full_numeric_chain_is_always_valid():
    from bot.aggregate import aggregate_percentiles, extend_tails, widen_percentiles

    rng = random.Random(4242)
    for _ in range(1500):
        count = rng.choice(COUNTS)
        lo_open, hi_open = rng.choice(BOUNDS)
        rmin = rng.uniform(-1e5, 1e5)
        rmax = rmin + rng.choice([1e-2, 1.0, 500.0, 1e7])
        zero = rmin - rng.uniform(0.01, 100.0) if rng.random() < 0.25 else None
        s = Scaling(rmin, rmax, zero)

        runs = []
        for _ in range(rng.randint(1, 4)):
            centre = rng.uniform(rmin - (rmax - rmin), rmax + (rmax - rmin))
            width = (rmax - rmin) * rng.choice([1e-6, 0.001, 0.05, 0.5, 5.0])
            runs.append(
                [
                    (0.05, centre - 2 * width),
                    (0.5, centre),
                    (0.95, centre + 2 * width),
                ]
            )

        points = extend_tails(widen_percentiles(aggregate_percentiles(runs), rng.uniform(1.0, 1.6)))
        cdf, _ = build_cdf(points, s, lo_open, hi_open, count)
        assert_valid(cdf, count, lo_open, hi_open)


def test_extend_tails_edge_cases():
    from bot.aggregate import extend_tails

    assert extend_tails([]) == []
    assert extend_tails([(0.5, 10.0)]) == [(0.5, 10.0)]
    # Identical values must not produce a non-monotone set.
    flat = extend_tails([(0.1, 5.0), (0.5, 5.0), (0.9, 5.0)])
    values = [v for _, v in flat]
    assert values == sorted(values)
    probs = [p for p, _ in flat]
    assert all(0.0 < p < 1.0 for p in probs)
    assert probs == sorted(probs)
