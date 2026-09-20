"""Checked against values Metaculus publishes, so the harness cannot flatter us."""

import math

import pytest


from bot.scaling import Scaling
from bot.scoring import (
    baseline_score,
    binary_pmf,
    bucket_index,
    continuous_pmf,
    peer_score,
    prize_share,
    resolution_bucket_continuous,
)


def test_binary_baseline_matches_published_range():
    """Metaculus documents the binary baseline score as running -897 to +100."""
    assert baseline_score(binary_pmf(0.5), 1, continuous=False) == pytest.approx(0.0)
    assert baseline_score(binary_pmf(0.999), 1, continuous=False) == pytest.approx(99.86, abs=0.1)
    assert baseline_score(binary_pmf(0.001), 1, continuous=False) == pytest.approx(-896.6, abs=0.5)


def test_binary_baseline_is_symmetric_about_the_outcome():
    for p in (0.1, 0.3, 0.7, 0.9):
        yes = baseline_score(binary_pmf(p), 1, continuous=False)
        no = baseline_score(binary_pmf(1 - p), 0, continuous=False)
        assert yes == pytest.approx(no)


def test_peer_score_is_zero_against_an_identical_field():
    assert peer_score(0.4, [0.4, 0.4, 0.4]) == pytest.approx(0.0, abs=1e-9)


def test_peer_score_rewards_beating_the_geometric_mean():
    assert peer_score(0.8, [0.4, 0.4, 0.4]) > 0
    assert peer_score(0.2, [0.4, 0.4, 0.4]) < 0


def test_peer_score_matches_the_published_winning_average():
    """Spring 2026's winner averaged +18.90, meaning about 1.21x the field's odds."""
    ratio = math.exp(18.90 / 100)
    assert ratio == pytest.approx(1.208, abs=0.005)
    n = 50
    field = [0.5] * (n - 1)
    mine = 0.5 * ratio
    assert peer_score(mine, field) == pytest.approx(18.90, abs=0.6)


def test_continuous_peer_score_is_halved():
    field = [0.01] * 20
    assert peer_score(0.02, field, continuous=True) == pytest.approx(
        peer_score(0.02, field, continuous=False) / 2
    )


def test_capping_arithmetic_that_the_bot_relies_on():
    """A 0.1% blowup against a 50% field costs roughly 33 good questions."""
    field = [0.5] * 99
    disaster = peer_score(0.001, field)
    capped = peer_score(0.05, field)
    assert disaster < -600
    assert capped > -240
    assert abs(disaster) / 19 > 30  # good questions needed to undo it


def test_bucket_index_edges():
    n = 200
    assert bucket_index(-0.1, n) == 0
    assert bucket_index(0.0, n) == 1, "exactly at the lower bound is inbound"
    assert bucket_index(1.0, n) == n
    assert bucket_index(1.5, n) == n + 1
    assert bucket_index(0.5, n) == 100


def test_continuous_pmf_length_and_sum():
    cdf = [i / 200 for i in range(201)]
    pmf = continuous_pmf(cdf)
    assert len(pmf) == 202
    assert sum(pmf) == pytest.approx(1.0)


def test_resolution_bucket_on_a_real_scaling():
    s = Scaling(0, 1000)
    assert resolution_bucket_continuous(500, s, 200) == 100
    assert resolution_bucket_continuous(-5, s, 200) == 0
    assert resolution_bucket_continuous(1500, s, 200) == 201


def test_continuous_baseline_uses_the_out_of_range_baseline_at_the_edges():
    cdf = [min(1.0, 0.005 + i / 200 * 0.99) for i in range(201)]
    pmf = continuous_pmf(cdf)
    edge = baseline_score(pmf, 0, continuous=True, open_bounds_count=1)
    middle = baseline_score(pmf, 100, continuous=True, open_bounds_count=1)
    assert edge != middle


def test_prize_share_is_quadratic():
    """Doubling your score quadruples your share, which is why coverage compounds."""
    field = [1000, 2000, 3000]
    assert prize_share(2000, field) == pytest.approx(4 * prize_share(1000, field))
    assert prize_share(-5, field) == 0.0
