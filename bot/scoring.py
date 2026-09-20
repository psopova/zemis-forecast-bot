"""The tournament's own scoring rules, reimplemented so we can measure ourselves.

Mirrors scoring/score_math.py and utils/the_math/formulas.py on the Metaculus
server. Kept faithful rather than tidy: the point is to be able to say what a
forecast would actually have scored, not what a reasonable scoring rule would
give it.

Tournament rank is the sum of spot peer scores over all questions, and the
share of the prize pool is proportional to that sum SQUARED. Two consequences
drive every design choice in this bot: a question not forecast contributes 0
and is therefore a forfeited fifteen-odd points, and one confident blowup can
undo dozens of good questions.
"""

from __future__ import annotations

import math
from typing import Sequence

from .scaling import Scaling

DEFAULT_INBOUND_OUTCOME_COUNT = 200


# -- turning a forecast into the pmf the server scores ---------------------
def binary_pmf(probability_yes: float) -> list[float]:
    return [1.0 - probability_yes, probability_yes]


def continuous_pmf(cdf: Sequence[float]) -> list[float]:
    pmf = [cdf[0]]
    for i in range(1, len(cdf)):
        pmf.append(cdf[i] - cdf[i - 1])
    pmf.append(1.0 - cdf[-1])
    return pmf


def bucket_index(u: float, inbound_outcome_count: int) -> int:
    """Which pmf bucket an unscaled outcome falls in.

    Bucket 0 is strictly below the lower bound and the last bucket is above the
    upper bound, so an outcome exactly at range_min belongs in bucket 1.
    """
    if u < 0:
        return 0
    if u > 1:
        return inbound_outcome_count + 1
    if u == 1:
        return inbound_outcome_count
    return max(int(u * inbound_outcome_count + 1 - 1e-10), 1)


def resolution_bucket_continuous(
    resolution_value: float, scaling: Scaling, inbound_outcome_count: int
) -> int:
    return bucket_index(scaling.to_unscaled(resolution_value), inbound_outcome_count)


# -- scores ----------------------------------------------------------------
def baseline_score(
    pmf: Sequence[float],
    resolution_bucket: int,
    continuous: bool,
    open_bounds_count: int = 0,
) -> float:
    """Score against a fixed uninformed baseline. Range -897 to +100 on binary."""
    if not continuous:
        options_at_time = sum(1 for v in pmf if not math.isnan(v))
        p = pmf[resolution_bucket]
        if math.isnan(p):
            p = pmf[-1]
        p = max(p, 1e-12)
        return 100.0 * math.log(p * options_at_time) / math.log(options_at_time)

    if resolution_bucket in (0, len(pmf) - 1):
        baseline = 0.05
    else:
        baseline = (1 - 0.05 * open_bounds_count) / (len(pmf) - 2)
    p = max(pmf[resolution_bucket], 1e-12)
    return 100.0 * math.log(p / baseline) / 2


def peer_score(
    p: float,
    others: Sequence[float],
    continuous: bool = False,
) -> float:
    """Score against the field, the number the leaderboard actually ranks on.

    100 * N/(N-1) * ln(your probability on the truth / the geometric mean of
    everyone's), halved for continuous questions. ``others`` are the other
    forecasters' probabilities on the realised outcome; the geometric mean
    includes your own, exactly as the server computes it.
    """
    values = [float(v) for v in others if v is not None and v > 0] + [float(p)]
    n = len(values)
    if n < 2:
        return 0.0
    log_mean = sum(math.log(max(v, 1e-12)) for v in values) / n
    gmp = math.exp(log_mean)
    score = 100.0 * (n / (n - 1)) * math.log(max(p, 1e-12) / gmp)
    return score / 2 if continuous else score


def brier(p: float, outcome: int) -> float:
    return (p - outcome) ** 2


def log_score(p: float, outcome: int) -> float:
    """Natural log score. Higher is better, 0 is perfect."""
    q = p if outcome == 1 else 1 - p
    return math.log(max(q, 1e-12))


def prize_share(my_total: float, all_totals: Sequence[float]) -> float:
    """Fraction of the pool a total score earns, under the squared rule.

    Negative totals earn nothing, and below roughly a 480 total the $50 minimum
    payout rule drops you entirely, so this is optimistic at the bottom.
    """
    if my_total <= 0:
        return 0.0
    positives = [t ** 2 for t in all_totals if t > 0]
    denom = sum(positives)
    return (my_total ** 2) / denom if denom > 0 else 0.0
