"""Turning several model runs into one submitted forecast.

Two interventions here, both taken from measured results rather than taste.

Ensembling across model families is the single most reproducible improvement in
the bot-maker surveys (+1,799 points, 86% of winners did it), and the median is
used rather than the mean because one member hallucinating a wild number should
not move the answer at all.

Extremity capping was the strongest within-winner differentiator (r = +0.48,
p = 0.005). The reason is arithmetic. Tournament rank is the sum of peer scores,
100 * ln(your probability on the truth / the field's geometric mean), and prize
money is proportional to that sum squared. Saying 0.2% and being wrong, against
a field at 50%, scores about -621. Saying 5% instead scores about -230. The
insurance costs almost nothing on the other side: 95% instead of 99.9% when you
are right gives up about 5 points. So capping pays whenever the model's stated
0.2% confidence is wrong more than roughly 1 time in 30, and LLMs at that
confidence are wrong far more often than that.

Capping alone is a blunt clip, so the probability is first shrunk toward even
odds in log-odds space (logistic recalibration, measured at a Brier improvement
of 0.016 on binary questions), and the hard cap is left as a backstop.
"""

from __future__ import annotations

import math
import statistics
from typing import Sequence

# Defaults. SHRINK is deliberately mild until the backtest fits it on real
# resolved questions; 1.0 would mean no shrinkage at all.
BINARY_SHRINK = 0.90
BINARY_FLOOR = 0.05
BINARY_CEILING = 0.95

# Multiple choice is the question type where bots lose most heavily to human
# professionals, so it gets more shrinkage toward the uniform, not less.
MC_UNIFORM_MIX = 0.08
MC_FLOOR = 0.01

# Percentile spreads from LLMs are too narrow; widening is the continuous
# analogue of capping a binary probability.
NUMERIC_WIDEN = 1.15

# Models are asked for percentiles from the 5th to the 95th, which leaves the
# outer 10% of the distribution undescribed. Interpolating out from the
# outermost pair extrapolates the steep middle slope, which produces a box with
# cliff edges: an outcome one bucket outside the elicited range scores the same
# as one at the far end of the scale. These anchors give the distribution real
# tails instead. The multipliers say that each tenfold drop in tail probability
# costs another 1.5 half-widths, which is fatter than a normal (about 0.8) and
# deliberately so, since the documented failure mode is tails that are too thin.
TAIL_ANCHORS: tuple[tuple[float, float], ...] = ((0.1, 1.5), (0.01, 3.0))


def logit(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def aggregate_binary(values: Sequence[float]) -> float | None:
    """Median of the ensemble. One bad member must not move the answer."""
    clean = [float(v) for v in values if v is not None and 0.0 < float(v) < 1.0]
    if not clean:
        return None
    return float(statistics.median(clean))


def spread(values: Sequence[float]) -> float:
    clean = [float(v) for v in values if v is not None]
    if len(clean) < 2:
        return 0.0
    return max(clean) - min(clean)


def calibrate_binary(
    p: float,
    shrink: float = BINARY_SHRINK,
    floor: float = BINARY_FLOOR,
    ceiling: float = BINARY_CEILING,
) -> float:
    """Shrink toward even odds in log-odds space, then clip."""
    shrunk = sigmoid(shrink * logit(p))
    return min(max(shrunk, floor), ceiling)


def break_even_error_rate(stated: float, cap: float) -> float:
    """How often a stated confidence must be wrong for capping it to pay.

    Used by the tests to keep the chosen floor honest rather than assumed.
    Compares what capping gives up when the forecast is right against what it
    saves when the forecast is wrong, both in log score.
    """
    stated = min(max(stated, 1e-6), 1 - 1e-6)
    gain_when_wrong = math.log(cap / stated)
    loss_when_right = math.log((1 - stated) / (1 - cap))
    if gain_when_wrong <= 0:
        return 1.0
    return loss_when_right / (loss_when_right + gain_when_wrong)


def aggregate_percentiles(
    runs: Sequence[Sequence[tuple[float, float]]],
) -> list[tuple[float, float]]:
    """Median value at each percentile across runs, then forced monotone.

    Taking the median percentile by percentile keeps the ensemble's shape. A run
    that omits a percentile simply does not vote on it.
    """
    buckets: dict[float, list[float]] = {}
    for run in runs:
        for p, value in run or []:
            if value is None or not math.isfinite(float(value)):
                continue
            buckets.setdefault(round(float(p), 6), []).append(float(value))
    if not buckets:
        return []
    merged = [(p, float(statistics.median(vs))) for p, vs in sorted(buckets.items())]

    out: list[tuple[float, float]] = []
    for p, value in merged:
        if out and value <= out[-1][1]:
            value = out[-1][1] + abs(out[-1][1]) * 1e-9 + 1e-9
        out.append((p, value))
    return out


def widen_percentiles(
    points: Sequence[tuple[float, float]], factor: float = NUMERIC_WIDEN
) -> list[tuple[float, float]]:
    """Push percentiles away from the median to counter narrow LLM spreads."""
    if not points or factor == 1.0:
        return list(points)
    values = [v for _, v in points]
    centre = float(statistics.median(values))
    return [(p, centre + (v - centre) * factor) for p, v in points]


def extend_tails(
    points: Sequence[tuple[float, float]],
    anchors: Sequence[tuple[float, float]] = TAIL_ANCHORS,
) -> list[tuple[float, float]]:
    """Add outer percentile anchors so the distribution decays instead of stopping."""
    pts = sorted((float(p), float(v)) for p, v in points)
    if len(pts) < 2:
        return pts
    values = [v for _, v in pts]
    centre = float(statistics.median(values))
    (p_lo, v_lo), (p_hi, v_hi) = pts[0], pts[-1]

    lower_width = max(centre - v_lo, 1e-12)
    upper_width = max(v_hi - centre, 1e-12)

    out = list(pts)
    for prob_mult, width_mult in anchors:
        low_p = p_lo * prob_mult
        high_p = 1.0 - (1.0 - p_hi) * prob_mult
        if 0.0 < low_p < p_lo:
            out.append((low_p, centre - lower_width * (1.0 + width_mult)))
        if p_hi < high_p < 1.0:
            out.append((high_p, centre + upper_width * (1.0 + width_mult)))
    return sorted(out)


def aggregate_multiple_choice(
    runs: Sequence[dict[str, float] | None], options: Sequence[str]
) -> dict[str, float] | None:
    """Median per option across runs, renormalised."""
    usable = [r for r in runs if r]
    if not usable:
        return None
    merged: dict[str, float] = {}
    for option in options:
        votes = [float(r[option]) for r in usable if option in r]
        merged[option] = float(statistics.median(votes)) if votes else 0.0
    total = sum(merged.values())
    if total <= 0:
        return {o: 1.0 / len(options) for o in options}
    return {o: v / total for o, v in merged.items()}


def calibrate_multiple_choice(
    probs: dict[str, float],
    options: Sequence[str],
    uniform_mix: float = MC_UNIFORM_MIX,
    floor: float = MC_FLOOR,
) -> dict[str, float]:
    """Mix toward the uniform, enforce a floor, and make the result sum to one.

    The server rejects any option outside [0.001, 0.999] and any set that does
    not sum to 1.0, and a rejected forecast scores zero, so this ends with an
    exact renormalisation rather than a hopeful one.
    """
    k = len(options)
    if k == 0:
        return {}
    mixed = {o: (1 - uniform_mix) * float(probs.get(o, 0.0)) + uniform_mix / k for o in options}
    floor = min(floor, 0.9 / k)
    mixed = {o: max(v, floor) for o, v in mixed.items()}

    total = sum(mixed.values())
    out = {o: v / total for o, v in mixed.items()}

    # Renormalising can push a value a hair under the floor; settle the residue
    # on the largest option, which has the most room to absorb it.
    out = {o: min(max(v, 0.0011), 0.9989) for o, v in out.items()}
    residual = 1.0 - sum(out.values())
    biggest = max(out, key=lambda o: out[o])
    out[biggest] = round(out[biggest] + residual, 12)
    return out
