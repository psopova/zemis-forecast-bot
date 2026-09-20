"""Build continuous CDFs that the Metaculus API will actually accept.

The server validator (questions/serializers/common.py::continuous_validation)
rounds the CDF to 10 decimal places and the implied bucket masses to 9, then
requires:

  * exactly ``inbound_outcome_count + 1`` values, evenly spaced in unscaled space
  * every bucket mass >= round(0.01 / inbound_outcome_count, 9)
  * every bucket mass <= 0.2 * 200 / inbound_outcome_count
  * cdf[0] == 0.0 exactly when the lower bound is closed, else >= 0.001
  * cdf[-1] == 1.0 exactly when the upper bound is closed, else <= 0.999

A rejected CDF scores zero on that question, which under a squared-score prize
rule is far more expensive than a slightly blunt distribution. So every CDF is
checked against a local mirror of the validator before it leaves this process,
and anything that fails falls back to a wide but legal distribution.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

from .scaling import Scaling

DEFAULT_INBOUND_OUTCOME_COUNT = 200

# Fraction of mass spread uniformly across the range. This is deliberate
# insurance: it bounds how badly a confidently wrong distribution can score,
# and it is what guarantees the minimum-step rule without any repair pass.
UNIFORM_MIX = 0.05

# Hard floor/ceiling on out-of-range mass for open bounds. 0.001 is the server
# limit; the margin absorbs rounding.
MIN_TAIL = 0.005
MAX_TAIL = 0.20

# Report, but do not treat as an error, a model that puts this much mass outside
# a bound the question says is impossible. Shows up in the run log as a prompt bug.
OUT_OF_RANGE_NOTE_THRESHOLD = 0.02


def min_bucket_mass(inbound_outcome_count: int) -> float:
    return float(np.round(0.01 / inbound_outcome_count, 9))


def max_bucket_mass(inbound_outcome_count: int) -> float:
    return 0.2 * DEFAULT_INBOUND_OUTCOME_COUNT / inbound_outcome_count


def validate_cdf(
    cdf: Sequence[float],
    inbound_outcome_count: int,
    open_lower_bound: bool,
    open_upper_bound: bool,
) -> list[str]:
    """Local mirror of the server validator. Empty list means it will be accepted."""
    errors: list[str] = []
    rounded = np.round(np.asarray(cdf, dtype=float), 10).tolist()
    pmf = np.round([rounded[i + 1] - rounded[i] for i in range(len(rounded) - 1)], 9)

    if len(rounded) != inbound_outcome_count + 1:
        errors.append(
            f"length {len(rounded)}, expected {inbound_outcome_count + 1}"
        )
        return errors  # everything below assumes the right length

    lo = min_bucket_mass(inbound_outcome_count)
    hi = max_bucket_mass(inbound_outcome_count)
    if not all(pmf >= lo):
        errors.append(f"bucket mass below minimum {lo} (min was {pmf.min()})")
    if not all(pmf <= hi):
        errors.append(f"bucket mass above maximum {hi} (max was {pmf.max()})")

    if open_lower_bound:
        if not rounded[0] >= 0.001:
            errors.append(f"open lower bound needs cdf[0] >= 0.001, got {rounded[0]}")
    elif rounded[0] != 0.00:
        errors.append(f"closed lower bound needs cdf[0] == 0.0, got {rounded[0]}")

    if open_upper_bound:
        if not rounded[-1] <= 0.999:
            errors.append(f"open upper bound needs cdf[-1] <= 0.999, got {rounded[-1]}")
    elif rounded[-1] != 1.00:
        errors.append(f"closed upper bound needs cdf[-1] == 1.0, got {rounded[-1]}")

    return errors


def _monotone_interp(xs: np.ndarray, ys: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Monotone cubic Hermite (Fritsch-Carlson) with linear extrapolation.

    Linear interpolation of a CDF gives a blocky step density; monotone cubic
    gives a smooth one without ever introducing a decrease, which is what the
    server's minimum-step rule cares about.
    """
    n = len(xs)
    if n == 1:
        return np.full_like(grid, ys[0])
    h = np.diff(xs)
    delta = np.diff(ys) / h

    if n == 2:
        m = np.array([delta[0], delta[0]])
    else:
        m = np.zeros(n)
        m[1:-1] = (delta[:-1] + delta[1:]) / 2.0
        m[0] = delta[0]
        m[-1] = delta[-1]
        # Fritsch-Carlson: clamp slopes so no overshoot can occur.
        for i in range(n - 1):
            if delta[i] == 0:
                m[i] = m[i + 1] = 0.0
            else:
                a = m[i] / delta[i]
                b = m[i + 1] / delta[i]
                s = a * a + b * b
                if s > 9.0:
                    t = 3.0 / math.sqrt(s)
                    m[i] = t * a * delta[i]
                    m[i + 1] = t * b * delta[i]
        m = np.clip(m, 0.0, None)  # ys is non-decreasing, so slopes are too

    out = np.empty_like(grid, dtype=float)
    idx = np.clip(np.searchsorted(xs, grid, side="right") - 1, 0, n - 2)
    for k, g in enumerate(grid):
        if g <= xs[0]:
            out[k] = ys[0] + m[0] * (g - xs[0])
            continue
        if g >= xs[-1]:
            out[k] = ys[-1] + m[-1] * (g - xs[-1])
            continue
        i = idx[k]
        t = (g - xs[i]) / h[i]
        t2, t3 = t * t, t * t * t
        out[k] = (
            (2 * t3 - 3 * t2 + 1) * ys[i]
            + (t3 - 2 * t2 + t) * h[i] * m[i]
            + (-2 * t3 + 3 * t2) * ys[i + 1]
            + (t3 - t2) * h[i] * m[i + 1]
        )
    return out


def _clean_anchors(points: Iterable[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    """Sort by position, drop duplicates, force strictly increasing probabilities."""
    pts = sorted((float(u), float(p)) for u, p in points)
    us: list[float] = []
    ps: list[float] = []
    for u, p in pts:
        p = min(max(p, 1e-6), 1.0 - 1e-6)
        if us and u - us[-1] < 1e-9:
            ps[-1] = max(ps[-1], p)  # same position, keep the higher probability
            continue
        if ps and p <= ps[-1]:
            p = ps[-1] + 1e-6  # an LLM that emits a non-monotone set gets repaired
        us.append(u)
        ps.append(p)
    return np.asarray(us, dtype=float), np.asarray(ps, dtype=float)


def _assemble(
    lo_mass: float,
    hi_mass: float,
    interior: np.ndarray,
    inbound_outcome_count: int,
    uniform_mix: float = UNIFORM_MIX,
) -> list[float]:
    """Turn bucket masses into a rounded, exactly-summing CDF."""
    target = 1.0 - lo_mass - hi_mass
    interior = np.clip(interior, 0.0, None)
    total = interior.sum()
    interior = interior / total * target if total > 0 else np.full_like(interior, target / len(interior))

    m = len(interior)
    uniform_mix = min(max(uniform_mix, 0.01), 0.5)
    interior = (1.0 - uniform_mix) * interior + uniform_mix * target / m

    # Cap any spike, spreading the excess over the buckets that are not capped.
    cap = max_bucket_mass(inbound_outcome_count) * 0.95
    for _ in range(50):
        over = interior > cap
        if not over.any():
            break
        excess = float((interior[over] - cap).sum())
        interior[over] = cap
        room = ~over
        if not room.any():
            break
        interior[room] += excess / int(room.sum())

    interior = np.round(interior, 10)
    residual = target - float(math.fsum(interior.tolist()))
    interior[int(np.argmax(interior))] += residual
    interior = np.round(interior, 10)

    vals = [round(lo_mass, 10)]
    running = interior.tolist()
    for i in range(m):
        vals.append(round(lo_mass + math.fsum(running[: i + 1]), 10))
    return vals


def safe_cdf(
    inbound_outcome_count: int = DEFAULT_INBOUND_OUTCOME_COUNT,
    open_lower_bound: bool = False,
    open_upper_bound: bool = False,
    uniform_mix: float = UNIFORM_MIX,
) -> list[float]:
    """A legal, deliberately uninformative fallback: uniform over the range."""
    lo = MIN_TAIL if open_lower_bound else 0.0
    hi = MIN_TAIL if open_upper_bound else 0.0
    interior = np.full(inbound_outcome_count, (1.0 - lo - hi) / inbound_outcome_count)
    return _assemble(lo, hi, interior, inbound_outcome_count, uniform_mix)


def build_cdf(
    percentile_points: Sequence[tuple[float, float]],
    scaling: Scaling,
    open_lower_bound: bool,
    open_upper_bound: bool,
    inbound_outcome_count: int = DEFAULT_INBOUND_OUTCOME_COUNT,
    uniform_mix: float = UNIFORM_MIX,
) -> tuple[list[float], list[str]]:
    """Build a submittable CDF from ``(probability, nominal value)`` pairs.

    Returns the CDF and a list of notes about repairs that were applied. The
    returned CDF is always valid: if construction fails it degrades to
    ``safe_cdf`` rather than raising, because a blunt forecast scores and a
    rejected one does not.
    """
    notes: list[str] = []
    n = inbound_outcome_count + 1

    usable = []
    for p, value in percentile_points:
        try:
            p = float(p)
            value = float(value)
        except (TypeError, ValueError):
            notes.append(f"dropped non-numeric percentile point {(p, value)!r}")
            continue
        if not math.isfinite(p) or not math.isfinite(value) or not (0.0 < p < 1.0):
            notes.append(f"dropped out-of-range percentile point {(p, value)!r}")
            continue
        usable.append((scaling.to_unscaled(value), p))

    if len(usable) < 2:
        notes.append("fewer than two usable percentile points, falling back to uniform")
        return safe_cdf(inbound_outcome_count, open_lower_bound, open_upper_bound, uniform_mix), notes

    us, ps = _clean_anchors(usable)
    grid = np.linspace(0.0, 1.0, n)
    raw = _monotone_interp(us, ps, grid)
    raw = np.clip(np.maximum.accumulate(raw), 0.0, 1.0)

    lo_mass = float(raw[0])
    hi_mass = float(1.0 - raw[-1])

    if open_lower_bound:
        lo_mass = min(max(lo_mass, MIN_TAIL), MAX_TAIL)
    else:
        # A closed bound means the outcome cannot land outside the range, so mass
        # the model put there is a scale error on its part. Renormalising the
        # interior conditions the model's own shape on the range, which keeps the
        # distribution honest; piling that mass on the edge bucket instead would
        # assert a spike at the cap that the model never actually claimed.
        if lo_mass > OUT_OF_RANGE_NOTE_THRESHOLD:
            notes.append(
                f"model put {lo_mass:.1%} of mass below a closed lower bound; "
                "conditioned on the range"
            )
        lo_mass = 0.0
    if open_upper_bound:
        hi_mass = min(max(hi_mass, MIN_TAIL), MAX_TAIL)
    else:
        if hi_mass > OUT_OF_RANGE_NOTE_THRESHOLD:
            notes.append(
                f"model put {hi_mass:.1%} of mass above a closed upper bound; "
                "conditioned on the range"
            )
        hi_mass = 0.0
    lo_mass = round(lo_mass, 10)
    hi_mass = round(hi_mass, 10)

    interior = np.diff(raw)
    cdf = _assemble(lo_mass, hi_mass, interior, inbound_outcome_count, uniform_mix)

    errors = validate_cdf(cdf, inbound_outcome_count, open_lower_bound, open_upper_bound)
    if errors:
        notes.append("built CDF failed local validation: " + "; ".join(errors))
        cdf = safe_cdf(inbound_outcome_count, open_lower_bound, open_upper_bound, uniform_mix)
    return cdf, notes


def percentiles_from_cdf(
    cdf: Sequence[float], scaling: Scaling, probs: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9)
) -> dict[float, float]:
    """Read nominal values back out of a CDF, for logging and comments."""
    arr = np.asarray(cdf, dtype=float)
    grid = np.linspace(0.0, 1.0, len(arr))
    out: dict[float, float] = {}
    for p in probs:
        i = int(np.searchsorted(arr, p, side="left"))
        if i <= 0:
            u = 0.0
        elif i >= len(arr):
            u = 1.0
        else:
            span = arr[i] - arr[i - 1]
            frac = 0.0 if span <= 0 else (p - arr[i - 1]) / span
            u = grid[i - 1] + frac * (grid[i] - grid[i - 1])
        out[p] = scaling.to_nominal(float(u))
    return out
