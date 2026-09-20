"""Conversions between Metaculus unscaled [0, 1] coordinates and nominal values.

Mirrors ``utils/the_math/formulas.py`` on the Metaculus server
(``unscaled_location_to_scaled_location`` and its inverse).

A question is log scaled when ``scaling.zero_point`` is not None. Date questions
use unix timestamps as their nominal values, so the same code handles them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Scaling:
    range_min: float
    range_max: float
    zero_point: float | None = None

    @classmethod
    def from_question(cls, question: dict) -> "Scaling":
        s = question.get("scaling") or {}
        rmin = s.get("range_min")
        rmax = s.get("range_max")
        if rmin is None or rmax is None:
            raise ValueError(
                f"question {question.get('id')} has no numeric range: {s!r}"
            )
        return cls(float(rmin), float(rmax), _float_or_none(s.get("zero_point")))

    @property
    def span(self) -> float:
        return self.range_max - self.range_min

    @property
    def deriv_ratio(self) -> float | None:
        if self.zero_point is None:
            return None
        denom = self.range_min - self.zero_point
        if denom == 0:
            return None
        ratio = (self.range_max - self.zero_point) / denom
        # A ratio at or below zero makes the log transform undefined; treat the
        # question as linear rather than producing nan and failing the whole run.
        if ratio <= 0 or math.isclose(ratio, 1.0):
            return None
        return ratio

    def to_nominal(self, u: float) -> float:
        """Unscaled position in [0, 1] -> the value a reader would see."""
        ratio = self.deriv_ratio
        if ratio is None:
            return self.range_min + self.span * u
        return self.range_min + self.span * (ratio ** u - 1.0) / (ratio - 1.0)

    def to_unscaled(self, value: float) -> float:
        """Nominal value -> position in [0, 1]. May fall outside [0, 1]."""
        ratio = self.deriv_ratio
        if ratio is None:
            if self.span == 0:
                return 0.0
            return (value - self.range_min) / self.span
        inner = (value - self.range_min) * (ratio - 1.0) + self.span
        if inner <= 0:
            # Below the representable floor of a log scaled question.
            return -1.0
        return (math.log(inner) - math.log(self.span)) / math.log(ratio)


def _float_or_none(v) -> float | None:
    return None if v is None else float(v)
