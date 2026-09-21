"""Reading numbers back out of model prose.

Models are told to end with a fixed block, and mostly do, but they also bold it,
add commas, write "1.2 million", append a percent sign to something that is not
a percentage, and occasionally emit the block twice. Every one of those is a
wasted ensemble member if parsing is strict, so parsing is not strict.
"""

from __future__ import annotations

import math
import re
from typing import Sequence

_MULTIPLIERS = {
    "k": 1e3,
    "thousand": 1e3,
    "m": 1e6,
    "mn": 1e6,
    "million": 1e6,
    "b": 1e9,
    "bn": 1e9,
    "billion": 1e9,
    "t": 1e12,
    "tn": 1e12,
    "trillion": 1e12,
}

# A leading dot is allowed because models write ".5" as readily as "0.5", and a
# probability that fails to parse costs the whole ensemble member.
_NUM = r"[-+]?(?:\d[\d,_\s]*(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?"


def parse_number(raw: str) -> float | None:
    """Turn a model-written quantity into a float, or None if it is not one."""
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if not text:
        return None
    text = text.replace("−", "-").replace("–", "-")
    text = re.sub(r"[\$£€¥₹]", "", text)
    text = text.replace("%", "")

    multiplier = 1.0
    for suffix, factor in sorted(_MULTIPLIERS.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"(?<=[\d\s]){re.escape(suffix)}\b", text):
            multiplier = factor
            text = re.sub(rf"(?<=[\d\s]){re.escape(suffix)}\b", "", text)
            break

    match = re.search(_NUM, text)
    if not match:
        return None
    cleaned = re.sub(r"[,\s_]", "", match.group(0))
    try:
        value = float(cleaned) * multiplier
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def parse_probability(text: str) -> float | None:
    """Find the final stated probability, as a fraction in (0, 1).

    The percent sign decides the scale, and it has to, because a bare 1 is
    ambiguous between one percent and certainty. Reading "PROBABILITY: 1%" as
    1.0 turned a model saying "almost certainly not" into a submitted 95% yes.
    On a log scoring rule that single confusion is worth roughly six hundred
    peer points in the wrong direction, so the rule is explicit: a percent sign
    always means percent, a bare value below one is already a fraction, and a
    bare value from one to a hundred is a percentage, which is the form the
    prompt asks for.
    """
    if not text:
        return None
    patterns = [
        r"PROBABILITY\s*[:=]\s*\**\s*(" + _NUM + r")\s*(%?)",
        r"probability\s*(?:is|of)?\s*[:=]?\s*\**\s*(" + _NUM + r")\s*(%)",
        r"\bP\s*\(\s*yes\s*\)\s*[:=]\s*(" + _NUM + r")\s*(%?)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.I)
        if not matches:
            continue
        raw, percent_sign = matches[-1]
        value = parse_number(raw)
        if value is None:
            continue
        if percent_sign:
            value /= 100.0
        elif value >= 1.0:
            value /= 100.0
        if value <= 0.0 or value >= 1.0:
            value = min(max(value, 0.001), 0.999)
        if 0.0 < value < 1.0:
            return value
    return None


_PERCENTILE_LINE = re.compile(
    r"^\s*[-*•]?\s*\**\s*(?:p|percentile)?\s*(\d{1,2}(?:\.\d+)?)\s*(?:st|nd|rd|th)?\s*(?:percentile)?\s*\**\s*[:=]\s*\**\s*(.+?)\s*\**\s*$",
    re.I | re.M,
)


def parse_percentiles(text: str) -> list[tuple[float, float]]:
    """Return ``(probability, value)`` pairs read from a PERCENTILES block."""
    if not text:
        return []
    block = text
    marker = re.search(r"PERCENTILES\s*[:=]?", text, flags=re.I)
    if marker:
        block = text[marker.end():]

    found: dict[float, float] = {}
    for raw_p, raw_v in _PERCENTILE_LINE.findall(block):
        try:
            p = float(raw_p)
        except ValueError:
            continue
        if not (0 < p < 100):
            continue
        value = parse_number(raw_v)
        if value is None:
            continue
        found[p / 100.0] = value

    if len(found) < 2 and marker:
        # The model ignored the block format; fall back to the whole message.
        return parse_percentiles(text.replace("PERCENTILES", "", 1)) if "PERCENTILES" in text else []
    return sorted(found.items())


def parse_multiple_choice(text: str, options: Sequence[str]) -> dict[str, float] | None:
    """Match stated percentages back to the exact option strings."""
    if not text or not options:
        return None
    block = text
    marker = re.search(r"PROBABILITIES\s*[:=]?", text, flags=re.I)
    if marker:
        block = text[marker.end():]

    result: dict[str, float] = {}
    for option in options:
        pattern = re.compile(
            r"^\s*[-*•]?\s*\**\s*" + re.escape(option) + r"\s*\**\s*[:=]\s*\**\s*(" + _NUM + r")\s*%?",
            re.I | re.M,
        )
        matches = pattern.findall(block) or pattern.findall(text)
        if matches:
            value = parse_number(matches[-1])
            if value is not None:
                result[option] = value

    if len(result) < len(options):
        # Positional fallback: lines in the given order, one number each.
        numbers = [
            parse_number(m)
            for m in re.findall(r"[:=]\s*\**\s*(" + _NUM + r")\s*%", block)
        ]
        numbers = [n for n in numbers if n is not None]
        if len(numbers) == len(options):
            result = dict(zip(options, numbers))
        else:
            return None

    total = sum(result.values())
    if total <= 0:
        return None
    return {k: v / total for k, v in result.items()}
