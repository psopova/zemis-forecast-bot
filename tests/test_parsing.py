import sys

import pytest


from bot.parsing import (
    parse_multiple_choice,
    parse_number,
    parse_percentiles,
    parse_probability,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("42", 42.0),
        ("42.5", 42.5),
        ("1,234", 1234.0),
        ("1 234 567", 1234567.0),
        ("$1,500", 1500.0),
        ("€2.5", 2.5),
        ("37%", 37.0),
        ("1.2 million", 1_200_000.0),
        ("3.4bn", 3.4e9),
        ("250k", 250_000.0),
        ("-15", -15.0),
        ("−15", -15.0),
        ("1e6", 1e6),
        ("about 88 units", 88.0),
        ("", None),
        ("none", None),
        ("N/A", None),
    ],
)
def test_parse_number(raw, expected):
    got = parse_number(raw)
    if expected is None:
        assert got is None
    else:
        assert got == pytest.approx(expected)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("blah\nPROBABILITY: 37%", 0.37),
        ("PROBABILITY: 37", 0.37),
        ("**PROBABILITY: 8%**", 0.08),
        ("probability: 0.42", 0.42),
        ("The probability is 65%.", 0.65),
        ("P(yes) = 0.9", 0.9),
        ("PROBABILITY: 5%\nsome trailing note\nPROBABILITY: 12%", 0.12),
        ("PROBABILITY: 0%", 0.001),
        ("PROBABILITY: 100%", 0.999),
        ("no number here", None),
        ("", None),
    ],
)
def test_parse_probability(text, expected):
    got = parse_probability(text)
    if expected is None:
        assert got is None
    else:
        assert got == pytest.approx(expected, abs=1e-9)


def test_parse_percentiles_clean():
    text = """Reasoning...
PERCENTILES:
5: 10
10: 12
20: 15
40: 20
60: 25
80: 32
90: 40
95: 55
"""
    got = parse_percentiles(text)
    assert [p for p, _ in got] == [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95]
    assert [v for _, v in got] == [10, 12, 15, 20, 25, 32, 40, 55]


def test_parse_percentiles_messy():
    text = """PERCENTILES:
**P5:** $1,200
10th percentile: 1.5k
- 20: 2,000
40 = 3.2 million
60: 4e6
80: 5,500,000
**90**: 7M
95 : 9.9 million
"""
    got = dict(parse_percentiles(text))
    assert got[0.05] == pytest.approx(1200)
    assert got[0.1] == pytest.approx(1500)
    assert got[0.2] == pytest.approx(2000)
    assert got[0.4] == pytest.approx(3.2e6)
    assert got[0.6] == pytest.approx(4e6)
    assert got[0.8] == pytest.approx(5.5e6)
    assert got[0.9] == pytest.approx(7e6)
    assert got[0.95] == pytest.approx(9.9e6)


def test_parse_percentiles_rejects_prose():
    assert parse_percentiles("I think it will be around fifty.") == []
    assert parse_percentiles("") == []


def test_parse_multiple_choice_exact():
    options = ["Fewer than 5", "5 to 10", "More than 10"]
    text = """PROBABILITIES:
Fewer than 5: 20%
5 to 10: 50%
More than 10: 30%
"""
    got = parse_multiple_choice(text, options)
    assert got == pytest.approx({"Fewer than 5": 0.2, "5 to 10": 0.5, "More than 10": 0.3})


def test_parse_multiple_choice_renormalises():
    options = ["A", "B"]
    got = parse_multiple_choice("PROBABILITIES:\nA: 30%\nB: 80%", options)
    assert sum(got.values()) == pytest.approx(1.0)
    assert got["B"] > got["A"]


def test_parse_multiple_choice_with_bullets_and_bold():
    options = ["Option one", "Option two"]
    text = "PROBABILITIES:\n- **Option one**: 70%\n- **Option two**: 30%"
    got = parse_multiple_choice(text, options)
    assert got == pytest.approx({"Option one": 0.7, "Option two": 0.3})


def test_parse_multiple_choice_gives_up_cleanly():
    assert parse_multiple_choice("nothing useful", ["A", "B"]) is None
    assert parse_multiple_choice("", ["A"]) is None


# --------------------------------------------------------------------------
# From the first run that produced real forecasts. "PROBABILITY: 1%" was read
# as certainty and submitted as a 95% yes on a question whose true answer is
# close to zero. The percent sign has to decide the scale.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text,expected",
    [
        ("PROBABILITY: 1%", 0.01),
        ("PROBABILITY: 2%", 0.02),
        ("PROBABILITY: 0.5%", 0.005),
        ("PROBABILITY: 99%", 0.99),
        ("PROBABILITY: 1", 0.01),
        ("PROBABILITY: 99", 0.99),
        ("PROBABILITY: 0.02", 0.02),
        ("PROBABILITY: .5", 0.5),
        ("probability: 0.02", 0.02),
        ("P(yes) = 0.9", 0.9),
        ("P(yes) = 1%", 0.01),
    ],
)
def test_percent_sign_decides_the_scale(text, expected):
    assert parse_probability(text) == pytest.approx(expected)


def test_a_confident_no_never_becomes_a_confident_yes():
    """The regression that matters: low stated probability stays low."""
    from bot.aggregate import calibrate_binary

    for text in ("PROBABILITY: 1%", "PROBABILITY: 0%", "PROBABILITY: 2%", "PROBABILITY: 1"):
        p = parse_probability(text)
        assert p is not None and p < 0.05, (text, p)
        assert calibrate_binary(p) < 0.5, (text, calibrate_binary(p))
