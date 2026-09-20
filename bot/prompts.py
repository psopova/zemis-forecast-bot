"""Prompts.

Shaped by what the published bot-maker surveys actually found, not by intuition:

  * Explicit base rates correlate with winning (r = +0.38; 40% of the top 15
    used them versus 7% of the bottom half).
  * Looking up how similar past questions resolved: 34% of winners, 0% of
    non-winners.
  * Status quo bias in the world, not the model, is the single most repeated
    piece of advice: most things do not change inside a short window.
  * Multi-persona prompting and "think like a Bayesian" framing both measured
    WORSE than plain reasoning, so neither appears here.
  * Date confusion is a top failure mode, so today's date, the open date and the
    resolution date are stated explicitly every time.
"""

from __future__ import annotations

from datetime import datetime, timezone

SYSTEM = (
    "You are a superforecaster producing a calibrated probability estimate. "
    "You are scored by a logarithmic rule against other forecasters, so "
    "overconfidence is punished far more harshly than it is rewarded. "
    "Reason concisely and concretely. Do not hedge in prose; put your uncertainty "
    "in the numbers."
)

_COMMON_STEPS = """Work through these steps before answering:
(a) The status quo outcome, if nothing changes between now and the resolution date.
(b) A reference class and its base rate. Say how often outcomes like this one occur, and over what sample. If you are guessing at the base rate, say so.
(c) What the evidence below actually establishes, separating reported fact from speculation. Note anything that looks stale.
(d) The strongest case for a higher answer and the strongest case for a lower one.
(e) How much can realistically change in the time remaining.

Weight the status quo heavily. Most questions resolve the boring way, and the world changes more slowly than news coverage suggests."""


def _header(ctx: dict) -> str:
    now = datetime.now(timezone.utc)
    parts = [
        f"Today's date: {now:%Y-%m-%d %H:%M} UTC.",
        f"Question: {ctx['title']}",
    ]
    if ctx.get("resolve_time"):
        parts.append(f"Scheduled resolution date: {ctx['resolve_time']}")
    if ctx.get("close_time"):
        parts.append(f"Question closes: {ctx['close_time']}")
    if ctx.get("days_left") is not None:
        parts.append(f"Time until resolution: about {ctx['days_left']} days.")
    return "\n".join(parts)


def _body(ctx: dict, research: str) -> str:
    chunks = [
        f"Resolution criteria:\n{ctx.get('resolution_criteria') or '(none given)'}",
        f"Background:\n{ctx.get('description') or '(none given)'}",
    ]
    if ctx.get("fine_print"):
        chunks.append(f"Fine print:\n{ctx['fine_print']}")
    chunks.append(f"Evidence gathered today:\n{research}")
    return "\n\n".join(chunks)


def binary_prompt(ctx: dict, research: str) -> list[dict]:
    user = f"""{_header(ctx)}

{_body(ctx, research)}

{_COMMON_STEPS}

Finish with exactly this line and nothing after it:
PROBABILITY: X%

where X is a number between 1 and 99."""
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def numeric_prompt(ctx: dict, research: str) -> list[dict]:
    bounds = []
    unit = ctx.get("unit") or ""
    if ctx.get("open_lower_bound"):
        bounds.append(f"Values below {ctx['range_min_label']} are possible.")
    else:
        bounds.append(f"The outcome cannot be below {ctx['range_min_label']}.")
    if ctx.get("open_upper_bound"):
        bounds.append(f"Values above {ctx['range_max_label']} are possible.")
    else:
        bounds.append(f"The outcome cannot be above {ctx['range_max_label']}.")

    user = f"""{_header(ctx)}

{_body(ctx, research)}

Units: {unit or 'as stated in the question'}
{' '.join(bounds)}

{_COMMON_STEPS}
(f) What an unexpectedly low outcome would look like, and an unexpectedly high one. Real distributions have fatter tails than they feel like they should.

Then give your distribution as percentiles. Use plain numbers in the question's units, with no commas, currency symbols, percent signs or words.

Finish with exactly this block and nothing after it:
PERCENTILES:
5: <value>
10: <value>
20: <value>
40: <value>
60: <value>
80: <value>
90: <value>
95: <value>

The values must increase down the list."""
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def multiple_choice_prompt(ctx: dict, research: str) -> list[dict]:
    options = ctx.get("options") or []
    listed = "\n".join(f"- {o}" for o in options)
    user = f"""{_header(ctx)}

{_body(ctx, research)}

The options are:
{listed}

{_COMMON_STEPS}

Leave real probability on options that look unlikely but are not impossible. Do not let any option fall below 1 percent unless it is genuinely ruled out by the resolution criteria.

Finish with exactly this block, one line per option, using the option text exactly as written above, and nothing after it:
PROBABILITIES:
<option text>: X%
<option text>: X%

The percentages must sum to 100."""
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def search_queries_prompt(ctx: dict) -> list[dict]:
    user = f"""Today is {datetime.now(timezone.utc):%Y-%m-%d}.

I need to research this forecasting question:
{ctx['title']}

{(ctx.get('resolution_criteria') or '')[:1200]}

Give me three news search queries that would surface the most decision-relevant
recent reporting. Make them short keyword queries, not questions, and make them
different from each other: one on the immediate subject, one on the wider
situation it sits in, and one on the mechanism or process that would decide it.

Answer with a JSON array of three strings and nothing else."""
    return [{"role": "system", "content": "You write precise search queries."}, {"role": "user", "content": user}]


SUMMARY_TEMPLATE = """Forecast produced automatically.

Models: {models}
Research sources: {sources}
Ensemble: {n_runs} runs, aggregated by {method}
{calibration}

{prediction_line}

Reasoning from one run:
{reasoning}
"""
