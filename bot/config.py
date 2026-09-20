"""Run configuration.

Tournament ids are read from the environment with slugs as defaults, because a
slug keeps working when Metaculus renumbers a season and the official SDK's
hardcoded id does not. TOURNAMENTS can be overridden entirely for a test run.
"""

from __future__ import annotations

import os

# Fall 2026 FutureEval runs 28 Sep 2026 to 6 Jan 2027, $50k.
# MiniBench is a rolling $1k round every two weeks.
SEASONAL_SLUG = os.environ.get("SEASONAL_TOURNAMENT", "fall-futureeval-2026")
MINIBENCH_SLUG = os.environ.get("MINIBENCH_TOURNAMENT", "minibench")
TEST_SLUG = os.environ.get("TEST_TOURNAMENT", "bot-testing-area")

MODES: dict[str, list[str]] = {
    "tournament": [SEASONAL_SLUG, MINIBENCH_SLUG],
    "seasonal": [SEASONAL_SLUG],
    "minibench": [MINIBENCH_SLUG],
    "test": [TEST_SLUG],
}


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# Five runs per question across three model families. The bot-maker survey put
# winners at roughly 28 LLM calls per question against 7 for non-winners, so
# this is the floor rather than the ceiling; raise it once the credit
# allocation is known.
RUNS_PER_QUESTION = _int_env("RUNS_PER_QUESTION", 5)
ENSEMBLE_MODELS = _int_env("ENSEMBLE_MODELS", 3)

# Questions are open to bots for about three hours, so a tick that takes longer
# than a few minutes risks missing the window entirely.
MAX_QUESTIONS_PER_TICK = _int_env("MAX_QUESTIONS_PER_TICK", 25)
QUESTION_WORKERS = _int_env("QUESTION_WORKERS", 3)
