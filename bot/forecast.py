"""Turning one Metaculus question into one submittable forecast.

Every question type ends in the same place: a payload the API will accept and a
comment explaining it, because prize eligibility requires a comment under every
question the bot forecasts.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

from . import aggregate, parsing, prompts
from .cdf import DEFAULT_INBOUND_OUTCOME_COUNT, build_cdf, safe_cdf
from .llm import (
    LLMError,
    NoModelsAvailable,
    chat_with_fallback,
    extract_json,
    run_parallel,
)
from .scaling import Scaling

log = logging.getLogger(__name__)

CONTINUOUS_TYPES = {"numeric", "discrete", "date"}


@dataclass
class Forecast:
    question_id: int
    post_id: int
    question_type: str
    payload: dict
    comment: str
    headline: str
    notes: list[str] = field(default_factory=list)
    runs_used: int = 0
    models_used: list[str] = field(default_factory=list)


def _iso(value: Any) -> str:
    if not value:
        return ""
    return str(value)[:19].replace("T", " ")


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def date_to_epoch(raw: str) -> float | None:
    """Read a date a model wrote, as unix seconds."""
    if raw is None:
        return None
    text = str(raw).strip()
    match = re.search(r"(\d{4})[-/](\d{1,2})(?:[-/](\d{1,2}))?", text)
    if match:
        year, month, day = match.group(1), match.group(2), match.group(3) or "1"
        try:
            return datetime(int(year), int(month), int(day), tzinfo=timezone.utc).timestamp()
        except ValueError:
            return None
    match = re.search(r"\b(19|20)\d{2}\b", text)
    if match:
        try:
            return datetime(int(match.group(0)), 7, 1, tzinfo=timezone.utc).timestamp()
        except ValueError:
            return None
    return None


def build_context(post: dict, question: dict) -> dict:
    qtype = question.get("type") or ""
    close = _parse_dt(question.get("scheduled_close_time"))
    resolve = _parse_dt(question.get("scheduled_resolve_time"))
    now = datetime.now(timezone.utc)
    days_left = None
    if resolve:
        days_left = max(0, int((resolve - now).total_seconds() // 86400))

    ctx: dict[str, Any] = {
        "post_id": post.get("id"),
        "question_id": question.get("id"),
        "type": qtype,
        "title": question.get("title") or post.get("title") or "",
        "description": (question.get("description") or "")[:6000],
        "resolution_criteria": (question.get("resolution_criteria") or "")[:4000],
        "fine_print": (question.get("fine_print") or "")[:2000],
        "close_time": _iso(question.get("scheduled_close_time")),
        "resolve_time": _iso(question.get("scheduled_resolve_time")),
        "days_left": days_left,
        "options": list(question.get("options") or []),
        "unit": question.get("unit") or "",
    }

    if qtype in CONTINUOUS_TYPES:
        scaling = Scaling.from_question(question)
        ctx["scaling"] = scaling
        ctx["open_lower_bound"] = bool(question.get("open_lower_bound"))
        ctx["open_upper_bound"] = bool(question.get("open_upper_bound"))
        ctx["inbound_outcome_count"] = int(
            question.get("inbound_outcome_count") or DEFAULT_INBOUND_OUTCOME_COUNT
        )
        if qtype == "date":
            ctx["range_min_label"] = _epoch_label(scaling.range_min)
            ctx["range_max_label"] = _epoch_label(scaling.range_max)
            ctx["unit"] = "date (YYYY-MM-DD)"
        else:
            ctx["range_min_label"] = _num_label(scaling.range_min)
            ctx["range_max_label"] = _num_label(scaling.range_max)
    return ctx


def _epoch_label(ts: float) -> str:
    try:
        return datetime.fromtimestamp(float(ts), timezone.utc).strftime("%Y-%m-%d")
    except (OverflowError, OSError, ValueError):
        return str(ts)


def _num_label(v: float) -> str:
    if abs(v - round(v)) < 1e-9 and abs(v) < 1e15:
        return f"{int(round(v)):,}"
    return f"{v:,.4g}"


# -- research --------------------------------------------------------------
def search_queries(ctx: dict, models: Sequence[str]) -> list[str]:
    fallback = [ctx["title"][:200]]
    try:
        text, _ = chat_with_fallback(
            prompts.search_queries_prompt(ctx), models, temperature=0.3, max_tokens=300
        )
        parsed = extract_json(text)
        queries = [str(q).strip() for q in parsed if str(q).strip()] if isinstance(parsed, list) else []
        return (queries or fallback)[:3]
    except (LLMError, ValueError) as exc:
        log.warning("query generation failed, using the title: %s", str(exc)[:200])
        return fallback


# -- per-type pipelines ----------------------------------------------------
def _run_ensemble(messages, models: Sequence[str], runs: int, temperature: float) -> list[tuple[str, str]]:
    """Fire ``runs`` calls, rotating across model families for diversity."""
    tasks = []
    for i in range(runs):
        ordered = list(models[i % len(models):]) + list(models[: i % len(models)])
        tasks.append(lambda o=ordered: chat_with_fallback(messages, o, temperature=temperature))
    out: list[tuple[str, str]] = []
    dead = 0
    for result in run_parallel(tasks, workers=min(runs, 5)):
        if isinstance(result, Exception):
            if isinstance(result, NoModelsAvailable):
                dead += 1
            log.warning("ensemble member failed: %s", str(result)[:200])
            continue
        out.append(result)
    if not out and dead:
        # Nothing answered at all. That is the LLM layer being down, not the
        # models being unsure, and the two must not be confused: see the callers.
        raise NoModelsAvailable(
            f"all {runs} ensemble members failed because every model is unavailable"
        )
    return out


def forecast_binary(ctx: dict, research: str, models: Sequence[str], runs: int) -> tuple[dict, str, list[str], int, list[str]]:
    results = _run_ensemble(prompts.binary_prompt(ctx, research), models, runs, 0.4)
    notes: list[str] = []
    probs, texts, used = [], [], []
    for text, model in results:
        p = parsing.parse_probability(text)
        if p is None:
            notes.append(f"{model}: no probability found")
            continue
        probs.append(p)
        texts.append(text)
        used.append(model)

    if not probs:
        raise LLMError("no ensemble member produced a usable probability")

    raw = aggregate.aggregate_binary(probs)
    disagreement = aggregate.spread(probs)
    final = aggregate.calibrate_binary(raw)
    notes.append(
        f"ensemble {[round(p, 3) for p in probs]} -> median {raw:.3f} -> calibrated {final:.3f} "
        f"(spread {disagreement:.3f})"
    )
    headline = f"{final:.1%}"
    return (
        {"question": ctx["question_id"], "probability_yes": round(final, 6)},
        headline,
        notes,
        len(probs),
        used,
    )


def forecast_numeric(ctx: dict, research: str, models: Sequence[str], runs: int):
    is_date = ctx["type"] == "date"
    results = _run_ensemble(prompts.numeric_prompt(ctx, research), models, runs, 0.4)
    notes: list[str] = []
    parsed_runs, used = [], []
    for text, model in results:
        points = parsing.parse_percentiles(text)
        if is_date:
            converted = []
            for p, raw in points:
                epoch = date_to_epoch(str(raw)) if not _looks_like_epoch(raw) else float(raw)
                if epoch is not None:
                    converted.append((p, epoch))
            points = converted
        if len(points) < 2:
            notes.append(f"{model}: no usable percentiles")
            continue
        parsed_runs.append(points)
        used.append(model)

    scaling: Scaling = ctx["scaling"]
    count = ctx["inbound_outcome_count"]
    if not parsed_runs and not results:
        # Submitting a uniform here would be the worst of both worlds: it scores
        # badly AND marks the question as forecast, so the bot never returns to
        # it once the outage clears. Fail instead, and let the next poll retry.
        raise NoModelsAvailable(
            "no ensemble member responded; refusing to submit a placeholder distribution"
        )
    if not parsed_runs:
        notes.append("models responded but no usable percentiles; submitting a uniform distribution")
        cdf = safe_cdf(count, ctx["open_lower_bound"], ctx["open_upper_bound"])
        return (
            {"question": ctx["question_id"], "continuous_cdf": cdf},
            "uniform (no usable model output)",
            notes,
            0,
            used,
        )

    merged = aggregate.aggregate_percentiles(parsed_runs)
    widened = aggregate.extend_tails(aggregate.widen_percentiles(merged))
    cdf, build_notes = build_cdf(
        widened,
        scaling,
        ctx["open_lower_bound"],
        ctx["open_upper_bound"],
        count,
    )
    notes.extend(build_notes)

    def fmt(v: float) -> str:
        return _epoch_label(v) if is_date else _num_label(v)

    lookup = dict(widened)
    median = lookup.get(0.5) or lookup.get(0.4) or list(lookup.values())[len(lookup) // 2]
    low = min(v for _, v in widened)
    high = max(v for _, v in widened)
    notes.append(f"{len(parsed_runs)} usable runs; merged median {fmt(median)}")
    headline = f"median {fmt(median)} (range {fmt(low)} to {fmt(high)})"
    return (
        {"question": ctx["question_id"], "continuous_cdf": cdf},
        headline,
        notes,
        len(parsed_runs),
        used,
    )


def _looks_like_epoch(v: Any) -> bool:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return 1e8 < f < 4e9


def forecast_multiple_choice(ctx: dict, research: str, models: Sequence[str], runs: int):
    options = ctx["options"]
    results = _run_ensemble(prompts.multiple_choice_prompt(ctx, research), models, runs, 0.4)
    notes: list[str] = []
    parsed_runs, used = [], []
    for text, model in results:
        got = parsing.parse_multiple_choice(text, options)
        if not got:
            notes.append(f"{model}: no usable option probabilities")
            continue
        parsed_runs.append(got)
        used.append(model)

    if not parsed_runs and not results:
        raise NoModelsAvailable(
            "no ensemble member responded; refusing to submit a placeholder distribution"
        )
    merged = aggregate.aggregate_multiple_choice(parsed_runs, options)
    if merged is None:
        notes.append("models responded but no usable option probabilities; using the uniform")
        merged = {o: 1.0 / len(options) for o in options}
    final = aggregate.calibrate_multiple_choice(merged, options)
    top = max(final, key=lambda o: final[o])
    notes.append(f"{len(parsed_runs)} usable runs; top option {top} at {final[top]:.1%}")
    headline = ", ".join(f"{o} {final[o]:.0%}" for o in options[:6])
    return (
        {
            "question": ctx["question_id"],
            "probability_yes_per_category": {o: round(v, 6) for o, v in final.items()},
        },
        headline,
        notes,
        len(parsed_runs),
        used,
    )


# -- entry point -----------------------------------------------------------
def forecast_question(
    post: dict,
    question: dict,
    research_text: str,
    research_sources: Sequence[str],
    models: Sequence[str],
    runs: int = 5,
    sample_reasoning: str = "",
) -> Forecast:
    ctx = build_context(post, question)
    qtype = ctx["type"]

    if qtype == "binary":
        payload, headline, notes, n, used = forecast_binary(ctx, research_text, models, runs)
        method = "median of ensemble, log-odds shrink then 5% cap"
    elif qtype in CONTINUOUS_TYPES:
        payload, headline, notes, n, used = forecast_numeric(ctx, research_text, models, runs)
        method = "median percentile by percentile, widened, 5% uniform mix"
    elif qtype == "multiple_choice":
        payload, headline, notes, n, used = forecast_multiple_choice(ctx, research_text, models, runs)
        method = "median per option, mixed toward uniform"
    else:
        raise ValueError(f"unsupported question type {qtype!r} on question {ctx['question_id']}")

    comment = prompts.SUMMARY_TEMPLATE.format(
        models=", ".join(sorted(set(used))) or ", ".join(models),
        sources=", ".join(research_sources) or "none reached",
        n_runs=n,
        method=method,
        calibration="\n".join(f"- {note}" for note in notes),
        prediction_line=f"Forecast: {headline}",
        reasoning=(sample_reasoning or "").strip()[:4000] or "(not captured)",
    )

    return Forecast(
        question_id=ctx["question_id"],
        post_id=ctx["post_id"],
        question_type=qtype,
        payload=payload,
        comment=comment,
        headline=headline,
        notes=notes,
        runs_used=n,
        models_used=sorted(set(used)),
    )
