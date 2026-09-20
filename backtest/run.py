"""Measure the bot against questions that have already resolved.

The tournament rules permit this explicitly: "it is acceptable to test on
questions that have already closed in the competition or test against questions
on the main site." Testing on OPEN tournament questions is not permitted, and
this script refuses to touch them.

A large warning, because it changes how the output should be read: a model can
already know how a resolved question turned out. Published work measures a 52%
Brier gap when models are asked to pretend they do not know an outcome. That
contamination makes a bot look better calibrated than it is, which pushes any
fitted shrinkage toward zero, exactly the wrong direction. So this harness is
for finding plumbing bugs and for ranking configurations against each other on
the same questions, not for believing the absolute numbers. Honest calibration
comes from MiniBench, which is live, unresolved and runs every two weeks.

Two phases, so one round of paid LLM calls can fit every calibration parameter:

  python -m backtest.run collect --limit 60 --out backtest/cache/runs.json
  python -m backtest.run fit --cache backtest/cache/runs.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import pathlib
import statistics
import sys
from datetime import datetime, timezone

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from bot import aggregate, parsing, prompts, research as research_mod, scoring
from bot.cdf import build_cdf
from bot.client import MetaculusClient, sub_questions
from bot.config import ENSEMBLE_MODELS, RUNS_PER_QUESTION
from bot.forecast import build_context, date_to_epoch, _looks_like_epoch
from bot.llm import USAGE, LLMError, chat_with_fallback, resolve_models
from bot.runner import setup_logging

log = logging.getLogger("backtest")

# Past bot tournaments. Their questions are closed and resolved, so they are
# fair game and they match the distribution of what the bot will actually face.
PAST_TOURNAMENTS = [
    "summer-futureeval-2026",
    "spring-aib-2026",
    "fall-aib-2025",
    "minibench",
]


def collect(args: argparse.Namespace) -> int:
    client = MetaculusClient()
    models = resolve_models(ENSEMBLE_MODELS)
    log.info("ensemble: %s", ", ".join(models))

    rows: list[dict] = []
    for tournament in args.tournament or PAST_TOURNAMENTS:
        if len(rows) >= args.limit:
            break
        for post in client.iter_posts(tournament, statuses="resolved"):
            if len(rows) >= args.limit:
                break
            for question in sub_questions(post):
                resolution = question.get("resolution")
                if resolution in (None, "", "annulled", "ambiguous"):
                    continue
                if question.get("type") not in {"binary", "numeric", "discrete", "date", "multiple_choice"}:
                    continue
                if question.get("actual_resolve_time") is None:
                    continue
                if args.min_open_date:
                    opened = str(question.get("open_time") or "")[:10]
                    if opened < args.min_open_date:
                        continue

                try:
                    row = _run_one(post, question, models, args.runs)
                except (LLMError, ValueError) as exc:
                    log.warning("q%s skipped: %s", question.get("id"), str(exc)[:200])
                    continue
                row["tournament"] = tournament
                rows.append(row)
                log.info("%d/%d collected (q%s)", len(rows), args.limit, question.get("id"))

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"collected_at": datetime.now(timezone.utc).isoformat(), "rows": rows}, indent=1))
    log.info("wrote %d rows to %s. LLM usage: %s", len(rows), out, USAGE.summary())
    return 0


def _run_one(post: dict, question: dict, models: list[str], runs: int) -> dict:
    ctx = build_context(post, question)
    report = research_mod.gather([ctx["title"][:200]])
    research_text = report.render()

    qtype = ctx["type"]
    if qtype == "binary":
        messages = prompts.binary_prompt(ctx, research_text)
    elif qtype == "multiple_choice":
        messages = prompts.multiple_choice_prompt(ctx, research_text)
    else:
        messages = prompts.numeric_prompt(ctx, research_text)

    raw: list[str] = []
    for i in range(runs):
        ordered = models[i % len(models):] + models[: i % len(models)]
        try:
            text, _ = chat_with_fallback(messages, ordered, temperature=0.4)
            raw.append(text)
        except LLMError as exc:
            log.debug("member failed: %s", exc)
    if not raw:
        raise LLMError("no usable ensemble members")

    return {
        "question_id": ctx["question_id"],
        "type": qtype,
        "title": ctx["title"],
        "resolution": question.get("resolution"),
        "options": ctx.get("options") or [],
        "open_time": question.get("open_time"),
        "actual_resolve_time": question.get("actual_resolve_time"),
        "scaling": question.get("scaling"),
        "open_lower_bound": ctx.get("open_lower_bound"),
        "open_upper_bound": ctx.get("open_upper_bound"),
        "inbound_outcome_count": ctx.get("inbound_outcome_count"),
        "raw_outputs": raw,
        "research_sources": report.source_names,
    }


# -- fitting ---------------------------------------------------------------
def _score_binary(row: dict, shrink: float, floor: float) -> float | None:
    probs = [p for p in (parsing.parse_probability(t) for t in row["raw_outputs"]) if p is not None]
    if not probs:
        return None
    merged = aggregate.aggregate_binary(probs)
    if merged is None:
        return None
    final = aggregate.calibrate_binary(merged, shrink=shrink, floor=floor, ceiling=1 - floor)
    outcome = 1 if str(row["resolution"]).lower() in {"yes", "true", "1"} else 0
    return scoring.baseline_score(scoring.binary_pmf(final), outcome, continuous=False)


def _score_numeric(row: dict, widen: float, uniform_mix: float = 0.05) -> float | None:
    from bot.scaling import Scaling

    try:
        scaling = Scaling(
            float(row["scaling"]["range_min"]),
            float(row["scaling"]["range_max"]),
            row["scaling"].get("zero_point"),
        )
    except (KeyError, TypeError, ValueError):
        return None

    is_date = row["type"] == "date"
    runs = []
    for text in row["raw_outputs"]:
        points = parsing.parse_percentiles(text)
        if is_date:
            points = [
                (p, float(v) if _looks_like_epoch(v) else date_to_epoch(str(v)))
                for p, v in points
            ]
            points = [(p, v) for p, v in points if v is not None]
        if len(points) >= 2:
            runs.append(points)
    if not runs:
        return None

    merged = aggregate.extend_tails(
        aggregate.widen_percentiles(aggregate.aggregate_percentiles(runs), widen)
    )
    count = int(row.get("inbound_outcome_count") or 200)
    cdf, _ = build_cdf(
        merged,
        scaling,
        bool(row["open_lower_bound"]),
        bool(row["open_upper_bound"]),
        count,
        uniform_mix,
    )

    truth = row["resolution"]
    value = date_to_epoch(str(truth)) if is_date else parsing.parse_number(str(truth))
    if value is None:
        return None
    pmf = scoring.continuous_pmf(cdf)
    bucket = scoring.resolution_bucket_continuous(value, scaling, count)
    open_bounds = int(bool(row["open_lower_bound"])) + int(bool(row["open_upper_bound"]))
    return scoring.baseline_score(pmf, bucket, continuous=True, open_bounds_count=open_bounds)


def _score_mc(row: dict, mix: float) -> float | None:
    options = row["options"]
    if not options or row["resolution"] not in options:
        return None
    runs = [parsing.parse_multiple_choice(t, options) for t in row["raw_outputs"]]
    merged = aggregate.aggregate_multiple_choice(runs, options)
    if merged is None:
        return None
    final = aggregate.calibrate_multiple_choice(merged, options, uniform_mix=mix)
    pmf = [final[o] for o in options]
    return scoring.baseline_score(pmf, options.index(row["resolution"]), continuous=False)


def fit(args: argparse.Namespace) -> int:
    data = json.loads(pathlib.Path(args.cache).read_text())
    rows = data["rows"]
    log.info("loaded %d rows collected at %s", len(rows), data.get("collected_at"))

    binary = [r for r in rows if r["type"] == "binary"]
    numeric = [r for r in rows if r["type"] in {"numeric", "discrete", "date"}]
    mc = [r for r in rows if r["type"] == "multiple_choice"]
    log.info("%d binary, %d numeric, %d multiple choice", len(binary), len(numeric), len(mc))

    results: dict[str, object] = {}

    if binary:
        best = None
        grid = []
        for shrink, floor in itertools.product(
            [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0], [0.02, 0.03, 0.05, 0.08, 0.10]
        ):
            scores = [s for s in (_score_binary(r, shrink, floor) for r in binary) if s is not None]
            if not scores:
                continue
            mean = statistics.mean(scores)
            grid.append({"shrink": shrink, "floor": floor, "mean_baseline": round(mean, 2), "n": len(scores)})
            if best is None or mean > best["mean_baseline"]:
                best = grid[-1]
        results["binary"] = {"best": best, "grid": sorted(grid, key=lambda g: -g["mean_baseline"])[:8]}

    if numeric:
        grid = []
        for widen, mix in itertools.product(
            [1.0, 1.1, 1.15, 1.25, 1.4], [0.02, 0.05, 0.08, 0.12, 0.2]
        ):
            scores = [s for s in (_score_numeric(r, widen, mix) for r in numeric) if s is not None]
            if not scores:
                continue
            grid.append(
                {
                    "widen": widen,
                    "uniform_mix": mix,
                    "mean_baseline": round(statistics.mean(scores), 2),
                    "n": len(scores),
                }
            )
        grid.sort(key=lambda g: -g["mean_baseline"])
        grid = grid[:10]
        results["numeric"] = {
            "best": max(grid, key=lambda g: g["mean_baseline"]) if grid else None,
            "grid": grid,
        }

    if mc:
        grid = []
        for mix in [0.0, 0.03, 0.05, 0.08, 0.12, 0.2]:
            scores = [s for s in (_score_mc(r, mix) for r in mc) if s is not None]
            if not scores:
                continue
            grid.append({"uniform_mix": mix, "mean_baseline": round(statistics.mean(scores), 2), "n": len(scores)})
        results["multiple_choice"] = {
            "best": max(grid, key=lambda g: g["mean_baseline"]) if grid else None,
            "grid": grid,
        }

    print(json.dumps(results, indent=2))
    print(
        "\nRead these as a ranking of configurations against each other, not as an\n"
        "estimate of live performance. The models may already know how these\n"
        "questions resolved, which flatters low shrinkage. Confirm on MiniBench."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="run the ensemble over resolved questions and cache raw output")
    c.add_argument("--limit", type=int, default=60)
    c.add_argument("--runs", type=int, default=RUNS_PER_QUESTION)
    c.add_argument("--tournament", action="append")
    c.add_argument("--min-open-date", default="", help="YYYY-MM-DD, to limit training-data leakage")
    c.add_argument("--out", default="backtest/cache/runs.json")
    c.set_defaults(func=collect)

    f = sub.add_parser("fit", help="sweep calibration parameters over cached output")
    f.add_argument("--cache", default="backtest/cache/runs.json")
    f.set_defaults(func=fit)

    args = parser.parse_args(argv)
    setup_logging()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
