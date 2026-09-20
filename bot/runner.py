"""The loop that actually earns the score.

Questions open at random hours and close to bots about three hours later, and a
question the bot never sees scores zero. Under a prize rule proportional to the
square of the summed score, missed questions are the most expensive failure
available, well ahead of any forecast being slightly off. So this module is
built around not missing things: every question is attempted independently,
every failure is caught and logged rather than allowed to end the tick, and the
bot's own already-submitted forecasts are the source of truth for what to skip.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timezone

from . import config, research as research_mod
from .client import MetaculusClient, MetaculusError, already_forecast, sub_questions
from .forecast import CONTINUOUS_TYPES, build_context, forecast_question, search_queries
from .llm import DEAD_MODELS, USAGE, LLMError, NoModelsAvailable, metaculus_proxy_models, resolve_models

log = logging.getLogger("bot")

SUPPORTED = {"binary", "multiple_choice"} | CONTINUOUS_TYPES


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def collect_targets(client: MetaculusClient, tournaments: list[str]) -> list[tuple[dict, dict]]:
    """Open questions in these tournaments that the bot has not forecast yet."""
    targets: list[tuple[dict, dict]] = []
    seen: set[int] = set()
    for tournament in tournaments:
        try:
            posts = list(client.iter_posts(tournament))
        except MetaculusError as exc:
            log.error("could not list %s: %s", tournament, exc)
            continue
        log.info("%s: %d open post(s)", tournament, len(posts))
        for post in posts:
            for question in sub_questions(post):
                qid = question.get("id")
                qtype = question.get("type")
                if not qid or qid in seen:
                    continue
                seen.add(qid)
                if qtype not in SUPPORTED:
                    log.info("skipping question %s: unsupported type %r", qid, qtype)
                    continue
                if already_forecast(question):
                    continue
                targets.append((post, question))
    return targets


def handle_one(client: MetaculusClient, post: dict, question: dict, models: list[str], runs: int) -> str:
    ctx = build_context(post, question)
    qid = ctx["question_id"]
    title = ctx["title"][:90]

    queries = search_queries(ctx, models)
    report = research_mod.gather(queries)
    if report.errors:
        log.warning("q%s research issues: %s", qid, "; ".join(report.errors)[:300])

    forecast = forecast_question(
        post=post,
        question=question,
        research_text=report.render(),
        research_sources=report.source_names,
        models=models,
        runs=runs,
    )

    client.submit_forecasts([forecast.payload])
    # The comment is a prize-eligibility requirement, so a failure here is
    # logged loudly rather than swallowed, but it must not undo the forecast.
    try:
        client.post_comment(forecast.post_id, forecast.comment)
    except MetaculusError as exc:
        log.error("q%s forecast submitted but comment FAILED: %s", qid, exc)

    log.info("q%s %-14s %s | %s", qid, forecast.question_type, forecast.headline, title)
    for note in forecast.notes:
        log.debug("  q%s: %s", qid, note)
    return forecast.headline


def run_tick(client: MetaculusClient, tournaments: list[str], models: list[str], runs: int, limit: int) -> int:
    targets = collect_targets(client, tournaments)
    if not targets:
        log.info("nothing new to forecast")
        return 0
    if len(targets) > limit:
        # Oldest first: those are closest to closing.
        log.warning("%d questions pending, taking the %d nearest to closing", len(targets), limit)
        targets = targets[:limit]

    log.info("forecasting %d question(s) with %s", len(targets), ", ".join(models))
    done = 0
    outage = 0
    with cf.ThreadPoolExecutor(max_workers=config.QUESTION_WORKERS) as pool:
        futures = {
            pool.submit(handle_one, client, post, question, models, runs): question.get("id")
            for post, question in targets
        }
        for fut in cf.as_completed(futures):
            qid = futures[fut]
            try:
                fut.result()
                done += 1
            except NoModelsAvailable as exc:
                outage += 1
                log.error("q%s skipped, no model available: %s", qid, str(exc)[:300])
            except (LLMError, MetaculusError, ValueError) as exc:
                log.error("q%s failed: %s", qid, str(exc)[:400])
            except Exception:  # noqa: BLE001 - one bad question must not end the tick
                log.error("q%s crashed:\n%s", qid, traceback.format_exc()[:1500])
    if outage:
        # Grinding through the rest of the list would be hundreds of doomed
        # requests against a shared proxy. The questions are untouched and the
        # next poll picks them up.
        log.error(
            "%d question(s) skipped because no model was reachable. Dead models: %s. "
            "Nothing was submitted for them, so they will be retried next run.",
            outage,
            ", ".join(sorted(DEAD_MODELS)) or "none recorded",
        )
    return done


def check_sources() -> int:
    """Run every research source once and report which actually returned data.

    Worth running from the environment the bot runs in, not a developer machine:
    a source can be reachable in one place and blocked in another.
    """
    setup_logging()
    report = research_mod.gather(["european central bank interest rate decision"])
    by_source: dict[str, int] = {}
    for item in report.items:
        by_source[item.source] = by_source.get(item.source, 0) + 1
    print(json.dumps({"items_by_source": by_source, "errors": report.errors}, indent=2))
    try:
        models = resolve_models(config.ENSEMBLE_MODELS)
        print(json.dumps({"models_resolved": models}, indent=2))
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"models_error": str(exc)}, indent=2))

    # Guessing a name the proxy does not serve costs a whole run, so ask it.
    proxy = metaculus_proxy_models()
    print(json.dumps({"metaculus_proxy_models": proxy or "none listed"}, indent=2))
    return 0 if by_source else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Forecast Metaculus tournament questions.")
    parser.add_argument("--mode", choices=sorted(config.MODES), default="tournament")
    parser.add_argument("--tournament", action="append", help="override the tournament slug or id")
    parser.add_argument("--runs", type=int, default=config.RUNS_PER_QUESTION)
    parser.add_argument("--limit", type=int, default=config.MAX_QUESTIONS_PER_TICK)
    parser.add_argument(
        "--watch",
        type=int,
        default=0,
        metavar="SECONDS",
        help="keep polling for this long inside one process, for continuous coverage",
    )
    parser.add_argument("--interval", type=int, default=300, help="seconds between polls in watch mode")
    parser.add_argument("--dry-run", action="store_true", help="do everything except submit")
    parser.add_argument("--check-sources", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    if args.check_sources:
        return check_sources()

    setup_logging(args.verbose)
    tournaments = args.tournament or config.MODES[args.mode]
    log.info("mode=%s tournaments=%s dry_run=%s", args.mode, tournaments, args.dry_run)

    try:
        client = MetaculusClient(dry_run=args.dry_run)
    except MetaculusError as exc:
        log.error("%s", exc)
        return 2

    models = resolve_models(config.ENSEMBLE_MODELS)
    log.info("ensemble: %s", ", ".join(models))

    deadline = time.monotonic() + args.watch if args.watch else None
    total = 0
    while True:
        started = time.monotonic()
        try:
            total += run_tick(client, tournaments, models, args.runs, args.limit)
        except Exception:  # noqa: BLE001 - a watch loop must outlive one bad tick
            log.error("tick crashed:\n%s", traceback.format_exc()[:2000])

        if deadline is None:
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        nap = max(30.0, min(args.interval - (time.monotonic() - started), remaining))
        log.info("sleeping %.0fs (%.0fs left in this watch window)", nap, remaining)
        time.sleep(nap)

    log.info("forecast %d question(s) this run. LLM usage: %s", total, USAGE.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
