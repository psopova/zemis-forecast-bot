"""Gathering evidence, from sources that cost nothing.

The tournament rules are explicit that a bot "may use any resources that are
generally available to human forecasters. This includes using publicly
available forecasts on questions found on other platforms or on Metaculus
itself." So prediction market odds are fair game, and they are free and
unauthenticated on all three of Polymarket, Kalshi and Manifold.

AskNews is used when a key is present (Metaculus gives entrants 1,000 calls a
month free). Everything else here needs no key at all, so the bot still works
if that allocation runs out mid-season.
"""

from __future__ import annotations

import html
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote_plus, urlencode

import requests

log = logging.getLogger(__name__)

TIMEOUT = 25.0
UA = {"User-Agent": "metaculus-forecast-bot/1.0 (research; contact via Metaculus)"}


@dataclass
class Evidence:
    source: str
    title: str
    detail: str
    url: str = ""
    published: str = ""

    def render(self) -> str:
        head = f"[{self.source}] {self.title}".strip()
        if self.published:
            head += f" ({self.published})"
        body = self.detail.strip()
        return f"{head}\n{body}" if body else head


@dataclass
class ResearchReport:
    items: list[Evidence] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def add(self, item: Evidence) -> None:
        self.items.append(item)

    def render(self, limit: int = 40, max_chars: int = 14000) -> str:
        if not self.items:
            return "No external evidence was retrieved. Reason from background knowledge and say so."
        chunks: list[str] = []
        total = 0
        for item in self.items[:limit]:
            text = item.render()
            if total + len(text) > max_chars:
                break
            chunks.append(text)
            total += len(text)
        return "\n\n".join(chunks)

    @property
    def source_names(self) -> list[str]:
        return sorted({item.source for item in self.items})


def _clean(text: str, limit: int = 600) -> str:
    text = html.unescape(re.sub(r"<[^>]+>", " ", text or ""))
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


# -- news ------------------------------------------------------------------
def asknews(query: str, report: ResearchReport, n: int = 8) -> None:
    """Metaculus gives entrants a free AskNews allocation; use it first if present."""
    client_id = os.environ.get("ASKNEWS_CLIENT_ID")
    secret = os.environ.get("ASKNEWS_SECRET")
    if not (client_id and secret):
        return
    try:
        token = requests.post(
            "https://auth.asknews.app/oauth2/token",
            data={"grant_type": "client_credentials", "scope": "news"},
            auth=(client_id, secret),
            timeout=TIMEOUT,
        )
        token.raise_for_status()
        access = token.json()["access_token"]
        resp = requests.get(
            "https://api.asknews.app/v1/news/search",
            params={"query": query, "n_articles": n, "return_type": "dicts", "method": "kw"},
            headers={"Authorization": f"Bearer {access}", **UA},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        for art in (resp.json().get("as_dicts") or [])[:n]:
            report.add(
                Evidence(
                    source="AskNews",
                    title=_clean(art.get("eng_title") or art.get("title") or "", 220),
                    detail=_clean(art.get("summary") or art.get("text") or ""),
                    url=art.get("article_url") or "",
                    published=str(art.get("pub_date") or "")[:10],
                )
            )
    except Exception as exc:  # noqa: BLE001
        report.errors.append(f"asknews: {exc}")


def gdelt(query: str, report: ResearchReport, n: int = 8, days: int = 21) -> None:
    """GDELT's document API is public, keyless and unmetered."""
    try:
        params = {
            "query": query,
            "mode": "artlist",
            "format": "json",
            "maxrecords": n,
            "sort": "datedesc",
            "timespan": f"{days}d",
        }
        resp = requests.get(
            "https://api.gdeltproject.org/api/v2/doc/doc",
            params=params,
            headers=UA,
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        for art in (resp.json().get("articles") or [])[:n]:
            report.add(
                Evidence(
                    source="GDELT",
                    title=_clean(art.get("title") or "", 220),
                    detail=_clean(art.get("domain") or ""),
                    url=art.get("url") or "",
                    published=str(art.get("seendate") or "")[:8],
                )
            )
    except Exception as exc:  # noqa: BLE001
        report.errors.append(f"gdelt: {exc}")


def wikipedia(query: str, report: ResearchReport, n: int = 2) -> None:
    """Base rates and background, keyless."""
    try:
        search = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": n,
                "format": "json",
            },
            headers=UA,
            timeout=TIMEOUT,
        )
        search.raise_for_status()
        for hit in (search.json().get("query", {}).get("search") or [])[:n]:
            title = hit.get("title")
            if not title:
                continue
            extract = requests.get(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote_plus(title)}",
                headers=UA,
                timeout=TIMEOUT,
            )
            detail = ""
            if extract.ok:
                detail = _clean(extract.json().get("extract") or "", 900)
            report.add(
                Evidence(
                    source="Wikipedia",
                    title=title,
                    detail=detail or _clean(hit.get("snippet") or ""),
                    url=f"https://en.wikipedia.org/wiki/{quote_plus(title)}",
                )
            )
    except Exception as exc:  # noqa: BLE001
        report.errors.append(f"wikipedia: {exc}")


# -- prediction markets ----------------------------------------------------
def manifold(query: str, report: ResearchReport, n: int = 4) -> None:
    try:
        resp = requests.get(
            "https://api.manifold.markets/v0/search-markets",
            params={"term": query, "limit": n, "filter": "open", "sort": "score"},
            headers=UA,
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        for m in (resp.json() or [])[:n]:
            prob = m.get("probability")
            if prob is None:
                continue
            report.add(
                Evidence(
                    source="Manifold",
                    title=_clean(m.get("question") or "", 220),
                    detail=(
                        f"market probability {float(prob):.0%}, "
                        f"{m.get('volume', 0):.0f} volume, "
                        f"closes {str(m.get('closeTime') and datetime.fromtimestamp(m['closeTime'] / 1000, timezone.utc).date())}"
                    ),
                    url=m.get("url") or "",
                )
            )
    except Exception as exc:  # noqa: BLE001
        report.errors.append(f"manifold: {exc}")


def polymarket(query: str, report: ResearchReport, n: int = 4) -> None:
    try:
        resp = requests.get(
            "https://gamma-api.polymarket.com/public-search",
            params={"q": query, "limit_per_type": n, "events_status": "active"},
            headers=UA,
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json() or {}
        events = data.get("events") or []
        for ev in events[:n]:
            for market in (ev.get("markets") or [])[:2]:
                prices = market.get("outcomePrices")
                outcomes = market.get("outcomes")
                if isinstance(prices, str):
                    prices = _loads_list(prices)
                if isinstance(outcomes, str):
                    outcomes = _loads_list(outcomes)
                if not prices or not outcomes:
                    continue
                pairs = ", ".join(
                    f"{o} {float(p):.0%}" for o, p in zip(outcomes, prices) if _is_float(p)
                )
                if not pairs:
                    continue
                report.add(
                    Evidence(
                        source="Polymarket",
                        title=_clean(market.get("question") or ev.get("title") or "", 220),
                        detail=f"market prices: {pairs}",
                        url=f"https://polymarket.com/event/{ev.get('slug', '')}",
                    )
                )
    except Exception as exc:  # noqa: BLE001
        report.errors.append(f"polymarket: {exc}")


def _loads_list(raw: str) -> list:
    import json

    try:
        value = json.loads(raw)
        return value if isinstance(value, list) else []
    except Exception:  # noqa: BLE001
        return []


def _is_float(v: Any) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


# -- orchestration ---------------------------------------------------------
NEWS_SOURCES = (asknews, gdelt)
BACKGROUND_SOURCES = (wikipedia,)
MARKET_SOURCES = (manifold, polymarket)


def gather(queries: list[str], include_markets: bool = True) -> ResearchReport:
    """Run every free source over the supplied queries.

    Sources fail independently. A dead endpoint costs a line in the error list,
    never the forecast, because a question skipped over a transient 503 scores
    zero and zero is the expensive outcome.
    """
    report = ResearchReport()
    if not queries:
        return report
    primary = queries[0]

    for fn in NEWS_SOURCES:
        for q in queries[:3]:
            fn(q, report)
            if fn is asknews and report.items:
                break  # the free allocation is finite; one good call is enough
    for fn in BACKGROUND_SOURCES:
        fn(primary, report)
    if include_markets:
        for fn in MARKET_SOURCES:
            fn(primary, report)

    seen: set[str] = set()
    deduped: list[Evidence] = []
    for item in report.items:
        key = (item.url or item.title).lower()[:160]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    report.items = deduped
    return report
