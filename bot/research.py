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
import time
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

    @property
    def source_mix(self) -> str:
        """What each source actually contributed, for the log.

        A source that silently returns nothing looks exactly like a source that
        is working, which is how a feature stays switched off for a season
        without anyone noticing.
        """
        counts: dict[str, int] = {}
        for item in self.items:
            counts[item.source] = counts.get(item.source, 0) + 1
        if not counts:
            return "nothing"
        return ", ".join(f"{name} {n}" for name, n in sorted(counts.items()))


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


_GDELT_STOP = {"the", "and", "for", "will", "any", "are", "was", "has", "had", "its", "this", "that", "with", "from", "before", "after", "than", "these", "their", "what", "which", "does"}


def gdelt_query(text: str) -> str:
    """GDELT has its own query grammar and rejects most natural phrasing.

    Live failures were "Parentheses may only be used around OR'd statements"
    and "Your search contained a keyword that was too short", both from passing
    a question title through unchanged. Reduce it to plain keywords.
    """
    words = re.findall(r"[A-Za-z0-9]+", text or "")
    keep = [w for w in words if len(w) >= 3 and w.lower() not in _GDELT_STOP]
    return " ".join(keep[:8])


# GDELT rate limits by egress address, and GitHub Actions runners share a small
# pool of them, so a 429 is a statement about the runner rather than about this
# query. A watcher runs a pass every four minutes for four hours; without a
# cooldown it pays two round trips and a three second sleep per question, every
# pass, for nothing. Sit out for a while after a refusal instead.
GDELT_COOLDOWN_SECONDS = 900.0
_gdelt_blocked_until = 0.0


def gdelt(query: str, report: ResearchReport, n: int = 8, days: int = 21) -> None:
    """GDELT's document API is public and keyless, but it rate-limits hard.

    One short retry is worth it; a long one is not, because the question closes
    in three hours and every other source is still available.
    """
    global _gdelt_blocked_until

    query = gdelt_query(query)
    if len(query.split()) < 2:
        return
    if time.monotonic() < _gdelt_blocked_until:
        report.errors.append("gdelt: rate limited, sitting out")
        return
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
        if resp.status_code == 429:
            time.sleep(3)
            resp = requests.get(
                "https://api.gdeltproject.org/api/v2/doc/doc",
                params=params,
                headers=UA,
                timeout=TIMEOUT,
            )
        if resp.status_code == 429:
            _gdelt_blocked_until = time.monotonic() + GDELT_COOLDOWN_SECONDS
            report.errors.append("gdelt: rate limited, sitting out")
            return
        resp.raise_for_status()
        if "json" not in (resp.headers.get("content-type") or ""):
            raise ValueError(f"non-JSON response: {resp.text[:80]!r}")
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


# -- the page the question will be resolved from ---------------------------
# Static web scraping was the highest scoring free feature in the Spring 2026
# survey (r = +0.33, p = 0.032), and in Fall 2025 it was used by 21% of prize
# winners and 0% of non-winners. It is also the most obviously sensible thing
# in the list: a question about the FAO Cereal Price Index names the FAO page
# that decides it, and news coverage is a worse source than the page itself.

# Hosts that cost a request and return nothing useful: Metaculus is the
# question we are already reading, and the rest serve a login wall or a
# JavaScript shell to a plain HTTP client.
_UNHELPFUL_HOSTS = re.compile(
    r"(metaculus\.com|twitter\.com|x\.com|facebook\.com|instagram\.com|linkedin\.com"
    r"|t\.co|bit\.ly|youtube\.com|youtu\.be|reddit\.com)",
    re.I,
)
_URL = re.compile(r"https?://[^\s<>\])\"\',]+")
_MAX_SOURCE_PAGES = 2
_SOURCE_CHARS = 1500


def resolution_source_urls(ctx: dict, limit: int = _MAX_SOURCE_PAGES) -> list[str]:
    """Links the question itself gives for how it will be decided.

    Read in order of authority: the resolution criteria name the deciding
    source, the fine print qualifies it, the description is background.
    """
    seen: set[str] = set()
    out: list[str] = []
    for field in ("resolution_criteria", "fine_print", "description"):
        for raw in _URL.findall(ctx.get(field) or ""):
            url = raw.rstrip(".,;:)\'\"")
            if _UNHELPFUL_HOSTS.search(url):
                continue
            key = url.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(url)
            if len(out) >= limit:
                return out
    return out


def resolution_sources(ctx: dict, report: ResearchReport, limit: int = _MAX_SOURCE_PAGES) -> None:
    """Fetch the pages the question points at and hand over their text.

    The text is whatever a stranger published on the internet, exactly like the
    news evidence, and it is labelled as a quoted page rather than as fact.
    """
    for url in resolution_source_urls(ctx, limit):
        try:
            resp = requests.get(url, headers=UA, timeout=TIMEOUT, allow_redirects=True)
            resp.raise_for_status()
            ctype = (resp.headers.get("content-type") or "").lower()
            if "html" not in ctype and "text" not in ctype and "json" not in ctype:
                report.errors.append(f"resolution source: {url[:60]} is {ctype[:30]}")
                continue
            body = _strip_page(resp.text)
            # A page that strips to almost nothing is a cookie banner or a
            # JavaScript shell. Low quality research measurably hurts: in the
            # Fall 2025 comparison the bot that did no search at all outscored
            # several search-equipped ones.
            if len(body) < 40:
                report.errors.append(f"resolution source: {url[:60]} had no readable text")
                continue
            report.add(
                Evidence(
                    source="Resolution source",
                    title=url,
                    detail=body[:_SOURCE_CHARS],
                )
            )
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"resolution source {url[:50]}: {str(exc)[:80]}")


_SCRIPTY = re.compile(r"<(script|style|noscript|svg)[^>]*>.*?</\1>", re.S | re.I)


def _strip_page(html_text: str) -> str:
    text = _SCRIPTY.sub(" ", html_text or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


# -- prediction markets ----------------------------------------------------
def manifold(query: str, report: ResearchReport, n: int = 4) -> None:
    try:
        resp = requests.get(
            "https://api.manifold.markets/v0/search-markets",
            # Without contractType the search returns perpetual and multi
            # outcome markets, which carry no "probability" field and are
            # dropped below, so a query can spend all four slots and return
            # nothing. Ask only for the kind that answers the question.
            params={
                "term": query,
                "limit": n,
                "filter": "open",
                "sort": "score",
                "contractType": "BINARY",
            },
            headers=UA,
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        rows = resp.json() or []
        usable = 0
        for m in rows[:n]:
            prob = m.get("probability")
            if prob is None:
                continue
            usable += 1
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
        if not usable:
            # Silence here is ambiguous: no market exists, or the search shape
            # changed again. Say which, so the next log answers it.
            report.errors.append(f"manifold: {len(rows)} market(s) matched, none usable")
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


def gather(
    queries: list[str],
    include_markets: bool = True,
    ctx: dict | None = None,
) -> ResearchReport:
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
        for q in queries[:2 if fn is not gdelt else 1]:
            fn(q, report)
            if fn is asknews and report.items:
                break  # the free allocation is finite; one good call is enough
    for fn in BACKGROUND_SOURCES:
        fn(primary, report)
    if ctx:
        resolution_sources(ctx, report)
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
