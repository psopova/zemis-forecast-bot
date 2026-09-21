"""What the research layer does when a source refuses to answer.

GDELT rate limits by egress address, and every GitHub Actions runner shares a
small pool of them, so in production a 429 is the normal case and not a blip.
A watcher makes a pass every four minutes for four hours; if each pass pays two
round trips and a three second sleep per question for a source that is not going
to answer, that time comes out of the questions.
"""

import time

import pytest

from bot import research
from bot.research import ResearchReport


class FakeResponse:
    _UNSET = object()

    def __init__(self, status_code=429, payload=_UNSET):
        self.status_code = status_code
        self.headers = {"content-type": "application/json"}
        # An empty list is a real answer ("no markets matched"), so it must not
        # fall through to the default the way `payload or {...}` would.
        self._payload = {"articles": []} if payload is FakeResponse._UNSET else payload
        self.text = "{}"

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError(f"unexpected raise_for_status on {self.status_code}")


@pytest.fixture(autouse=True)
def reset_cooldown():
    research._gdelt_blocked_until = 0.0
    yield
    research._gdelt_blocked_until = 0.0


def test_a_rate_limited_gdelt_is_left_alone_for_a_while(monkeypatch):
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        return FakeResponse(429)

    monkeypatch.setattr(research.requests, "get", fake_get)
    monkeypatch.setattr(research.time, "sleep", lambda s: None)

    first = ResearchReport()
    research.gdelt("Trump net approval rating December 2026", first)
    # One attempt plus the single short retry, and then it gives up.
    assert len(calls) == 2
    assert any("rate limited" in e for e in first.errors)

    second = ResearchReport()
    research.gdelt("FAO Vegetable Oil Price Index September 2026", second)
    # The second question must not pay for the first question's refusal again.
    assert len(calls) == 2
    assert any("rate limited" in e for e in second.errors)


def test_the_cooldown_expires_rather_than_disabling_the_source_forever(monkeypatch):
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        return FakeResponse(429)

    monkeypatch.setattr(research.requests, "get", fake_get)
    monkeypatch.setattr(research.time, "sleep", lambda s: None)
    research.gdelt("Trump net approval rating December 2026", ResearchReport())
    assert len(calls) == 2

    # Move past the cooldown; GDELT gets another chance.
    real_monotonic = time.monotonic
    monkeypatch.setattr(
        research.time,
        "monotonic",
        lambda: real_monotonic() + research.GDELT_COOLDOWN_SECONDS + 1,
    )
    research.gdelt("Trump net approval rating December 2026", ResearchReport())
    assert len(calls) == 4


def test_a_working_gdelt_still_returns_evidence(monkeypatch):
    payload = {
        "articles": [
            {
                "title": "Approval rating slips again",
                "domain": "example.com",
                "url": "https://example.com/a",
                "seendate": "20260919T000000Z",
            }
        ]
    }
    monkeypatch.setattr(
        research.requests, "get", lambda url, **kw: FakeResponse(200, payload)
    )
    report = ResearchReport()
    research.gdelt("Trump net approval rating December 2026", report)
    assert not report.errors
    assert any("Approval rating slips again" in e.title for e in report.items)


# -- prediction markets ----------------------------------------------------
# 34% of prize winners looked up related markets or resolved questions against
# 0% of non-winners (p = 0.04), and it ranked second of 33 features the next
# season. It is the strongest free signal published, so it has to actually run.


def test_manifold_asks_only_for_markets_that_carry_a_probability(monkeypatch):
    seen = {}

    def fake_get(url, params=None, **kwargs):
        seen.update(params or {})
        return FakeResponse(200, [])

    monkeypatch.setattr(research.requests, "get", fake_get)
    research.manifold("will the shutdown end", ResearchReport())
    # Perpetual and multi-outcome markets have no "probability" and get
    # dropped, so without this filter a query can spend every slot and return
    # nothing while looking perfectly healthy.
    assert seen.get("contractType") == "BINARY"
    assert seen.get("filter") == "open"


def test_a_market_price_arrives_as_weighable_evidence(monkeypatch):
    payload = [
        {
            "question": "US government shutdown on October 1st 2026?",
            "probability": 0.0384,
            "volume": 8544,
            "closeTime": 1790812740000,
            "url": "https://manifold.markets/x/y",
        }
    ]
    monkeypatch.setattr(research.requests, "get", lambda url, **kw: FakeResponse(200, payload))
    report = ResearchReport()
    research.manifold("government shutdown", report)
    assert not report.errors
    item = report.items[0]
    assert item.source == "Manifold"
    assert "4%" in item.detail, item.detail
    assert "8544" in item.detail, "volume is how the model tells a real price from a thin one"


def test_the_evidence_mix_is_reportable(monkeypatch):
    report = ResearchReport()
    report.add(research.Evidence(source="Manifold", title="a", detail=""))
    report.add(research.Evidence(source="Manifold", title="b", detail=""))
    report.add(research.Evidence(source="GDELT", title="c", detail=""))
    # A source that quietly returns nothing looks identical to one that works.
    assert report.source_mix == "GDELT 1, Manifold 2"
    assert ResearchReport().source_mix == "nothing"


# -- the page the question resolves from -----------------------------------
# Static web scraping was the highest scoring free feature in the Spring 2026
# survey (r = +0.33, p = 0.032) and was used by 21% of Fall 2025 prize winners
# against 0% of non-winners. A question about a price index names the page that
# decides it, and that page beats news coverage of it.


def test_the_deciding_link_is_preferred_over_background_links():
    ctx = {
        "description": "Background at https://example.org/background",
        "resolution_criteria": "Resolves per https://www.fao.org/foodpricesindex/en/ .",
        "fine_print": "Fallback: https://tradingeconomics.com/commodity/food",
    }
    urls = research.resolution_source_urls(ctx, limit=3)
    assert urls[0] == "https://www.fao.org/foodpricesindex/en/", urls
    assert urls[1] == "https://tradingeconomics.com/commodity/food", urls


def test_links_that_cost_a_request_and_give_nothing_are_skipped():
    ctx = {
        "resolution_criteria": (
            "See https://twitter.com/FAO and https://www.metaculus.com/questions/1/ "
            "and https://x.com/a and https://www.fao.org/real"
        ),
        "fine_print": "",
        "description": "",
    }
    # Metaculus is the question we are already reading; the social hosts serve a
    # login wall or a JavaScript shell to a plain HTTP client.
    assert research.resolution_source_urls(ctx) == ["https://www.fao.org/real"]


def test_trailing_punctuation_is_not_part_of_the_link():
    ctx = {"resolution_criteria": "per https://www.fao.org/a/b.html, checked monthly.",
           "fine_print": "", "description": ""}
    assert research.resolution_source_urls(ctx) == ["https://www.fao.org/a/b.html"]


def test_the_page_arrives_as_readable_text(monkeypatch):
    html_page = (
        "<html><head><style>p{color:red}</style><script>track()</script></head>"
        "<body><h1>FAO Cereal Price Index</h1>"
        "<p>The FAO Cereal Price Index averaged 117.2 points in September 2026, "
        "down 1.4 points from August, with wheat quotations easing.</p></body></html>"
    )

    class R:
        status_code = 200
        headers = {"content-type": "text/html; charset=utf-8"}
        text = html_page

        def raise_for_status(self):
            pass

    monkeypatch.setattr(research.requests, "get", lambda url, **kw: R())
    report = ResearchReport()
    research.resolution_sources(
        {"resolution_criteria": "per https://www.fao.org/x", "fine_print": "", "description": ""},
        report,
    )
    item = report.items[0]
    assert item.source == "Resolution source"
    assert "117.2" in item.detail
    assert "track()" not in item.detail and "color:red" not in item.detail


def test_a_dead_source_page_costs_a_note_and_not_the_forecast(monkeypatch):
    def boom(url, **kw):
        raise research.requests.RequestException("connection reset")

    monkeypatch.setattr(research.requests, "get", boom)
    report = ResearchReport()
    research.resolution_sources(
        {"resolution_criteria": "per https://www.fao.org/x", "fine_print": "", "description": ""},
        report,
    )
    assert report.items == []
    assert any("resolution source" in e for e in report.errors)


def test_a_question_with_no_links_costs_no_requests(monkeypatch):
    def blow_up(url, **kw):
        raise AssertionError("should not have fetched anything")

    monkeypatch.setattr(research.requests, "get", blow_up)
    research.resolution_sources({"resolution_criteria": "no links here"}, ResearchReport())


def test_an_empty_manifold_says_so_rather_than_looking_healthy(monkeypatch):
    monkeypatch.setattr(research.requests, "get", lambda url, **kw: FakeResponse(200, []))
    report = ResearchReport()
    research.manifold("something obscure", report)
    # Absence and breakage look identical without this.
    assert any("none usable" in e for e in report.errors), report.errors
