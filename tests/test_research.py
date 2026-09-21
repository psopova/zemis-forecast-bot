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
    def __init__(self, status_code=429, payload=None):
        self.status_code = status_code
        self.headers = {"content-type": "application/json"}
        self._payload = payload or {"articles": []}
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
