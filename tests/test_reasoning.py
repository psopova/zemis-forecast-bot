"""Asking the model to think harder, without letting that break anything.

Metaculus ran eight pairs of bots differing only in reasoning effort. The
higher-effort bot won all eight, one-sided sign test p = 0.004, and their
example pair scored 11.3 against 4.56 peer points per question. It is the only
controlled result in their Spring 2026 analysis that reached significance.

The reason this file exists is that turning it on has two ways to backfire, and
both of them are silent. A provider that does not know the parameter answers
400, which this bot treats as "this model will never work" and blacklists. And
a model that does know it bills its thinking against the output budget, so it
can spend the entire allowance reasoning and hand back an empty string, which
parses as nothing and scores as nothing.
"""

import pytest

from bot import llm


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text="", finish_reason="stop", content="ok"):
        self.status_code = status_code
        self.headers = {}
        self.text = text
        self._payload = payload or {
            "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
            "usage": {},
        }

    @property
    def ok(self):
        return self.status_code < 400

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    llm.NO_REASONING.clear()
    llm.DEAD_MODELS.clear()
    monkeypatch.setattr(llm.LIMITER, "wait", lambda key: None)
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    yield
    llm.NO_REASONING.clear()
    llm.DEAD_MODELS.clear()


def _capture(monkeypatch, responses):
    sent = []
    queue = list(responses)

    def post(url, headers=None, json=None, timeout=None):
        sent.append(dict(json))
        return queue.pop(0) if queue else FakeResponse()

    monkeypatch.setattr(llm.requests, "post", post)
    return sent


def test_the_model_is_asked_to_think_hard_by_default(monkeypatch):
    monkeypatch.setattr(llm, "REASONING_EFFORT", "high")
    sent = _capture(monkeypatch, [FakeResponse()])
    llm.chat([{"role": "user", "content": "x"}], "gemini/models/gemini-3.8-flash")
    assert sent[0]["reasoning_effort"] == "high"


def test_openrouter_gets_its_own_spelling(monkeypatch):
    monkeypatch.setattr(llm, "REASONING_EFFORT", "high")
    sent = _capture(monkeypatch, [FakeResponse()])
    llm.chat([{"role": "user", "content": "x"}], "openrouter/openai/gpt-5")
    assert sent[0]["reasoning"] == {"effort": "high"}
    assert "reasoning_effort" not in sent[0]


def test_it_can_be_turned_off(monkeypatch):
    monkeypatch.setattr(llm, "REASONING_EFFORT", "off")
    sent = _capture(monkeypatch, [FakeResponse()])
    llm.chat([{"role": "user", "content": "x"}], "gemini/models/gemini-3.8-flash")
    assert "reasoning_effort" not in sent[0] and "reasoning" not in sent[0]


def test_thinking_gets_room_so_it_does_not_eat_the_answer(monkeypatch):
    monkeypatch.setattr(llm, "REASONING_EFFORT", "high")
    sent = _capture(monkeypatch, [FakeResponse()])
    llm.chat([{"role": "user", "content": "x"}], "gemini/models/gemini-3.8-flash", max_tokens=3000)
    assert sent[0]["max_tokens"] >= llm.REASONING_MAX_TOKENS


def test_a_model_that_rejects_the_parameter_is_not_blacklisted(monkeypatch):
    monkeypatch.setattr(llm, "REASONING_EFFORT", "high")
    rejection = FakeResponse(400, text='{"error":{"message":"Unknown parameter: reasoning_effort"}}')
    sent = _capture(monkeypatch, [rejection, FakeResponse(content="answered")])

    out = llm.chat([{"role": "user", "content": "x"}], "gemini/models/gemini-3.8-flash")

    assert out == "answered"
    assert "gemini/models/gemini-3.8-flash" not in llm.DEAD_MODELS, (
        "a model must not be written off for declining one optional setting"
    )
    assert "reasoning_effort" in sent[0]
    assert "reasoning_effort" not in sent[1], "the retry must drop the parameter"


def test_a_real_400_still_kills_the_model(monkeypatch):
    monkeypatch.setattr(llm, "REASONING_EFFORT", "high")
    _capture(monkeypatch, [FakeResponse(400, text='{"error":"no allowance for this model"}')])
    with pytest.raises(llm.ModelUnavailable):
        llm.chat([{"role": "user", "content": "x"}], "gemini/models/gemini-3.8-flash")
    assert "gemini/models/gemini-3.8-flash" in llm.DEAD_MODELS


def test_thinking_until_the_answer_is_cut_off_falls_back(monkeypatch):
    monkeypatch.setattr(llm, "REASONING_EFFORT", "high")
    truncated = FakeResponse(content="", finish_reason="length")
    sent = _capture(monkeypatch, [truncated, FakeResponse(content="PROBABILITY: 12%")])

    out = llm.chat([{"role": "user", "content": "x"}], "gemini/models/gemini-3.8-flash")

    assert out == "PROBABILITY: 12%"
    assert "reasoning_effort" not in sent[1]
    assert sent[1]["max_tokens"] == 3000, "the ceiling goes back down with the thinking"


def test_the_lesson_is_remembered_for_the_next_question(monkeypatch):
    monkeypatch.setattr(llm, "REASONING_EFFORT", "high")
    rejection = FakeResponse(400, text='{"error":{"message":"reasoning_effort is not supported"}}')
    sent = _capture(monkeypatch, [rejection, FakeResponse(), FakeResponse()])

    llm.chat([{"role": "user", "content": "x"}], "gemini/models/gemini-3.8-flash")
    llm.chat([{"role": "user", "content": "y"}], "gemini/models/gemini-3.8-flash")

    assert "reasoning_effort" not in sent[2], (
        "one probe per model, not one per call: a watcher makes thousands"
    )


def test_a_truncated_answer_with_content_is_kept(monkeypatch):
    # Only an EMPTY truncated answer is worth retrying. A long one that ran out
    # at the end may still carry the forecast line.
    monkeypatch.setattr(llm, "REASONING_EFFORT", "high")
    sent = _capture(monkeypatch, [FakeResponse(content="PROBABILITY: 7%", finish_reason="length")])
    out = llm.chat([{"role": "user", "content": "x"}], "gemini/models/gemini-3.8-flash")
    assert out == "PROBABILITY: 7%"
    assert len(sent) == 1
