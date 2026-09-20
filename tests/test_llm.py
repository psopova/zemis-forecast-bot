"""Model output is prose with JSON somewhere inside it. Parsing must not be brittle."""


import pytest


from bot.llm import MODEL_PREFERENCES, STATIC_FALLBACK, extract_json, resolve_models


@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"a": 1}', {"a": 1}),
        ("Here you go:\n```json\n{\"a\": 2}\n```\nHope that helps", {"a": 2}),
        ('Sure!\n{"a": 3, "b": [1,2,3,],}\n', {"a": 3, "b": [1, 2, 3]}),
        ("prose [1, 2, 3] more prose", [1, 2, 3]),
        ('```\n[{"p":0.1,"v":5}]\n```', [{"p": 0.1, "v": 5}]),
        ('[{"p":0.1},{"p":0.9}]', [{"p": 0.1}, {"p": 0.9}]),
        ('Reasoning follows.\n\n```json\n{"probability": 0.42}\n```', {"probability": 0.42}),
        ('{"nested": {"a": [1, {"b": 2}]}}', {"nested": {"a": [1, {"b": 2}]}}),
        ("```json\n{\"x\": 1}\n```\n```json\n{\"y\": 2}\n```", {"x": 1}),
        ('{"s": "a string with } and ] inside"}', {"s": "a string with } and ] inside"}),
    ],
)
def test_extract_json(text, expected):
    assert extract_json(text) == expected


@pytest.mark.parametrize("text", ["", "no json at all", "{unclosed", None])
def test_extract_json_rejects_garbage(text):
    with pytest.raises(ValueError):
        extract_json(text)


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_resolve_models_falls_back_when_the_catalogue_is_unreachable(monkeypatch):
    """An unreachable catalogue must degrade to the static list, not crash."""
    monkeypatch.delenv("BOT_MODELS", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    import bot.llm as llm

    def boom(*args, **kwargs):
        raise OSError("no route to host")

    monkeypatch.setattr(llm.requests, "get", boom)
    llm._CATALOGUE_CACHE = None
    try:
        assert resolve_models(3) == STATIC_FALLBACK[:3]
    finally:
        llm._CATALOGUE_CACHE = None


def test_the_catalogue_is_read_without_a_key(monkeypatch):
    """The endpoint is public, and gating it on a key was the original bug.

    Before the fix, a run with no OpenRouter key skipped the catalogue entirely
    and used hardcoded model names from training data, which the Metaculus proxy
    then rejected. The resolver must use the live catalogue either way.
    """
    monkeypatch.delenv("BOT_MODELS", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    import bot.llm as llm

    seen = {}

    def fake_get(url, headers=None, timeout=None):
        seen["url"] = url
        seen["headers"] = headers
        return _FakeResponse(
            {
                "data": [
                    {"id": "openai/gpt-9-turbo", "created": 30},
                    {"id": "anthropic/claude-opus-9", "created": 20},
                    {"id": "google/gemini-9-pro", "created": 25},
                ]
            }
        )

    monkeypatch.setattr(llm.requests, "get", fake_get)
    llm._CATALOGUE_CACHE = None
    try:
        picked = resolve_models(3)
    finally:
        llm._CATALOGUE_CACHE = None

    assert seen["headers"] == {}, "no credentials should be sent"
    assert picked == ["openai/gpt-9-turbo", "anthropic/claude-opus-9", "google/gemini-9-pro"]
    assert picked != STATIC_FALLBACK[:3]


def test_resolve_models_honours_an_explicit_override(monkeypatch):
    monkeypatch.setenv("BOT_MODELS", " a/one , b/two ")
    assert resolve_models(5) == ["a/one", "b/two"]


def test_resolve_models_prefers_one_model_per_vendor(monkeypatch):
    monkeypatch.delenv("BOT_MODELS", raising=False)
    import bot.llm as llm

    llm._CATALOGUE_CACHE = [
        {"id": "openai/gpt-6-turbo", "created": 30},
        {"id": "openai/gpt-6-mini", "created": 31},
        {"id": "anthropic/claude-opus-5", "created": 20},
        {"id": "google/gemini-4-pro", "created": 25},
        {"id": "google/gemini-4-flash", "created": 26},
        {"id": "meta/llama-9", "created": 40},
        {"id": "openai/gpt-6-turbo:free", "created": 50},
    ]
    picked = resolve_models(3)
    llm._CATALOGUE_CACHE = None
    vendors = [m.split("/")[0] for m in picked]
    assert len(set(vendors)) == 3, picked
    assert not any(":free" in m for m in picked)
    assert "meta/llama-9" not in picked


def test_preference_patterns_compile():
    import re

    for pattern, score in MODEL_PREFERENCES:
        re.compile(pattern)
        assert 0 < score <= 100
