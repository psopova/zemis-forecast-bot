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


def test_resolve_models_falls_back_without_a_catalogue(monkeypatch):
    monkeypatch.delenv("BOT_MODELS", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    import bot.llm as llm

    llm._CATALOGUE_CACHE = None
    picked = resolve_models(3)
    assert picked == STATIC_FALLBACK[:3]


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
