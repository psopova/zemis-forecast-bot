"""LLM access with a fallback chain, speaking one dialect to four providers.

Every provider here exposes an OpenAI-compatible /chat/completions endpoint, so
there is a single code path and no litellm (which is 133 MB and pulls in boto3).

Model ids are resolved at runtime from the provider catalogue rather than
pinned, because the season runs four months and model names churn inside that
window. BOT_MODELS overrides the resolver when a specific set is wanted.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import requests

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    key_env: str
    auth_scheme: str = "Bearer"

    @property
    def key(self) -> str | None:
        return os.environ.get(self.key_env) or None


PROVIDERS: dict[str, Provider] = {
    # Where the sponsored tournament credits live.
    "openrouter": Provider("openrouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    # Metaculus's own proxy, authenticated with the bot token. Documented as a
    # fallback: it does not support every model.
    "metaculus": Provider(
        "metaculus",
        "https://llm-proxy.metaculus.com/proxy/openai/v1",
        "METACULUS_TOKEN",
        auth_scheme="Token",
    ),
    "gemini": Provider(
        "gemini", "https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"
    ),
    "groq": Provider("groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY"),
}

# Ordered preference patterns. Higher score wins; ties break on how recently the
# model was published. Only vendors whose credits Metaculus sponsors are listed,
# because those are the ones that cost nothing.
MODEL_PREFERENCES: list[tuple[str, int]] = [
    (r"^openai/(gpt-[6-9]|o[5-9])(?!.*mini)", 100),
    (r"^anthropic/claude.*(opus|fable)", 100),
    (r"^google/gemini.*pro", 95),
    (r"^openai/gpt-[5-9]", 85),
    (r"^anthropic/claude.*sonnet", 85),
    (r"^google/gemini.*flash(?!.*lite)", 80),
    (r"^openai/", 50),
    (r"^anthropic/", 50),
    (r"^google/gemini", 50),
]

EXCLUDE = re.compile(r"(:free|:online|-preview|audio|image|tts|embed|moderation|search)", re.I)

# Fallback if the catalogue cannot be read. Deliberately family-diverse.
STATIC_FALLBACK = ["openai/gpt-5", "anthropic/claude-sonnet-4", "google/gemini-2.5-pro"]


class LLMError(RuntimeError):
    pass


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    by_model: dict[str, int] = field(default_factory=dict)

    def record(self, model: str, usage: dict | None, cost: float = 0.0) -> None:
        self.calls += 1
        self.by_model[model] = self.by_model.get(model, 0) + 1
        self.cost_usd += cost
        if usage:
            self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
            self.completion_tokens += int(usage.get("completion_tokens") or 0)

    def summary(self) -> str:
        return (
            f"{self.calls} calls, {self.prompt_tokens}+{self.completion_tokens} tokens, "
            f"${self.cost_usd:.3f}, models={self.by_model}"
        )


USAGE = Usage()


def _provider_for(model: str) -> tuple[Provider, str]:
    """Route by an explicit prefix, else to whichever provider has a key."""
    for name, prov in PROVIDERS.items():
        prefix = f"{name}/"
        if model.startswith(prefix):
            return prov, model[len(prefix):]
    order = ["openrouter", "metaculus", "gemini", "groq"]
    for name in order:
        if PROVIDERS[name].key:
            return PROVIDERS[name], model
    raise LLMError(
        "No LLM credentials found. Set OPENROUTER_API_KEY (tournament credits), "
        "or METACULUS_TOKEN to use the Metaculus proxy, or GEMINI_API_KEY / GROQ_API_KEY."
    )


def chat(
    messages: Sequence[dict],
    model: str,
    temperature: float = 0.3,
    max_tokens: int = 3000,
    timeout: float = 240.0,
    attempts: int = 3,
) -> str:
    prov, bare_model = _provider_for(model)
    key = prov.key
    if not key:
        raise LLMError(f"{prov.key_env} is not set, needed for model {model}")

    headers = {
        "Authorization": f"{prov.auth_scheme} {key}",
        "Content-Type": "application/json",
    }
    if prov.name == "openrouter":
        headers["X-Title"] = "metaculus-forecast-bot"

    body = {
        "model": bare_model,
        "messages": list(messages),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    last = None
    for attempt in range(1, attempts + 1):
        try:
            resp = requests.post(
                f"{prov.base_url}/chat/completions",
                headers=headers,
                json=body,
                timeout=timeout,
            )
        except requests.RequestException as exc:
            last = f"{type(exc).__name__}: {exc}"
            time.sleep(min(2 ** attempt, 30))
            continue

        if resp.status_code in (429, 500, 502, 503, 504):
            last = f"HTTP {resp.status_code}"
            time.sleep(min(2 ** attempt, 30))
            continue
        if not resp.ok:
            raise LLMError(f"{prov.name} {bare_model} -> HTTP {resp.status_code}: {resp.text[:400]}")

        data = resp.json()
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"{prov.name} returned no content: {json.dumps(data)[:400]}")
        cost = 0.0
        usage = data.get("usage") or {}
        if isinstance(usage.get("cost"), (int, float)):
            cost = float(usage["cost"])
        USAGE.record(model, usage, cost)
        return text or ""

    raise LLMError(f"{prov.name} {bare_model} failed after {attempts} attempts: {last}")


def chat_with_fallback(
    messages: Sequence[dict],
    models: Sequence[str],
    **kwargs,
) -> tuple[str, str]:
    """Try each model in turn. Returns (text, model_that_answered)."""
    errors = []
    for model in models:
        try:
            return chat(messages, model, **kwargs), model
        except LLMError as exc:
            errors.append(f"{model}: {exc}")
            log.warning("model %s failed, trying next: %s", model, str(exc)[:200])
    raise LLMError("all models failed:\n" + "\n".join(errors))


# -- model resolution ------------------------------------------------------
_CATALOGUE_CACHE: list[dict] | None = None


def _catalogue() -> list[dict]:
    global _CATALOGUE_CACHE
    if _CATALOGUE_CACHE is not None:
        return _CATALOGUE_CACHE
    prov = PROVIDERS["openrouter"]
    if not prov.key:
        _CATALOGUE_CACHE = []
        return _CATALOGUE_CACHE
    try:
        resp = requests.get(
            f"{prov.base_url}/models",
            headers={"Authorization": f"Bearer {prov.key}"},
            timeout=30,
        )
        resp.raise_for_status()
        _CATALOGUE_CACHE = resp.json().get("data") or []
    except Exception as exc:  # catalogue is an optimisation, never fatal
        log.warning("could not read the model catalogue: %s", exc)
        _CATALOGUE_CACHE = []
    return _CATALOGUE_CACHE


def resolve_models(count: int = 3) -> list[str]:
    """Pick ``count`` capable models from distinct vendors.

    Vendor diversity is the point: the bot-maker surveys found ensembling across
    model families is worth far more than which single model is best, because
    correlated errors are what sink an ensemble.
    """
    override = os.environ.get("BOT_MODELS", "").strip()
    if override:
        return [m.strip() for m in override.split(",") if m.strip()]

    catalogue = _catalogue()
    if not catalogue:
        return STATIC_FALLBACK[:count] if count <= len(STATIC_FALLBACK) else STATIC_FALLBACK

    scored: list[tuple[int, int, str, str]] = []
    for entry in catalogue:
        mid = entry.get("id") or ""
        if EXCLUDE.search(mid):
            continue
        for pattern, score in MODEL_PREFERENCES:
            if re.search(pattern, mid):
                vendor = mid.split("/", 1)[0]
                scored.append((score, int(entry.get("created") or 0), vendor, mid))
                break

    scored.sort(reverse=True)
    picked: list[str] = []
    seen_vendors: set[str] = set()
    for _, _, vendor, mid in scored:
        if vendor in seen_vendors:
            continue
        picked.append(mid)
        seen_vendors.add(vendor)
        if len(picked) >= count:
            break
    # Top up from the remainder if fewer vendors were available than requested.
    if len(picked) < count:
        for _, _, _, mid in scored:
            if mid not in picked:
                picked.append(mid)
            if len(picked) >= count:
                break
    return picked or STATIC_FALLBACK[:count]


# -- structured output -----------------------------------------------------
_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> Any:
    """Pull a JSON value out of model prose.

    Models wrap JSON in fences, prefix it with commentary, and occasionally emit
    trailing commas. All of that is recoverable and none of it is worth a retry.
    """
    if text is None:
        raise ValueError("no text to parse")
    candidates: list[str] = []
    for match in _JSON_BLOCK.finditer(text):
        candidates.append(match.group(1))
    candidates.append(text)

    for chunk in candidates:
        chunk = chunk.strip()
        # Try whichever bracket opens first, so a list of objects is read as a
        # list rather than as its first element.
        pairs = [("{", "}"), ("[", "]")]
        pairs.sort(key=lambda pair: chunk.find(pair[0]) if chunk.find(pair[0]) != -1 else len(chunk) + 1)
        for opener, closer in pairs:
            start = chunk.find(opener)
            end = chunk.rfind(closer)
            if start == -1 or end <= start:
                continue
            body = chunk[start : end + 1]
            for attempt in (body, re.sub(r",\s*([}\]])", r"\1", body)):
                try:
                    return json.loads(attempt)
                except json.JSONDecodeError:
                    continue
    raise ValueError(f"no JSON found in model output: {text[:300]!r}")


def run_parallel(tasks: Sequence[Callable[[], Any]], workers: int = 6) -> list[Any]:
    """Run independent LLM calls concurrently; failures come back as exceptions."""
    results: list[Any] = [None] * len(tasks)
    if not tasks:
        return results
    with cf.ThreadPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
        futures = {pool.submit(task): i for i, task in enumerate(tasks)}
        for fut in cf.as_completed(futures):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception as exc:  # noqa: BLE001 - surfaced to the caller
                results[i] = exc
    return results
