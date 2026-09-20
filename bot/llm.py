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

import threading

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

# Any ":suffix" on an OpenRouter id is a routing variant, not a different
# model. The live catalogue handed back "openai/gpt-6-astra:batch", and batch
# routing can take hours to return, against a question window of three. Reject
# every variant rather than blocklisting them one at a time.
VARIANT = re.compile(r":")
EXCLUDE = re.compile(r"(-preview|audio|image|tts|embed|moderation|search)", re.I)

# Fallback if the catalogue cannot be read. Deliberately family-diverse.
STATIC_FALLBACK = ["openai/gpt-5", "anthropic/claude-sonnet-4", "google/gemini-2.5-pro"]


# Requests per minute and maximum concurrency per provider. The first live run
# fired fifteen requests at once at a free Gemini key and got nothing but 429s
# for four minutes. Sending fewer requests is what makes them succeed.
PROVIDER_LIMITS: dict[str, tuple[float, int]] = {
    "openrouter": (120.0, 6),
    "gemini": (10.0, 2),
    "groq": (25.0, 3),
    "metaculus": (20.0, 2),
}


class _RateLimiter:
    """A minimum spacing plus a concurrency cap, per provider."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next_free: dict[str, float] = {}
        self._slots: dict[str, threading.Semaphore] = {}

    def slot(self, name: str) -> threading.Semaphore:
        with self._lock:
            if name not in self._slots:
                _, concurrency = PROVIDER_LIMITS.get(name, (60.0, 4))
                self._slots[name] = threading.Semaphore(concurrency)
            return self._slots[name]

    def wait(self, name: str) -> None:
        rpm, _ = PROVIDER_LIMITS.get(name, (60.0, 4))
        spacing = 60.0 / max(rpm, 1.0)
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next_free.get(name, 0.0))
            self._next_free[name] = start + spacing
        delay = start - time.monotonic()
        if delay > 0:
            time.sleep(delay)


LIMITER = _RateLimiter()


class LLMError(RuntimeError):
    pass


class ModelUnavailable(LLMError):
    """The model will never work with these credentials: wrong name, or no allowance.

    Distinct from a transient failure. Retrying it on the next question wastes a
    request and, when the whole ensemble is unavailable, turns one dead provider
    into hundreds of pointless calls a minute against a shared proxy.
    """


class NoModelsAvailable(LLMError):
    """Every model in the ensemble is unavailable. The run cannot forecast at all."""


# Models that answered with an allowance or authentication error this process.
DEAD_MODELS: set[str] = set()


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
            return prov, _adapt_model_name(prov, model[len(prefix):])
    order = ["openrouter", "metaculus", "gemini", "groq"]
    for name in order:
        if PROVIDERS[name].key:
            return PROVIDERS[name], _adapt_model_name(PROVIDERS[name], model)
    raise LLMError(
        "No LLM credentials found. Set OPENROUTER_API_KEY (tournament credits), "
        "or METACULUS_TOKEN to use the Metaculus proxy, or GEMINI_API_KEY / GROQ_API_KEY."
    )


def _adapt_model_name(prov: Provider, model: str) -> str:
    """OpenRouter uses ``vendor/model``; the other providers want the bare name.

    Sending "openai/gpt-5" to the Metaculus proxy produced
    "You don\'t have an allowance for model <openai/gpt-5> on <Openai>", because
    the vendor prefix is part of the name it looks up. The same single-segment
    strip also turns Google's catalogue form "models/gemini-2.5-pro" into the
    "gemini-2.5-pro" its OpenAI-compatible endpoint is documented with, and
    leaves a bare Groq id such as "llama-3.3-70b-versatile" untouched.
    """
    if prov.name == "openrouter":
        return model
    return model.split("/", 1)[1] if "/" in model else model


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
        with LIMITER.slot(prov.name):
            LIMITER.wait(prov.name)
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

        if resp.status_code == 429:
            # Free tiers count per minute, so a few seconds of backoff just
            # spends another request against the same exhausted window.
            retry_after = resp.headers.get("Retry-After")
            delay = 30.0 * attempt
            if retry_after:
                try:
                    delay = max(float(retry_after), 5.0)
                except ValueError:
                    pass
            last = f"HTTP 429, waited {delay:.0f}s"
            log.info("%s rate limited, sleeping %.0fs", prov.name, delay)
            time.sleep(min(delay, 90.0))
            continue

        if resp.status_code in (500, 502, 503, 504):
            last = f"HTTP {resp.status_code}"
            time.sleep(min(2 ** attempt, 30))
            continue
        if resp.status_code in (400, 401, 403, 404):
            DEAD_MODELS.add(model)
            raise ModelUnavailable(
                f"{prov.name} {bare_model} -> HTTP {resp.status_code}: {resp.text[:300]}"
            )
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
    live = [m for m in models if m not in DEAD_MODELS]
    if not live:
        raise NoModelsAvailable(
            "every model is unavailable with the current credentials: "
            + ", ".join(sorted(DEAD_MODELS))
        )
    for model in live:
        try:
            return chat(messages, model, **kwargs), model
        except ModelUnavailable as exc:
            errors.append(f"{model}: {exc}")
            log.warning("model %s is unavailable, will not retry it: %s", model, str(exc)[:200])
        except LLMError as exc:
            errors.append(f"{model}: {exc}")
            log.warning("model %s failed, trying next: %s", model, str(exc)[:200])
    if all(m in DEAD_MODELS for m in models):
        raise NoModelsAvailable("all models failed permanently:\n" + "\n".join(errors))
    raise LLMError("all models failed:\n" + "\n".join(errors))


# -- model resolution ------------------------------------------------------
_CATALOGUE_CACHE: list[dict] | None = None

# Not every id a provider lists is a chat model. The first live run picked
# "models/aqa", which is an attributed-question-answering endpoint and returns
# 404 for generateContent, and "gemini-2.5-pro", which Google has closed to new
# keys. Both cost a whole run.
_NOT_A_CHAT_MODEL = re.compile(
    r"(aqa|embed|imagen|veo|image-gen|^models/text-|tts|speech|audio|live|vision-only|learnlm)",
    re.I,
)

# On a free tier the flash class is what actually answers: the pro class has a
# per-minute limit low enough that a five member ensemble exhausts it on the
# first question. Capability is worth less than a reply.
_FLASH = re.compile(r"flash", re.I)
_LITE = re.compile(r"(lite|mini|nano|tiny|8b|instant)", re.I)
_VERSION = re.compile(r"(\d+(?:\.\d+)?)")


def active_provider() -> Provider | None:
    """Whichever provider this run can actually reach, in preference order."""
    for name in ("openrouter", "gemini", "groq", "metaculus"):
        if PROVIDERS[name].key:
            return PROVIDERS[name]
    return None


def provider_catalogue(prov: Provider) -> list[str]:
    """Ask a provider which models it serves. Empty list if it will not say."""
    headers = {"Authorization": f"{prov.auth_scheme} {prov.key}"} if prov.key else {}
    try:
        resp = requests.get(f"{prov.base_url}/models", headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # noqa: BLE001
        log.warning("could not list %s models: %s", prov.name, str(exc)[:200])
        return []
    entries = data.get("data") if isinstance(data, dict) else data
    out = []
    for entry in entries or []:
        mid = entry.get("id") if isinstance(entry, dict) else entry
        if mid and not EXCLUDE.search(str(mid)):
            out.append(str(mid))
    return out


def _rank_bare(ids: Sequence[str]) -> list[str]:
    """Order a provider's own model ids: answerable first, newest first."""

    def score(mid: str) -> tuple[float, float, str]:
        tier = 0.0 if _FLASH.search(mid) else 1.0
        if _LITE.search(mid):
            tier += 0.5
        versions = [float(v) for v in _VERSION.findall(mid)] or [0.0]
        return (tier, -max(versions), mid)

    usable = [m for m in ids if not _NOT_A_CHAT_MODEL.search(m)]
    return sorted(usable, key=score)


def _catalogue() -> list[dict]:
    global _CATALOGUE_CACHE
    if _CATALOGUE_CACHE is not None:
        return _CATALOGUE_CACHE
    prov = PROVIDERS["openrouter"]
    # This endpoint needs no credentials, so the resolver still works before the
    # tournament credits arrive. Gating it on a key was why a run with no key
    # fell back to hardcoded model names that no longer exist.
    headers = {"Authorization": f"Bearer {prov.key}"} if prov.key else {}
    try:
        resp = requests.get(
            f"{prov.base_url}/models",
            headers=headers,
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

    prov = active_provider()
    if prov is not None and prov.name != "openrouter":
        # No OpenRouter key, so the ensemble comes from whichever provider we
        # do have. Ask it what it serves; hardcoding names is what produced
        # "You don\'t have an allowance for model <openai/gpt-5>".
        served = _rank_bare(provider_catalogue(prov))
        if served:
            return [f"{prov.name}/{mid}" for mid in served[:count]]
        log.warning("%s served no model list; falling back", prov.name)

    catalogue = _catalogue()
    if not catalogue:
        return STATIC_FALLBACK[:count] if count <= len(STATIC_FALLBACK) else STATIC_FALLBACK

    scored: list[tuple[int, int, str, str]] = []
    for entry in catalogue:
        mid = entry.get("id") or ""
        if VARIANT.search(mid) or EXCLUDE.search(mid):
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


def metaculus_proxy_models() -> list[str]:
    """Ask the Metaculus LLM proxy which models the bot token is allowed to use.

    The proxy rejects a name it does not recognise with an "allowance" error, so
    guessing is expensive. This is reported by --check-sources.
    """
    prov = PROVIDERS["metaculus"]
    if not prov.key:
        return []
    try:
        resp = requests.get(
            f"{prov.base_url}/models",
            headers={"Authorization": f"Token {prov.key}"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # noqa: BLE001
        log.warning("could not list the Metaculus proxy models: %s", str(exc)[:200])
        return []
    entries = data.get("data") if isinstance(data, dict) else data
    out = []
    for entry in entries or []:
        mid = entry.get("id") if isinstance(entry, dict) else entry
        if mid:
            out.append(str(mid))
    return sorted(out)


def provider_is_metered(threshold: float = 30.0) -> bool:
    """True when the run is on a low rate limit and should spend calls sparingly."""
    prov = active_provider()
    if prov is None:
        return True
    rpm, _ = PROVIDER_LIMITS.get(prov.name, (60.0, 4))
    return rpm < threshold
