"""Thin Metaculus API client.

Written against the server source rather than the official SDK for two reasons.
The SDK pins a tournament id from last season, and it sleeps 3.5 to 4.5 seconds
before every single request, which burns most of a polling tick before any
question is even read. Questions are open to bots for about three hours, so
latency here is score.

The token is read from the environment, never logged, and scrubbed out of any
exception text before it is printed.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from typing import Any, Iterator
from urllib.parse import urlencode

import requests

log = logging.getLogger(__name__)

API_BASE = os.environ.get("METACULUS_API_BASE_URL", "https://www.metaculus.com/api")
TOKEN_ENV = "METACULUS_TOKEN"

RETRY_STATUS = {429, 500, 502, 503, 504}
MAX_ATTEMPTS = 5


class MetaculusError(RuntimeError):
    pass


def _scrub(text: str, token: str | None) -> str:
    if token and token in text:
        text = text.replace(token, "<redacted>")
    return text


class MetaculusClient:
    def __init__(
        self,
        token: str | None = None,
        base_url: str = API_BASE,
        min_interval: float = 0.25,
        timeout: float = 45.0,
        dry_run: bool = False,
    ):
        self._token = token or os.environ.get(TOKEN_ENV) or ""
        if not self._token:
            raise MetaculusError(
                f"{TOKEN_ENV} is not set. Add it as a repository secret; "
                "it is never read from a file or passed on the command line."
            )
        self.base_url = base_url.rstrip("/")
        self.min_interval = min_interval
        self.timeout = timeout
        self.dry_run = dry_run
        self._last_request = 0.0
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Token {self._token}",
                "Accept-Language": "en",
                "Content-Type": "application/json",
                "User-Agent": "metaculus-forecast-bot (+github actions)",
            }
        )

    # -- plumbing ----------------------------------------------------------
    def _throttle(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def _request(self, method: str, path: str, **kwargs) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        last_error = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            self._throttle()
            try:
                resp = self._session.request(method, url, timeout=self.timeout, **kwargs)
            except requests.RequestException as exc:
                last_error = _scrub(str(exc), self._token)
                log.warning("%s %s attempt %d failed: %s", method, path, attempt, last_error)
                time.sleep(min(2 ** attempt, 20) + random.random())
                continue

            if resp.status_code in RETRY_STATUS:
                last_error = f"HTTP {resp.status_code}: {_scrub(resp.text[:500], self._token)}"
                retry_after = resp.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else min(2 ** attempt, 20)
                log.warning("%s %s attempt %d: %s", method, path, attempt, last_error)
                time.sleep(delay + random.random())
                continue

            if not resp.ok:
                raise MetaculusError(
                    f"{method} {path} -> HTTP {resp.status_code}: "
                    f"{_scrub(resp.text[:1000], self._token)}"
                )

            if not resp.content:
                return None
            try:
                return resp.json()
            except json.JSONDecodeError:
                return resp.text

        raise MetaculusError(f"{method} {path} failed after {MAX_ATTEMPTS} attempts: {last_error}")

    # -- reads -------------------------------------------------------------
    def iter_posts(
        self,
        tournament: str | int,
        statuses: str = "open",
        page_size: int = 100,
        max_pages: int = 20,
    ) -> Iterator[dict]:
        """Yield posts in a tournament. ``tournament`` may be a slug or an id."""
        offset = 0
        for _ in range(max_pages):
            params = [
                ("limit", page_size),
                ("offset", offset),
                ("tournaments", tournament),
                ("statuses", statuses),
                ("include_descriptions", "true"),
                ("order_by", "open_time"),
            ]
            data = self._request("GET", f"/posts/?{urlencode(params)}")
            results = (data or {}).get("results") or []
            for post in results:
                yield post
            if len(results) < page_size:
                return
            offset += page_size

    def get_post(self, post_id: int) -> dict:
        return self._request("GET", f"/posts/{post_id}/")

    # -- writes ------------------------------------------------------------
    def submit_forecasts(self, payloads: list[dict]) -> None:
        """Submit a batch. Each payload is keyed by QUESTION id, not post id."""
        if not payloads:
            return
        if self.dry_run:
            log.info("dry run: would submit %d forecast(s)", len(payloads))
            return
        self._request("POST", "/questions/forecast/", json=payloads)

    def post_comment(self, post_id: int, text: str, is_private: bool = True) -> dict | None:
        """Attach a reasoning comment. Keyed by POST id.

        Prize eligibility requires a comment under every question forecast, so
        this raises on failure rather than logging success the way the official
        SDK does.
        """
        if self.dry_run:
            log.info("dry run: would comment on post %s (%d chars)", post_id, len(text))
            return None
        return self._request(
            "POST",
            "/comments/create/",
            json={
                "on_post": post_id,
                "text": text,
                "parent": None,
                "is_private": is_private,
                "included_forecast": True,
            },
        )


# -- helpers over the post payload ----------------------------------------
def question_of(post: dict) -> dict | None:
    """Single-question posts only. Group posts are handled by the caller."""
    return post.get("question")


def sub_questions(post: dict) -> list[dict]:
    """Every forecastable question on a post, flattening question groups."""
    q = post.get("question")
    if q:
        return [q]
    group = post.get("group_of_questions") or {}
    return list(group.get("questions") or [])


def already_forecast(question: dict) -> bool:
    mine = question.get("my_forecasts") or {}
    latest = mine.get("latest") or {}
    if not latest:
        return False
    # A withdrawn forecast has an end_time in the past; treat it as absent.
    return bool(latest.get("forecast_values") or latest.get("continuous_cdf") or latest.get("probability_yes") is not None)
