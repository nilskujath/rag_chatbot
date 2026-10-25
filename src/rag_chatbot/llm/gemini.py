"""Text generation through Google AI Studio (Gemini).

The free tier is throttled client side (`RateLimiter`) and transient API errors
(429 quota, 503 overload) are retried with exponential backoff, so long runs
such as the evaluation survive a busy API.
"""

import logging
import threading
import time
from collections.abc import Callable, Iterator
from functools import cache
from typing import TypeVar

from google import genai
from google.genai import errors, types

from rag_chatbot.settings import settings

logger = logging.getLogger(__name__)
T = TypeVar("T")

RETRY_CODES = (429, 503)


class RateLimiter:
    """Space calls at least 60 / `requests_per_minute` seconds apart. Thread safe.

    `clock` and `sleep` are injectable so that the spacing can be tested without
    waiting.
    """

    def __init__(
        self,
        requests_per_minute: int,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._interval = 60.0 / requests_per_minute
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def wait(self) -> None:
        """Block until the next slot, then reserve the one after it."""
        with self._lock:
            now = self._clock()
            slot = max(now, self._next_slot)
            self._next_slot = slot + self._interval
        if slot > now:
            logger.debug("rate limit: wait %.1fs", slot - now)
            self._sleep(slot - now)


limiter = RateLimiter(settings.llm_requests_per_minute)


@cache
def _client() -> genai.Client:
    """Build the client on first use, so that imports never need an API key."""
    return genai.Client(api_key=settings.google_api_key)


def _with_retry(call: Callable[[], T]) -> T:
    """Run an API call up to `llm_max_attempts` times.

    Only 429 and 503 are retried, after 10 s, 20 s, 40 s, ...; any other error,
    and the last failed attempt, propagate to the caller.
    """
    attempt = 0
    while True:
        limiter.wait()
        try:
            return call()
        except errors.APIError as error:
            attempt += 1
            if error.code not in RETRY_CODES or attempt == settings.llm_max_attempts:
                raise
            delay = 10 * 2 ** (attempt - 1)
            logger.warning("LLM error %s, retry in %ds", error.code, delay)
            time.sleep(delay)


def generate(
    prompt: str,
    model: str | None = None,
    json_output: bool = False,
    temperature: float | None = None,
) -> str:
    """Return the text of a single-turn completion.

    `json_output` switches the API to JSON mode; `temperature` is passed through
    (0 for the router, the SQL writer and the RAGAS judge, default otherwise).
    """
    model = model or settings.llm_model
    config = types.GenerateContentConfig(
        response_mime_type="application/json" if json_output else None,
        temperature=temperature,
    )
    start = time.perf_counter()
    response = _with_retry(
        lambda: _client().models.generate_content(
            model=model, contents=prompt, config=config
        )
    )
    logger.info("LLM %s: %.1fs", model, time.perf_counter() - start)
    return response.text or ""


def stream(prompt: str, model: str | None = None) -> Iterator[str]:
    """Yield the text of a single-turn completion as it is produced.

    A quota error surfaces on the first chunk, not when the stream is opened,
    so the retried step covers both; later chunks are not retried.
    """
    model = model or settings.llm_model
    start = time.perf_counter()

    def open_stream():
        parts = iter(
            _client().models.generate_content_stream(model=model, contents=prompt)
        )
        return parts, next(parts, None)

    parts, part = _with_retry(open_stream)
    while part is not None:
        if part.text:
            yield part.text
        part = next(parts, None)
    logger.info("LLM %s (stream): %.1fs", model, time.perf_counter() - start)
