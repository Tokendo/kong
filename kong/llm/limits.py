"""Model-specific limits and rate limiting for LLM API calls."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class ModelLimits:
    """Context window and chunking parameters for a specific model.

    max_prompt_chars: safe upper bound on prompt length in characters,
        leaving room for system prompt and output within the context window.
    max_chunk_functions: cap on how many functions to pack into a single
        batch call (prevents the LLM from losing track of results).
    max_output_tokens: maximum output tokens for batch calls.
    """

    max_prompt_chars: int
    max_chunk_functions: int
    max_output_tokens: int


# Conservative fallback for models we have no context-window figure for.
_DEFAULT_LIMITS = ModelLimits(
    max_prompt_chars=400_000,
    max_chunk_functions=120,
    max_output_tokens=16384,
)

# Anthropic models with a 1M token context (~4M chars). 900k chars is ~225k
# tokens, which leaves the rest of the window for the system prompt and output.
#
# max_chunk_functions stays at 120 because batch size is bound by *output*
# tokens, not by the context window: analyze_function_batch() asks for 16384
# output tokens, and each function's JSON result costs ~130 of them. Raising
# the function count without also raising the output budget truncates the
# response mid-JSON.
_LARGE_CONTEXT_LIMITS = ModelLimits(
    max_prompt_chars=900_000,
    max_chunk_functions=120,
    max_output_tokens=16384,
)

# Z.ai's GLM-5.3 carries the same 1M token context, and needs different numbers
# anyway: it reasons before it answers, and the reasoning is charged to the
# completion budget the answer has to fit in too. 120 functions × ~130 tokens
# of JSON is already 95% of a 16k budget, so every chunk truncates before the
# model reaches its answer — an empty reply, paid for in full, and a whole
# chunk of functions marked failed at once.
#
# Measured on a 1475-function binary: at 16k every chunk of 120 truncated, 3
# for 3. Retried at 32k the same chunks succeeded once in three, taking 18 and
# 37 minutes to fail the other two. So ~273 completion tokens per function is
# the *marginal* rate, not a safe one. 40 functions against a 32k budget leaves
# roughly 800 per function — about three times what the model was seen to need,
# which is the margin that keeps a bad chunk from costing half an hour.
_GLM_LIMITS = ModelLimits(
    max_prompt_chars=900_000,
    max_chunk_functions=40,
    max_output_tokens=32768,
)

MODEL_LIMITS: dict[str, ModelLimits] = {
    # Anthropic — 1M token context.
    "claude-fable-5": _LARGE_CONTEXT_LIMITS,
    "claude-opus-5": _LARGE_CONTEXT_LIMITS,
    "claude-opus-4-8": _LARGE_CONTEXT_LIMITS,
    "claude-opus-4-7": _LARGE_CONTEXT_LIMITS,
    "claude-opus-4-6": _LARGE_CONTEXT_LIMITS,
    "claude-sonnet-5": _LARGE_CONTEXT_LIMITS,
    "claude-sonnet-4-6": _LARGE_CONTEXT_LIMITS,
    # Anthropic — 200k token context (~800k chars) so 400k chars is a safe cap.
    "claude-sonnet-4-20250514": _DEFAULT_LIMITS,
    "claude-haiku-4-5": _DEFAULT_LIMITS,
    "claude-haiku-4-5-20251001": _DEFAULT_LIMITS,
    # Z.ai — GLM-5.3 and its flash sibling carry a 1M token context, and both
    # reason into the completion budget. The flash model has not been measured
    # separately; it gets the same numbers until it has been, since the failure
    # this avoids costs a chunk of functions rather than a retry.
    "glm-5.3": _GLM_LIMITS,
    "glm-5.3-flash": _GLM_LIMITS,
    # OpenAI — 128k token context (~512k chars) so 350k chars leaves room for overhead.
    "gpt-4o": ModelLimits(350_000, 80, 16384),
    "gpt-4o-2024-11-20": ModelLimits(350_000, 80, 16384),
    "gpt-4o-mini": ModelLimits(350_000, 80, 16384),
    "gpt-4o-mini-2024-07-18": ModelLimits(350_000, 80, 16384),
    # OpenAI reasoning models — 200k context so more expensive, smaller batches.
    "o1": ModelLimits(400_000, 40, 32768),
    "o3-mini": ModelLimits(400_000, 60, 32768),
}


def get_model_limits(model: str) -> ModelLimits:
    """Look up chunking limits for a model, falling back to defaults."""
    return MODEL_LIMITS.get(model, _DEFAULT_LIMITS)


def take_within_budget(
    items: Iterable[T],
    render: Callable[[T], str],
    budget: int,
) -> list[T]:
    """Return the longest prefix of *items* whose rendered text fits *budget* chars.

    Prompt sections that grow with the size of the binary (the list of names
    resolved so far, recovered struct definitions, per-function bodies) have to
    be cut somewhere, or they eventually crowd out the code the prompt is
    actually about. Callers pass the items in priority order.
    """
    kept: list[T] = []
    used = 0
    for item in items:
        cost = len(render(item)) + 1  # the newline that joins sections
        if used + cost > budget:
            break
        kept.append(item)
        used += cost
    return kept


class RateLimiter:
    """Token-bucket style rate limiter for API calls.

    Tracks request timestamps and sleeps proactively to stay below the
    configured requests-per-minute (RPM) and tokens-per-minute (TPM) limits.
    """

    def __init__(
        self,
        requests_per_minute: int | None = None,
        tokens_per_minute: int | None = None,
    ) -> None:
        self._rpm = requests_per_minute
        self._tpm = tokens_per_minute
        self._lock = threading.Lock()
        self._request_times: list[float] = []
        self._token_log: list[tuple[float, int]] = []

    def wait_if_needed(self, estimated_tokens: int = 0) -> None:
        """Block until it's safe to make the next request."""
        while True:
            with self._lock:
                now = time.monotonic()
                window_start = now - 60.0
                wait = 0.0

                if self._rpm is not None:
                    self._request_times = [
                        t for t in self._request_times if t > window_start
                    ]
                    if len(self._request_times) >= self._rpm:
                        sleep_until = self._request_times[0] + 60.0
                        wait = max(wait, sleep_until - now)

                if self._tpm is not None and estimated_tokens > 0:
                    self._token_log = [
                        (t, n) for t, n in self._token_log if t > window_start
                    ]
                    total_tokens = sum(n for _, n in self._token_log)
                    if total_tokens + estimated_tokens > self._tpm:
                        sleep_until = (
                            self._token_log[0][0] + 60.0
                            if self._token_log
                            else now
                        )
                        wait = max(wait, sleep_until - now)

                if wait <= 0:
                    return

            # Sleep outside the lock so other threads aren't blocked.
            time.sleep(wait)

    def record_request(self, tokens_used: int = 0) -> None:
        """Record that a request was made."""
        with self._lock:
            now = time.monotonic()
            self._request_times.append(now)
            if tokens_used > 0:
                self._token_log.append((now, tokens_used))
