"""Knowing when Kong is waiting on the model.

Most of a run's wall clock is spent inside one HTTP request: a chunk of forty
functions goes out and nothing happens for a minute or two. From the outside
that is indistinguishable from a hung program, which is why the interface has
to be able to say *which* call it is waiting on and for how long.

The information is collected here rather than in each client so that it does
not have to be added to every provider, and so that a client handed in by a
test or a plugin is tracked too: `TrackedLLMClient` wraps whatever satisfies
the LLMClient protocol and forwards everything it does not time.
"""

from __future__ import annotations

import itertools
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

#: What the three tracked entry points are called in the interface.
_KIND_LABELS = {
    "batch": "batch of functions",
    "function": "one function",
    "tools": "agentic round",
}


@dataclass(frozen=True)
class LLMCall:
    """One request that has been sent and not yet answered."""

    id: int
    kind: str
    model: str
    prompt_chars: int
    max_tokens: int | None
    started_at: float

    def waiting_seconds(self, now: float | None = None) -> float:
        return max(0.0, (now if now is not None else time.time()) - self.started_at)

    @property
    def label(self) -> str:
        kind = _KIND_LABELS.get(self.kind, self.kind)
        size = f"{self.prompt_chars / 1000:.0f}k chars" if self.prompt_chars else ""
        budget = f"{self.max_tokens} token budget" if self.max_tokens else ""
        detail = " · ".join(part for part in (kind, size, budget) if part)
        return f"{self.model} — {detail}" if detail else self.model


@dataclass(frozen=True)
class ActivitySnapshot:
    """What the interface renders. Taken under the lock, safe to keep."""

    waiting: bool = False
    in_flight: int = 0
    #: The longest-running request, which is the one worth naming.
    label: str = ""
    model: str = ""
    kind: str = ""
    waiting_seconds: float = 0.0
    #: Every request that has come back, and the time they spent in flight.
    completed: int = 0
    total_wait_seconds: float = 0.0
    last_wait_seconds: float = 0.0
    calls: tuple[LLMCall, ...] = ()


class LLMActivity:
    """Thread-safe registry of the requests currently in flight.

    The finishing pass and the coherence review each run on their own thread
    and can be in flight while the analysis is paused mid-chunk, so this is
    shared state rather than a single "current call".
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ids = itertools.count(1)
        self._active: dict[int, LLMCall] = {}
        self._completed = 0
        self._total_wait = 0.0
        self._last_wait = 0.0

    def begin(
        self,
        kind: str,
        model: str,
        prompt_chars: int = 0,
        max_tokens: int | None = None,
    ) -> LLMCall:
        with self._lock:
            call = LLMCall(
                id=next(self._ids),
                kind=kind,
                model=model,
                prompt_chars=prompt_chars,
                max_tokens=max_tokens,
                started_at=time.time(),
            )
            self._active[call.id] = call
            return call

    def end(self, call: LLMCall) -> float:
        """Mark *call* answered. Returns how long it was in flight."""
        waited = call.waiting_seconds()
        with self._lock:
            if self._active.pop(call.id, None) is None:
                # Ended twice: the first one already paid for the statistics.
                return waited
            self._completed += 1
            self._total_wait += waited
            self._last_wait = waited
        return waited

    @contextmanager
    def track(
        self,
        kind: str,
        model: str,
        prompt_chars: int = 0,
        max_tokens: int | None = None,
    ) -> Iterator[LLMCall]:
        call = self.begin(kind, model, prompt_chars, max_tokens)
        try:
            yield call
        finally:
            # A request that fails is a request that stopped being waited on:
            # leaving it in flight would pin the indicator on forever.
            self.end(call)

    def snapshot(self) -> ActivitySnapshot:
        now = time.time()
        with self._lock:
            calls = tuple(sorted(self._active.values(), key=lambda c: c.started_at))
            completed = self._completed
            total_wait = self._total_wait
            last_wait = self._last_wait

        if not calls:
            return ActivitySnapshot(
                completed=completed,
                total_wait_seconds=total_wait,
                last_wait_seconds=last_wait,
            )

        oldest = calls[0]
        return ActivitySnapshot(
            waiting=True,
            in_flight=len(calls),
            label=oldest.label,
            model=oldest.model,
            kind=oldest.kind,
            waiting_seconds=oldest.waiting_seconds(now),
            completed=completed,
            total_wait_seconds=total_wait,
            last_wait_seconds=last_wait,
            calls=calls,
        )


class TrackedLLMClient:
    """An LLM client that says when it is waiting for an answer.

    Wraps the three entry points that actually cross the network and forwards
    everything else — `model`, `usage`, `total_cost_usd` — to the client
    underneath, so it is a drop-in wherever an LLMClient is expected.
    """

    def __init__(self, client: Any, activity: LLMActivity) -> None:
        self._client = client
        self._activity = activity

    @property
    def activity(self) -> LLMActivity:
        return self._activity

    @property
    def wrapped(self) -> Any:
        return self._client

    def _model_name(self, model: str | None) -> str:
        return model or getattr(self._client, "model", "") or "model"

    def analyze_function(self, prompt: str, *, model: str | None = None) -> Any:
        with self._activity.track(
            "function",
            self._model_name(model),
            len(prompt),
            getattr(self._client, "max_tokens", None),
        ):
            return self._client.analyze_function(prompt, model=model)

    def analyze_function_batch(
        self,
        prompt: str,
        *,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> Any:
        with self._activity.track(
            "batch", self._model_name(model), len(prompt), max_tokens
        ):
            return self._client.analyze_function_batch(
                prompt, model=model, max_tokens=max_tokens
            )

    def analyze_with_tools(self, prompt: str, *args: Any, **kwargs: Any) -> Any:
        with self._activity.track(
            "tools",
            self._model_name(kwargs.get("model")),
            len(prompt),
            getattr(self._client, "max_tokens", None),
        ):
            return self._client.analyze_with_tools(prompt, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        # Guarded so that an attribute missing during __init__ raises rather
        # than recursing through _client, which is not set yet.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._client, name)
