"""Tests for the in-flight LLM request tracker."""

from __future__ import annotations

import threading
import time

import pytest

from kong.llm.activity import LLMActivity, TrackedLLMClient


class _Client:
    """A stand-in LLM client that can be held mid-call."""

    model = "claude-opus-5"
    max_tokens = 2048

    def __init__(self, hold: threading.Event | None = None) -> None:
        self.hold = hold
        self.usage = "the usage object"
        self.calls: list[tuple[str, str | None]] = []

    def analyze_function(self, prompt, *, model=None):
        self.calls.append(("function", model))
        if self.hold is not None:
            self.hold.wait(timeout=5)
        return "one answer"

    def analyze_function_batch(self, prompt, *, model=None, max_tokens=None):
        self.calls.append(("batch", model))
        if self.hold is not None:
            self.hold.wait(timeout=5)
        return ["answers"]

    def analyze_with_tools(self, prompt, system, tools, executor, max_rounds=10):
        self.calls.append(("tools", None))
        return "tool answer"

    @property
    def total_cost_usd(self) -> float:
        return 2.5


class TestLLMActivity:
    def test_nothing_in_flight_reads_as_idle(self):
        snapshot = LLMActivity().snapshot()
        assert snapshot.waiting is False
        assert snapshot.in_flight == 0
        assert snapshot.label == ""

    def test_a_call_in_flight_is_named(self):
        activity = LLMActivity()
        activity.begin("batch", "glm-5.3", prompt_chars=82_000, max_tokens=32768)

        snapshot = activity.snapshot()
        assert snapshot.waiting is True
        assert snapshot.in_flight == 1
        assert "glm-5.3" in snapshot.label
        assert "batch of functions" in snapshot.label
        assert "82k chars" in snapshot.label
        assert "32768 token budget" in snapshot.label

    def test_the_oldest_call_is_the_one_reported(self):
        activity = LLMActivity()
        first = activity.begin("batch", "first-model")
        time.sleep(0.01)
        activity.begin("function", "second-model")

        snapshot = activity.snapshot()
        assert snapshot.in_flight == 2
        assert snapshot.model == "first-model"
        assert snapshot.calls[0].id == first.id

    def test_an_answered_call_stops_being_waited_on(self):
        activity = LLMActivity()
        call = activity.begin("function", "m")
        activity.end(call)

        snapshot = activity.snapshot()
        assert snapshot.waiting is False
        assert snapshot.completed == 1
        assert snapshot.total_wait_seconds >= 0

    def test_ending_twice_counts_once(self):
        activity = LLMActivity()
        call = activity.begin("function", "m")
        activity.end(call)
        activity.end(call)

        assert activity.snapshot().completed == 1

    def test_a_failed_call_is_not_left_in_flight(self):
        """An exception is the end of the waiting, not a reason to pin it on."""
        activity = LLMActivity()
        with pytest.raises(RuntimeError):
            with activity.track("function", "m"):
                raise RuntimeError("the endpoint hung up")

        assert activity.snapshot().waiting is False


class TestTrackedLLMClient:
    def test_the_call_is_visible_while_it_is_in_flight(self):
        hold = threading.Event()
        activity = LLMActivity()
        client = TrackedLLMClient(_Client(hold), activity)

        worker = threading.Thread(
            target=lambda: client.analyze_function_batch("x" * 4000, model="glm-5.3")
        )
        worker.start()
        try:
            deadline = time.time() + 5
            while time.time() < deadline and not activity.snapshot().waiting:
                time.sleep(0.01)
            snapshot = activity.snapshot()
            assert snapshot.waiting is True
            assert snapshot.kind == "batch"
            assert snapshot.model == "glm-5.3"
        finally:
            hold.set()
            worker.join(timeout=5)

        assert activity.snapshot().waiting is False
        assert activity.snapshot().completed == 1

    def test_the_answer_is_handed_back_untouched(self):
        client = TrackedLLMClient(_Client(), LLMActivity())

        assert client.analyze_function("p") == "one answer"
        assert client.analyze_function_batch("p", max_tokens=99) == ["answers"]
        assert client.analyze_with_tools("p", "sys", [], None) == "tool answer"

    def test_everything_else_reaches_the_client_underneath(self):
        wrapped = _Client()
        client = TrackedLLMClient(wrapped, LLMActivity())

        assert client.model == "claude-opus-5"
        assert client.usage == "the usage object"
        assert client.total_cost_usd == 2.5
        assert client.wrapped is wrapped

    def test_the_model_of_the_call_wins_over_the_client_default(self):
        activity = LLMActivity()
        client = TrackedLLMClient(_Client(), activity)
        seen: list[str] = []

        original = activity.begin

        def record(kind, model, *args, **kwargs):
            seen.append(model)
            return original(kind, model, *args, **kwargs)

        activity.begin = record  # type: ignore[method-assign]
        client.analyze_function("p", model="claude-haiku-4-5")
        client.analyze_function("p")

        assert seen == ["claude-haiku-4-5", "claude-opus-5"]

    def test_a_missing_attribute_still_raises(self):
        client = TrackedLLMClient(_Client(), LLMActivity())

        with pytest.raises(AttributeError):
            client.no_such_method
