"""Tests for retrying a response that spent its budget without answering."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

from kong.llm.client import AnthropicClient
from kong.llm.openai_client import OpenAIClient
from kong.llm.truncation import MAX_TOKENS_CAP, call_with_budget, next_budget


class TestNextBudget:
    def test_doubles(self):
        assert next_budget(2048) == 4096

    def test_stops_at_the_cap(self):
        assert next_budget(MAX_TOKENS_CAP // 2 + 1) == MAX_TOKENS_CAP

    def test_no_room_left_at_the_cap(self):
        assert next_budget(MAX_TOKENS_CAP) is None


class TestCallWithBudget:
    def test_an_answer_on_the_first_try_is_not_retried(self):
        budgets = []

        result = call_with_budget(
            lambda b: budgets.append(b) or "answer",
            budget=2048,
            is_truncated=lambda r: r != "answer",
            label="test",
        )

        assert result == "answer"
        assert budgets == [2048]

    def test_a_truncated_answer_is_retried_at_double(self):
        budgets = []

        def send(budget):
            budgets.append(budget)
            return "" if len(budgets) == 1 else "answer"

        result = call_with_budget(
            send, budget=2048, is_truncated=lambda r: r == "", label="test",
        )

        assert result == "answer"
        assert budgets == [2048, 4096]

    def test_it_retries_once_and_returns_what_it_got(self):
        """A truncated attempt is paid for, so it does not double forever."""
        budgets = []

        result = call_with_budget(
            lambda b: budgets.append(b) or "",
            budget=2048,
            is_truncated=lambda r: r == "",
            label="test",
        )

        assert result == ""
        assert budgets == [2048, 4096]

    def test_more_attempts_can_be_asked_for(self):
        budgets = []

        call_with_budget(
            lambda b: budgets.append(b) or "",
            budget=1024,
            is_truncated=lambda r: r == "",
            label="test",
            attempts=3,
        )

        assert budgets == [1024, 2048, 4096]

    def test_a_budget_at_the_cap_is_not_retried(self):
        budgets = []

        call_with_budget(
            lambda b: budgets.append(b) or "",
            budget=MAX_TOKENS_CAP,
            is_truncated=lambda r: r == "",
            label="test",
        )

        assert budgets == [MAX_TOKENS_CAP]

    def test_the_retry_is_logged(self, caplog):
        with caplog.at_level(logging.WARNING, logger="kong.llm.truncation"):
            call_with_budget(
                lambda b: "",
                budget=2048,
                is_truncated=lambda r: r == "",
                label="chunk analysis",
            )

        assert "chunk analysis" in caplog.text
        assert "retrying at 4096" in caplog.text
        assert "giving up" in caplog.text


def _openai_response(text, finish_reason="stop", tool_calls=None):
    message = MagicMock()
    message.content = text
    message.tool_calls = tool_calls

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason

    usage = MagicMock()
    usage.prompt_tokens = 100
    usage.completion_tokens = 50
    usage.prompt_tokens_details = None

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


def _anthropic_message(text, stop_reason="end_turn"):
    block = MagicMock()
    block.type = "text"
    block.text = text

    usage = MagicMock()
    usage.input_tokens = 100
    usage.output_tokens = 50
    usage.cache_creation_input_tokens = 0
    usage.cache_read_input_tokens = 0

    message = MagicMock()
    message.content = [block] if text else []
    message.usage = usage
    message.stop_reason = stop_reason
    return message


BATCH_JSON = '[{"address": "0x401000", "name": "parse_header", "confidence": 88}]'


class TestOpenAIChunkRetry:
    """The chunk path matters most: a truncated batch costs every function."""

    @patch("kong.llm.openai_client.openai.OpenAI")
    def test_an_empty_truncated_chunk_is_retried_at_double(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = [
            _openai_response("", finish_reason="length"),
            _openai_response(BATCH_JSON),
        ]

        client = OpenAIClient(api_key="k")
        responses = client.analyze_function_batch("prompt", max_tokens=16384)

        assert [r.name for r in responses] == ["parse_header"]
        calls = mock_client.chat.completions.create.call_args_list
        assert [c.kwargs["max_tokens"] for c in calls] == [16384, 32768]

    @patch("kong.llm.openai_client.openai.OpenAI")
    def test_the_truncated_attempt_is_still_billed(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = [
            _openai_response("", finish_reason="length"),
            _openai_response(BATCH_JSON),
        ]

        client = OpenAIClient(api_key="k")
        client.analyze_function_batch("prompt")

        assert client.usage.calls == 2

    @patch("kong.llm.openai_client.openai.OpenAI")
    def test_a_truncated_chunk_that_answered_is_not_retried(self, mock_openai_cls):
        """Content plus finish_reason=length means a cut-off answer, not none."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _openai_response(
            BATCH_JSON, finish_reason="length",
        )

        client = OpenAIClient(api_key="k")
        client.analyze_function_batch("prompt")

        assert mock_client.chat.completions.create.call_count == 1

    @patch("kong.llm.openai_client.openai.OpenAI")
    def test_an_ordinary_empty_answer_is_not_retried(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _openai_response("")

        client = OpenAIClient(api_key="k")
        client.analyze_function_batch("prompt")

        assert mock_client.chat.completions.create.call_count == 1


class TestAnthropicChunkRetry:
    @patch("kong.llm.client.anthropic.Anthropic")
    def test_an_empty_truncated_chunk_is_retried_at_double(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.side_effect = [
            _anthropic_message("", stop_reason="max_tokens"),
            _anthropic_message(BATCH_JSON),
        ]

        client = AnthropicClient(api_key="k")
        responses = client.analyze_function_batch("prompt", max_tokens=8192)

        assert [r.name for r in responses] == ["parse_header"]
        calls = mock_client.messages.create.call_args_list
        assert [c.kwargs["max_tokens"] for c in calls] == [8192, 16384]

    @patch("kong.llm.client.anthropic.Anthropic")
    def test_a_tool_use_stop_is_not_a_truncation(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _anthropic_message(
            "", stop_reason="tool_use",
        )

        client = AnthropicClient(api_key="k")
        client.analyze_function_batch("prompt")

        assert mock_client.messages.create.call_count == 1


class TestToolCallsWithoutFinishReason:
    """Some OpenAI-compatible servers omit finish_reason on a tool call."""

    @patch("kong.llm.openai_client.openai.OpenAI")
    def test_the_tool_is_executed_anyway(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.function.name = "read_memory"
        tool_call.function.arguments = '{"address": "0x401020"}'

        mock_client.chat.completions.create.side_effect = [
            _openai_response(None, finish_reason="stop", tool_calls=[tool_call]),
            _openai_response('{"name": "decoded", "confidence": 70}'),
        ]
        executor = MagicMock()
        executor.execute.return_value = "de ad be ef"

        client = OpenAIClient(api_key="k")
        result = client.analyze_with_tools(
            "prompt", "system", [
                {"name": "read_memory", "description": "d", "input_schema": {}},
            ], executor,
        )

        executor.execute.assert_called_once_with("read_memory", {"address": "0x401020"})
        assert result.name == "decoded"

    @patch("kong.llm.openai_client.openai.OpenAI")
    def test_no_tool_calls_still_ends_the_loop(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _openai_response(
            '{"name": "done", "confidence": 60}', finish_reason="tool_calls",
        )

        client = OpenAIClient(api_key="k")
        result = client.analyze_with_tools("prompt", "system", [], MagicMock())

        assert result.name == "done"
        assert mock_client.chat.completions.create.call_count == 1


class TestBudgetCap:
    """A budget the endpoint cannot generate before the deadline is not asked for.

    Kong does not stream, so a completion comes back only once it is finished.
    Retrying a truncated call at a budget the endpoint has no time to produce
    spends the whole deadline and returns just as empty — the failure that cost
    the FA18 run two hours across two functions.
    """

    def test_no_cap_before_the_endpoint_has_been_measured(self):
        client = OpenAIClient(api_key="k")
        assert client._budget_cap() == MAX_TOKENS_CAP

    def test_a_slow_endpoint_caps_the_retry(self):
        client = OpenAIClient(api_key="k", timeout=600.0)
        # 20 tokens/second, the rate the FA18 run actually saw.
        for _ in range(5):
            client._observe(_openai_response("x"), 50 / 20)

        assert client._budget_cap() == 12000
        assert next_budget(32000, client._budget_cap()) is None

    def test_a_fast_endpoint_does_not_cap_the_retry(self):
        client = OpenAIClient(api_key="k", timeout=600.0)
        for _ in range(5):
            client._observe(_openai_response("x"), 50 / 500)

        assert next_budget(16384, client._budget_cap()) == 32768

    def test_a_longer_deadline_buys_a_bigger_budget(self):
        slow = OpenAIClient(api_key="k", timeout=600.0)
        patient = OpenAIClient(api_key="k", timeout=3600.0)
        for client in (slow, patient):
            for _ in range(5):
                client._observe(_openai_response("x"), 50 / 20)

        assert patient._budget_cap() == 6 * slow._budget_cap()

    def test_an_untimed_call_does_not_count_as_a_sample(self):
        client = OpenAIClient(api_key="k")
        for _ in range(5):
            client._observe(_openai_response("x"), 0.0)

        assert client._budget_cap() == MAX_TOKENS_CAP

    def test_the_capped_retry_is_skipped_rather_than_sent(self, caplog):
        sent = []

        with caplog.at_level(logging.WARNING, logger="kong.llm.truncation"):
            call_with_budget(
                lambda b: sent.append(b) or "",
                budget=32000,
                is_truncated=lambda r: r == "",
                label="slow-model function analysis",
                max_budget=12000,
            )

        assert sent == [32000]
        assert "not retrying" in caplog.text

    def test_the_anthropic_client_caps_the_same_way(self):
        client = AnthropicClient(api_key="k", timeout=600.0)
        assert client._budget_cap() == MAX_TOKENS_CAP
        for _ in range(5):
            client._observe(_anthropic_message("x"), 50 / 20)

        assert client._budget_cap() == 12000


class TestRequestDeadline:
    """The deadline is explicit, and bounded by a small retry count.

    Left to the SDK a request waits 600s per attempt and Kong asked for five
    retries on top, so a request that could never be answered was not reported
    for an hour.
    """

    @patch("kong.llm.openai_client.openai.OpenAI")
    def test_openai_gets_an_explicit_deadline_and_few_retries(self, mock_openai_cls):
        OpenAIClient(api_key="k", timeout=120.0, max_retries=1)

        kwargs = mock_openai_cls.call_args.kwargs
        assert kwargs["max_retries"] == 1
        assert kwargs["timeout"].read == 120.0
        assert kwargs["timeout"].connect == 10.0

    @patch("kong.llm.client.anthropic.Anthropic")
    def test_anthropic_gets_an_explicit_deadline_and_few_retries(self, mock_cls):
        AnthropicClient(api_key="k", timeout=120.0, max_retries=1)

        kwargs = mock_cls.call_args.kwargs
        assert kwargs["max_retries"] == 1
        assert kwargs["timeout"].read == 120.0

    @patch("kong.llm.openai_client.openai.OpenAI")
    def test_every_request_carries_the_deadline(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _openai_response(
            '{"name": "f", "confidence": 90}'
        )

        client = OpenAIClient(api_key="k", timeout=42.0)
        client.analyze_function("prompt")

        assert mock_client.chat.completions.create.call_args.kwargs["timeout"] == 42.0
