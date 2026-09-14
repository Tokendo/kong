"""Anthropic SDK wrapper for function analysis.

Implements the LLMClient protocol expected by the Analyzer.
Tracks token usage and cost per call.  Supports both simple
(single-shot JSON) and tool-use (agentic loop) interactions.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import anthropic
import httpx

from kong.agent.analyzer import Analyzer, LLMResponse
from kong.agent.prompts import BATCH_OUTPUT_SCHEMA, BATCH_SYSTEM_PROMPT, OUTPUT_SCHEMA, SYSTEM_PROMPT
from kong.llm.tools import ToolExecutor
from kong.llm.truncation import MAX_TOKENS_CAP, call_with_budget
from kong.llm.usage import TokenUsage

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5"

# Output budget for a batch call when the caller does not set one.
DEFAULT_BATCH_MAX_TOKENS = 16384

#: Ceiling on any single request, in seconds. Left to itself the SDK applies
#: its own default to every attempt and multiplies it by the retry count, so a
#: request the endpoint never answers is not reported for the best part of an
#: hour. See kong.llm.openai_client for the same constants on the other side.
DEFAULT_TIMEOUT_SECONDS = 600.0

#: A connection that has not opened by now is not going to.
CONNECT_TIMEOUT_SECONDS = 10.0

#: Completed calls to time before the measured output rate is trusted. Below
#: this the client makes no assumption about the endpoint's speed.
MIN_RATE_SAMPLES = 3

#: Attempts the SDK makes on a connection error or a timeout. A timeout is
#: usually a budget the endpoint cannot deliver in time: it reproduces on the
#: retry and is paid for in full each round, so keep the multiplier small.
DEFAULT_MAX_RETRIES = 2


def _extract_text(message: Any) -> str:
    return "".join(block.text for block in message.content if block.type == "text")


def _answered_nothing(message: Any) -> bool:
    """True when the output budget ran out before the model said anything.

    Thinking tokens are charged to the same budget as the answer, so this
    arrives as a successful message with no text in it. A tool call counts as
    an answer: a turn that only calls a tool is not truncated.
    """
    if _extract_text(message) or message.stop_reason == "tool_use":
        return False
    return message.stop_reason == "max_tokens"


class AnthropicClient:
    """Concrete LLM client using the Anthropic SDK.

    Satisfies the LLMClient protocol from kong.agent.analyzer.

    Usage::

        client = AnthropicClient()
        response = client.analyze_function(prompt)
        response = client.analyze_with_tools(prompt, system, tools, executor)
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 2048,
        api_key: str | None = None,
        timeout: float | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout if timeout is not None else DEFAULT_TIMEOUT_SECONDS
        self._client = anthropic.Anthropic(
            api_key=api_key,
            max_retries=max_retries,
            timeout=httpx.Timeout(self.timeout, connect=CONNECT_TIMEOUT_SECONDS),
        )
        self.usage = TokenUsage()
        #: Output tokens generated and seconds spent generating them, summed
        #: over completed calls. See `_budget_cap`.
        self._tokens_generated = 0
        self._generation_seconds = 0.0
        self._timed_calls = 0

    def _observe(self, message: Any, seconds: float) -> None:
        """Record how fast the endpoint generated one message."""
        generated = getattr(message.usage, "output_tokens", 0) or 0
        if generated <= 0 or seconds <= 0:
            return
        self._tokens_generated += generated
        self._generation_seconds += seconds
        self._timed_calls += 1

    def _budget_cap(self) -> int:
        """The largest output budget this endpoint can deliver before the deadline.

        Kong does not stream, so a message returns nothing until it is
        finished: a budget the endpoint cannot generate within `timeout`
        cannot come back, however often it is asked for, and retrying a
        truncated call at one spends the whole deadline to arrive just as
        empty. Measured rather than assumed — see the OpenAI client for the
        reasoning in full. Until there are enough samples this is the module
        ceiling, i.e. no cap at all.
        """
        if self._timed_calls < MIN_RATE_SAMPLES or self._generation_seconds <= 0:
            return MAX_TOKENS_CAP
        rate = self._tokens_generated / self._generation_seconds
        return int(rate * self.timeout)

    def analyze_function(self, prompt: str, *, model: str | None = None) -> LLMResponse:
        """Send an analysis prompt and return parsed response (no tools)."""
        effective_model = model or self.model

        def send(budget: int) -> Any:
            started = time.monotonic()
            message = self._client.messages.create(
                model=effective_model,
                max_tokens=budget,
                timeout=self.timeout,
                system=[{
                    "type": "text",
                    "text": f"{SYSTEM_PROMPT}\n\n{OUTPUT_SCHEMA}",
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )
            self._record_usage(message, effective_model)
            self._observe(message, time.monotonic() - started)
            return message

        message = call_with_budget(
            send,
            budget=self.max_tokens,
            is_truncated=_answered_nothing,
            label=f"{effective_model} function analysis",
            max_budget=self._budget_cap(),
        )

        raw_text = self._extract_text(message)
        response = Analyzer.parse_llm_json(raw_text)
        response.input_tokens = message.usage.input_tokens
        response.output_tokens = message.usage.output_tokens
        response.raw = raw_text
        return response

    def analyze_function_batch(
        self,
        prompt: str,
        *,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> list[LLMResponse]:
        """Send a batch analysis prompt and return parsed list of responses."""
        effective_model = model or self.model

        def send(budget: int) -> Any:
            started = time.monotonic()
            message = self._client.messages.create(
                model=effective_model,
                max_tokens=budget,
                timeout=self.timeout,
                system=[{
                    "type": "text",
                    "text": f"{BATCH_SYSTEM_PROMPT}\n\n{BATCH_OUTPUT_SCHEMA}",
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )
            self._record_usage(message, effective_model)
            self._observe(message, time.monotonic() - started)
            return message

        # A truncated batch costs every function in the chunk, not one.
        message = call_with_budget(
            send,
            budget=max_tokens or DEFAULT_BATCH_MAX_TOKENS,
            is_truncated=_answered_nothing,
            label=f"{effective_model} chunk analysis",
            max_budget=self._budget_cap(),
        )

        raw_text = self._extract_text(message)
        responses = Analyzer.parse_llm_json_batch(raw_text)
        for resp in responses:
            resp.input_tokens = message.usage.input_tokens
            resp.output_tokens = message.usage.output_tokens
        return responses

    def analyze_with_tools(
        self,
        prompt: str,
        system: str,
        tools: list[dict[str, Any]],
        tool_executor: ToolExecutor,
        max_rounds: int = 10,
    ) -> LLMResponse:
        """Run an agentic tool-use loop.

        Sends the prompt with tool definitions.  When the model returns
        ``tool_use`` blocks, executes each tool via *tool_executor* and
        feeds results back.  Repeats until the model returns a final text
        response or *max_rounds* is exhausted.
        """
        cached_system = [{
            "type": "text",
            "text": f"{system}\n\n{OUTPUT_SCHEMA}",
            "cache_control": {"type": "ephemeral"},
        }]

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": prompt},
        ]

        total_input = 0
        total_output = 0

        def send(budget: int) -> Any:
            nonlocal total_input, total_output
            started = time.monotonic()
            message = self._client.messages.create(
                model=self.model,
                max_tokens=budget,
                timeout=self.timeout,
                system=cached_system,
                tools=tools,
                messages=messages,
            )
            total_input += message.usage.input_tokens
            total_output += message.usage.output_tokens
            self._record_usage(message, self.model)
            self._observe(message, time.monotonic() - started)
            return message

        for _ in range(max_rounds):
            message = call_with_budget(
                send,
                budget=self.max_tokens,
                is_truncated=_answered_nothing,
                label=f"{self.model} tool round",
                max_budget=self._budget_cap(),
            )

            if message.stop_reason != "tool_use":
                raw_text = self._extract_text(message)
                response = Analyzer.parse_llm_json(raw_text)
                response.input_tokens = total_input
                response.output_tokens = total_output
                response.raw = raw_text
                return response

            messages.append({"role": "assistant", "content": message.content})

            tool_results: list[dict[str, Any]] = []
            for block in message.content:
                if block.type != "tool_use":
                    continue
                result_str = tool_executor.execute(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_str,
                })

            messages.append({"role": "user", "content": tool_results})

        raw_text = self._extract_text(message)
        response = Analyzer.parse_llm_json(raw_text)
        response.input_tokens = total_input
        response.output_tokens = total_output
        response.raw = raw_text
        return response

    @property
    def total_cost_usd(self) -> float:
        return self.usage.total_cost_usd

    def _extract_text(self, message: Any) -> str:
        return _extract_text(message)

    def _record_usage(self, message: Any, model: str | None = None) -> None:
        effective_model = model or self.model
        usage = message.usage
        mu = self.usage._get(effective_model)
        mu.input_tokens += usage.input_tokens
        mu.output_tokens += usage.output_tokens
        mu.cache_creation_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0
        mu.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
        mu.calls += 1
        logger.debug(
            "LLM [%s]: %d in / %d out / %d cache_write / %d cache_read tokens "
            "(total: %d calls, $%.4f)",
            effective_model,
            usage.input_tokens,
            usage.output_tokens,
            getattr(usage, "cache_creation_input_tokens", 0) or 0,
            getattr(usage, "cache_read_input_tokens", 0) or 0,
            self.usage.calls,
            self.usage.total_cost_usd,
        )
