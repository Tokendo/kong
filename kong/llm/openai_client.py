"""OpenAI SDK wrapper for function analysis.

Implements the LLMClient protocol expected by the Analyzer.
Tracks token usage and cost per call.  Supports both simple
(single-shot JSON) and tool-use (agentic loop) interactions.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx
import openai

from kong.agent.analyzer import Analyzer, LLMResponse
from kong.agent.prompts import BATCH_OUTPUT_SCHEMA, BATCH_SYSTEM_PROMPT, OUTPUT_SCHEMA, SYSTEM_PROMPT
from kong.llm.tools import ToolExecutor
from kong.llm.truncation import MAX_TOKENS_CAP, call_with_budget
from kong.llm.usage import TokenUsage

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4o"

# Output budget for a batch call when the caller does not set one.
DEFAULT_BATCH_MAX_TOKENS = 16384

#: Ceiling on any single request, in seconds. The SDK's own default is 600s,
#: which it applies to every attempt and multiplies by its retry count: a call
#: an endpoint could never answer used to sit there for an hour before raising
#: APITimeoutError, and a long run could spend a third of its wall clock that
#: way. Set it from the slowest answer the endpoint is expected to produce.
DEFAULT_TIMEOUT_SECONDS = 600.0

#: A connection that has not opened by now is not going to.
CONNECT_TIMEOUT_SECONDS = 10.0

#: Completed calls to time before the measured output rate is trusted. Below
#: this the client makes no assumption about the endpoint's speed and lets the
#: truncation retry grow as far as it likes.
MIN_RATE_SAMPLES = 3

#: Attempts the SDK makes on a connection error or a timeout. A timeout is
#: usually a budget the endpoint cannot deliver in time, which reproduces
#: exactly on a retry and is paid for in full each round, so keep the
#: multiplier small — the SDK's default of 2 with Kong's old 5 meant six
#: full-length waits before anything was reported.
DEFAULT_MAX_RETRIES = 2


def _answered_nothing(response: Any) -> bool:
    """True when the completion budget ran out before the model said anything.

    Reasoning models charge their chain-of-thought to the same budget as the
    answer, so this arrives as a successful response with empty content. Tool
    calls count as an answer: a round that only calls a tool is not truncated.
    """
    choice = response.choices[0]
    if choice.message.content or choice.message.tool_calls:
        return False
    return choice.finish_reason == "length"


def _convert_tools_to_openai(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Anthropic-style tool schemas to OpenAI function-calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"],
            },
        }
        for tool in tools
    ]


class OpenAIClient:
    """Concrete LLM client using the OpenAI SDK.

    Satisfies the LLMClient protocol from kong.agent.analyzer.

    Usage::

        client = OpenAIClient()
        response = client.analyze_function(prompt)
        response = client.analyze_with_tools(prompt, system, tools, executor)
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 2048,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout if timeout is not None else DEFAULT_TIMEOUT_SECONDS
        self._client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=max_retries,
            timeout=httpx.Timeout(self.timeout, connect=CONNECT_TIMEOUT_SECONDS),
        )
        self.usage = TokenUsage()
        #: Output tokens generated and seconds spent generating them, summed
        #: over completed calls. See `_budget_cap`.
        self._tokens_generated = 0
        self._generation_seconds = 0.0
        self._timed_calls = 0

    def _observe(self, response: Any, seconds: float) -> None:
        """Record how fast the endpoint generated one completion."""
        generated = getattr(response.usage, "completion_tokens", 0) or 0
        if generated <= 0 or seconds <= 0:
            return
        self._tokens_generated += generated
        self._generation_seconds += seconds
        self._timed_calls += 1

    def _budget_cap(self) -> int:
        """The largest output budget this endpoint can deliver before the deadline.

        Kong does not stream, so a completion returns nothing at all until it
        is finished: a budget the endpoint cannot generate within `timeout`
        cannot come back, however many times it is asked for. Retrying a
        truncated call at such a budget spends the whole deadline to arrive at
        the same empty answer, which is how a single unanswerable function
        used to cost an hour.

        Measured rather than assumed: hosted endpoints and a local server on a
        laptop are two orders of magnitude apart, and only the endpoint on the
        day knows which this is. Until there are enough samples the cap is the
        module ceiling, i.e. no cap at all.
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
            response = self._client.chat.completions.create(
                model=effective_model,
                max_tokens=budget,
                timeout=self.timeout,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": f"{SYSTEM_PROMPT}\n\n{OUTPUT_SCHEMA}"},
                    {"role": "user", "content": prompt},
                ],
            )
            self._record_usage(response, effective_model)
            self._observe(response, time.monotonic() - started)
            return response

        response = call_with_budget(
            send,
            budget=self.max_tokens,
            is_truncated=_answered_nothing,
            label=f"{effective_model} function analysis",
            max_budget=self._budget_cap(),
        )

        raw_text = response.choices[0].message.content or ""
        result = Analyzer.parse_llm_json(raw_text)
        result.input_tokens = response.usage.prompt_tokens
        result.output_tokens = response.usage.completion_tokens
        result.raw = raw_text
        return result

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
            response = self._client.chat.completions.create(
                model=effective_model,
                max_tokens=budget,
                timeout=self.timeout,
                messages=[
                    {"role": "system", "content": f"{BATCH_SYSTEM_PROMPT}\n\n{BATCH_OUTPUT_SCHEMA}"},
                    {"role": "user", "content": prompt},
                ],
            )
            self._record_usage(response, effective_model)
            self._observe(response, time.monotonic() - started)
            return response

        # A truncated batch costs every function in the chunk, not one.
        response = call_with_budget(
            send,
            budget=max_tokens or DEFAULT_BATCH_MAX_TOKENS,
            is_truncated=_answered_nothing,
            label=f"{effective_model} chunk analysis",
            max_budget=self._budget_cap(),
        )

        raw_text = response.choices[0].message.content or ""
        responses = Analyzer.parse_llm_json_batch(raw_text)
        for resp in responses:
            resp.input_tokens = response.usage.prompt_tokens
            resp.output_tokens = response.usage.completion_tokens
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
        tool_calls, executes each tool via *tool_executor* and feeds results
        back.  Repeats until the model returns a final text response or
        *max_rounds* is exhausted.
        """
        openai_tools = _convert_tools_to_openai(tools)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": f"{system}\n\n{OUTPUT_SCHEMA}"},
            {"role": "user", "content": prompt},
        ]

        total_input = 0
        total_output = 0

        last_text = ""

        def send(budget: int) -> Any:
            nonlocal total_input, total_output
            started = time.monotonic()
            response = self._client.chat.completions.create(
                model=self.model,
                max_tokens=budget,
                timeout=self.timeout,
                tools=openai_tools,
                messages=messages,
            )
            total_input += response.usage.prompt_tokens
            total_output += response.usage.completion_tokens
            self._record_usage(response, self.model)
            self._observe(response, time.monotonic() - started)
            return response

        for _ in range(max_rounds):
            response = call_with_budget(
                send,
                budget=self.max_tokens,
                is_truncated=_answered_nothing,
                label=f"{self.model} tool round",
                max_budget=self._budget_cap(),
            )

            choice = response.choices[0]
            last_text = choice.message.content or ""
            tool_calls = choice.message.tool_calls or []

            # Some OpenAI-compatible servers return tool calls without setting
            # finish_reason, so the calls themselves decide, not the reason.
            if not tool_calls:
                result = Analyzer.parse_llm_json(last_text)
                result.input_tokens = total_input
                result.output_tokens = total_output
                result.raw = last_text
                return result

            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": choice.message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }
            messages.append(assistant_msg)

            for tool_call in tool_calls:
                arguments = json.loads(tool_call.function.arguments)
                result_str = tool_executor.execute(tool_call.function.name, arguments)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_str,
                })

        result = Analyzer.parse_llm_json(last_text)
        result.input_tokens = total_input
        result.output_tokens = total_output
        result.raw = last_text
        return result

    @property
    def total_cost_usd(self) -> float:
        return self.usage.total_cost_usd

    def _record_usage(self, response: Any, model: str | None = None) -> None:
        effective_model = model or self.model
        usage = response.usage
        mu = self.usage._get(effective_model)
        mu.input_tokens += usage.prompt_tokens
        mu.output_tokens += usage.completion_tokens
        cached = getattr(
            getattr(usage, "prompt_tokens_details", None),
            "cached_tokens",
            0,
        ) or 0
        mu.cache_read_tokens += cached
        mu.calls += 1
        logger.debug(
            "LLM [%s]: %d in / %d out / %d cached tokens "
            "(total: %d calls, $%.4f)",
            effective_model,
            usage.prompt_tokens,
            usage.completion_tokens,
            cached,
            self.usage.calls,
            self.usage.total_cost_usd,
        )
