"""The obfuscation decision is taken over a binary, not one function at a time.

The per-function heuristics read structure, and structure does not separate
obfuscated code from ordinary code shaped like it. On a clean 1998 game binary
they matched the CRT's printf and scanf engines, `memmove` and the float-to-
string converter, and the agentic loop they routed to spent three of a six-hour
run's hours on those four functions — one of which timed out after 1h48 and
produced nothing.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from kong.agent.deobfuscator import (
    BINARY_OBFUSCATION_THRESHOLD,
    DEOBFUSCATION_TIME_BUDGET,
    Deobfuscator,
    ObfuscationType,
    classify_obfuscation,
    is_known_library_code,
    obfuscation_verdict,
)

# What the CRT's format engines look like: a dispatch loop with many cases.
CRT_FORMAT_ENGINE = (
    "void _output(void)\n{\n  while (true) {\n    switch(state) {\n"
    + "".join(f"    case 0x{case:x}:\n      state = {case + 1};\n      break;\n"
             for case in range(1, 25))
    + "    }\n  }\n}\n"
    + "  /* padding */\n" * 220
)


class TestTheHeuristicsStillReportShape:
    """The per-function classifier is unchanged; only what is done with it is."""

    def test_a_dispatch_loop_still_trips_it(self):
        assert ObfuscationType.CONTROL_FLOW_FLATTENING in classify_obfuscation(
            CRT_FORMAT_ENGINE
        )

    def test_a_large_dispatch_loop_still_reads_as_vmprotect(self):
        assert ObfuscationType.VM_PROTECTION in classify_obfuscation(CRT_FORMAT_ENGINE)

    def test_plain_code_trips_nothing(self):
        assert classify_obfuscation("int add(int a, int b) { return a + b; }") == []


class TestBinaryVerdict:
    def test_a_handful_in_a_thousand_is_not_a_protected_binary(self):
        verdict = obfuscation_verdict(flagged=6, total=1351)

        assert not verdict.believed
        assert "false positives" in verdict.describe()

    def test_a_protected_binary_is_believed(self):
        verdict = obfuscation_verdict(flagged=900, total=1000)

        assert verdict.believed
        assert "deobfuscating" in verdict.describe()

    def test_a_small_binary_that_is_entirely_protected_is_believed(self):
        """A share test alone would dismiss a 40-function protected binary."""
        verdict = obfuscation_verdict(flagged=30, total=40)

        assert verdict.believed

    def test_a_threshold_of_zero_believes_everything(self):
        assert obfuscation_verdict(flagged=1, total=5000, threshold=0.0).believed

    def test_nothing_flagged_is_not_a_verdict_to_act_on(self):
        assert not obfuscation_verdict(flagged=0, total=1000).believed

    def test_an_empty_binary_decides_nothing(self):
        assert not obfuscation_verdict(flagged=0, total=0).believed

    def test_the_share_is_reported_for_the_log(self):
        assert obfuscation_verdict(flagged=5, total=100).share == pytest.approx(0.05)

    def test_the_default_threshold_clears_the_fa18_case(self):
        """6 of 1351 was the real measurement; it must not be believed."""
        assert 6 / 1351 < BINARY_OBFUSCATION_THRESHOLD


class TestLibraryGate:
    def test_a_signature_match_is_known_library_code(self):
        assert is_known_library_code(0x45F2C0, {0x45F2C0, 0x401000})

    def test_anything_else_is_not(self):
        assert not is_known_library_code(0x401234, {0x45F2C0})

    def test_no_matches_means_nothing_is_excused(self):
        assert not is_known_library_code(0x401234, frozenset())


class _SlowToolLLM:
    """Answers tool calls forever, taking `seconds` of clock per round."""

    def __init__(self, clock, seconds=400.0):
        self.clock = clock
        self.seconds = seconds
        self.rounds = 0

    def analyze_with_tools(self, prompt, system, tools, tool_executor,
                           max_rounds=10, max_seconds=None):
        from kong.agent.analyzer import LLMResponse
        started = self.clock()
        for round_number in range(max_rounds):
            if max_seconds is not None and round_number > 0:
                if self.clock() - started >= max_seconds:
                    break
            self.rounds += 1
            self.clock.advance(self.seconds)
        return LLMResponse(name="whatever_it_had")


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class TestTimeBudget:
    """Ten rounds against a slow endpoint is hours on one function."""

    def _context(self):
        from kong.agent.analyzer import AnalysisContext
        from kong.ghidra.types import BinaryInfo, FunctionClassification, FunctionInfo

        return AnalysisContext(
            function=FunctionInfo(
                address=0x45F2C0, name="FUN_0045f2c0", size=900,
                classification=FunctionClassification.LARGE,
            ),
            decompilation=CRT_FORMAT_ENGINE,
            binary_info=BinaryInfo(
                arch="x86", format="PE", endianness="little",
                word_size=4, compiler="windows", name="FA18.exe",
            ),
        )

    def test_the_loop_stops_at_its_budget(self):
        clock = _Clock()
        llm = _SlowToolLLM(clock, seconds=400.0)

        Deobfuscator(MagicMock(), llm).deobfuscate(
            self._context(), [ObfuscationType.VM_PROTECTION], max_seconds=900.0,
        )

        # 3 rounds x 400s crosses 900s; a tenth round would be 4000s.
        assert llm.rounds == 3

    def test_no_budget_runs_every_round(self):
        clock = _Clock()
        llm = _SlowToolLLM(clock, seconds=400.0)

        Deobfuscator(MagicMock(), llm).deobfuscate(
            self._context(), [ObfuscationType.VM_PROTECTION], max_seconds=None,
        )

        assert llm.rounds == 10

    def test_a_fast_endpoint_is_not_cut_short(self):
        clock = _Clock()
        llm = _SlowToolLLM(clock, seconds=1.0)

        Deobfuscator(MagicMock(), llm).deobfuscate(
            self._context(), [ObfuscationType.VM_PROTECTION], max_seconds=900.0,
        )

        assert llm.rounds == 10

    def test_there_is_a_budget_by_default(self):
        assert DEOBFUSCATION_TIME_BUDGET is not None


class TestTheClientLoopHonoursTheBudget:
    """The budget has to bite in the client, not only in the call that sets it."""

    def _client_with_tool_rounds(self, mock_openai_cls):
        """An endpoint that always asks for another tool call, never answering."""
        from kong.llm.openai_client import OpenAIClient

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        def a_tool_call(*args, **kwargs):
            call = MagicMock()
            call.id = "call_1"
            call.function.name = "read_memory"
            call.function.arguments = "{}"
            message = MagicMock()
            message.content = ""
            message.tool_calls = [call]
            choice = MagicMock()
            choice.message = message
            choice.finish_reason = "tool_calls"
            usage = MagicMock()
            usage.prompt_tokens = 10
            usage.completion_tokens = 10
            usage.prompt_tokens_details = None
            response = MagicMock()
            response.choices = [choice]
            response.usage = usage
            return response

        mock_client.chat.completions.create.side_effect = a_tool_call
        return mock_client, OpenAIClient(api_key="k")

    @patch("kong.llm.openai_client.openai.OpenAI")
    def test_it_stops_between_rounds_once_the_budget_is_spent(
        self, mock_openai_cls, monkeypatch,
    ):
        mock_client, client = self._client_with_tool_rounds(mock_openai_cls)
        clock = _Clock()
        monkeypatch.setattr("kong.llm.openai_client.time.monotonic", clock)
        original = mock_client.chat.completions.create.side_effect

        def slow(*args, **kwargs):
            clock.advance(400.0)
            return original(*args, **kwargs)

        mock_client.chat.completions.create.side_effect = slow
        executor = MagicMock()
        executor.execute.return_value = "0x00"

        client.analyze_with_tools("p", "s", [], executor, max_seconds=900.0)

        assert mock_client.chat.completions.create.call_count == 3

    @patch("kong.llm.openai_client.openai.OpenAI")
    def test_without_a_budget_it_runs_every_round(
        self, mock_openai_cls, monkeypatch,
    ):
        mock_client, client = self._client_with_tool_rounds(mock_openai_cls)
        clock = _Clock()
        monkeypatch.setattr("kong.llm.openai_client.time.monotonic", clock)
        original = mock_client.chat.completions.create.side_effect

        def slow(*args, **kwargs):
            clock.advance(4000.0)
            return original(*args, **kwargs)

        mock_client.chat.completions.create.side_effect = slow
        executor = MagicMock()
        executor.execute.return_value = "0x00"

        client.analyze_with_tools("p", "s", [], executor, max_rounds=4)

        assert mock_client.chat.completions.create.call_count == 4
