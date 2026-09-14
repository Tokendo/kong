"""Prompt sections that grow with the binary must stay inside the budget.

The batch path is covered in tests/test_agent_supervisor.py; this file covers
the other three prompts a run sends: the per-function pass, the deobfuscation
pass (which shares the same context object) and the synthesis pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from kong.agent.analyzer import Analyzer, LLMResponse
from kong.agent.models import FunctionResult
from kong.agent.queue import WorkItem
from kong.ghidra.types import (
    BinaryInfo,
    FunctionClassification,
    FunctionInfo,
    StructDefinition,
    StructField,
)
from kong.llm.limits import take_within_budget
from kong.synthesis.semantic import SYNTHESIS_FUNCTION_CAP, SemanticSynthesizer

BUDGET = 8000


def _func(addr=0x1000, name="FUN_00001000", size=100):
    return FunctionInfo(
        address=addr, name=name, size=size,
        classification=FunctionClassification.MEDIUM,
    )


def _item(addr=0x1000, name="FUN_00001000"):
    return WorkItem(function=_func(addr, name), callers=[], callees=[])


def _binary_info():
    return BinaryInfo(
        arch="x86-64", format="ELF", endianness="little",
        word_size=8, compiler="GCC",
    )


def _mock_client(decompilation="void f(void) { return; }"):
    client = MagicMock()
    client.get_decompilation.return_value = decompilation
    client.get_function_info.return_value = _func()
    client.get_xrefs_from.return_value = []
    return client


def _known_results(count: int) -> dict[int, FunctionResult]:
    return {
        0x2000 + i: FunctionResult(
            address=0x2000 + i,
            original_name=f"FUN_{0x2000 + i:08x}",
            name=f"resolved_function_number_{i}",
            confidence=90,
        )
        for i in range(count)
    }


def _struct(index: int) -> StructDefinition:
    return StructDefinition(
        name=f"recovered_struct_{index}",
        size=64,
        fields=[
            StructField(name=f"field_{f}", data_type="int", offset=f * 4, size=4)
            for f in range(8)
        ],
    )


class TestTakeWithinBudget:
    def test_keeps_what_fits(self):
        assert take_within_budget(["aaa", "bbb", "ccc"], str, 8) == ["aaa", "bbb"]

    def test_stops_at_the_first_item_that_does_not_fit(self):
        # "aaaa" costs 5 with its newline, so nothing fits in 4.
        assert take_within_budget(["aaaa", "b"], str, 4) == []

    def test_a_zero_budget_keeps_nothing(self):
        assert take_within_budget(["a", "b"], str, 0) == []

    def test_everything_fits_under_a_large_budget(self):
        items = list(range(50))
        assert take_within_budget(items, str, 10_000) == items

    def test_accepts_an_iterator(self):
        assert take_within_budget(iter(["aa", "bb"]), str, 100) == ["aa", "bb"]


class TestAnalyzerPromptBudget:
    def test_known_functions_are_capped(self):
        analyzer = Analyzer(_mock_client(), MagicMock(), max_prompt_chars=BUDGET)

        context = analyzer._build_context(
            _item(), _binary_info(), _known_results(5000), []
        )

        assert len(context.known_functions) < 5000
        rendered = sum(
            len(f"- 0x{addr:08x}: {name}") + 1
            for addr, name in context.known_functions.items()
        )
        assert rendered <= BUDGET // 10

    def test_the_newest_names_are_the_ones_kept(self):
        analyzer = Analyzer(_mock_client(), MagicMock(), max_prompt_chars=BUDGET)

        context = analyzer._build_context(
            _item(), _binary_info(), _known_results(5000), []
        )

        assert "resolved_function_number_4999" in context.known_functions.values()
        assert "resolved_function_number_0" not in context.known_functions.values()

    def test_recovered_structs_are_capped(self):
        analyzer = Analyzer(_mock_client(), MagicMock(), max_prompt_chars=BUDGET)

        context = analyzer._build_context(
            _item(), _binary_info(), {}, [], [_struct(i) for i in range(500)]
        )

        assert 0 < len(context.known_types) < 500

    def test_the_whole_prompt_stays_within_the_budget(self):
        analyzer = Analyzer(_mock_client(), MagicMock(), max_prompt_chars=BUDGET)

        context = analyzer._build_context(
            _item(),
            _binary_info(),
            _known_results(5000),
            [],
            [_struct(i) for i in range(500)],
        )

        assert len(analyzer._build_prompt(context)) <= BUDGET

    def test_a_body_at_the_allowance_still_fits_with_full_sections(self):
        """The cap has to hold for the worst case, not just for small functions."""
        budget = 40000
        allowance = budget - 4 * (budget // 10)
        body = "  ptr[0] = ptr[1] + 3;" * (allowance // 22)
        analyzer = Analyzer(_mock_client(body), MagicMock(), max_prompt_chars=budget)

        context = analyzer._build_context(
            _item(),
            _binary_info(),
            _known_results(5000),
            [],
            [_struct(i) for i in range(500)],
        )

        assert len(context.decompilation) > budget // 2
        assert len(analyzer._build_prompt(context)) <= budget

    def test_the_deobfuscation_prompt_inherits_the_bounded_context(self):
        from kong.agent.deobfuscator import Deobfuscator, ObfuscationType

        analyzer = Analyzer(_mock_client(), MagicMock(), max_prompt_chars=BUDGET)
        context = analyzer._build_context(
            _item(), _binary_info(), _known_results(5000), []
        )

        deobfuscator = Deobfuscator(_mock_client(), MagicMock())
        prompt = deobfuscator._build_prompt(
            context, [ObfuscationType.CONTROL_FLOW_FLATTENING], ""
        )

        assert len(prompt) <= BUDGET

    def test_no_budget_means_no_truncation(self):
        analyzer = Analyzer(_mock_client(), MagicMock())

        context = analyzer._build_context(
            _item(), _binary_info(), _known_results(300), []
        )

        assert len(context.known_functions) == 300

    def test_a_function_over_the_budget_is_not_sent(self):
        llm = MagicMock()
        llm.analyze_function.return_value = LLMResponse(name="whatever")
        client = _mock_client("void huge(void) {}" * 2000)
        analyzer = Analyzer(client, llm, max_prompt_chars=BUDGET)

        result = analyzer.analyze(_item(), _binary_info(), {}, [])

        llm.analyze_function.assert_not_called()
        assert "prompt budget" in result.error
        assert result.name == ""

    def test_a_function_within_the_budget_is_analyzed(self):
        llm = MagicMock()
        llm.analyze_function.return_value = LLMResponse(name="init_module")
        analyzer = Analyzer(_mock_client(), llm, max_prompt_chars=BUDGET)

        result = analyzer.analyze(_item(), _binary_info(), {}, [])

        assert result.name == "init_module"
        assert not result.error


@dataclass
class _FakeResponse:
    raw: str = "{}"
    input_tokens: int = 0
    output_tokens: int = 0


class _RecordingLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    @property
    def last_prompt(self) -> str:
        return self.prompts[-1] if self.prompts else ""

    @property
    def every_prompt(self) -> str:
        """Synthesis sends one call per group, so a claim about "the prompt"
        is now a claim about all of them together."""
        return "\n".join(self.prompts)

    def analyze_function(self, prompt: str, *, model: str | None = None):
        self.prompts.append(prompt)
        return _FakeResponse()


def _synthesis_inputs(count: int, body_chars: int = 2000):
    results = [
        FunctionResult(
            address=0x1000 + i,
            original_name=f"FUN_{0x1000 + i:08x}",
            name=f"analyzed_function_{i}",
            confidence=90,
            classification="utility",
        )
        for i in range(count)
    ]
    decompilations = {
        r.address: f"void analyzed_function(void) {{ DAT_{r.address:08x}; }}"
        + "x" * body_chars
        for r in results
    }
    return results, decompilations


class TestSynthesisPromptBudget:
    def test_prompt_stays_within_the_budget(self):
        llm = _RecordingLLM()
        results, decompilations = _synthesis_inputs(40)
        synthesizer = SemanticSynthesizer(llm, max_prompt_chars=BUDGET)

        synthesizer.synthesize(results, decompilations)

        assert len(llm.last_prompt) <= BUDGET

    def test_some_functions_still_make_it_in(self):
        llm = _RecordingLLM()
        results, decompilations = _synthesis_inputs(40, body_chars=200)
        synthesizer = SemanticSynthesizer(llm, max_prompt_chars=BUDGET)

        synthesizer.synthesize(results, decompilations)

        assert "analyzed_function_" in llm.last_prompt

    def test_a_budget_below_the_scaffolding_sends_no_bodies(self):
        llm = _RecordingLLM()
        results, decompilations = _synthesis_inputs(40)
        synthesizer = SemanticSynthesizer(llm, max_prompt_chars=10)

        synthesizer.synthesize(results, decompilations)

        assert "```c" not in llm.last_prompt.split("Functions and Decompilations")[-1]

    def test_without_a_budget_the_function_cap_still_applies(self):
        llm = _RecordingLLM()
        results, decompilations = _synthesis_inputs(SYNTHESIS_FUNCTION_CAP + 20, 100)
        synthesizer = SemanticSynthesizer(llm)

        synthesizer.synthesize(results, decompilations)

        headers = llm.last_prompt.count("Classification: utility")
        assert headers == SYNTHESIS_FUNCTION_CAP

    def test_the_globals_section_is_capped(self):
        llm = _RecordingLLM()
        results = [
            FunctionResult(
                address=0x1000 + i,
                original_name=f"FUN_{0x1000 + i:08x}",
                name=f"analyzed_function_{i}",
                confidence=90,
            )
            for i in range(200)
        ]
        # Every function touches every global, so each line is long.
        shared = " ".join(f"DAT_{0x9000 + g:08x};" for g in range(300))
        decompilations = {r.address: f"void f(void) {{ {shared} }}" for r in results}
        synthesizer = SemanticSynthesizer(llm, max_prompt_chars=BUDGET)

        synthesizer.synthesize(results, decompilations)

        globals_section = llm.last_prompt.split("## Global Variables")[-1]
        globals_section = globals_section.split("## Functions")[0]
        assert len(globals_section) <= BUDGET // 10 + 200

    def test_the_most_cross_referenced_functions_are_kept(self):
        llm = _RecordingLLM()
        results, decompilations = _synthesis_inputs(30, body_chars=500)
        # Give one function many more global references than the others.
        star = results[17]
        decompilations[star.address] = " ".join(
            f"DAT_{0x9000 + g:08x};" for g in range(50)
        )
        synthesizer = SemanticSynthesizer(llm, max_prompt_chars=BUDGET)

        synthesizer.synthesize(results, decompilations)

        # Sent first, in the group that carries the most shared structure.
        assert "analyzed_function_17" in llm.prompts[0]


class TestSupervisorWiring:
    def test_the_sequential_path_receives_the_budget(self, tmp_path):
        from kong.agent.supervisor import Supervisor
        from kong.config import KongConfig, LLMConfig, LLMProvider, OutputConfig

        config = KongConfig(
            llm=LLMConfig(
                provider=LLMProvider.CUSTOM,
                model="kong-local",
                base_url="http://127.0.0.1:8080/v1",
                max_prompt_chars=BUDGET,
            ),
            output=OutputConfig(directory=tmp_path / "out"),
        )
        sup = Supervisor(MagicMock(), config, llm_client=MagicMock())

        assert sup._get_effective_limits().max_prompt_chars == BUDGET
        assert sup._known_functions_budget() == BUDGET // 10


@pytest.mark.parametrize("budget", [1000, 4000, 16000])
def test_analyzer_budget_scales_with_the_configured_limit(budget):
    analyzer = Analyzer(_mock_client(), MagicMock(), max_prompt_chars=budget)
    context = analyzer._build_context(
        _item(), _binary_info(), _known_results(5000), []
    )
    assert len(analyzer._build_prompt(context)) <= budget
