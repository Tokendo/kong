"""Synthesis is several calls, not one request over a whole program.

One call over everything is the natural shape for a cross-function pass and the
most fragile request a run makes: on a 1351-function binary it exhausted a 32k
output budget, was retried at 64k, timed out an hour later, and took the whole
phase with it.
"""

from __future__ import annotations

from dataclasses import dataclass

from kong.agent.models import FunctionResult
from kong.synthesis.semantic import SYNTHESIS_FUNCTION_CAP, SemanticSynthesizer


@dataclass
class _Answer:
    raw: str
    input_tokens: int = 0
    output_tokens: int = 0


class _Groups:
    """Records each call, and can be told which of them fail."""

    def __init__(self, answers=None, fail_on=()):
        self.prompts: list[str] = []
        self.answers = answers or []
        self.fail_on = set(fail_on)

    def analyze_function(self, prompt: str, *, model: str | None = None):
        index = len(self.prompts)
        self.prompts.append(prompt)
        if index in self.fail_on:
            raise TimeoutError("Request timed out.")
        if index < len(self.answers):
            return _Answer(raw=self.answers[index])
        return _Answer(raw="{}")


def _inputs(count, body_chars=400):
    results = [
        FunctionResult(
            address=0x1000 + i,
            original_name=f"FUN_{0x1000 + i:08x}",
            name=f"function_{i}",
            confidence=80,
            classification="utility",
        )
        for i in range(count)
    ]
    decompilations = {
        r.address: f"void function_{i}(void) {{ DAT_00009000 = 1; }}"
        + "\n// " + "x" * body_chars
        for i, r in enumerate(results)
    }
    return results, decompilations


class TestGrouping:
    def test_a_small_program_is_still_one_call(self):
        llm = _Groups()
        results, decompilations = _inputs(4)

        SemanticSynthesizer(llm, max_prompt_chars=100_000).synthesize(
            results, decompilations
        )

        assert len(llm.prompts) == 1

    def test_a_budget_that_cannot_hold_everything_splits_the_pass(self):
        llm = _Groups()
        results, decompilations = _inputs(20, body_chars=800)

        SemanticSynthesizer(llm, max_prompt_chars=9000).synthesize(
            results, decompilations
        )

        assert len(llm.prompts) > 1

    def test_every_prompt_stays_within_the_budget(self):
        budget = 9000
        llm = _Groups()
        results, decompilations = _inputs(20, body_chars=800)

        SemanticSynthesizer(llm, max_prompt_chars=budget).synthesize(
            results, decompilations
        )

        assert llm.prompts
        for prompt in llm.prompts:
            assert len(prompt) <= budget

    def test_no_budget_means_no_splitting(self):
        llm = _Groups()
        results, decompilations = _inputs(30)

        SemanticSynthesizer(llm).synthesize(results, decompilations)

        assert len(llm.prompts) == 1

    def test_the_cap_applies_to_the_pass_not_to_each_call(self):
        llm = _Groups()
        results, decompilations = _inputs(200, body_chars=200)

        SemanticSynthesizer(llm, max_prompt_chars=9000).synthesize(
            results, decompilations
        )

        sent = sum(prompt.count("### function_") for prompt in llm.prompts)
        assert sent <= SYNTHESIS_FUNCTION_CAP

    def test_nothing_to_synthesise_sends_nothing(self):
        llm = _Groups()

        result = SemanticSynthesizer(llm).synthesize([], {})

        assert llm.prompts == []
        assert result.globals == {}


class TestPartialResults:
    def test_a_group_that_fails_costs_only_its_own_functions(self):
        llm = _Groups(
            answers=[
                '{"globals": {"DAT_00009000": "g_tick"}}',
                "{}",
                '{"globals": {"DAT_0000a000": "g_mode"}}',
            ],
            fail_on=[1],
        )
        results, decompilations = _inputs(20, body_chars=800)

        outcome = SemanticSynthesizer(llm, max_prompt_chars=9000).synthesize(
            results, decompilations
        )

        assert outcome.partial
        assert outcome.failed_passes == 1
        assert outcome.passes >= 1
        assert "DAT_00009000" in outcome.globals

    def test_a_pass_where_everything_fails_still_raises(self):
        """The caller has to be able to tell "nothing came back" from "partial"."""
        import pytest

        llm = _Groups(fail_on=range(50))
        results, decompilations = _inputs(8)

        with pytest.raises(TimeoutError):
            SemanticSynthesizer(llm, max_prompt_chars=100_000).synthesize(
                results, decompilations
            )

    def test_a_complete_pass_is_not_marked_partial(self):
        llm = _Groups()
        results, decompilations = _inputs(8)

        outcome = SemanticSynthesizer(llm, max_prompt_chars=100_000).synthesize(
            results, decompilations
        )

        assert not outcome.partial


class TestMerging:
    def _result(self, **kwargs):
        from kong.synthesis.semantic import SynthesisResult

        return SynthesisResult(**kwargs)

    def test_globals_from_every_group_are_kept(self):
        first = self._result(globals={"DAT_1": "a"})
        first.merge(self._result(globals={"DAT_2": "b"}))

        assert first.globals == {"DAT_1": "a", "DAT_2": "b"}

    def test_the_earlier_group_wins_a_disagreement(self):
        """Groups are sent most-cross-referenced first: it saw more evidence."""
        first = self._result(globals={"DAT_1": "frame_counter"})
        first.merge(self._result(globals={"DAT_1": "some_int"}))

        assert first.globals["DAT_1"] == "frame_counter"

    def test_a_struct_proposed_twice_is_kept_once(self):
        first = self._result(structs=[{"name": "entity", "fields": []}])
        first.merge(self._result(structs=[{"name": "entity", "fields": []}]))

        assert len(first.structs) == 1

    def test_a_different_struct_is_added(self):
        first = self._result(structs=[{"name": "entity"}])
        first.merge(self._result(structs=[{"name": "weapon"}]))

        assert [s["name"] for s in first.structs] == ["entity", "weapon"]

    def test_name_refinements_merge_the_same_way(self):
        first = self._result(name_refinements={"0x1000": "tick"})
        first.merge(self._result(name_refinements={"0x1000": "x", "0x2000": "draw"}))

        assert first.name_refinements == {"0x1000": "tick", "0x2000": "draw"}
