"""Tests for the draft/refine routing rules."""

from __future__ import annotations

from kong.agent.models import FunctionResult
from kong.agent.refinement import (
    DEFAULT_REFINE_BELOW,
    is_generic_name,
    needs_refinement,
    refinement_reason,
    should_draft,
)
from kong.ghidra.types import FunctionClassification, FunctionInfo


def _result(**kwargs) -> FunctionResult:
    base = {
        "address": 0x1000,
        "original_name": "FUN_00001000",
        "name": "parse_http_header",
        "confidence": 95,
        "model": "fast-model",
    }
    base.update(kwargs)
    return FunctionResult(**base)


def _func(size=100, cls=FunctionClassification.MEDIUM) -> FunctionInfo:
    return FunctionInfo(
        address=0x1000, name="FUN_00001000", size=size, classification=cls,
    )


class TestShouldDraft:
    def test_a_small_function_is_drafted(self):
        assert should_draft(_func(size=40, cls=FunctionClassification.SMALL))

    def test_a_large_function_goes_straight_to_the_strong_model(self):
        assert not should_draft(_func(size=4000, cls=FunctionClassification.LARGE))

    def test_size_decides_when_triage_left_no_classification(self):
        assert should_draft(_func(size=64, cls=None))
        assert not should_draft(_func(size=5000, cls=None))


class TestGenericNames:
    def test_decompiler_placeholders_say_nothing(self):
        assert is_generic_name("FUN_00401a30")
        assert is_generic_name("sub_1234")
        assert is_generic_name("unk_80")

    def test_the_words_a_model_falls_back_on_say_nothing(self):
        assert is_generic_name("helper")
        assert is_generic_name("Process_Data")
        assert is_generic_name("   wrapper  ")

    def test_a_real_name_is_kept(self):
        assert not is_generic_name("parse_http_header")
        assert not is_generic_name("chacha20_encrypt")

    def test_a_name_that_merely_starts_with_a_word_is_kept(self):
        assert not is_generic_name("process_dns_answer")


class TestRefinementReason:
    def test_a_confident_specific_answer_is_final(self):
        assert refinement_reason(_result()) == ""

    def test_a_low_score_is_a_reason(self):
        reason = refinement_reason(_result(confidence=40))
        assert "40" in reason and str(DEFAULT_REFINE_BELOW) in reason

    def test_the_threshold_is_configurable(self):
        assert refinement_reason(_result(confidence=70), threshold=50) == ""
        assert refinement_reason(_result(confidence=70), threshold=90)

    def test_an_empty_name_is_a_reason(self):
        assert refinement_reason(_result(name="")) == "draft returned no name"

    def test_a_name_that_says_nothing_is_a_reason_whatever_the_score(self):
        assert refinement_reason(_result(name="handler", confidence=100))

    def test_a_failed_draft_is_a_reason_and_the_error_is_quoted(self):
        reason = refinement_reason(_result(error="Chunk call failed: HTTP 503"))
        assert "HTTP 503" in reason

    def test_a_very_long_error_does_not_run_into_the_event_log(self):
        reason = refinement_reason(_result(error="x" * 5000))
        assert len(reason) < 200

    def test_a_skipped_function_is_never_re_analyzed(self):
        skipped = _result(skipped=True, skip_reason="trivial", name="", confidence=0)
        assert refinement_reason(skipped) == ""

    def test_needs_refinement_mirrors_the_reason(self):
        assert needs_refinement(_result(confidence=10))
        assert not needs_refinement(_result())
