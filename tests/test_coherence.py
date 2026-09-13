"""Tests for the coherence pass: what it finds, and what it makes of it.

Detection is text and arithmetic over results that already exist, so almost
none of this needs Ghidra or a model.
"""

from __future__ import annotations

import json

from kong.agent.coherence import (
    CoherenceReport,
    CoherenceReviewer,
    Conflict,
    ConflictKind,
    Resolution,
    detect_conflicts,
    find_call_sites,
    parse_resolutions,
    parse_signature,
)
from kong.agent.models import FunctionResult
from kong.agent.signatures import SignatureDB


def _result(addr: int, name: str, signature: str = "", confidence: int = 90, **kwargs):
    kwargs.setdefault("original_name", f"FUN_{addr:08x}")
    return FunctionResult(
        address=addr,
        name=name,
        signature=signature,
        confidence=confidence,
        **kwargs,
    )


def _results(*results: FunctionResult) -> dict[int, FunctionResult]:
    return {r.address: r for r in results}


def _kinds(conflicts: list[Conflict]) -> list[ConflictKind]:
    return [c.kind for c in conflicts]


class TestParseSignature:
    def test_reads_the_parts_it_compares(self):
        parsed = parse_signature("int parse_header(char *buf, size_t len)")
        assert parsed is not None
        assert parsed.return_type == "int"
        assert parsed.name == "parse_header"
        assert parsed.param_count == 2
        assert not parsed.returns_void

    def test_void_parameters_mean_none(self):
        parsed = parse_signature("void reset(void)")
        assert parsed is not None
        assert parsed.param_count == 0
        assert parsed.returns_void

    def test_empty_parentheses_say_nothing(self):
        """`int f()` is not a claim about arity, so it cannot contradict one."""
        parsed = parse_signature("int f()")
        assert parsed is not None
        assert parsed.unspecified

    def test_a_variadic_is_flagged_and_not_counted(self):
        parsed = parse_signature("int printf(char *fmt, ...)")
        assert parsed is not None
        assert parsed.variadic
        assert parsed.param_count == 1

    def test_a_pointer_return_is_a_value(self):
        parsed = parse_signature("void *memcpy(void *d, const void *s, size_t n)")
        assert parsed is not None
        assert not parsed.returns_void

    def test_rubbish_is_not_a_signature(self):
        assert parse_signature("") is None
        assert parse_signature("some prose") is None


class TestFindCallSites:
    CODE = """
undefined8 FUN_00401000(int param_1)

{
  int iVar1;

  helper(param_1, 3, "a,b");
  iVar1 = helper(param_1, 3, other(1, 2));
  if (iVar1 == 0) {
    log_line();
  }
  else
    log_line();
  return 0;
}
"""

    def test_arguments_are_counted_at_the_top_level(self):
        sites = find_call_sites(self.CODE, ["helper"])
        assert [s.arg_count for s in sites] == [3, 3]

    def test_an_assignment_uses_the_return_value(self):
        sites = find_call_sites(self.CODE, ["helper"])
        assert [s.uses_return for s in sites] == [False, True]

    def test_a_statement_on_its_own_does_not(self):
        sites = find_call_sites(self.CODE, ["log_line"])
        assert [s.uses_return for s in sites] == [False, False]

    def test_a_nested_call_is_an_argument_of_its_own(self):
        sites = find_call_sites(self.CODE, ["other"])
        assert [(s.arg_count, s.uses_return) for s in sites] == [(2, True)]

    def test_the_function_header_is_not_a_call_to_itself(self):
        """Ghidra opens with the signature, which reads exactly like a call."""
        assert find_call_sites(self.CODE, ["FUN_00401000"]) == []

    def test_the_line_is_kept_for_the_prompt(self):
        site = find_call_sites(self.CODE, ["log_line"])[0]
        assert site.line == "log_line();"


class TestDuplicateNames:
    def test_two_functions_with_one_name_is_a_conflict(self):
        results = _results(
            _result(0x1000, "parse_header"),
            _result(0x2000, "parse_header"),
        )
        conflicts = detect_conflicts(results, {}, {})
        assert _kinds(conflicts) == [ConflictKind.DUPLICATE_NAME]
        assert conflicts[0].addresses == (0x1000, 0x2000)
        assert "parse_header" in conflicts[0].summary

    def test_failed_and_skipped_results_do_not_collide(self):
        results = _results(
            _result(0x1000, "parse_header"),
            _result(0x2000, "parse_header", skipped=True),
            _result(0x3000, "parse_header", error="chunk call failed"),
        )
        assert detect_conflicts(results, {}, {}) == []

    def test_distinct_names_agree(self):
        results = _results(
            _result(0x1000, "parse_header"), _result(0x2000, "write_body"),
        )
        assert detect_conflicts(results, {}, {}) == []


class TestCallSiteConflicts:
    CALLER = """
void FUN_00402000(void)

{
  parse_header(buf, 12, 0, 1);
  return;
}
"""

    def test_an_argument_count_that_contradicts_the_signature(self):
        results = _results(
            _result(0x401000, "parse_header", "int parse_header(char *b, int n)"),
            _result(0x402000, "read_message"),
        )
        conflicts = detect_conflicts(
            results, {0x402000: self.CALLER}, {0x401000: [0x402000]},
        )
        assert _kinds(conflicts) == [ConflictKind.ARGUMENT_COUNT]
        assert conflicts[0].addresses == (0x401000, 0x402000)
        assert "passes 4" in conflicts[0].summary
        assert "read_message" in conflicts[0].summary

    def test_a_matching_arity_is_not_a_conflict(self):
        results = _results(
            _result(
                0x401000,
                "parse_header",
                "int parse_header(char *b, int n, int f, int g)",
            ),
            _result(0x402000, "read_message"),
        )
        assert detect_conflicts(
            results, {0x402000: self.CALLER}, {0x401000: [0x402000]},
        ) == []

    def test_a_variadic_never_contradicts_a_call_site(self):
        results = _results(
            _result(0x401000, "parse_header", "int parse_header(char *b, ...)"),
            _result(0x402000, "read_message"),
        )
        assert detect_conflicts(
            results, {0x402000: self.CALLER}, {0x401000: [0x402000]},
        ) == []

    def test_a_void_function_whose_value_is_used(self):
        caller = """
int FUN_00402000(void)

{
  int iVar1;
  iVar1 = parse_header(buf, 12);
  return iVar1;
}
"""
        results = _results(
            _result(0x401000, "parse_header", "void parse_header(char *b, int n)"),
            _result(0x402000, "read_message"),
        )
        conflicts = detect_conflicts(results, {0x402000: caller}, {0x401000: [0x402000]})
        assert _kinds(conflicts) == [ConflictKind.RETURN_VALUE]
        assert "declared void" in conflicts[0].summary

    def test_the_old_decompiler_name_is_matched_too(self):
        """A caller decompiled before the rename still says FUN_00401000."""
        caller = """
void FUN_00402000(void)

{
  FUN_00401000(buf, 12, 0, 1);
  return;
}
"""
        results = _results(
            _result(0x401000, "parse_header", "int parse_header(char *b, int n)"),
            _result(0x402000, "read_message"),
        )
        conflicts = detect_conflicts(results, {0x402000: caller}, {0x401000: [0x402000]})
        assert _kinds(conflicts) == [ConflictKind.ARGUMENT_COUNT]

    def test_a_shared_name_does_not_drag_in_the_other_function(self):
        """A call site cannot say which of two same-named functions it meant."""
        results = _results(
            _result(0x401000, "parse_header", "int parse_header(char *b, int n)"),
            _result(0x402000, "read_message"),
            _result(0x403000, "parse_header", "void parse_header(void)"),
        )
        conflicts = detect_conflicts(
            results,
            {0x402000: self.CALLER},
            {0x401000: [0x402000], 0x403000: [0x402000]},
        )
        # The duplicate itself is reported; the arity of the function that
        # happens to share its name is not derived from the same call site.
        assert _kinds(conflicts) == [ConflictKind.DUPLICATE_NAME]

    def test_a_caller_that_never_calls_it_says_nothing(self):
        results = _results(
            _result(0x401000, "parse_header", "int parse_header(char *b, int n)"),
            _result(0x402000, "read_message"),
        )
        assert detect_conflicts(
            results, {0x402000: "void FUN_00402000(void)\n{\n  return;\n}"},
            {0x401000: [0x402000]},
        ) == []


class TestSignatureNames:
    def test_a_signature_that_names_another_function(self):
        results = _results(
            _result(0x1000, "parse_header", "int read_block(char *b)"),
        )
        conflicts = detect_conflicts(results, {}, {})
        assert _kinds(conflicts) == [ConflictKind.SIGNATURE_NAME]
        assert "read_block" in conflicts[0].summary

    def test_the_same_name_is_no_conflict(self):
        results = _results(
            _result(0x1000, "parse_header", "int parse_header(char *b)"),
        )
        assert detect_conflicts(results, {}, {}) == []


class TestKnownSignatures:
    def _db(self) -> SignatureDB:
        db = SignatureDB()
        db.load_directory()
        return db

    def test_a_claimed_identity_with_the_wrong_arity(self):
        results = _results(
            _result(0x1000, "memcpy", "void *memcpy(void *d, void *s)"),
        )
        conflicts = detect_conflicts(results, {}, {}, self._db())
        assert _kinds(conflicts) == [ConflictKind.KNOWN_SIGNATURE]
        assert "memcpy" in conflicts[0].summary

    def test_the_real_thing_passes(self):
        results = _results(
            _result(0x1000, "memcpy", "void *memcpy(void *d, const void *s, size_t n)"),
        )
        assert detect_conflicts(results, {}, {}, self._db()) == []

    def test_a_name_the_database_does_not_know_is_left_alone(self):
        results = _results(
            _result(0x1000, "parse_header", "int parse_header(char *b)"),
        )
        assert detect_conflicts(results, {}, {}, self._db()) == []


class TestUnsupportedConfidence:
    def test_certainty_about_something_it_did_not_understand(self):
        results = _results(_result(
            0x1000,
            "process_data",
            confidence=95,
            comments="Cannot determine what this does.",
        ))
        conflicts = detect_conflicts(results, {}, {})
        assert _kinds(conflicts) == [ConflictKind.UNSUPPORTED_CONFIDENCE]
        assert "95%" in conflicts[0].summary

    def test_honest_uncertainty_is_not_a_contradiction(self):
        results = _results(_result(
            0x1000,
            "process_data",
            confidence=40,
            comments="Cannot determine what this does.",
        ))
        assert detect_conflicts(results, {}, {}) == []


class TestOrdering:
    def test_the_worst_contradictions_come_first(self):
        results = _results(
            _result(0x1000, "parse_header", "int read_block(char *b)"),
            _result(0x2000, "parse_header", "int parse_header(char *b)"),
        )
        conflicts = detect_conflicts(results, {}, {})
        assert _kinds(conflicts) == [
            ConflictKind.DUPLICATE_NAME, ConflictKind.SIGNATURE_NAME,
        ]


class _FakeLLM:
    """Records the prompts it is given and answers with a scripted reply."""

    model = "test-model"

    def __init__(self, raw: str = '{"resolutions": []}') -> None:
        self.raw = raw
        self.prompts: list[str] = []

    def analyze_function(self, prompt, *, model=None):
        from kong.agent.analyzer import LLMResponse

        self.prompts.append(prompt)
        return LLMResponse(name="", raw=self.raw)


def _conflict(kind: ConflictKind, *addresses: int) -> Conflict:
    return Conflict(kind=kind, addresses=tuple(addresses), summary="disagrees")


class TestBatching:
    def test_a_batch_is_capped_by_the_number_of_conflicts(self):
        reviewer = CoherenceReviewer(_FakeLLM(), conflicts_per_call=2)
        conflicts = [
            _conflict(ConflictKind.DUPLICATE_NAME, addr)
            for addr in (0x1000, 0x2000, 0x3000, 0x4000, 0x5000)
        ]
        assert [len(b) for b in reviewer.batches(conflicts, {})] == [2, 2, 1]

    def test_long_decompilations_split_the_batch(self):
        reviewer = CoherenceReviewer(_FakeLLM(), max_prompt_chars=3_000)
        decompilations = {addr: "x" * 1_000 for addr in (0x1000, 0x2000, 0x3000)}
        conflicts = [
            _conflict(ConflictKind.DUPLICATE_NAME, addr)
            for addr in (0x1000, 0x2000, 0x3000)
        ]
        assert [len(b) for b in reviewer.batches(conflicts, decompilations)] == [2, 1]

    def test_a_shared_function_is_paid_for_once(self):
        """Two conflicts about one function cost one decompilation, not two."""
        reviewer = CoherenceReviewer(_FakeLLM(), max_prompt_chars=3_000)
        decompilations = {0x1000: "x" * 1_000}
        conflicts = [
            _conflict(ConflictKind.DUPLICATE_NAME, 0x1000),
            _conflict(ConflictKind.SIGNATURE_NAME, 0x1000),
        ]
        assert len(reviewer.batches(conflicts, decompilations)) == 1


class TestPrompt:
    def test_the_prompt_carries_the_code_and_the_conflicts(self):
        reviewer = CoherenceReviewer(_FakeLLM())
        results = _results(
            _result(0x1000, "parse_header"), _result(0x2000, "parse_header"),
        )
        conflict = _conflict(ConflictKind.DUPLICATE_NAME, 0x1000, 0x2000)

        prompt = reviewer.build_prompt(
            [conflict], results, {0x1000: "void a(void) {}", 0x2000: "void b(void) {}"},
        )

        assert conflict.id in prompt
        assert "void a(void) {}" in prompt
        assert "void b(void) {}" in prompt
        assert "no_change" in prompt

    def test_a_function_in_two_conflicts_is_shown_once(self):
        reviewer = CoherenceReviewer(_FakeLLM())
        results = _results(_result(0x1000, "parse_header"))
        prompt = reviewer.build_prompt(
            [
                _conflict(ConflictKind.DUPLICATE_NAME, 0x1000),
                _conflict(ConflictKind.SIGNATURE_NAME, 0x1000),
            ],
            results,
            {0x1000: "void a(void) {}"},
        )
        assert prompt.count("### 0x00001000: parse_header") == 1

    def test_the_review_parses_what_the_model_answers(self):
        conflict = _conflict(ConflictKind.DUPLICATE_NAME, 0x1000, 0x2000)
        llm = _FakeLLM(json.dumps({"resolutions": [{
            "conflict": conflict.id,
            "verdict": "resolved",
            "explanation": "one parses, the other writes",
            "changes": [{"address": "0x00002000", "name": "write_header"}],
        }]}))
        reviewer = CoherenceReviewer(llm)

        resolutions = reviewer.review([conflict], _results(), {})

        assert len(resolutions) == 1
        assert resolutions[0].changes[0].name == "write_header"
        assert llm.prompts


class TestParseResolutions:
    CONFLICT = _conflict(ConflictKind.DUPLICATE_NAME, 0x1000, 0x2000)

    def test_a_bare_array_is_accepted(self):
        raw = json.dumps([{
            "conflict": self.CONFLICT.id,
            "changes": [{"address": 0x1000, "name": "parse_request"}],
        }])
        resolutions = parse_resolutions(raw, [self.CONFLICT])
        assert resolutions[0].conflict_id == self.CONFLICT.id
        # A verdict the model left out is read off what it actually asked for.
        assert resolutions[0].verdict == "resolved"

    def test_markdown_fences_are_stripped(self):
        raw = (
            "```json\n"
            + json.dumps({"resolutions": [{
                "conflict": self.CONFLICT.id, "verdict": "no_change",
            }]})
            + "\n```"
        )
        assert parse_resolutions(raw, [self.CONFLICT])[0].verdict == "no_change"

    def test_an_unknown_conflict_is_dropped(self):
        raw = json.dumps({"resolutions": [{"conflict": "made_up:0000"}]})
        assert parse_resolutions(raw, [self.CONFLICT]) == []

    def test_a_change_outside_the_batch_is_dropped(self):
        """The answer lands in Ghidra, so a stray address is not followed."""
        raw = json.dumps({"resolutions": [{
            "conflict": self.CONFLICT.id,
            "changes": [
                {"address": "0x00009999", "name": "elsewhere"},
                {"address": "0x00001000", "name": "parse_request"},
            ],
        }]})
        changes = parse_resolutions(raw, [self.CONFLICT])[0].changes
        assert [c.address for c in changes] == [0x1000]

    def test_a_confidence_written_as_a_percentage_is_read(self):
        raw = json.dumps({"resolutions": [{
            "conflict": self.CONFLICT.id,
            "changes": [{"address": "0x00001000", "confidence": "55%"}],
        }]})
        assert parse_resolutions(raw, [self.CONFLICT])[0].changes[0].confidence == 55

    def test_a_change_that_asks_for_nothing_is_not_a_change(self):
        raw = json.dumps({"resolutions": [{
            "conflict": self.CONFLICT.id,
            "changes": [{"address": "0x00001000"}],
        }]})
        resolution = parse_resolutions(raw, [self.CONFLICT])[0]
        assert resolution.changes == []
        assert resolution.verdict == "no_change"

    def test_an_answer_cut_off_mid_array_keeps_what_arrived(self):
        """Each resolution stands on its own, so a truncated reply is not lost."""
        raw = (
            '{"resolutions": [{"conflict": "' + self.CONFLICT.id + '", '
            '"verdict": "resolved", "changes": [{"address": "0x00001000", '
            '"name": "parse_request"'
        )
        changes = parse_resolutions(raw, [self.CONFLICT])[0].changes
        assert [c.name for c in changes] == ["parse_request"]

    def test_prose_instead_of_json_resolves_nothing(self):
        assert parse_resolutions("I could not decide.", [self.CONFLICT]) == []
        assert parse_resolutions("", [self.CONFLICT]) == []


class TestReport:
    def test_the_report_pairs_each_conflict_with_its_answer(self):
        conflict = _conflict(ConflictKind.DUPLICATE_NAME, 0x1000, 0x2000)
        report = CoherenceReport(
            functions_checked=12,
            conflicts=[conflict],
            resolutions={conflict.id: Resolution(
                conflict_id=conflict.id,
                verdict="resolved",
                explanation="one parses, the other writes",
            )},
            applied=["0x00002000: parse_header to write_header"],
        )

        document = report.as_dict()

        assert document["functions_checked"] == 12
        assert document["conflicts_found"] == 1
        assert document["changes_applied"] == 1
        assert document["conflicts"][0]["resolution"]["verdict"] == "resolved"
        # Serialisable as it stands: it is written next to the analysis.
        json.dumps(document)

    def test_an_unanswered_conflict_is_reported_without_one(self):
        conflict = _conflict(ConflictKind.DUPLICATE_NAME, 0x1000)
        report = CoherenceReport(conflicts=[conflict], unreviewed=1)
        assert report.reviewed == 0
        assert report.as_dict()["conflicts"][0]["resolution"] is None
