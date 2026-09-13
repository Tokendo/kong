"""Tests for the supervisor agent loop."""

from __future__ import annotations

import json
import re
from unittest.mock import MagicMock

import pytest

from kong.agent.analyzer import LLMResponse
from kong.agent.coherence import REPORT_NAME
from kong.agent.events import EventType, Phase
from kong.agent.models import AnalysisStats, FunctionResult
from kong.agent.supervisor import Supervisor
from kong.config import KongConfig, LLMConfig, LLMProvider, OutputConfig, RunStage
from kong.ghidra.types import BinaryInfo, FunctionClassification, FunctionInfo


def _make_client(functions=None, binary_info=None):
    """Create a mock GhidraClient."""
    client = MagicMock()
    client.get_binary_info.return_value = binary_info or BinaryInfo(
        arch="x86-64", format="ELF", endianness="little", word_size=8,
        compiler="GCC", name="test_binary",
    )
    client.list_functions.return_value = functions or []
    client.get_strings.return_value = []
    client.get_callers.return_value = []
    client.get_callees.return_value = []
    client.get_decompilation.return_value = "void stub(void) { return; }"
    return client


def _func(addr, name, size=100, cls=FunctionClassification.MEDIUM):
    return FunctionInfo(address=addr, name=name, size=size, classification=cls)


def _batch_response_with_addresses(prompt: str, **kwargs: object) -> list[LLMResponse]:
    """Mock batch response that extracts addresses from chunk prompt headers."""
    import re
    addresses = re.findall(r"### (0x[0-9a-fA-F]+):", prompt)
    return [
        LLMResponse(
            name=f"func_{i}",
            confidence=80,
            classification="utility",
            address=int(addr, 16),
        )
        for i, addr in enumerate(addresses)
    ]


class TestSupervisorLifecycle:
    def test_run_emits_start_and_complete(self, tmp_path):
        client = _make_client()
        config = KongConfig(output=OutputConfig(directory=tmp_path / "out"))
        sup = Supervisor(client, config)

        events = []
        sup.on_event(events.append)
        sup.run()

        types = [e.type for e in events]
        assert types[0] == EventType.RUN_START
        assert types[-1] == EventType.RUN_COMPLETE

    def test_all_five_phases_run(self, tmp_path):
        client = _make_client()
        config = KongConfig(output=OutputConfig(directory=tmp_path / "out"))
        sup = Supervisor(client, config)

        events = []
        sup.on_event(events.append)
        sup.run()

        phase_starts = [e.phase for e in events if e.type == EventType.PHASE_START]
        expected = [Phase.TRIAGE, Phase.ANALYSIS, Phase.CLEANUP, Phase.SYNTHESIS, Phase.EXPORT]
        assert phase_starts == expected

    def test_run_with_no_functions(self, tmp_path):
        client = _make_client(functions=[])
        config = KongConfig(output=OutputConfig(directory=tmp_path / "out"))
        sup = Supervisor(client, config)
        results = sup.run()
        assert results == {}


class TestSupervisorTriage:
    def test_triage_enumerates_functions(self, tmp_path):
        funcs = [_func(0x1000, "a"), _func(0x2000, "b")]
        client = _make_client(functions=funcs)
        config = KongConfig(output=OutputConfig(directory=tmp_path / "out"))
        sup = Supervisor(client, config)

        events = []
        sup.on_event(events.append)
        sup.run()

        enum_events = [e for e in events if e.type == EventType.TRIAGE_FUNCTIONS_ENUMERATED]
        assert len(enum_events) == 1
        assert enum_events[0].data["total"] == 2

    def test_triage_builds_queue(self, tmp_path):
        funcs = [
            _func(0x1000, "a"),
            _func(0x2000, "b", cls=FunctionClassification.THUNK),
        ]
        client = _make_client(functions=funcs)
        config = KongConfig(output=OutputConfig(directory=tmp_path / "out"))
        sup = Supervisor(client, config)

        events = []
        sup.on_event(events.append)
        sup.run()

        queue_events = [e for e in events if e.type == EventType.TRIAGE_QUEUE_BUILT]
        assert len(queue_events) == 1
        assert queue_events[0].data["queue_size"] == 1


class TestSupervisorAnalysis:
    def test_skips_trivial_functions(self, tmp_path):
        funcs = [_func(0x1000, "tiny", size=10, cls=FunctionClassification.TRIVIAL)]
        client = _make_client(functions=funcs)
        config = KongConfig(output=OutputConfig(directory=tmp_path / "out"))
        sup = Supervisor(client, config)

        events = []
        sup.on_event(events.append)
        results = sup.run()

        assert 0x1000 in results
        assert results[0x1000].skipped is True
        assert results[0x1000].skip_reason == "trivial"

    def test_produces_results_for_each_function(self, tmp_path):
        funcs = [_func(0x1000, "a"), _func(0x2000, "b"), _func(0x3000, "c")]
        client = _make_client(functions=funcs)
        config = KongConfig(output=OutputConfig(directory=tmp_path / "out"))
        sup = Supervisor(client, config)
        results = sup.run()

        assert len(results) == 3
        for addr in [0x1000, 0x2000, 0x3000]:
            assert addr in results

    def test_chunk_analysis_emits_events(self, tmp_path):
        funcs = [_func(0x1000, "a")]
        client = _make_client(functions=funcs)
        client.get_decompilation.return_value = "void a(void) { return; }"
        config = KongConfig(output=OutputConfig(directory=tmp_path / "out"))

        mock_llm = MagicMock()
        mock_llm.analyze_function_batch.side_effect = _batch_response_with_addresses
        mock_llm.analyze_function.return_value = LLMResponse(
            name="", raw='{"globals":{},"structs":[],"name_refinements":{}}',
        )

        sup = Supervisor(client, config, llm_client=mock_llm)

        events = []
        sup.on_event(events.append)
        sup.run()

        starts = [e for e in events if e.type == EventType.FUNCTION_START]
        assert len(starts) >= 1
        assert any(e.data["address"] == 0x1000 for e in starts)

        completes = [e for e in events if e.type == EventType.FUNCTION_COMPLETE]
        assert len(completes) >= 1

    def test_handles_chunk_error(self, tmp_path):
        funcs = [_func(0x1000, "a")]
        client = _make_client(functions=funcs)
        client.get_decompilation.return_value = "void a(void) { return; }"
        config = KongConfig(output=OutputConfig(directory=tmp_path / "out"))

        mock_llm = MagicMock()
        mock_llm.analyze_function_batch.side_effect = RuntimeError("LLM timeout")
        mock_llm.analyze_function.return_value = LLMResponse(
            name="", raw='{"globals":{},"structs":[],"name_refinements":{}}',
        )

        sup = Supervisor(client, config, llm_client=mock_llm)

        events = []
        sup.on_event(events.append)
        results = sup.run()

        assert results[0x1000].error
        error_events = [e for e in events if e.type == EventType.FUNCTION_ERROR]
        assert len(error_events) == 1


class TestAnalysisStats:
    def test_record_result_renamed(self):
        stats = AnalysisStats(total_functions=10)
        result = FunctionResult(
            address=0x1000, original_name="FUN_1000",
            name="init_module", confidence=85, llm_calls=2,
        )
        stats.record_result(result)

        assert stats.analyzed == 1
        assert stats.renamed == 1
        assert stats.named == 1
        assert stats.high_confidence == 1
        assert stats.llm_calls == 2

    def test_record_result_skipped(self):
        stats = AnalysisStats(total_functions=10)
        result = FunctionResult(
            address=0x1000, original_name="FUN_1000",
            skipped=True, skip_reason="trivial",
        )
        stats.record_result(result)

        assert stats.skipped == 1
        assert stats.analyzed == 0

    def test_record_result_error(self):
        stats = AnalysisStats(total_functions=10)
        result = FunctionResult(
            address=0x1000, original_name="FUN_1000",
            error="timeout",
        )
        stats.record_result(result)

        assert stats.errors == 1
        assert stats.analyzed == 0

    def test_record_result_unchanged_name_is_confirmed(self):
        stats = AnalysisStats(total_functions=10)
        result = FunctionResult(
            address=0x1000, original_name="FUN_1000",
            name="FUN_1000", confidence=30, llm_calls=1,
        )
        stats.record_result(result)

        assert stats.analyzed == 1
        assert stats.renamed == 0
        assert stats.confirmed == 1
        assert stats.named == 1
        assert stats.low_confidence == 1

    def test_replace_result_moves_the_function_between_buckets(self):
        stats = AnalysisStats(total_functions=10)
        draft = FunctionResult(
            address=0x1000, original_name="FUN_1000",
            name="FUN_1000", confidence=30, llm_calls=1,
        )
        stats.record_result(draft)

        refined = FunctionResult(
            address=0x1000, original_name="FUN_1000",
            name="parse_http_header", confidence=95, llm_calls=2,
        )
        stats.replace_result(draft, refined)

        assert stats.analyzed == 1
        assert stats.renamed == 1
        assert stats.confirmed == 0
        assert stats.low_confidence == 0
        assert stats.high_confidence == 1
        assert stats.llm_calls == 2

    def test_replace_result_takes_a_function_out_of_the_error_count(self):
        stats = AnalysisStats(total_functions=10)
        failed = FunctionResult(
            address=0x1000, original_name="FUN_1000", error="HTTP 503",
        )
        stats.record_result(failed)

        stats.replace_result(failed, FunctionResult(
            address=0x1000, original_name="FUN_1000",
            name="parse", confidence=90, llm_calls=1,
        ))

        assert stats.errors == 0
        assert stats.analyzed == 1

    def test_confidence_buckets(self):
        stats = AnalysisStats(total_functions=10)

        stats.record_result(FunctionResult(address=1, original_name="a", name="x", confidence=90, llm_calls=1))
        stats.record_result(FunctionResult(address=2, original_name="b", name="y", confidence=60, llm_calls=1))
        stats.record_result(FunctionResult(address=3, original_name="c", name="z", confidence=30, llm_calls=1))

        assert stats.high_confidence == 1
        assert stats.medium_confidence == 1
        assert stats.low_confidence == 1

    def test_name_rate(self):
        stats = AnalysisStats(total_functions=4)
        stats.renamed = 2
        stats.confirmed = 1
        assert stats.name_rate == 0.75

class TestSupervisorExport:
    def test_export_creates_source_file(self, tmp_path):
        funcs = [_func(0x1000, "a")]
        client = _make_client(functions=funcs)
        client.get_decompilation.return_value = "void a(void) { return; }"
        config = KongConfig(output=OutputConfig(
            directory=tmp_path / "out",
            formats=["source"],
        ))
        sup = Supervisor(client, config)
        sup.run()

        assert (tmp_path / "out" / "decompiled.c").exists()

    def test_export_creates_json_file(self, tmp_path):
        funcs = [_func(0x1000, "a")]
        client = _make_client(functions=funcs)
        client.get_decompilation.return_value = "void a(void) { return; }"
        config = KongConfig(output=OutputConfig(
            directory=tmp_path / "out",
            formats=["json"],
        ))
        sup = Supervisor(client, config)
        sup.run()

        assert (tmp_path / "out" / "analysis.json").exists()

    def test_export_emits_export_file_events(self, tmp_path):
        funcs = [_func(0x1000, "a")]
        client = _make_client(functions=funcs)
        client.get_decompilation.return_value = "void a(void) { return; }"
        config = KongConfig(output=OutputConfig(
            directory=tmp_path / "out",
            formats=["source", "json"],
        ))
        sup = Supervisor(client, config)

        events = []
        sup.on_event(events.append)
        sup.run()

        export_events = [e for e in events if e.type == EventType.EXPORT_FILE]
        assert len(export_events) == 2
        formats_emitted = {e.data["format"] for e in export_events}
        assert formats_emitted == {"source", "json"}

    def test_export_skips_ghidra_format(self, tmp_path):
        client = _make_client()
        config = KongConfig(output=OutputConfig(
            directory=tmp_path / "out",
            formats=["ghidra"],
        ))
        sup = Supervisor(client, config)
        sup.run()


class TestSupervisorSynthesis:
    def test_synthesis_phase_runs(self, tmp_path):
        funcs = [_func(0x1000, "a")]
        client = _make_client(functions=funcs)
        client.get_decompilation.return_value = "void a(void) { return; }"
        config = KongConfig(output=OutputConfig(directory=tmp_path / "out"))

        mock_llm = MagicMock()
        mock_llm.analyze_function.return_value = LLMResponse(
            name="init", confidence=80, raw='{"globals":{},"structs":[],"name_refinements":{}}',
        )
        mock_llm.analyze_function_batch.side_effect = _batch_response_with_addresses

        sup = Supervisor(client, config, llm_client=mock_llm)

        events = []
        sup.on_event(events.append)
        sup.run()

        phase_starts = [e.phase for e in events if e.type == EventType.PHASE_START]
        assert Phase.SYNTHESIS in phase_starts

        phase_completes = [e.phase for e in events if e.type == EventType.PHASE_COMPLETE]
        assert Phase.SYNTHESIS in phase_completes

    def test_synthesis_skipped_without_llm_client(self, tmp_path):
        funcs = [_func(0x1000, "a")]
        client = _make_client(functions=funcs)
        config = KongConfig(output=OutputConfig(directory=tmp_path / "out"))
        sup = Supervisor(client, config)

        events = []
        sup.on_event(events.append)
        sup.run()

        synthesis_completes = [
            e for e in events
            if e.type == EventType.PHASE_COMPLETE and e.phase == Phase.SYNTHESIS
        ]
        assert len(synthesis_completes) == 1
        assert "skipped" in synthesis_completes[0].message.lower()

    def test_synthesis_skipped_with_no_results(self, tmp_path):
        client = _make_client(functions=[])
        config = KongConfig(output=OutputConfig(directory=tmp_path / "out"))

        mock_llm = MagicMock()
        sup = Supervisor(client, config, llm_client=mock_llm)

        events = []
        sup.on_event(events.append)
        sup.run()

        synthesis_completes = [
            e for e in events
            if e.type == EventType.PHASE_COMPLETE and e.phase == Phase.SYNTHESIS
        ]
        assert len(synthesis_completes) == 1
        assert "skipped" in synthesis_completes[0].message.lower()


class TestDecompilationCache:
    def test_cached_decompilation_avoids_redundant_ghidra_calls(self, tmp_path):
        client = _make_client(functions=[_func(0x1000, "FUN_1000")])
        config = KongConfig(output=OutputConfig(directory=tmp_path / "out"))
        sup = Supervisor(client, config)
        sup._decompilation_cache[0x1000] = "cached code"

        result = sup._get_decompilation(0x1000)
        assert result == "cached code"
        client.get_decompilation.assert_not_called()

    def test_uncached_decompilation_calls_ghidra_and_caches(self, tmp_path):
        client = _make_client(functions=[_func(0x1000, "FUN_1000")])
        client.get_decompilation.return_value = "void foo(void) {}"
        config = KongConfig(output=OutputConfig(directory=tmp_path / "out"))
        sup = Supervisor(client, config)

        result = sup._get_decompilation(0x1000)
        assert result == "void foo(void) {}"
        assert 0x1000 in sup._decompilation_cache
        client.get_decompilation.assert_called_once_with(0x1000)


class TestCleanup:
    def test_cleanup_retries_signatures(self, tmp_path):
        client = _make_client()
        config = KongConfig(output=OutputConfig(directory=tmp_path / "out"))
        sup = Supervisor(client, config)
        sup.binary_info = client.get_binary_info()
        sup.results[0x1000] = FunctionResult(
            address=0x1000, original_name="FUN_1000",
            name="init", confidence=80, signature="void init(void)",
            signature_applied=False,
        )

        sup._run_cleanup()

        client.set_function_signature.assert_called_once_with(0x1000, "void init(void)")
        assert sup.results[0x1000].signature_applied is True


class TestChunkedPipelineIntegration:
    def test_full_pipeline_uses_chunk_calls(self, tmp_path):
        funcs = [_func(0x1000 + i * 0x100, f"FUN_{i}", size=64) for i in range(5)]
        client = _make_client(functions=funcs)
        client.get_decompilation.return_value = "void f(void) { return; }"
        client.get_xrefs_from.return_value = []
        client.get_function_info.return_value = funcs[0]
        client.list_custom_types.return_value = []
        config = KongConfig(output=OutputConfig(directory=tmp_path / "out"))

        mock_llm = MagicMock()
        mock_llm.analyze_function_batch.side_effect = _batch_response_with_addresses
        mock_llm.analyze_function.return_value = LLMResponse(
            name="", raw='{"globals":{},"structs":[],"name_refinements":{}}',
        )

        sup = Supervisor(client, config, llm_client=mock_llm)
        sup.run()

        assert mock_llm.analyze_function_batch.call_count > 0
        named = [r for r in sup.results.values() if r.name and not r.skipped]
        assert len(named) == 5

    def test_chunk_prompt_includes_decompilations(self, tmp_path):
        funcs = [_func(0x1000, "FUN_1000", size=64)]
        client = _make_client(functions=funcs)
        client.get_decompilation.return_value = "void special_func(void) { return; }"
        config = KongConfig(output=OutputConfig(directory=tmp_path / "out"))

        mock_llm = MagicMock()
        mock_llm.analyze_function_batch.side_effect = _batch_response_with_addresses
        mock_llm.analyze_function.return_value = LLMResponse(
            name="", raw='{"globals":{},"structs":[],"name_refinements":{}}',
        )

        sup = Supervisor(client, config, llm_client=mock_llm)
        sup.run()

        prompt = mock_llm.analyze_function_batch.call_args[0][0]
        assert "0x00001000" in prompt
        assert "special_func" in prompt
        assert "x86-64" in prompt

    def test_all_functions_in_single_chunk(self, tmp_path):
        """With < CHUNK_SIZE functions, everything goes in one LLM call."""
        funcs = [_func(0x1000 + i * 0x100, f"FUN_{i}", size=64) for i in range(10)]
        client = _make_client(functions=funcs)
        client.get_decompilation.return_value = "void f(void) { return; }"
        config = KongConfig(output=OutputConfig(directory=tmp_path / "out"))

        mock_llm = MagicMock()
        mock_llm.analyze_function_batch.side_effect = _batch_response_with_addresses
        mock_llm.analyze_function.return_value = LLMResponse(
            name="", raw='{"globals":{},"structs":[],"name_refinements":{}}',
        )

        sup = Supervisor(client, config, llm_client=mock_llm)
        sup.run()

        assert mock_llm.analyze_function_batch.call_count == 1


class TestGetEffectiveLimits:
    def test_non_custom_returns_base_limits(self, tmp_path):
        from kong.llm.limits import _DEFAULT_LIMITS

        config = KongConfig(
            llm=LLMConfig(provider=LLMProvider.ANTHROPIC),
            output=OutputConfig(directory=tmp_path / "out"),
        )
        sup = Supervisor(_make_client(), config)
        limits = sup._get_effective_limits()
        assert limits == _DEFAULT_LIMITS

    def test_a_hosted_provider_honours_an_explicit_batch_size(self, tmp_path):
        """Rate limits and long-batch amnesia are not in the model table."""
        config = KongConfig(
            llm=LLMConfig(
                provider=LLMProvider.ANTHROPIC,
                model="claude-opus-5",
                max_chunk_functions=25,
            ),
            output=OutputConfig(directory=tmp_path / "out"),
        )
        sup = Supervisor(_make_client(), config)
        limits = sup._limits_for("claude-opus-5")

        assert limits.max_chunk_functions == 25
        # Untouched: only what was asked for is overridden.
        assert limits.max_prompt_chars == 900_000

    def test_a_hosted_batch_size_reaches_the_chunker(self, tmp_path):
        funcs = [_func(addr, f"FUN_{addr:x}", size=32) for addr in range(0x1000, 0x1500, 0x100)]
        client = _make_client(functions=funcs)
        client.get_decompilation.return_value = "void f(void) { return; }"
        config = KongConfig(
            llm=LLMConfig(
                provider=LLMProvider.ANTHROPIC,
                model="claude-opus-5",
                max_chunk_functions=2,
            ),
            output=OutputConfig(directory=tmp_path / "out"),
        )
        mock_llm = MagicMock()
        mock_llm.model = "claude-opus-5"
        mock_llm.analyze_function_batch.side_effect = _batch_response_with_addresses
        mock_llm.analyze_function.return_value = LLMResponse(
            name="", raw='{"globals":{},"structs":[],"name_refinements":{}}',
        )

        sup = Supervisor(client, config, llm_client=mock_llm)
        sup.run()

        sizes = [
            len(re.findall(r"### 0x", call.args[0]))
            for call in mock_llm.analyze_function_batch.call_args_list
        ]
        assert sizes == [2, 2, 1]

    def test_custom_with_overrides(self, tmp_path):
        config = KongConfig(
            llm=LLMConfig(
                provider=LLMProvider.CUSTOM,
                model="llama3:8b",
                max_prompt_chars=32000,
                max_chunk_functions=20,
                max_output_tokens=4096,
            ),
            output=OutputConfig(directory=tmp_path / "out"),
        )
        sup = Supervisor(_make_client(), config)
        limits = sup._get_effective_limits()
        assert limits.max_prompt_chars == 32000
        assert limits.max_chunk_functions == 20
        assert limits.max_output_tokens == 4096

    def test_custom_partial_overrides_fall_back(self, tmp_path):
        from kong.llm.limits import _DEFAULT_LIMITS

        config = KongConfig(
            llm=LLMConfig(
                provider=LLMProvider.CUSTOM,
                model="llama3:8b",
                max_prompt_chars=32000,
            ),
            output=OutputConfig(directory=tmp_path / "out"),
        )
        sup = Supervisor(_make_client(), config)
        limits = sup._get_effective_limits()
        assert limits.max_prompt_chars == 32000
        assert limits.max_chunk_functions == _DEFAULT_LIMITS.max_chunk_functions
        assert limits.max_output_tokens == _DEFAULT_LIMITS.max_output_tokens


def _small_context_config(tmp_path, **overrides):
    """A supervisor config sized like a local llama.cpp server."""
    params = dict(
        provider=LLMProvider.CUSTOM,
        model="qwen2.5-coder-7b",
        base_url="http://localhost:8080/v1",
        max_prompt_chars=8000,
        max_chunk_functions=3,
        max_output_tokens=1024,
    )
    params.update(overrides)
    return KongConfig(
        llm=LLMConfig(**params),
        output=OutputConfig(directory=tmp_path / "out"),
    )


class TestLimitedContextChunking:
    """A local model with a small window must never be sent an over-budget prompt."""

    def test_batch_call_uses_the_configured_output_budget(self, tmp_path):
        funcs = [_func(0x1000, "FUN_1000", size=64)]
        client = _make_client(functions=funcs)
        client.get_decompilation.return_value = "void f(void) { return; }"
        config = _small_context_config(tmp_path)

        mock_llm = MagicMock()
        mock_llm.analyze_function_batch.side_effect = _batch_response_with_addresses
        mock_llm.analyze_function.return_value = LLMResponse(
            name="", raw='{"globals":{},"structs":[],"name_refinements":{}}',
        )

        sup = Supervisor(client, config, llm_client=mock_llm)
        sup.run()

        assert mock_llm.analyze_function_batch.call_args.kwargs["max_tokens"] == 1024

    def test_every_prompt_stays_within_the_budget(self, tmp_path):
        """The 'already identified' preamble grows; the budget must still hold."""
        funcs = [_func(0x1000 + i * 0x100, f"FUN_{i}", size=64) for i in range(40)]
        client = _make_client(functions=funcs)
        client.get_decompilation.return_value = "void f(void) { return; }" * 10
        config = _small_context_config(tmp_path)

        mock_llm = MagicMock()
        mock_llm.analyze_function_batch.side_effect = _batch_response_with_addresses
        mock_llm.analyze_function.return_value = LLMResponse(
            name="", raw='{"globals":{},"structs":[],"name_refinements":{}}',
        )

        sup = Supervisor(client, config, llm_client=mock_llm)
        sup.run()

        assert mock_llm.analyze_function_batch.call_count > 1
        prompts = [c[0][0] for c in mock_llm.analyze_function_batch.call_args_list]
        assert max(len(p) for p in prompts) <= 8000

    def test_known_functions_preamble_is_bounded(self, tmp_path):
        config = _small_context_config(tmp_path)
        sup = Supervisor(_make_client(), config)
        sup.binary_info = BinaryInfo(
            arch="x86-64", format="ELF", endianness="little", word_size=8,
            compiler="GCC", name="test_binary",
        )
        sup.results = {
            0x1000 + i: FunctionResult(
                address=0x1000 + i,
                original_name=f"FUN_{i}",
                name=f"resolved_name_number_{i}",
                confidence=90,
            )
            for i in range(5000)
        }

        prompt = sup._build_chunk_prompt([])

        assert "resolved_name_number_4999" in prompt  # newest kept
        assert len(prompt) <= sup._known_functions_budget() + 500

    def test_function_larger_than_the_budget_is_reported_not_sent(self, tmp_path):
        funcs = [_func(0x1000, "FUN_1000", size=64), _func(0x2000, "FUN_2000", size=64)]
        client = _make_client(functions=funcs)

        def decompilation(addr):
            return "void huge(void) {}" * 2000 if addr == 0x2000 else "void f(void) {}"

        client.get_decompilation.side_effect = decompilation
        config = _small_context_config(tmp_path)

        mock_llm = MagicMock()
        mock_llm.analyze_function_batch.side_effect = _batch_response_with_addresses
        mock_llm.analyze_function.return_value = LLMResponse(
            name="", raw='{"globals":{},"structs":[],"name_refinements":{}}',
        )

        sup = Supervisor(client, config, llm_client=mock_llm)
        events = []
        sup.on_event(events.append)
        sup.run()

        prompts = [c[0][0] for c in mock_llm.analyze_function_batch.call_args_list]
        assert all("0x00002000" not in p for p in prompts)
        assert "prompt budget" in sup.results[0x2000].error
        assert sup.results[0x1000].name  # the rest of the run continues
        errors = [e for e in events if e.type == EventType.FUNCTION_ERROR]
        assert any(e.data["address"] == 0x2000 for e in errors)

    def test_budget_smaller_than_the_scaffolding_is_rejected(self, tmp_path):
        config = _small_context_config(tmp_path, max_prompt_chars=50)
        sup = Supervisor(_make_client(), config)
        sup.binary_info = BinaryInfo(
            arch="x86-64", format="ELF", endianness="little", word_size=8,
            compiler="GCC", name="test_binary",
        )

        with pytest.raises(ValueError, match="too small"):
            sup._split_into_chunks([])

    def test_split_returns_chunks_and_oversized(self, tmp_path):
        config = _small_context_config(tmp_path)
        sup = Supervisor(_make_client(), config)
        sup.binary_info = BinaryInfo(
            arch="x86-64", format="ELF", endianness="little", word_size=8,
            compiler="GCC", name="test_binary",
        )

        chunks, oversized = sup._split_into_chunks([])
        assert chunks == []
        assert oversized == []


class TestConfidenceRegression:
    """A string score used to kill the run inside AnalysisStats.record_result."""

    @staticmethod
    def _llm_returning(confidence):
        """An LLM whose batch replies go through the real JSON parsing."""
        import json

        from kong.agent.analyzer import Analyzer

        def batch(prompt, **kwargs):
            import re

            addresses = re.findall(r"### (0x[0-9a-fA-F]+):", prompt)
            payload = json.dumps([
                {
                    "address": address,
                    "name": f"named_{index}",
                    "confidence": confidence,
                    "classification": "utility",
                }
                for index, address in enumerate(addresses)
            ])
            return Analyzer.parse_llm_json_batch(payload)

        llm = MagicMock()
        llm.analyze_function_batch.side_effect = batch
        llm.analyze_function.return_value = LLMResponse(
            name="", raw='{"globals":{},"structs":[],"name_refinements":{}}',
        )
        return llm

    def _run(self, tmp_path, confidence):
        funcs = [_func(0x1000, "FUN_1000", size=64)]
        client = _make_client(functions=funcs)
        client.get_decompilation.return_value = "void f(void) { return; }"
        config = KongConfig(output=OutputConfig(directory=tmp_path / "out"))

        sup = Supervisor(client, config, llm_client=self._llm_returning(confidence))
        sup.run()
        return sup

    def test_a_string_score_no_longer_kills_the_run(self, tmp_path):
        sup = self._run(tmp_path, "85")

        assert sup.stats.errors == 0
        assert sup.stats.high_confidence == 1
        assert sup.results[0x1000].confidence == 85

    def test_a_percent_string_lands_in_the_right_bucket(self, tmp_path):
        sup = self._run(tmp_path, "60%")

        assert sup.stats.medium_confidence == 1

    def test_a_fractional_score_is_not_flattened_to_zero(self, tmp_path):
        sup = self._run(tmp_path, 0.9)

        assert sup.stats.high_confidence == 1

    def test_an_unusable_score_is_recorded_low_not_fatal(self, tmp_path):
        sup = self._run(tmp_path, "very confident")

        assert sup.stats.errors == 0
        assert sup.stats.low_confidence == 1

    def test_the_exported_json_holds_an_int(self, tmp_path):
        import json

        self._run(tmp_path, "85")
        document = json.loads((tmp_path / "out" / "analysis.json").read_text())

        assert document["functions"][0]["confidence"] == 85


class TestRunLog:
    """Every run leaves a trace next to its output."""

    def test_a_run_writes_events_log(self, tmp_path):
        client = _make_client()
        config = KongConfig(output=OutputConfig(directory=tmp_path / "out"))

        Supervisor(client, config).run()

        contents = (tmp_path / "out" / "events.log").read_text(encoding="utf-8")
        assert "run_start" in contents
        assert "run_complete" in contents

    def test_the_start_event_names_the_trace(self, tmp_path):
        client = _make_client()
        config = KongConfig(output=OutputConfig(directory=tmp_path / "out"))
        sup = Supervisor(client, config)

        events = []
        sup.on_event(events.append)
        sup.run()

        start = next(e for e in events if e.type is EventType.RUN_START)
        assert start.data["log_path"].endswith("events.log")

    def test_a_fatal_error_is_recorded_before_it_propagates(self, tmp_path):
        client = _make_client()
        client.get_binary_info.side_effect = RuntimeError("Ghidra went away")
        config = KongConfig(output=OutputConfig(directory=tmp_path / "out"))

        with pytest.raises(RuntimeError):
            Supervisor(client, config).run()

        contents = (tmp_path / "out" / "events.log").read_text(encoding="utf-8")
        assert "Ghidra went away" in contents
        assert "Traceback (most recent call last)" in contents

    def test_the_listener_does_not_survive_the_run(self, tmp_path):
        """Otherwise a second run() on the same supervisor logs twice."""
        client = _make_client()
        config = KongConfig(output=OutputConfig(directory=tmp_path / "out"))
        sup = Supervisor(client, config)

        sup.run()

        assert sup._listeners == []


class TestDecompilationCacheInvalidation:
    """Cleanup applies types; synthesis and export must see them."""

    def _supervisor(self, tmp_path):
        client = _make_client()
        config = KongConfig(output=OutputConfig(directory=tmp_path / "out"))
        return client, Supervisor(client, config)

    def test_a_retyped_function_is_dropped_from_the_cache(self, tmp_path):
        client, sup = self._supervisor(tmp_path)
        client.get_callers.return_value = []
        sup._decompilation_cache = {0x1000: "stale", 0x2000: "fresh"}

        dropped = sup._invalidate_decompilation([0x1000])

        assert dropped == 1
        assert sup._decompilation_cache == {0x2000: "fresh"}

    def test_callers_are_dropped_too(self, tmp_path):
        """A caller's call site shows the parameters that were just retyped."""
        client, sup = self._supervisor(tmp_path)
        client.get_callers.return_value = [0x2000]
        sup._decompilation_cache = {0x1000: "stale", 0x2000: "also stale"}

        assert sup._invalidate_decompilation([0x1000]) == 2
        assert sup._decompilation_cache == {}

    def test_an_uncached_function_costs_nothing(self, tmp_path):
        client, sup = self._supervisor(tmp_path)
        client.get_callers.return_value = []

        assert sup._invalidate_decompilation([0x9999]) == 0

    def test_nothing_changed_means_no_ghidra_calls(self, tmp_path):
        client, sup = self._supervisor(tmp_path)

        assert sup._invalidate_decompilation([]) == 0
        client.get_callers.assert_not_called()

    def test_a_failing_caller_lookup_is_survivable(self, tmp_path):
        client, sup = self._supervisor(tmp_path)
        client.get_callers.side_effect = RuntimeError("Ghidra said no")
        sup._decompilation_cache = {0x1000: "stale"}

        assert sup._invalidate_decompilation([0x1000]) == 1

    def test_cleanup_invalidates_what_it_retyped(self, tmp_path, monkeypatch):
        import kong.agent.supervisor as supervisor_module

        client, sup = self._supervisor(tmp_path)
        client.get_callers.return_value = []
        sup._decompilation_cache = {0x1000: "stale", 0x3000: "untouched"}

        # One struct proposal, applied to the function at 0x1000.
        monkeypatch.setattr(
            type(sup.struct_accumulator), "proposal_count", property(lambda self: 1),
        )
        monkeypatch.setattr(sup.struct_accumulator, "unify", lambda: [])
        monkeypatch.setattr(
            supervisor_module, "apply_unified_structs", lambda c, u: [0x1000],
        )

        sup._run_cleanup()

        assert sup._decompilation_cache == {0x3000: "untouched"}

    def test_cleanup_invalidates_a_retried_signature(self, tmp_path):
        client, sup = self._supervisor(tmp_path)
        client.get_callers.return_value = []
        sup.results = {
            0x1000: FunctionResult(
                address=0x1000,
                original_name="FUN_00001000",
                name="parse",
                signature="int parse(char *)",
                signature_applied=False,
            ),
        }
        sup._decompilation_cache = {0x1000: "stale"}

        sup._run_cleanup()

        assert sup._decompilation_cache == {}
        assert sup.results[0x1000].signature_applied


class TestResume:
    """A run checkpoints itself so the next one need not pay twice."""

    def _run(self, tmp_path, resume=False, functions=None):
        funcs = functions or [_func(0x1000, "FUN_00001000", size=64)]
        client = _make_client(functions=funcs)
        client.get_decompilation.return_value = "void f(void) { return; }"
        client.get_xrefs_from.return_value = []
        client.get_function_info.return_value = funcs[0]
        client.list_custom_types.return_value = []
        client.get_callers.return_value = []

        mock_llm = MagicMock()
        mock_llm.analyze_function_batch.side_effect = _batch_response_with_addresses
        mock_llm.analyze_function.return_value = LLMResponse(
            name="", raw='{"globals":{},"structs":[],"name_refinements":{}}',
        )

        config = KongConfig(
            output=OutputConfig(directory=tmp_path / "out"), resume=resume,
        )
        sup = Supervisor(client, config, llm_client=mock_llm)
        sup.run()
        return sup, mock_llm

    def test_a_run_leaves_a_state_file(self, tmp_path):
        self._run(tmp_path)

        assert (tmp_path / "out" / "analysis_state.json").exists()

    def test_without_the_flag_everything_is_re_analyzed(self, tmp_path):
        self._run(tmp_path)
        _, second_llm = self._run(tmp_path)

        assert second_llm.analyze_function_batch.call_count == 1

    def test_with_the_flag_a_named_function_is_not_sent_again(self, tmp_path):
        first, _ = self._run(tmp_path)
        assert first.results[0x1000].name

        second, second_llm = self._run(tmp_path, resume=True)

        assert second.results[0x1000].name == first.results[0x1000].name
        assert second_llm.analyze_function_batch.call_count == 0

    def test_a_restored_name_is_written_back_to_ghidra(self, tmp_path):
        """The analysis pass that would normally write it is being skipped."""
        self._run(tmp_path)
        second, _ = self._run(tmp_path, resume=True)

        second.client.rename_function.assert_any_call(
            0x1000, second.results[0x1000].name,
        )

    def test_a_failed_function_is_retried(self, tmp_path):
        from kong.state.persistence import save_state

        save_state(
            {0x1000: FunctionResult(
                address=0x1000, original_name="FUN_00001000", error="HTTP 429",
            )},
            tmp_path / "out",
        )

        second, second_llm = self._run(tmp_path, resume=True)

        assert second_llm.analyze_function_batch.call_count == 1
        assert not second.results[0x1000].error

    def test_resuming_without_a_state_file_just_runs(self, tmp_path):
        sup, llm = self._run(tmp_path, resume=True)

        assert llm.analyze_function_batch.call_count == 1
        assert sup.results[0x1000].name

    def test_an_unwritable_state_directory_does_not_kill_the_run(
        self, tmp_path, monkeypatch,
    ):
        import kong.agent.supervisor as supervisor_module

        def explode(results, output_dir, binary=None):
            raise OSError("read-only file system")

        monkeypatch.setattr(supervisor_module, "save_state", explode)

        sup, _ = self._run(tmp_path)

        assert sup.results[0x1000].name



class TestClosingTheProgram:
    """Whatever the run has paid for has to be on disk when it stops."""

    def _sup(self, tmp_path, funcs=None):
        client = _make_client(functions=funcs or [_func(0x1000, "FUN_1000", size=64)])
        client.get_decompilation.return_value = "void f(void) { return; }"
        binary = tmp_path / "target.bin"
        binary.write_bytes(b"\x7fELF")
        client.binary_path = str(binary)

        mock_llm = MagicMock()
        mock_llm.model = "test-model"
        mock_llm.analyze_function_batch.side_effect = _batch_response_with_addresses
        mock_llm.analyze_function.return_value = LLMResponse(
            name="", raw='{"globals":{},"structs":[],"name_refinements":{}}',
        )
        config = KongConfig(output=OutputConfig(directory=tmp_path / "out"))
        return Supervisor(client, config, llm_client=mock_llm), binary

    def test_checkpoint_writes_what_is_there_so_far(self, tmp_path):
        from kong.state.persistence import load_state, state_path

        sup, _ = self._sup(tmp_path)
        sup.results[0x1000] = FunctionResult(
            address=0x1000, original_name="FUN_1000", name="parse", confidence=90,
        )

        sup.checkpoint()

        assert state_path(tmp_path / "out").exists()
        assert load_state(tmp_path / "out")[0x1000].name == "parse"

    def test_checkpoint_on_an_untouched_supervisor_is_harmless(self, tmp_path):
        sup, _ = self._sup(tmp_path)

        sup.checkpoint()  # no results, no binary opened, no crash

    def test_a_crash_still_leaves_a_checkpoint(self, tmp_path, monkeypatch):
        from kong.state.persistence import load_state

        sup, _ = self._sup(tmp_path)

        def explode() -> None:
            raise RuntimeError("Ghidra went away")

        monkeypatch.setattr(sup, "_run_cleanup", explode)

        with pytest.raises(RuntimeError):
            sup.run()

        assert load_state(tmp_path / "out")[0x1000].name

    def test_the_state_records_the_binary_it_describes(self, tmp_path):
        import json

        from kong.state.persistence import state_path

        sup, binary = self._sup(tmp_path)
        sup.run()

        stored = json.loads(state_path(tmp_path / "out").read_text())["binary"]
        assert stored["path"] == str(binary.resolve())
        assert stored["size"] == binary.stat().st_size

    def test_a_second_run_over_the_same_binary_resumes_on_its_own(self, tmp_path):
        """No --resume: picking up where the last run stopped is the default."""
        sup, binary = self._sup(tmp_path)
        sup.run()

        second, _ = self._sup(tmp_path)
        second.client.binary_path = str(binary)
        second.run()

        assert second.llm_client.analyze_function_batch.call_count == 0
        assert second.results[0x1000].name

    def test_another_binary_in_the_same_directory_is_analyzed_from_scratch(
        self, tmp_path,
    ):
        sup, _ = self._sup(tmp_path)
        sup.run()

        other = tmp_path / "other.bin"
        other.write_bytes(b"\x7fELF a different program entirely")
        second, _ = self._sup(tmp_path)
        second.client.binary_path = str(other)
        second.run()

        assert second.llm_client.analyze_function_batch.call_count == 1

    def test_fresh_ignores_the_saved_state(self, tmp_path):
        sup, binary = self._sup(tmp_path)
        sup.run()

        second, _ = self._sup(tmp_path)
        second.client.binary_path = str(binary)
        second.config.resume = False
        second.run()

        assert second.llm_client.analyze_function_batch.call_count == 1


class TestTwoPassAnalysis:
    """A fast model drafts, the strong model redoes what the draft got wrong."""

    DRAFT = "fast-model"
    STRONG = "strong-model"

    def _llm(self, draft_confidence=40, refined_name="parse_http_header"):
        llm = MagicMock()
        llm.model = self.STRONG

        def batch(prompt, **kwargs):
            import re

            addresses = re.findall(r"### (0x[0-9a-fA-F]+):", prompt)
            return [
                LLMResponse(
                    name=f"draft_{index}",
                    confidence=draft_confidence,
                    classification="utility",
                    address=int(address, 16),
                )
                for index, address in enumerate(addresses)
            ]

        llm.analyze_function_batch.side_effect = batch
        # Serves both the second pass and the synthesis call: the analyzer
        # reads the parsed fields, the synthesizer reads the raw payload.
        llm.analyze_function.return_value = LLMResponse(
            name=refined_name,
            confidence=95,
            raw='{"globals":{},"structs":[],"name_refinements":{}}',
        )
        return llm

    def _config(self, tmp_path, **llm_kwargs):
        return KongConfig(
            llm=LLMConfig(model=self.STRONG, draft_model=self.DRAFT, **llm_kwargs),
            output=OutputConfig(directory=tmp_path / "out"),
        )

    def _run(self, tmp_path, funcs, llm=None, **llm_kwargs):
        client = _make_client(functions=funcs)
        client.get_decompilation.return_value = "void f(void) { return; }"
        sup = Supervisor(
            client, self._config(tmp_path, **llm_kwargs),
            llm_client=llm or self._llm(),
        )
        sup.run()
        return sup

    @staticmethod
    def _batch_models(llm):
        return [
            call.kwargs["model"] for call in llm.analyze_function_batch.call_args_list
        ]

    def test_the_draft_pass_runs_on_the_draft_model(self, tmp_path):
        llm = self._llm()
        self._run(tmp_path, [_func(0x1000, "FUN_1000", size=64)], llm=llm)

        assert self._batch_models(llm) == [self.DRAFT]

    def test_a_large_function_never_sees_the_draft_model(self, tmp_path):
        llm = self._llm()
        funcs = [
            _func(0x1000, "FUN_1000", size=64, cls=FunctionClassification.SMALL),
            _func(0x2000, "FUN_2000", size=4000, cls=FunctionClassification.LARGE),
        ]
        self._run(tmp_path, funcs, llm=llm)

        by_model = {
            call.kwargs["model"]: call.args[0]
            for call in llm.analyze_function_batch.call_args_list
        }
        assert "0x00001000" in by_model[self.DRAFT]
        assert "0x00002000" in by_model[self.STRONG]
        assert "0x00002000" not in by_model[self.DRAFT]

    def test_a_weak_draft_is_re_analyzed_by_the_strong_model(self, tmp_path):
        llm = self._llm(draft_confidence=40)
        sup = self._run(tmp_path, [_func(0x1000, "FUN_1000", size=64)], llm=llm)

        result = sup.results[0x1000]
        assert result.name == "parse_http_header"
        assert result.refined
        assert result.model == self.STRONG
        assert result.llm_calls == 2  # the draft call was paid for too

    def test_a_confident_draft_is_left_alone(self, tmp_path):
        llm = self._llm(draft_confidence=95)
        sup = self._run(tmp_path, [_func(0x1000, "FUN_1000", size=64)], llm=llm)

        result = sup.results[0x1000]
        assert result.name == "draft_0"
        assert not result.refined
        assert result.model == self.DRAFT

    def test_a_name_that_says_nothing_is_re_analyzed_however_confident(self, tmp_path):
        llm = self._llm()
        llm.analyze_function_batch.side_effect = lambda prompt, **kwargs: [
            LLMResponse(name="helper", confidence=100, address=0x1000),
        ]
        sup = self._run(tmp_path, [_func(0x1000, "FUN_1000", size=64)], llm=llm)

        assert sup.results[0x1000].refined

    def test_the_second_pass_counts_the_function_once(self, tmp_path):
        sup = self._run(tmp_path, [_func(0x1000, "FUN_1000", size=64)])

        assert sup.stats.analyzed == 1
        assert sup.stats.renamed == 1
        assert sup.stats.high_confidence == 1
        assert sup.stats.low_confidence == 0

    def test_a_second_pass_that_answers_nothing_keeps_the_draft_name(self, tmp_path):
        llm = self._llm()
        llm.analyze_function.return_value = LLMResponse(
            name="", raw='{"globals":{},"structs":[],"name_refinements":{}}',
        )
        sup = self._run(tmp_path, [_func(0x1000, "FUN_1000", size=64)], llm=llm)

        result = sup.results[0x1000]
        assert result.name == "draft_0"
        assert not result.refined

    def test_a_second_pass_that_raises_keeps_the_draft_name(self, tmp_path):
        llm = self._llm()
        calls = {"n": 0}

        def analyze(prompt, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:  # the refinement call; synthesis comes after
                raise RuntimeError("connection reset")
            return LLMResponse(
                name="", raw='{"globals":{},"structs":[],"name_refinements":{}}',
            )

        llm.analyze_function.side_effect = analyze
        sup = self._run(tmp_path, [_func(0x1000, "FUN_1000", size=64)], llm=llm)

        assert sup.results[0x1000].name == "draft_0"
        assert sup.stats.errors == 0

    def test_the_second_pass_re_reads_the_decompilation(self, tmp_path):
        """What is cached is the text the draft saw, before it renamed anything."""
        llm = self._llm()
        client = _make_client(functions=[_func(0x1000, "FUN_1000", size=64)])
        client.get_decompilation.return_value = "void f(void) { return; }"
        sup = Supervisor(client, self._config(tmp_path), llm_client=llm)
        sup.run()

        assert client.get_decompilation.call_count >= 2

    def test_a_draft_name_is_kept_out_of_the_prompt_preamble(self, tmp_path):
        sup = Supervisor(_make_client(), self._config(tmp_path), llm_client=self._llm())
        sup.binary_info = _make_client().get_binary_info()
        sup.results = {
            0x1000: FunctionResult(
                address=0x1000, original_name="FUN_1000", name="unsure_name",
                confidence=30, model=self.DRAFT,
            ),
            0x2000: FunctionResult(
                address=0x2000, original_name="FUN_2000", name="confident_name",
                confidence=95, model=self.DRAFT,
            ),
        }

        prompt = sup._build_chunk_prompt([])

        assert "confident_name" in prompt
        assert "unsure_name" not in prompt

    def test_a_result_from_the_strong_model_is_never_a_candidate(self, tmp_path):
        sup = Supervisor(_make_client(), self._config(tmp_path), llm_client=self._llm())
        sup.results = {
            0x1000: FunctionResult(
                address=0x1000, original_name="FUN_1000", name="weak_name",
                confidence=10, model=self.STRONG,
            ),
        }

        assert sup._refinement_candidates() == []

    def test_without_a_draft_model_nothing_changes(self, tmp_path):
        llm = self._llm(draft_confidence=10)
        client = _make_client(functions=[_func(0x1000, "FUN_1000", size=64)])
        client.get_decompilation.return_value = "void f(void) { return; }"
        config = KongConfig(
            llm=LLMConfig(model=self.STRONG),
            output=OutputConfig(directory=tmp_path / "out"),
        )
        sup = Supervisor(client, config, llm_client=llm)
        sup.run()

        assert self._batch_models(llm) == [self.STRONG]
        assert sup.results[0x1000].name == "draft_0"
        assert not sup.results[0x1000].refined

    def test_a_draft_model_equal_to_the_main_one_is_a_single_pass(self, tmp_path):
        llm = self._llm(draft_confidence=10)
        client = _make_client(functions=[_func(0x1000, "FUN_1000", size=64)])
        client.get_decompilation.return_value = "void f(void) { return; }"
        config = KongConfig(
            llm=LLMConfig(model=self.STRONG, draft_model=self.STRONG),
            output=OutputConfig(directory=tmp_path / "out"),
        )
        sup = Supervisor(client, config, llm_client=llm)
        sup.run()

        assert sup._draft_model == ""
        assert self._batch_models(llm) == [self.STRONG]
        assert not sup.results[0x1000].refined

    def test_a_drafted_function_is_refined_after_a_resume(self, tmp_path):
        from kong.state.persistence import save_state

        save_state(
            {0x1000: FunctionResult(
                address=0x1000, original_name="FUN_1000", name="draft_0",
                confidence=30, model=self.DRAFT, llm_calls=1,
            )},
            tmp_path / "out",
        )

        llm = self._llm()
        client = _make_client(functions=[_func(0x1000, "FUN_1000", size=64)])
        client.get_decompilation.return_value = "void f(void) { return; }"
        config = KongConfig(
            llm=LLMConfig(model=self.STRONG),
            output=OutputConfig(directory=tmp_path / "out"),
            resume=True,
        )
        sup = Supervisor(client, config, llm_client=llm)
        sup.run()

        result = sup.results[0x1000]
        assert result.refined
        assert result.name == "parse_http_header"
        assert llm.analyze_function_batch.call_count == 0

    def test_limits_follow_the_model_of_the_pass(self, tmp_path):
        sup = Supervisor(_make_client(), self._config(tmp_path), llm_client=self._llm())

        assert sup._limits_for("claude-opus-5").max_prompt_chars == 900_000
        assert sup._limits_for("gpt-4o").max_prompt_chars == 350_000

    def test_a_local_endpoint_caps_both_models(self, tmp_path):
        config = KongConfig(
            llm=LLMConfig(
                provider=LLMProvider.CUSTOM,
                model="qwen2.5-coder-32b",
                draft_model="qwen2.5-coder-7b",
                max_prompt_chars=30_000,
            ),
            output=OutputConfig(directory=tmp_path / "out"),
        )
        sup = Supervisor(_make_client(), config, llm_client=self._llm())

        assert sup._limits_for("qwen2.5-coder-7b").max_prompt_chars == 30_000
        assert sup._limits_for("qwen2.5-coder-32b").max_prompt_chars == 30_000


class TestCoherenceReview:
    """The manual pass that reads the finished results against each other."""

    CALLER_CODE = """
void FUN_00402000(void)

{
  parse_header(buf, 12, 0, 1);
  return;
}
"""

    def _config(self, tmp_path):
        return KongConfig(
            llm=LLMConfig(model="claude-opus-5"),
            output=OutputConfig(directory=tmp_path / "out"),
        )

    def _llm(self, raw: str = '{"resolutions": []}'):
        llm = MagicMock()
        llm.model = "claude-opus-5"
        llm.analyze_function.return_value = LLMResponse(name="", raw=raw)
        return llm

    def _supervisor(self, tmp_path, results, llm=None, client=None):
        client = client or _make_client()
        sup = Supervisor(client, self._config(tmp_path), llm_client=llm)
        for result in results:
            sup._store_result(result.address, result)
            sup.stats.record_result(result)
        return sup

    @staticmethod
    def _result(addr, name, signature="", confidence=90):
        return FunctionResult(
            address=addr,
            original_name=f"FUN_{addr:08x}",
            name=name,
            signature=signature,
            confidence=confidence,
        )

    def test_an_analysis_that_agrees_with_itself_reports_nothing(self, tmp_path):
        sup = self._supervisor(tmp_path, [
            self._result(0x401000, "parse_header"),
            self._result(0x402000, "read_message"),
        ], llm=self._llm())

        events = []
        sup.on_event(events.append)
        report = sup.review_coherence()

        assert report.conflicts == []
        assert report.functions_checked == 2
        checked = [e for e in events if e.type == EventType.COHERENCE_CHECKED]
        assert checked[0].data == {"checked": 2, "conflicts": 0}
        assert sup.llm_client.analyze_function.call_count == 0

    def test_a_contradiction_is_found_and_reported(self, tmp_path):
        sup = self._supervisor(tmp_path, [
            self._result(0x401000, "parse_header"),
            self._result(0x402000, "parse_header"),
        ], llm=self._llm())

        events = []
        sup.on_event(events.append)
        report = sup.review_coherence()

        assert [c.kind.value for c in report.conflicts] == ["duplicate_name"]
        conflict_events = [
            e for e in events if e.type == EventType.COHERENCE_CONFLICT
        ]
        assert len(conflict_events) == 1
        assert conflict_events[0].data["addresses"] == [0x401000, 0x402000]

    def test_the_call_graph_comes_from_ghidra_without_a_queue(self, tmp_path):
        """A review can run on results restored from a state file."""
        client = _make_client()
        client.get_decompilation.side_effect = (
            lambda addr: self.CALLER_CODE if addr == 0x402000 else "void f(void) {}"
        )
        client.get_callers.side_effect = (
            lambda addr: [0x402000] if addr == 0x401000 else []
        )
        sup = self._supervisor(
            tmp_path,
            [
                self._result(
                    0x401000, "parse_header", "int parse_header(char *b, int n)",
                ),
                self._result(0x402000, "read_message"),
            ],
            llm=self._llm(),
            client=client,
        )

        report = sup.review_coherence()

        assert [c.kind.value for c in report.conflicts] == ["argument_count"]

    def test_a_resolution_reaches_ghidra_and_the_results(self, tmp_path):
        client = _make_client()
        sup = self._supervisor(tmp_path, [
            self._result(0x401000, "parse_header"),
            self._result(0x402000, "parse_header"),
        ], client=client)
        conflict_id = "duplicate_name:00401000-00402000"
        sup.llm_client = self._llm(json.dumps({"resolutions": [{
            "conflict": conflict_id,
            "verdict": "resolved",
            "explanation": "the second one writes rather than parses",
            "changes": [{
                "address": "0x00402000",
                "name": "write_header",
                "confidence": 60,
            }],
        }]}))

        events = []
        sup.on_event(events.append)
        report = sup.review_coherence()

        client.rename_function.assert_called_once_with(0x402000, "write_header")
        assert sup.results[0x402000].name == "write_header"
        assert sup.results[0x402000].confidence == 60
        assert report.applied
        resolved = [e for e in events if e.type == EventType.COHERENCE_RESOLVED]
        assert resolved[0].data["id"] == conflict_id

    def test_a_lowered_score_moves_the_confidence_buckets(self, tmp_path):
        sup = self._supervisor(tmp_path, [
            self._result(0x401000, "parse_header"),
            self._result(0x402000, "parse_header"),
        ])
        assert sup.stats.high_confidence == 2
        sup.llm_client = self._llm(json.dumps({"resolutions": [{
            "conflict": "duplicate_name:00401000-00402000",
            "changes": [{"address": "0x00402000", "confidence": 40}],
        }]}))

        sup.review_coherence()

        assert sup.stats.high_confidence == 1
        assert sup.stats.low_confidence == 1

    def test_a_rename_ghidra_refuses_is_not_recorded_as_done(self, tmp_path):
        client = _make_client()
        client.rename_function.side_effect = RuntimeError("name already in use")
        sup = self._supervisor(tmp_path, [
            self._result(0x401000, "parse_header"),
            self._result(0x402000, "parse_header"),
        ], client=client)
        sup.llm_client = self._llm(json.dumps({"resolutions": [{
            "conflict": "duplicate_name:00401000-00402000",
            "changes": [{"address": "0x00402000", "name": "write_header"}],
        }]}))

        report = sup.review_coherence()

        assert sup.results[0x402000].name == "parse_header"
        assert report.applied == []

    def test_without_a_model_the_contradictions_are_only_reported(self, tmp_path):
        sup = self._supervisor(tmp_path, [
            self._result(0x401000, "parse_header"),
            self._result(0x402000, "parse_header"),
        ], llm=None)

        events = []
        sup.on_event(events.append)
        report = sup.review_coherence()

        assert len(report.conflicts) == 1
        assert report.applied == []
        done = [e for e in events if e.type == EventType.PHASE_COMPLETE]
        assert "no model is configured" in done[-1].message

    def test_an_unreadable_answer_is_reported_rather_than_swallowed(self, tmp_path):
        """No events.log outside a run, so it has to reach the window."""
        sup = self._supervisor(tmp_path, [
            self._result(0x401000, "parse_header"),
            self._result(0x402000, "parse_header"),
        ], llm=self._llm("I had a think about it and could not decide."))

        events = []
        sup.on_event(events.append)
        report = sup.review_coherence()

        assert report.applied == []
        assert any(
            "could not be read" in e.message
            for e in events if e.type == EventType.PHASE_COMPLETE
        )

    def test_an_api_failure_leaves_the_analysis_as_it_was(self, tmp_path):
        sup = self._supervisor(tmp_path, [
            self._result(0x401000, "parse_header"),
            self._result(0x402000, "parse_header"),
        ], llm=self._llm())
        sup.llm_client.analyze_function.side_effect = RuntimeError("503")

        report = sup.review_coherence()

        assert report.applied == []
        assert sup.results[0x402000].name == "parse_header"

    def test_the_findings_are_left_on_disk(self, tmp_path):
        sup = self._supervisor(tmp_path, [
            self._result(0x401000, "parse_header"),
            self._result(0x402000, "parse_header"),
        ], llm=self._llm())

        events = []
        sup.on_event(events.append)
        sup.review_coherence()

        path = tmp_path / "out" / REPORT_NAME
        document = json.loads(path.read_text(encoding="utf-8"))
        assert document["conflicts_found"] == 1
        assert document["conflicts"][0]["kind"] == "duplicate_name"
        assert any(
            e.type == EventType.EXPORT_FILE and e.data.get("format") == "coherence"
            for e in events
        )

    def test_the_names_it_fixed_survive_the_session(self, tmp_path):
        """The state file is what a later run resumes from."""
        sup = self._supervisor(tmp_path, [
            self._result(0x401000, "parse_header"),
            self._result(0x402000, "parse_header"),
        ])
        sup.llm_client = self._llm(json.dumps({"resolutions": [{
            "conflict": "duplicate_name:00401000-00402000",
            "changes": [{"address": "0x00402000", "name": "write_header"}],
        }]}))

        sup.review_coherence()

        from kong.state.persistence import load_state

        saved = load_state(tmp_path / "out")
        assert saved[0x402000].name == "write_header"

    def test_only_one_review_writes_to_ghidra_at_a_time(self, tmp_path):
        sup = self._supervisor(tmp_path, [
            self._result(0x401000, "parse_header"),
            self._result(0x402000, "parse_header"),
        ], llm=self._llm())

        sup._coherence_lock.acquire()
        try:
            events = []
            sup.on_event(events.append)
            report = sup.review_coherence()
        finally:
            sup._coherence_lock.release()

        assert report.conflicts == []
        assert events == []

    def test_a_backlog_is_capped_and_says_what_it_left(self, tmp_path):
        sup = self._supervisor(tmp_path, [
            self._result(0x401000, "parse_header"),
            self._result(0x402000, "parse_header"),
            self._result(0x403000, "write_body", "int flush_body(void)"),
        ], llm=self._llm())

        events = []
        sup.on_event(events.append)
        report = sup.review_coherence(limit=1)

        assert len(report.conflicts) == 2
        assert report.unreviewed == 1
        assert sup.llm_client.analyze_function.call_count == 1
        done = [e for e in events if e.type == EventType.PHASE_COMPLETE]
        assert "left for a later pass" in done[-1].message


class TestStagedRun:
    """The draft and the finishing pass, started separately.

    A binary of a few thousand functions is a long unattended draft and a
    short expensive second pass. Splitting them is what lets the second one be
    a decision rather than a side effect of the first.
    """

    DRAFT = "fast-model"
    STRONG = "strong-model"

    def _llm(self, draft_confidence=40, refined_name="parse_http_header"):
        llm = MagicMock()
        llm.model = self.STRONG

        def batch(prompt, **kwargs):
            return [
                LLMResponse(
                    name=f"draft_{index}",
                    confidence=draft_confidence,
                    classification="utility",
                    address=int(address, 16),
                )
                for index, address in enumerate(
                    re.findall(r"### (0x[0-9a-fA-F]+):", prompt)
                )
            ]

        llm.analyze_function_batch.side_effect = batch
        llm.analyze_function.return_value = LLMResponse(
            name=refined_name,
            confidence=95,
            raw='{"globals":{},"structs":[],"name_refinements":{}}',
        )
        return llm

    def _run(self, tmp_path, funcs, llm, *, stage=RunStage.DRAFT, draft_model=DRAFT):
        client = _make_client(functions=funcs)
        client.get_decompilation.return_value = "void f(void) { return; }"
        config = KongConfig(
            llm=LLMConfig(model=self.STRONG, draft_model=draft_model),
            output=OutputConfig(directory=tmp_path / "out"),
            stage=stage,
        )
        sup = Supervisor(client, config, llm_client=llm)
        sup.run()
        return sup

    @staticmethod
    def _batch_models(llm):
        return [
            call.kwargs["model"] for call in llm.analyze_function_batch.call_args_list
        ]

    # ------------------------------------------------------------- the draft

    def test_the_draft_stage_sends_nothing_to_the_primary_model(self, tmp_path):
        llm = self._llm(draft_confidence=10)
        funcs = [
            _func(0x1000, "FUN_1000", size=64, cls=FunctionClassification.SMALL),
            _func(0x2000, "FUN_2000", size=4000, cls=FunctionClassification.LARGE),
        ]
        self._run(tmp_path, funcs, llm)

        assert set(self._batch_models(llm)) == {self.DRAFT}
        assert llm.analyze_function.call_args_list == []

    def test_a_large_function_is_drafted_too_rather_than_upgraded(self, tmp_path):
        """Outside a draft stage this one would go straight to the big model."""
        llm = self._llm()
        funcs = [_func(0x2000, "FUN_2000", size=4000, cls=FunctionClassification.LARGE)]
        self._run(tmp_path, funcs, llm)

        assert self._batch_models(llm) == [self.DRAFT]

    def test_obfuscated_code_is_held_back_instead_of_drafted(
        self, tmp_path, monkeypatch,
    ):
        from kong.agent.deobfuscator import ObfuscationType

        monkeypatch.setattr(
            "kong.agent.supervisor.classify_obfuscation",
            lambda _decompiled: [ObfuscationType.STRING_ENCRYPTION],
        )
        llm = self._llm(draft_confidence=95)
        sup = self._run(tmp_path, [_func(0x1000, "FUN_1000", size=64)], llm)

        assert llm.analyze_function_batch.call_args_list == []
        assert llm.analyze_function.call_args_list == []
        assert 0x1000 not in sup.results
        assert "obfuscated" in sup._deferred[0x1000]
        assert sup.pending_finish == 1

    def test_the_draft_stage_leaves_synthesis_for_the_finishing_pass(self, tmp_path):
        llm = self._llm(draft_confidence=95)
        client = _make_client(functions=[_func(0x1000, "FUN_1000", size=64)])
        client.get_decompilation.return_value = "void f(void) { return; }"
        config = KongConfig(
            llm=LLMConfig(model=self.STRONG, draft_model=self.DRAFT),
            output=OutputConfig(directory=tmp_path / "out"),
            stage=RunStage.DRAFT,
        )
        sup = Supervisor(client, config, llm_client=llm)
        events = []
        sup.on_event(events.append)
        sup.run()

        phases = [e.phase for e in events if e.type == EventType.PHASE_START]
        assert Phase.SYNTHESIS not in phases
        assert Phase.EXPORT in phases

    def test_the_draft_stage_says_what_is_left(self, tmp_path):
        llm = self._llm(draft_confidence=10)
        client = _make_client(functions=[_func(0x1000, "FUN_1000", size=64)])
        client.get_decompilation.return_value = "void f(void) { return; }"
        config = KongConfig(
            llm=LLMConfig(model=self.STRONG, draft_model=self.DRAFT),
            output=OutputConfig(directory=tmp_path / "out"),
            stage=RunStage.DRAFT,
        )
        sup = Supervisor(client, config, llm_client=llm)
        events = []
        sup.on_event(events.append)
        sup.run()

        complete = [e for e in events if e.type == EventType.RUN_COMPLETE][-1]
        assert complete.data["pending_finish"] == 1
        assert "finishing pass" in complete.message

    # ----------------------------------------------------- the finishing pass

    def test_the_finishing_pass_redoes_what_fell_under_the_threshold(self, tmp_path):
        llm = self._llm(draft_confidence=40)
        sup = self._run(tmp_path, [_func(0x1000, "FUN_1000", size=64)], llm)
        assert sup.results[0x1000].name == "draft_0"

        improved = sup.run_finishing_pass()

        assert improved == 1
        result = sup.results[0x1000]
        assert result.name == "parse_http_header"
        assert result.refined
        assert result.llm_calls == 2  # the draft call was paid for too
        assert sup.pending_finish == 0

    def test_a_confident_draft_is_left_alone_by_the_finishing_pass(self, tmp_path):
        llm = self._llm(draft_confidence=95)
        sup = self._run(tmp_path, [_func(0x1000, "FUN_1000", size=64)], llm)

        assert sup.pending_finish == 0
        assert sup.run_finishing_pass() == 0
        assert sup.results[0x1000].name == "draft_0"

    def test_the_finishing_pass_picks_up_what_was_never_analyzed(
        self, tmp_path, monkeypatch,
    ):
        from kong.agent.deobfuscator import ObfuscationType

        monkeypatch.setattr(
            "kong.agent.supervisor.classify_obfuscation",
            lambda _decompiled: [ObfuscationType.STRING_ENCRYPTION],
        )
        llm = self._llm()
        sup = self._run(tmp_path, [_func(0x1000, "FUN_1000", size=64)], llm)

        assert sup.run_finishing_pass() == 1
        assert sup.results[0x1000].name == "parse_http_header"
        assert sup._deferred == {}

    def test_the_finishing_pass_redoes_a_weak_answer_from_the_primary_model(
        self, tmp_path,
    ):
        """No draft model: the two stages are the batch pass and this one."""
        llm = self._llm(draft_confidence=20)
        sup = self._run(tmp_path, [_func(0x1000, "FUN_1000", size=64)], llm,
                        draft_model=None)
        assert sup.results[0x1000].model == self.STRONG
        # The automatic pass would leave this alone; asked for, it does not.
        assert sup._refinement_candidates() == []

        assert sup.run_finishing_pass() == 1
        assert sup.results[0x1000].name == "parse_http_header"

    def test_a_closed_program_stops_the_finishing_pass_before_the_first_call(
        self, tmp_path,
    ):
        """Whoever owns the client may have released it. Say so once."""
        llm = self._llm(draft_confidence=40)
        sup = self._run(tmp_path, [_func(0x1000, "FUN_1000", size=64)], llm)
        assert sup.pending_finish == 1
        sup.client.is_open = False
        events = []
        sup.on_event(events.append)

        assert sup.run_finishing_pass() == 0

        assert sup.results[0x1000].name == "draft_0"  # the draft is untouched
        messages = [e.message for e in events]
        assert any("open in Ghidra" in m for m in messages)
        assert not any("Re-analyzing" in m for m in messages)

    def test_the_finishing_pass_re_exports(self, tmp_path):
        llm = self._llm(draft_confidence=40)
        sup = self._run(tmp_path, [_func(0x1000, "FUN_1000", size=64)], llm)
        events = []
        sup.on_event(events.append)

        sup.run_finishing_pass()

        phases = [e.phase for e in events if e.type == EventType.PHASE_START]
        assert Phase.SYNTHESIS in phases
        assert Phase.EXPORT in phases

    def test_a_full_run_keeps_the_finishing_pass_off_unanalyzed_functions(
        self, tmp_path,
    ):
        """Mid-run, an unnamed function is one the batch pass has not reached."""
        llm = self._llm(draft_confidence=95)
        client = _make_client(functions=[_func(0x1000, "FUN_1000", size=64)])
        config = KongConfig(
            llm=LLMConfig(model=self.STRONG),
            output=OutputConfig(directory=tmp_path / "out"),
        )
        sup = Supervisor(client, config, llm_client=llm)
        sup._run_triage()

        assert sup.pending_finish == 0

    # --------------------------------------------------------- as its own run

    def test_a_finish_run_reads_the_draft_off_disk(self, tmp_path):
        llm = self._llm(draft_confidence=40)
        funcs = [_func(0x1000, "FUN_1000", size=64)]
        self._run(tmp_path, funcs, llm)

        second = self._llm()
        client = _make_client(functions=funcs)
        client.get_decompilation.return_value = "void f(void) { return; }"
        config = KongConfig(
            llm=LLMConfig(model=self.STRONG, draft_model=self.DRAFT),
            output=OutputConfig(directory=tmp_path / "out"),
            stage=RunStage.FINISH,
        )
        sup = Supervisor(client, config, llm_client=second)
        sup.run()

        assert second.analyze_function_batch.call_args_list == []
        assert sup.results[0x1000].name == "parse_http_header"
        assert sup.results[0x1000].refined

    def test_a_finish_run_without_a_draft_says_so(self, tmp_path):
        client = _make_client(functions=[_func(0x1000, "FUN_1000", size=64)])
        config = KongConfig(
            llm=LLMConfig(model=self.STRONG),
            output=OutputConfig(directory=tmp_path / "out"),
            stage=RunStage.FINISH,
        )
        sup = Supervisor(client, config, llm_client=self._llm())

        with pytest.raises(ValueError, match="No saved analysis"):
            sup.run()

    def test_a_finish_run_retries_what_the_draft_failed(self, tmp_path):
        """A failed function is dropped by a plain resume; here it is the point."""
        funcs = [_func(0x1000, "FUN_1000", size=64)]
        broken = self._llm()
        broken.analyze_function_batch.side_effect = RuntimeError("rate limited")
        self._run(tmp_path, funcs, broken)

        second = self._llm()
        client = _make_client(functions=funcs)
        client.get_decompilation.return_value = "void f(void) { return; }"
        config = KongConfig(
            llm=LLMConfig(model=self.STRONG, draft_model=self.DRAFT),
            output=OutputConfig(directory=tmp_path / "out"),
            stage=RunStage.FINISH,
        )
        sup = Supervisor(client, config, llm_client=second)
        sup.run()

        assert sup.results[0x1000].name == "parse_http_header"
        assert sup.stats.errors == 0
