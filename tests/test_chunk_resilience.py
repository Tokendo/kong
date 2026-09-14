"""Chunk calls: one failure must not cost every function in the chunk.

A chunk is one request covering many functions, so a single transient error
used to mark eight or sixteen functions failed at once — which is why failures
arrived in bursts. And the calls themselves ran strictly one after another,
although the work queue is ordered bottom-up precisely so that functions at the
same depth do not depend on each other.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

from kong.agent.analyzer import LLMResponse
from kong.agent.queue import WorkItem
from kong.config import AnalysisConfig, KongConfig, LLMConfig, LLMProvider, OutputConfig
from kong.ghidra.types import BinaryInfo, FunctionClassification, FunctionInfo
from kong.llm.limits import ModelLimits


def _func(addr, name="FUN_x", size=100):
    return FunctionInfo(
        address=addr, name=name or f"FUN_{addr:08x}", size=size,
        classification=FunctionClassification.MEDIUM,
    )


def _chunk(addresses):
    return [
        (WorkItem(function=_func(a, f"FUN_{a:08x}")), f"void f_{a:x}(void) {{}}")
        for a in addresses
    ]


def _limits():
    return ModelLimits(
        max_prompt_chars=100_000, max_chunk_functions=8, max_output_tokens=4096,
    )


def _supervisor(tmp_path, llm=None, **analysis):
    from kong.agent.supervisor import Supervisor

    config = KongConfig(
        llm=LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4o"),
        output=OutputConfig(directory=tmp_path / "out"),
        analysis=AnalysisConfig(**analysis),
    )
    sup = Supervisor(MagicMock(), config, llm_client=llm or MagicMock())
    sup.binary_info = BinaryInfo(
        arch="x86", format="PE", endianness="little",
        word_size=4, compiler="windows", name="FA18.exe",
    )
    return sup


class TestChunkRecovery:
    def test_a_transient_failure_is_retried_before_anything_is_lost(self, tmp_path):
        llm = MagicMock()
        llm.analyze_function_batch.side_effect = [
            RuntimeError("500 Internal Server Error"),
            [LLMResponse(name="ok", address=0x1000)],
        ]
        sup = _supervisor(tmp_path, llm, chunk_attempts=2)

        responses, error = sup._send_one("prompt", "gpt-4o", _limits())

        assert error is None
        assert [r.name for r in responses] == ["ok"]

    def test_a_persistent_failure_is_reported_not_raised(self, tmp_path):
        llm = MagicMock()
        llm.analyze_function_batch.side_effect = RuntimeError("bad request")
        sup = _supervisor(tmp_path, llm, chunk_attempts=2)

        responses, error = sup._send_one("prompt", "gpt-4o", _limits())

        assert responses == []
        assert "bad request" in error
        assert llm.analyze_function_batch.call_count == 2

    def test_a_failed_chunk_is_halved_and_most_of_it_survives(self, tmp_path):
        """The point of the whole thing: one bad function, not eight."""
        poison = 0x4000
        calls = []

        def batch(prompt, **kwargs):
            calls.append(prompt)
            if f"0x{poison:08x}" in prompt:
                raise RuntimeError("context length exceeded")
            addresses = [a for a in (0x1000, 0x2000, 0x3000) if f"0x{a:08x}" in prompt]
            return [LLMResponse(name=f"f_{a:x}", address=a) for a in addresses]

        llm = MagicMock()
        llm.analyze_function_batch.side_effect = batch
        sup = _supervisor(tmp_path, llm, chunk_attempts=1)
        chunk = _chunk([0x1000, 0x2000, 0x3000, poison])

        responses, failures = sup._recover_chunk(chunk, "gpt-4o", _limits(), "boom")

        assert sorted(r.address for r in responses) == [0x1000, 0x2000, 0x3000]
        assert list(failures) == [poison]
        assert "context length exceeded" in failures[poison]

    def test_a_single_function_that_cannot_be_split_is_the_one_that_fails(
        self, tmp_path,
    ):
        llm = MagicMock()
        llm.analyze_function_batch.side_effect = RuntimeError("nope")
        sup = _supervisor(tmp_path, llm, chunk_attempts=1)

        responses, failures = sup._recover_chunk(
            _chunk([0x1000]), "gpt-4o", _limits(), "nope",
        )

        assert responses == []
        assert list(failures) == [0x1000]

    def test_a_chunk_that_recovers_whole_reports_no_failures(self, tmp_path):
        llm = MagicMock()
        llm.analyze_function_batch.side_effect = lambda prompt, **kw: [
            LLMResponse(name="f", address=a)
            for a in (0x1000, 0x2000)
            if f"0x{a:08x}" in prompt
        ]
        sup = _supervisor(tmp_path, llm, chunk_attempts=1)

        _, failures = sup._recover_chunk(
            _chunk([0x1000, 0x2000]), "gpt-4o", _limits(), "transient",
        )

        assert failures == {}


class TestConcurrency:
    def test_a_local_endpoint_keeps_to_one_call(self, tmp_path):
        from kong.agent.supervisor import Supervisor

        config = KongConfig(
            llm=LLMConfig(
                provider=LLMProvider.CUSTOM, model="local",
                base_url="http://127.0.0.1:8080/v1",
            ),
            output=OutputConfig(directory=tmp_path / "out"),
        )
        sup = Supervisor(MagicMock(), config, llm_client=MagicMock())

        assert sup._chunk_concurrency() == 1

    def test_a_hosted_api_sends_several(self, tmp_path):
        assert _supervisor(tmp_path)._chunk_concurrency() > 1

    def test_the_setting_wins_over_the_provider(self, tmp_path):
        assert _supervisor(tmp_path, chunk_concurrency=7)._chunk_concurrency() == 7

    def test_zero_is_read_as_one_rather_than_no_calls_at_all(self, tmp_path):
        assert _supervisor(tmp_path, chunk_concurrency=0)._chunk_concurrency() == 1

    def test_calls_really_do_overlap(self, tmp_path):
        """Four calls that each sleep must not take four times as long."""
        in_flight = []
        peak = [0]
        guard = threading.Lock()

        def batch(prompt, **kwargs):
            with guard:
                in_flight.append(1)
                peak[0] = max(peak[0], len(in_flight))
            time.sleep(0.05)
            with guard:
                in_flight.pop()
            return []

        llm = MagicMock()
        llm.analyze_function_batch.side_effect = batch
        sup = _supervisor(tmp_path, llm, chunk_concurrency=4)

        sup._send_prompts(["a", "b", "c", "d"], "gpt-4o", _limits())

        assert peak[0] > 1

    def test_answers_come_back_in_the_order_they_were_sent(self, tmp_path):
        """However the endpoint reorders them, events and state must not."""
        def batch(prompt, **kwargs):
            # The later prompts answer first.
            time.sleep(0.05 if prompt == "first" else 0.0)
            return [LLMResponse(name=prompt, address=0x1000)]

        llm = MagicMock()
        llm.analyze_function_batch.side_effect = batch
        sup = _supervisor(tmp_path, llm, chunk_concurrency=4)

        sent = sup._send_prompts(["first", "second", "third"], "gpt-4o", _limits())

        assert [responses[0].name for responses, _ in sent] == [
            "first", "second", "third",
        ]

    def test_one_prompt_needs_no_pool(self, tmp_path):
        llm = MagicMock()
        llm.analyze_function_batch.return_value = [
            LLMResponse(name="only", address=0x1000)
        ]
        sup = _supervisor(tmp_path, llm, chunk_concurrency=4)

        sent = sup._send_prompts(["only"], "gpt-4o", _limits())

        assert len(sent) == 1
        assert sent[0][1] is None
