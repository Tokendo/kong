"""Tests for the Python/C# reconstruction export."""

from __future__ import annotations

import json
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from kong.agent.models import AnalysisStats, FunctionResult
from kong.export.source import ExportData
from kong.export.transpile import (
    TargetLanguage,
    TranslatedFunction,
    Transpiler,
    export_translated,
    transpile_and_export,
)
from kong.ghidra.types import BinaryInfo
from kong.llm.usage import TokenUsage


@dataclass
class _Response:
    raw: str = "[]"
    input_tokens: int = 0
    output_tokens: int = 0


class _ScriptedLLM:
    """Returns one canned response per call and records the prompts."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = responses or []
        self.prompts: list[str] = []

    def analyze_function(self, prompt: str, *, model: str | None = None) -> _Response:
        self.prompts.append(prompt)
        index = min(len(self.prompts) - 1, len(self._responses) - 1)
        if not self._responses:
            return _Response()
        return _Response(raw=self._responses[index])


def _result(addr, name, classification="utility", confidence=90):
    return FunctionResult(
        address=addr,
        original_name=f"FUN_{addr:08x}",
        name=name,
        signature=f"void {name}(uchar *buf, ulong len)",
        confidence=confidence,
        classification=classification,
        comments=f"{name} does something",
    )


def _data(count=3, body="void f(void) { return; }"):
    results = {}
    decompilations = {}
    for i in range(count):
        addr = 0x401000 + i * 0x40
        results[addr] = _result(addr, f"recovered_function_{i}")
        decompilations[addr] = body
    return ExportData(
        binary_info=BinaryInfo(
            arch="x86-64", format="ELF", endianness="little",
            word_size=8, compiler="GCC", name="target.bin",
        ),
        stats=AnalysisStats(total_functions=count, analyzed=count),
        results=results,
        decompilations=decompilations,
        token_usage=TokenUsage(),
        duration_seconds=1.0,
    )


def _payload(addresses, faithful=True, notes=""):
    return json.dumps([
        {
            "address": f"0x{addr:08x}",
            "code": f"def recovered_function_{i}(buf, length):\n    return 0",
            "faithful": faithful,
            "notes": notes,
        }
        for i, addr in enumerate(addresses)
    ])


class TestChunking:
    def test_chunk_size_follows_the_output_budget(self):
        transpiler = Transpiler(
            _ScriptedLLM(), TargetLanguage.PYTHON, max_output_tokens=1200
        )
        assert transpiler._functions_per_chunk() == 2

    def test_a_tiny_output_budget_still_sends_one_function(self):
        transpiler = Transpiler(
            _ScriptedLLM(), TargetLanguage.PYTHON, max_output_tokens=100
        )
        assert transpiler._functions_per_chunk() == 1

    def test_prompts_stay_within_the_prompt_budget(self):
        data = _data(count=20, body="void f(void) { int x = 1; return; }" * 20)
        llm = _ScriptedLLM(["[]"])
        transpiler = Transpiler(
            llm,
            TargetLanguage.PYTHON,
            max_prompt_chars=6000,
            max_output_tokens=4096,
        )

        transpiler.translate(data)

        assert len(llm.prompts) > 1
        assert max(len(p) for p in llm.prompts) <= 6000

    def test_every_function_appears_in_exactly_one_chunk(self):
        data = _data(count=9)
        llm = _ScriptedLLM(["[]"])
        transpiler = Transpiler(llm, TargetLanguage.PYTHON, max_output_tokens=1200)

        transpiler.translate(data)

        # Count function headers only: the output schema carries an example
        # address of its own.
        headers = [f"### 0x{addr:08x}:" for addr in data.results]
        seen = [sum(prompt.count(h) for prompt in llm.prompts) for h in headers]
        assert seen == [1] * len(headers)


    def test_a_function_over_the_budget_is_not_sent(self):
        data = _data(count=2)
        addrs = list(data.results)
        data.decompilations[addrs[1]] = "void huge(void) { return; }" * 400
        llm = _ScriptedLLM(["[]"])
        transpiler = Transpiler(
            llm, TargetLanguage.PYTHON, max_prompt_chars=6000, max_output_tokens=4096
        )

        transpiler.translate(data)

        assert all(f"### 0x{addrs[1]:08x}:" not in p for p in llm.prompts)
        assert any(f"### 0x{addrs[0]:08x}:" in p for p in llm.prompts)

    def test_an_untranslatable_function_is_still_listed_in_the_file(self, tmp_path):
        data = _data(count=2)
        addrs = list(data.results)
        data.decompilations[addrs[1]] = "void huge(void) { return; }" * 400
        llm = _ScriptedLLM([_payload([addrs[0]])])

        path = transpile_and_export(
            data,
            tmp_path,
            TargetLanguage.PYTHON,
            llm,
            max_prompt_chars=6000,
            max_output_tokens=4096,
        )
        text = path.read_text(encoding="utf-8")

        assert "not translated" in text
        assert f"0x{addrs[1]:08x}" in text

    def test_a_budget_below_the_instructions_is_rejected(self):
        data = _data(count=1)
        transpiler = Transpiler(
            _ScriptedLLM(), TargetLanguage.PYTHON, max_prompt_chars=100
        )

        with pytest.raises(ValueError, match="too small"):
            transpiler.translate(data)


class TestParsing:
    def test_translations_are_mapped_back_by_address(self):
        data = _data(count=2)
        llm = _ScriptedLLM([_payload(list(data.results))])
        transpiler = Transpiler(llm, TargetLanguage.PYTHON)

        translated = transpiler.translate(data)

        assert set(translated) == set(data.results)
        assert translated[0x401000].code.startswith("def recovered_function_0")

    def test_a_fenced_response_is_accepted(self):
        data = _data(count=1)
        payload = "```json\n" + _payload(list(data.results)) + "\n```"
        transpiler = Transpiler(_ScriptedLLM([payload]), TargetLanguage.PYTHON)

        assert len(transpiler.translate(data)) == 1

    def test_malformed_json_is_repaired_when_possible(self):
        data = _data(count=1)
        broken = '[{"address": "0x00401000", "code": "def f(): pass",}]'
        transpiler = Transpiler(_ScriptedLLM([broken]), TargetLanguage.PYTHON)

        assert len(transpiler.translate(data)) == 1

    def test_a_non_array_response_yields_nothing(self):
        data = _data(count=1)
        transpiler = Transpiler(_ScriptedLLM(['{"nope": 1}']), TargetLanguage.PYTHON)

        assert transpiler.translate(data) == {}

    def test_entries_without_code_are_dropped(self):
        data = _data(count=1)
        payload = json.dumps([{"address": "0x00401000", "code": "  "}])
        transpiler = Transpiler(_ScriptedLLM([payload]), TargetLanguage.PYTHON)

        assert transpiler.translate(data) == {}

    def test_a_failing_chunk_does_not_abort_the_pass(self):
        data = _data(count=2)

        class _Exploding(_ScriptedLLM):
            def analyze_function(self, prompt, *, model=None):
                raise RuntimeError("endpoint down")

        transpiler = Transpiler(_Exploding(), TargetLanguage.PYTHON)

        assert transpiler.translate(data) == {}


class TestFileLayout:
    def test_python_file_is_written_with_a_reconstruction_header(self, tmp_path):
        data = _data(count=1)
        translated = {
            0x401000: TranslatedFunction(0x401000, "def recovered_function_0():\n    pass")
        }

        path = export_translated(
            data, tmp_path / "decompiled.py", TargetLanguage.PYTHON, translated
        )
        text = path.read_text(encoding="utf-8")

        assert path.name == "decompiled.py"
        assert "NOT a runnable port" in text
        assert "def recovered_function_0" in text

    def test_csharp_file_wraps_methods_in_a_class(self, tmp_path):
        data = _data(count=1)
        translated = {
            0x401000: TranslatedFunction(
                0x401000, "public static void RecoveredFunction0() { }"
            )
        }

        path = export_translated(
            data, tmp_path / "Decompiled.cs", TargetLanguage.CSHARP, translated
        )
        text = path.read_text(encoding="utf-8")

        assert text.count("public static class Decompiled") == 1
        assert text.rstrip().endswith("}")
        assert "    public static void RecoveredFunction0() { }" in text

    def test_unfaithful_translations_are_marked(self, tmp_path):
        data = _data(count=1)
        translated = {
            0x401000: TranslatedFunction(
                0x401000,
                "def recovered_function_0():\n    pass",
                faithful=False,
                notes="raw memory write at 0x10",
            )
        }

        path = export_translated(
            data, tmp_path / "decompiled.py", TargetLanguage.PYTHON, translated
        )
        text = path.read_text(encoding="utf-8")

        assert "# NOT FAITHFUL: raw memory write at 0x10" in text

    def test_a_function_the_model_skipped_is_reported_in_place(self, tmp_path):
        data = _data(count=1)

        path = export_translated(
            data, tmp_path / "decompiled.py", TargetLanguage.PYTHON, {}
        )
        text = path.read_text(encoding="utf-8")

        assert "not translated" in text
        assert "0x00401000" in text

    def test_the_header_counts_faithful_translations(self, tmp_path):
        data = _data(count=2)
        addrs = list(data.results)
        translated = {
            addrs[0]: TranslatedFunction(addrs[0], "def a(): pass", faithful=True),
            addrs[1]: TranslatedFunction(addrs[1], "def b(): pass", faithful=False),
        }

        path = export_translated(
            data, tmp_path / "decompiled.py", TargetLanguage.PYTHON, translated
        )

        assert "1 of 2 functions translated without loss" in path.read_text(
            encoding="utf-8"
        )

    def test_functions_are_grouped_by_classification(self, tmp_path):
        data = _data(count=2)
        addrs = list(data.results)
        data.results[addrs[0]].classification = "crypto"
        data.results[addrs[1]].classification = "parser"
        translated = {
            addr: TranslatedFunction(addr, f"def f{i}(): pass")
            for i, addr in enumerate(addrs)
        }

        path = export_translated(
            data, tmp_path / "decompiled.py", TargetLanguage.PYTHON, translated
        )
        text = path.read_text(encoding="utf-8")

        assert text.index("Crypto") < text.index("Parsers")


class TestEndToEnd:
    def test_transpile_and_export_writes_the_file(self, tmp_path):
        data = _data(count=2)
        llm = _ScriptedLLM([_payload(list(data.results))])

        path = transpile_and_export(
            data, tmp_path, TargetLanguage.PYTHON, llm, max_output_tokens=4096
        )

        assert path.exists()
        assert "recovered_function_0" in path.read_text(encoding="utf-8")

    @pytest.mark.parametrize(
        "language,filename",
        [(TargetLanguage.PYTHON, "decompiled.py"), (TargetLanguage.CSHARP, "Decompiled.cs")],
    )
    def test_each_language_has_its_own_filename(self, tmp_path, language, filename):
        data = _data(count=1)
        llm = _ScriptedLLM([_payload(list(data.results))])

        path = transpile_and_export(data, tmp_path, language, llm)

        assert path.name == filename


class TestSupervisorIntegration:
    def test_the_format_triggers_the_extra_pass(self, tmp_path):
        from kong.agent.supervisor import Supervisor
        from kong.config import KongConfig, OutputConfig

        client = MagicMock()
        client.get_binary_info.return_value = BinaryInfo(
            arch="x86-64", format="ELF", endianness="little",
            word_size=8, compiler="GCC", name="target.bin",
        )
        client.list_functions.return_value = []
        client.get_strings.return_value = []
        client.get_callers.return_value = []
        client.get_callees.return_value = []
        client.get_decompilation.return_value = "void f(void) { return; }"

        config = KongConfig(
            output=OutputConfig(
                directory=tmp_path / "out", formats=["json", "python"]
            )
        )
        llm = MagicMock()
        llm.analyze_function.return_value = _Response(raw="[]")
        llm.analyze_function_batch.return_value = []

        sup = Supervisor(client, config, llm_client=llm)
        sup.run()

        assert (tmp_path / "out" / "decompiled.py").exists()

    def test_without_the_format_no_file_is_written(self, tmp_path):
        from kong.agent.supervisor import Supervisor
        from kong.config import KongConfig, OutputConfig

        client = MagicMock()
        client.get_binary_info.return_value = BinaryInfo(
            arch="x86-64", format="ELF", endianness="little",
            word_size=8, compiler="GCC", name="target.bin",
        )
        client.list_functions.return_value = []
        client.get_strings.return_value = []
        client.get_callers.return_value = []
        client.get_callees.return_value = []

        config = KongConfig(
            output=OutputConfig(directory=tmp_path / "out", formats=["json"])
        )

        sup = Supervisor(client, config)
        sup.run()

        assert not (tmp_path / "out" / "decompiled.py").exists()
