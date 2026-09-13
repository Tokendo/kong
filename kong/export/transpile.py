"""Translate the recovered C into another language, as an export format.

This is a reading aid, not a port. Ghidra's output is full of raw memory
access, pointer casts and calling-convention artifacts that have no faithful
equivalent in Python or C#, so the result is labelled a reconstruction and
every function the model could not translate faithfully is marked in place.

The translation is an extra LLM pass over the whole binary: budget for roughly
one more analysis run.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import json_repair

from kong.agent.models import FunctionResult
from kong.agent.prompts import (
    TRANSPILE_OUTPUT_SCHEMA,
    TRANSPILE_SYSTEM_PROMPT,
    transpile_target_rules,
)
from kong.export.source import SECTION_ORDER, ExportData, _includable_results

if TYPE_CHECKING:
    from kong.agent.analyzer import LLMClient

logger = logging.getLogger(__name__)

# Translated code is far longer per function than an analysis result, so a
# chunk is sized by output tokens rather than by the input budget alone.
_OUTPUT_TOKENS_PER_FUNCTION = 600

# The chunk header names the function count, so measuring the scaffolding with
# an empty chunk is a couple of characters short. Round it up.
_HEADER_SLACK = 16


class TargetLanguage(Enum):
    PYTHON = "python"
    CSHARP = "csharp"

    @property
    def display_name(self) -> str:
        return {TargetLanguage.PYTHON: "Python", TargetLanguage.CSHARP: "C#"}[self]

    @property
    def filename(self) -> str:
        return {
            TargetLanguage.PYTHON: "decompiled.py",
            TargetLanguage.CSHARP: "Decompiled.cs",
        }[self]

    @property
    def comment_prefix(self) -> str:
        return {TargetLanguage.PYTHON: "#", TargetLanguage.CSHARP: "//"}[self]

    @property
    def indent(self) -> str:
        """Indentation applied to a translated function in the assembled file."""
        return {TargetLanguage.PYTHON: "", TargetLanguage.CSHARP: "    "}[self]


@dataclass
class TranslatedFunction:
    address: int
    code: str
    faithful: bool = True
    notes: str = ""


class Transpiler:
    """Translates analyzed functions into *language*, one chunk per LLM call."""

    def __init__(
        self,
        llm: LLMClient,
        language: TargetLanguage,
        max_prompt_chars: int | None = None,
        max_output_tokens: int | None = None,
    ) -> None:
        self.llm = llm
        self.language = language
        self.max_prompt_chars = max_prompt_chars
        self.max_output_tokens = max_output_tokens

    # ------------------------------------------------------------------ chunking

    def _functions_per_chunk(self) -> int:
        if self.max_output_tokens is None:
            return 8
        return max(1, self.max_output_tokens // _OUTPUT_TOKENS_PER_FUNCTION)

    def _render_entry(self, result: FunctionResult, decompilation: str) -> str:
        header = (
            f"### 0x{result.address:08x}: {result.name} "
            f"(confidence {result.confidence}%, {result.classification})"
        )
        comment = f"Purpose: {result.comments}" if result.comments else ""
        signature = f"Recovered signature: {result.signature}" if result.signature else ""
        body = "\n".join(part for part in (comment, signature) if part)
        return f"{header}\n{body}\n```c\n{decompilation}\n```\n"

    def _split_into_chunks(
        self, entries: list[tuple[FunctionResult, str]]
    ) -> tuple[list[list[tuple[FunctionResult, str]]], list[FunctionResult]]:
        """Split into chunks that fit the budget, plus the functions that cannot."""
        per_chunk = self._functions_per_chunk()

        available: int | None = None
        if self.max_prompt_chars is not None:
            # The instructions, the target rules and the output schema are sent
            # with every chunk; only what is left belongs to the functions.
            overhead = len(self._build_prompt([])) + _HEADER_SLACK
            available = self.max_prompt_chars - overhead
            if available <= 0:
                raise ValueError(
                    f"max_prompt_chars={self.max_prompt_chars} is too small to "
                    f"transpile: the instructions alone need {overhead} chars."
                )

        chunks: list[list[tuple[FunctionResult, str]]] = []
        current: list[tuple[FunctionResult, str]] = []
        oversized: list[FunctionResult] = []
        used = 0

        for result, decompilation in entries:
            cost = len(self._render_entry(result, decompilation))

            # Too large even on its own: the export marks it as not translated
            # rather than sending a prompt we know overflows the window.
            if available is not None and cost > available:
                oversized.append(result)
                continue

            over_budget = available is not None and used + cost > available
            if current and (over_budget or len(current) >= per_chunk):
                chunks.append(current)
                current = []
                used = 0
            current.append((result, decompilation))
            used += cost

        if current:
            chunks.append(current)
        return chunks, oversized

    # --------------------------------------------------------------- translation

    def _build_prompt(self, chunk: list[tuple[FunctionResult, str]]) -> str:
        parts = [
            f"Translate the following {len(chunk)} functions into "
            f"{self.language.display_name}.",
            "",
            transpile_target_rules(self.language.value),
            "",
            TRANSPILE_OUTPUT_SCHEMA,
            "",
        ]
        parts += [self._render_entry(result, decomp) for result, decomp in chunk]
        return "\n".join(parts)

    def _translate_chunk(
        self, chunk: list[tuple[FunctionResult, str]]
    ) -> dict[int, TranslatedFunction]:
        prompt = self._build_prompt(chunk)
        response = self.llm.analyze_function(prompt, model=None)
        return self._parse(response.raw)

    @staticmethod
    def _parse(raw: str) -> dict[int, TranslatedFunction]:
        from kong.agent.analyzer import _safe_int, strip_markdown_fences

        text = strip_markdown_fences(raw)
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            data = json_repair.loads(text)

        if not isinstance(data, list):
            logger.warning("Transpiler response was not a JSON array")
            return {}

        translated: dict[int, TranslatedFunction] = {}
        for entry in data:
            if not isinstance(entry, dict):
                continue
            address = _safe_int(entry.get("address", 0))
            code = str(entry.get("code", "")).strip()
            if not address or not code:
                continue
            translated[address] = TranslatedFunction(
                address=address,
                code=code,
                faithful=bool(entry.get("faithful", True)),
                notes=str(entry.get("notes", "")),
            )
        return translated

    def translate(self, data: ExportData) -> dict[int, TranslatedFunction]:
        """Translate every exportable function. Never raises on a chunk failure."""
        entries = [
            (result, data.decompilations[result.address])
            for result in _includable_results(data)
        ]
        translated: dict[int, TranslatedFunction] = {}

        chunks, oversized = self._split_into_chunks(entries)
        for result in oversized:
            logger.warning(
                "Not transpiling %s: decompilation does not fit the %s char "
                "prompt budget.", result.name, self.max_prompt_chars,
            )
        for number, chunk in enumerate(chunks, start=1):
            logger.info(
                "Transpiling chunk %d/%d (%d functions) to %s...",
                number, len(chunks), len(chunk), self.language.display_name,
            )
            try:
                translated.update(self._translate_chunk(chunk))
            except Exception as exc:
                logger.warning("Transpile chunk %d failed: %s", number, exc)

        return translated


# ------------------------------------------------------------------- file layout


def _python_header(data: ExportData, faithful: int, total: int) -> list[str]:
    bi = data.binary_info
    return [
        '"""Reconstruction of ' + bi.name + " in Python.",
        "",
        f"Source: {bi.arch} {bi.format} ({bi.compiler}), translated from Ghidra",
        "decompiler output by Kong. This is a reading aid, NOT a runnable port:",
        f"{faithful} of {total} functions translated without loss; the rest carry a",
        "NOT FAITHFUL marker naming what could not be expressed.",
        '"""',
        "",
        "",
    ]


def _csharp_header(data: ExportData, faithful: int, total: int) -> list[str]:
    bi = data.binary_info
    return [
        "// Reconstruction of " + bi.name + " in C#.",
        "//",
        f"// Source: {bi.arch} {bi.format} ({bi.compiler}), translated from Ghidra",
        "// decompiler output by Kong. This is a reading aid, NOT a runnable port:",
        f"// {faithful} of {total} functions translated without loss; the rest carry",
        "// a NOT FAITHFUL marker naming what could not be expressed.",
        "",
        "namespace Kong.Reconstructed;",
        "",
        "public static class Decompiled",
        "{",
        "",
    ]


def _format_function(
    translated: TranslatedFunction, language: TargetLanguage
) -> list[str]:
    lines: list[str] = []
    prefix = language.comment_prefix
    if not translated.faithful:
        note = translated.notes or "behaviour lost in translation"
        lines.append(f"{prefix} NOT FAITHFUL: {note}")
    lines.extend(translated.code.splitlines())
    lines.append("")

    indent = language.indent
    return [f"{indent}{line}" if line else line for line in lines]


def _missing_function(
    result: FunctionResult, language: TargetLanguage
) -> list[str]:
    prefix = language.comment_prefix
    indent = language.indent
    return [
        f"{indent}{prefix} 0x{result.address:08x} {result.name}: "
        f"not translated (the model returned no entry for it).",
        "",
    ]


def export_translated(
    data: ExportData,
    output_path: Path,
    language: TargetLanguage,
    translated: dict[int, TranslatedFunction],
) -> Path:
    """Assemble the translated functions into one file, grouped like the C export."""
    results = _includable_results(data)
    faithful = sum(1 for t in translated.values() if t.faithful)

    header = (
        _python_header if language is TargetLanguage.PYTHON else _csharp_header
    )
    parts: list[str] = header(data, faithful, len(results))

    sections: dict[str, list[FunctionResult]] = {}
    rank = {key: index for index, (key, _) in enumerate(SECTION_ORDER)}
    for result in results:
        key = result.classification if result.classification in rank else "unknown"
        sections.setdefault(key, []).append(result)

    indent = language.indent
    prefix = language.comment_prefix
    for section_key, section_label in SECTION_ORDER:
        funcs = sections.get(section_key)
        if not funcs:
            continue
        funcs.sort(key=lambda r: r.address)
        parts.append(f"{indent}{prefix} {'=' * 20} {section_label} {'=' * 20}")
        parts.append("")
        for func in funcs:
            entry = translated.get(func.address)
            if entry is None:
                parts.extend(_missing_function(func, language))
            else:
                parts.extend(_format_function(entry, language))

    if language is TargetLanguage.CSHARP:
        parts.append("}")

    output_path.write_text("\n".join(parts), encoding="utf-8")
    return output_path


def transpile_and_export(
    data: ExportData,
    output_dir: Path,
    language: TargetLanguage,
    llm: LLMClient,
    max_prompt_chars: int | None = None,
    max_output_tokens: int | None = None,
) -> Path:
    """Run the translation pass and write the file. Returns the written path."""
    transpiler = Transpiler(
        llm,
        language,
        max_prompt_chars=max_prompt_chars,
        max_output_tokens=max_output_tokens,
    )
    translated = transpiler.translate(data)
    return export_translated(
        data, output_dir / language.filename, language, translated
    )
