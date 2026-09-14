"""Semantic synthesis: a cross-function pass that unifies globals, structs and names.

One call over a whole program is the natural shape for this — the point is to
see everything at once — and it is also the most fragile request a run makes.
On a 1351-function binary it exhausted a 32k output budget, was retried at 64k
and timed out an hour later, taking the whole phase with it and leaving no
trace of it in the exported document.

So the pass is split: as many calls as the prompt budget needs, each seeing a
group of functions and the same shared list of cross-referenced globals, and
their answers merged. A group that fails costs its own functions and nothing
else.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from kong.agent.analyzer import strip_markdown_fences
from kong.agent.models import FunctionResult
from kong.llm.limits import take_within_budget

if TYPE_CHECKING:
    from kong.agent.analyzer import LLMClient

logger = logging.getLogger(__name__)

SYNTHESIS_FUNCTION_CAP = 50

_DAT_PATTERN = re.compile(r"DAT_[0-9a-fA-F]+")


@dataclass
class SynthesisResult:
    globals: dict[str, str] = field(default_factory=dict)
    structs: list[dict[str, object]] = field(default_factory=list)
    name_refinements: dict[str, str] = field(default_factory=dict)
    #: How the pass went, so a caller can tell a complete synthesis from one
    #: that lost a group. Both 0 on a result nobody ran.
    passes: int = 0
    failed_passes: int = 0

    @property
    def partial(self) -> bool:
        return self.failed_passes > 0

    def merge(self, other: SynthesisResult) -> None:
        """Fold another group's answer into this one.

        First answer wins on a collision: groups are sent most-cross-referenced
        first, so the earlier group saw the global in more places and its name
        for it is the better evidenced.
        """
        for name, proposed in other.globals.items():
            self.globals.setdefault(name, proposed)
        for address, proposed in other.name_refinements.items():
            self.name_refinements.setdefault(address, proposed)
        known = {
            struct.get("name") for struct in self.structs if isinstance(struct, dict)
        }
        for struct in other.structs:
            if isinstance(struct, dict) and struct.get("name") in known:
                continue
            self.structs.append(struct)
            if isinstance(struct, dict):
                known.add(struct.get("name"))


class SemanticSynthesizer:
    """Makes one LLM call over complete post-naming analysis to unify globals, synthesize structs,
    and refine names."""

    def __init__(self, llm: LLMClient, max_prompt_chars: int | None = None) -> None:
        self.llm = llm
        self.max_prompt_chars = max_prompt_chars

    def synthesize(
        self,
        results: list[FunctionResult],
        decompilations: dict[int, str],
        model: str | None = None,
    ) -> SynthesisResult:
        """Run the pass, one call per group of functions, and merge the answers.

        Raises only when every group failed: a synthesis that lost one group of
        fifty functions is still worth the globals and structs the others
        found, and the caller is told it was partial rather than being handed
        an exception for the whole phase.
        """
        groups = self._group(results, decompilations)
        if not groups:
            return SynthesisResult()

        merged = SynthesisResult()
        last_error: Exception | None = None
        for number, group in enumerate(groups, start=1):
            prompt = self._build_synthesis_prompt(group, decompilations)
            logger.info(
                "Synthesis pass %d/%d: %d functions, %d chars.",
                number, len(groups), len(group), len(prompt),
            )
            try:
                response = self.llm.analyze_function(prompt, model=model)
            except Exception as e:
                last_error = e
                merged.failed_passes += 1
                logger.warning(
                    "Synthesis pass %d/%d failed: %s. The other groups stand.",
                    number, len(groups), e,
                )
                continue
            merged.passes += 1
            merged.merge(self._parse_response(response.raw))

        if merged.passes == 0 and last_error is not None:
            raise last_error
        return merged

    def _group(
        self,
        results: list[FunctionResult],
        decompilations: dict[int, str],
    ) -> list[list[FunctionResult]]:
        """Split the functions worth synthesising into prompt-sized groups.

        Ordered most-cross-referenced first, as one call always was: those
        carry the most shared structure, so they are the ones that must not be
        the group that gets dropped.
        """
        _, xref_counts = self._extract_globals(decompilations)
        eligible = [r for r in results if r.address in decompilations]
        eligible.sort(key=lambda r: xref_counts.get(r.address, 0), reverse=True)
        eligible = eligible[:SYNTHESIS_FUNCTION_CAP]
        if not eligible:
            return []

        if self.max_prompt_chars is None:
            return [eligible]

        # The scaffolding, the instructions and the globals list are sent with
        # every group, so only what is left over belongs to the bodies.
        overhead = len(self._build_synthesis_prompt([], decompilations))
        available = self.max_prompt_chars - overhead
        if available <= 0:
            logger.warning(
                "max_prompt_chars=%d leaves nothing for function bodies after "
                "%d chars of synthesis scaffolding; synthesising one function "
                "per call.", self.max_prompt_chars, overhead,
            )
            return [[result] for result in eligible]

        groups: list[list[FunctionResult]] = []
        current: list[FunctionResult] = []
        used = 0
        for result in eligible:
            cost = len(self._render_function(result, decompilations)) + 1
            if current and used + cost > available:
                groups.append(current)
                current = []
                used = 0
            current.append(result)
            used += cost
        if current:
            groups.append(current)
        return groups

    @staticmethod
    def _render_function(
        result: FunctionResult, decompilations: dict[int, str]
    ) -> str:
        return "\n".join([
            f"### {result.name} (0x{result.address:08x})",
            f"Classification: {result.classification}, Confidence: {result.confidence}",
            "```c",
            decompilations[result.address],
            "```",
            "",
        ])

    def _build_synthesis_prompt(
        self,
        results: list[FunctionResult],
        decompilations: dict[int, str],
    ) -> str:
        parts: list[str] = []

        parts.append(
            "You are performing a cross-function synthesis pass over an analyzed binary. "
            "Your three tasks are:"
        )
        parts.append("")
        parts.append(
            "1. **Rename global variables**: For each DAT_XXXXXXXX address that appears "
            "in multiple functions, propose a meaningful name based on how it is used."
        )
        parts.append(
            "2. **Synthesize structs**: If multiple functions access fields at consistent "
            "offsets from the same base pointer or global, propose a struct definition."
        )
        parts.append(
            "3. **Refine function names**: If seeing all functions together reveals better "
            "names than the per-function pass produced, propose refinements."
        )
        parts.append("")
        parts.append("Respond with a single JSON object (no other text):")
        parts.append("```json")
        parts.append("{")
        parts.append('  "globals": {"DAT_XXXXXXXX": "meaningful_name", ...},')
        parts.append('  "structs": [{"name": "StructName", "fields": [{"name": "field", "type": "int", "offset": 0}]}, ...],')
        parts.append('  "name_refinements": {"0xADDRESS": "better_name", ...}')
        parts.append("}")
        parts.append("```")

        globals_map, xref_counts = self._extract_globals(decompilations)
        results_by_addr = {r.address: r for r in results}

        multi_use_globals = {
            name: addrs for name, addrs in globals_map.items() if len(addrs) >= 2
        }

        def render_global(entry: tuple[str, set[int]]) -> str:
            dat_name, addrs = entry
            func_names = [
                (results_by_addr[addr].name if addr in results_by_addr else f"FUN_{addr:08x}")
                for addr in sorted(addrs)
            ]
            return f"- `{dat_name}` referenced by: {', '.join(func_names)}"

        if multi_use_globals:
            entries = sorted(multi_use_globals.items())
            if self.max_prompt_chars is not None:
                entries = take_within_budget(
                    entries, render_global, self.max_prompt_chars // 10
                )
            if entries:
                parts.append("")
                parts.append("## Global Variables")
                parts.append("")
                parts.extend(render_global(entry) for entry in entries)

        parts.append("")
        parts.append("## Functions and Decompilations")
        parts.append("")

        def render_function(result: FunctionResult) -> str:
            return self._render_function(result, decompilations)

        eligible = [r for r in results if r.address in decompilations]

        if self.max_prompt_chars is not None:
            # `_group` already sized this list against the same budget; this is
            # the backstop for a caller that passed its own list, and for the
            # one case grouping cannot fix — a single decompilation larger than
            # the whole window.
            spent = sum(len(part) + 1 for part in parts)
            eligible = take_within_budget(
                eligible, render_function, max(0, self.max_prompt_chars - spent)
            )

        for result in eligible:
            parts.append(f"### {result.name} (0x{result.address:08x})")
            parts.append(f"Classification: {result.classification}, Confidence: {result.confidence}")
            parts.append("```c")
            parts.append(decompilations[result.address])
            parts.append("```")
            parts.append("")

        return "\n".join(parts)

    @staticmethod
    def _extract_globals(decompilations: dict[int, str]) -> tuple[dict[str, set[int]], dict[int, int]]:
        """Extract globals and per-function xref counts in a single pass."""
        globals_map: dict[str, set[int]] = defaultdict(set)
        xref_counts: dict[int, int] = defaultdict(int)
        for addr, code in decompilations.items():
            for match in _DAT_PATTERN.finditer(code):
                globals_map[match.group(0)].add(addr)
                xref_counts[addr] += 1
        return dict(globals_map), dict(xref_counts)

    @staticmethod
    def _parse_response(raw: str) -> SynthesisResult:
        text = strip_markdown_fences(raw)

        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            logger.warning("Failed to parse synthesis response as JSON")
            return SynthesisResult()

        return SynthesisResult(
            globals=data.get("globals", {}),
            structs=data.get("structs", []),
            name_refinements=data.get("name_refinements", {}),
        )

    @staticmethod
    def _apply_globals(code: str, globals_map: dict[str, str]) -> str:
        for dat_name, meaningful_name in globals_map.items():
            code = code.replace(dat_name, meaningful_name)
        return code

    def apply_to_decompilations(
        self,
        result: SynthesisResult,
        decompilations: dict[int, str],
    ) -> dict[int, str]:
        return {
            addr: self._apply_globals(code, result.globals)
            for addr, code in decompilations.items()
        }
