"""Shared data models for the agent pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kong.agent.analyzer import StructProposal


@dataclass
class AnalysisStats:
    """Aggregate statistics for the full run."""
    total_functions: int = 0
    analyzed: int = 0
    renamed: int = 0
    confirmed: int = 0
    high_confidence: int = 0
    medium_confidence: int = 0
    low_confidence: int = 0
    skipped: int = 0
    errors: int = 0
    llm_calls: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    signature_matches: int = 0

    @property
    def named(self) -> int:
        return self.renamed + self.confirmed

    @property
    def duration_seconds(self) -> float:
        end = self.end_time or time.time()
        return end - self.start_time if self.start_time else 0.0

    @property
    def name_rate(self) -> float:
        return self.named / self.total_functions if self.total_functions else 0.0

    def record_result(self, result: FunctionResult) -> None:
        self._apply(result, 1)

    def replace_result(self, previous: FunctionResult, result: FunctionResult) -> None:
        """Swap an already-recorded result for a new one.

        The second analysis pass re-analyzes functions the draft pass already
        recorded. Recording the new result on top would count the function
        twice, in two different confidence buckets.
        """
        self._apply(previous, -1)
        self._apply(result, 1)

    def _apply(self, result: FunctionResult, sign: int) -> None:
        if result.skipped:
            self.skipped += sign
            return
        if result.error:
            self.errors += sign
            return
        self.analyzed += sign
        self.llm_calls += sign * result.llm_calls
        if result.name:
            if result.name != result.original_name:
                self.renamed += sign
            else:
                self.confirmed += sign
        # TODO: calibrate these eventually. these buckets are arbitrary, need eval data to
        # determine meaningful confidence tiers for the LLM's self-reported scores.
        if result.confidence >= 80:
            self.high_confidence += sign
        elif result.confidence >= 50:
            self.medium_confidence += sign
        else:
            self.low_confidence += sign


@dataclass
class FunctionResult:
    """Result of analyzing a single function."""
    address: int
    original_name: str
    name: str = ""
    signature: str = ""
    confidence: int = 0
    classification: str = ""
    comments: str = ""
    reasoning: str = ""
    error: str = ""
    #: Model that produced this result. Tells the second pass what the first
    #: one answered with, so it never re-reads its own work.
    model: str = ""
    #: True once the second pass has re-analyzed this function.
    refined: bool = False
    llm_calls: int = 0
    skipped: bool = False
    skip_reason: str = ""
    signature_applied: bool = False
    struct_proposals: list[StructProposal] = field(default_factory=list)
    obfuscation_techniques: list[str] = field(default_factory=list)
    deobfuscation_tool_calls: int = 0


@dataclass(frozen=True)
class PhaseFailure:
    """A pipeline phase that failed without stopping the run.

    Synthesis and the transpiling exporters are best-effort: when one raises,
    the supervisor logs it and moves on, because the function-level work is
    already done and worth exporting. That made the failure invisible in the
    deliverable — the run reported "complete", and analysis.json carried no
    sign that a phase was missing from it. These are what the export reports
    instead, alongside the per-function `failures`.
    """

    #: Phase.value, e.g. "synthesis".
    phase: str
    #: str() of the exception, as the user sees it in events.log.
    error: str
    #: What was being produced when it failed, when a phase has more than one
    #: product — the target language of an export, say. Empty otherwise.
    detail: str = ""
