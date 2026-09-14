from __future__ import annotations

import json
from pathlib import Path

from kong.agent.models import FunctionResult, PhaseFailure
from kong.config import LLMProvider
from kong.export.source import ExportData


def _build_binary_section(data: ExportData) -> dict[str, str | int]:
    bi = data.binary_info
    return {
        "name": bi.name,
        "path": bi.path,
        "arch": bi.arch,
        "format": bi.format,
        "endianness": bi.endianness,
        "word_size": bi.word_size,
        "compiler": bi.compiler,
    }


def _build_stats_section(data: ExportData) -> dict[str, int | float | bool]:
    s = data.stats
    cost_tracking = data.provider is not LLMProvider.CUSTOM if data.provider else True
    return {
        "total_functions": s.total_functions,
        "analyzed": s.analyzed,
        "named": s.named,
        "renamed": s.renamed,
        "confirmed": s.confirmed,
        "high_confidence": s.high_confidence,
        "medium_confidence": s.medium_confidence,
        "low_confidence": s.low_confidence,
        "skipped": s.skipped,
        "errors": s.errors,
        "llm_calls": s.llm_calls,
        "duration_seconds": data.duration_seconds,
        "cost_usd": data.token_usage.total_cost_usd,
        "cost_tracking": cost_tracking,
    }


def _build_function_entry(result: FunctionResult) -> dict[str, str | int | list[str]]:
    return {
        "address": f"0x{result.address:08x}",
        "original_name": result.original_name,
        "name": result.name,
        "signature": result.signature,
        "confidence": result.confidence,
        "classification": result.classification,
        "comments": result.comments,
        "reasoning": result.reasoning,
        "obfuscation_techniques": result.obfuscation_techniques,
        # Which model gets the credit, and whether a second pass looked at it.
        # On a two-model run this is what tells the two apart afterwards.
        "model": result.model,
        "refined": result.refined,
    }


def _build_failure_entry(result: FunctionResult) -> dict[str, str]:
    return {
        "address": f"0x{result.address:08x}",
        "original_name": result.original_name,
        "error": result.error,
    }


def _build_call_graph_section(data: ExportData) -> dict[str, object]:
    """Who calls whom, as triage read it from Ghidra.

    Kong has always built this to order the work queue bottom-up and has never
    written it down, so every consumer of analysis.json had to infer the edges
    back out of the C — which only recovers calls between functions that were
    named, and loses every call into one that was skipped. These are the edges
    themselves, including the ones that land on a function no pass analyzed.
    """
    graph = data.call_graph
    if graph is None:
        return {"edges": []}
    return {
        "edges": [
            [f"0x{caller:08x}", f"0x{callee:08x}"] for caller, callee in graph.edges()
        ],
    }


def _build_phase_failure_entry(failure: PhaseFailure) -> dict[str, str]:
    entry = {"phase": failure.phase, "error": failure.error}
    if failure.detail:
        entry["detail"] = failure.detail
    return entry


def export_json(data: ExportData, output_path: Path) -> Path:
    includable = [
        result for result in data.results.values()
        if not result.skipped and not result.error
    ]
    includable.sort(key=lambda r: r.address)

    # Listed separately rather than dropped: stats.errors counts them, and
    # without this the only record of why is events.log.
    failures = sorted(
        (r for r in data.results.values() if r.error),
        key=lambda r: r.address,
    )

    document = {
        "binary": _build_binary_section(data),
        "stats": _build_stats_section(data),
        "functions": [_build_function_entry(r) for r in includable],
        "failures": [_build_failure_entry(r) for r in failures],
        # Whole phases that fell over. A best-effort phase logs and returns, so
        # without this a run that lost its synthesis still reported success and
        # the document simply had nothing where the synthesis should have been.
        "phase_failures": [
            _build_phase_failure_entry(f) for f in data.phase_failures
        ],
        "call_graph": _build_call_graph_section(data),
    }

    output_path.write_text(json.dumps(document, indent=2))
    return output_path
