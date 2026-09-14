"""Supervisor agent — orchestrates the full analysis pipeline.

Drives: triage → analysis → cleanup → synthesis → export.
Emits structured events for TUI/CLI consumption.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace

from kong.agent.analyzer import Analyzer, LLMClient, LLMResponse
from kong.agent.coherence import (
    DEFAULT_REVIEW_LIMIT,
    REPORT_NAME,
    CoherenceReport,
    CoherenceReviewer,
    Resolution,
    detect_conflicts,
)
from kong.agent.deobfuscator import (
    Deobfuscator,
    classify_obfuscation,
    is_known_library_code,
    obfuscation_verdict,
)
from kong.agent.events import Event, EventCallback, EventType, Phase
from kong.agent.models import AnalysisStats, FunctionResult, PhaseFailure
from kong.agent.queue import WorkItem, WorkQueue
from kong.agent.refinement import refinement_reason, should_draft
from kong.agent.run_log import RunLog
from kong.agent.signatures import SignatureDB
from kong.state.persistence import (
    BinaryIdentity,
    load_call_edges,
    load_state,
    save_state,
    state_path,
)
from kong.agent.triage import CallGraph, TriageAgent, TriageResult
from kong.agent.type_recovery import StructAccumulator, apply_unified_structs
from kong.config import KongConfig, LLMProvider, RunStage
from kong.export.source import ExportData, export_source
from kong.export.structured import export_json
from kong.export.transpile import TargetLanguage, transpile_and_export
from kong.ghidra.client import GhidraClient
from kong.ghidra.types import BinaryInfo, FunctionInfo, StringEntry
from kong.llm.limits import ModelLimits, get_model_limits, take_within_budget
from kong.llm.usage import TokenUsage
from kong.normalizer.syntactic import normalize
from kong.synthesis.semantic import SemanticSynthesizer, SynthesisResult

logger = logging.getLogger(__name__)


def _clean_api_error(exc: Exception) -> str:
    """Extract a concise error message from LLM API exceptions.

    The OpenAI SDK embeds raw HTTP response bodies (often HTML from proxies)
    in exception messages. This extracts just the useful parts.
    """
    import openai

    if isinstance(exc, openai.APIStatusError):
        return f"HTTP {exc.status_code}: {exc.response.reason_phrase}"
    return str(exc)[:200]


@dataclass
class Chunk:
    """One batch call's worth of functions, and what is unusual about it.

    Most chunks are a handful of functions sharing a call. The two flags are
    for the odd one out: a function too large to share goes alone, without the
    names preamble, and if it does not fit even then its body is cut down and
    the fact recorded, so nothing downstream mistakes a partial reading for a
    complete one.
    """

    entries: list[tuple[WorkItem, str]]
    preamble: bool = True
    truncated: frozenset[int] = frozenset()

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)


class Supervisor:
    """Main agent loop that orchestrates the full analysis pipeline.

    Usage:
        supervisor = Supervisor(client, config)
        supervisor.on_event(my_callback)
        results = supervisor.run()
    """

    def __init__(
        self,
        client: GhidraClient,
        config: KongConfig,
        llm_client: LLMClient | None = None,
        signature_db: SignatureDB | None = None,
    ) -> None:
        self.client = client
        self.config = config
        self.llm_client = llm_client
        self.sig_db = signature_db or SignatureDB()
        self.queue = WorkQueue()
        self.stats = AnalysisStats()
        self.results: dict[int, FunctionResult] = {}
        #: Phases that failed without stopping the run. Exported alongside the
        #: per-function failures so a partial result says so in the artefact
        #: and not only in events.log.
        self.phase_failures: list[PhaseFailure] = []
        self.triage_result: TriageResult | None = None
        #: Whether this binary's obfuscation detections were believed. False
        #: until the analysis phase has seen enough of the binary to judge;
        #: see kong.agent.deobfuscator.obfuscation_verdict.
        self._obfuscation_believed: bool = False
        self.binary_info: BinaryInfo | None = None
        self.functions: list[FunctionInfo] = []
        self.strings: list[StringEntry] = []
        self.struct_accumulator = StructAccumulator()
        self._synthesis_result: SynthesisResult | None = None
        #: What the last coherence review found. None until one is asked for.
        self.coherence_report: CoherenceReport | None = None
        self._listeners: list[EventCallback] = []
        self._paused: bool = False
        self._resume_event = threading.Event()
        self._resume_event.set()
        self._decompilation_cache: dict[int, str] = {}
        self._functions_by_addr: dict[int, FunctionInfo] = {}
        self._results_since_save = 0
        #: Addresses a previous run drafted and left worth a second pass.
        self._pending_refinement: set[int] = set()
        #: Functions the draft stage did not analyze at all, and why. The
        #: draft stage holds back everything that needs the per-function path
        #: instead of sending it to the primary model, which is the whole
        #: point of stopping after the draft.
        self._deferred: dict[int, str] = {}
        #: Bumped on every write to `results` or `_deferred`, so the count a
        #: window polls can be cached against it.
        self._result_version = 0
        self._pending_finish_cache: tuple[int, int] | None = None
        # Results are written from the analysis thread and read by whoever
        # checkpoints — the GUI does it from the main thread when the window
        # closes, mid-run.
        self._results_lock = threading.Lock()
        # The coherence review writes names into Ghidra from whichever
        # thread asked for it, so only one of them may be in there.
        self._coherence_lock = threading.Lock()
        # Same for the finishing pass, which a window offers as a button.
        self._finish_lock = threading.Lock()
        # Same again for analyzing one function on demand, from a graph node
        # clicked in the window.
        self._function_lock = threading.Lock()
        self._binary_identity: BinaryIdentity | None = None
        self._identity_resolved = False

    def _program_is_open(self) -> bool:
        """True while Ghidra still holds the program a pass would work on.

        The passes that run after the analysis get their client from whoever
        built the supervisor, and that owner is free to release it; asking
        first is what turns a closed program into one message instead of one
        failure per function. Duck-typed: a client without `is_open` — a test
        double, mostly — is taken at its word.
        """
        return bool(getattr(self.client, "is_open", True))

    @property
    def _primary_model(self) -> str:
        """The model that has the final word on every function.

        The name is stamped onto every result and written to the state file, so
        a client that does not carry a usable one is reported as unknown rather
        than allowed to break the checkpoint at the end of the analysis.
        """
        model = getattr(self.llm_client, "model", "")
        return model if isinstance(model, str) else ""

    @property
    def _draft_model(self) -> str:
        """The fast first-pass model, or "" when the run is single-pass.

        A draft model equal to the primary one is not a two-pass run: it would
        send the same question to the same model twice.
        """
        draft = self.config.llm.draft_model
        if not draft or self.llm_client is None or draft == self._primary_model:
            return ""
        return draft

    def _get_decompilation(self, addr: int) -> str:
        if addr not in self._decompilation_cache:
            self._decompilation_cache[addr] = self.client.get_decompilation(addr)
        return self._decompilation_cache[addr]

    def _invalidate_decompilation(self, addrs: list[int]) -> int:
        """Drop cached decompilation for these functions and their callers.

        Ghidra re-decompiles with the new types on the next request. Callers
        count too: their call sites show the parameters that were just retyped.
        Returns the number of entries dropped.
        """
        if not addrs:
            return 0

        stale = set(addrs)
        for addr in addrs:
            try:
                stale.update(self.client.get_callers(addr))
            except Exception:
                logger.debug(
                    "Could not list callers of 0x%08x", addr, exc_info=True,
                )

        dropped = sum(
            1 for addr in stale if self._decompilation_cache.pop(addr, None) is not None
        )
        if dropped:
            logger.info(
                "Cleanup changed %d functions; dropped %d cached decompilations "
                "so synthesis and export see the new types.",
                len(addrs), dropped,
            )
        return dropped

    def _restore_previous_results(self, *, include_failed: bool = False) -> int:
        """Re-seed results from an earlier run and put them back into Ghidra.

        Only functions that were named successfully are restored: an errored
        one is left out so this run retries it, which is the whole point of
        resuming. Restored names are re-applied to the program database, since
        the analysis pass that would normally write them is being skipped.

        ``include_failed`` is for a finishing run, which has no analysis pass
        to retry them: there the failures are what it came for, so they are
        restored as they are and handed to the finishing pass.

        A name a draft model produced and never had checked is restored too —
        it is the best name available and its callers benefit from it — but its
        address is held for the second pass, so resuming does not quietly
        promote draft output to final output.
        """
        previous = load_state(self.config.output.directory, self._identity())
        if not previous:
            if state_path(self.config.output.directory).exists():
                # load_state logged the reason; the user needs to know that the
                # file they can see is not the one being used.
                self._emit(Event(
                    type=EventType.PHASE_START,
                    phase=Phase.ANALYSIS,
                    message=(
                        "The saved state in this output directory is for "
                        "another binary or another Kong version; analyzing "
                        "from scratch."
                    ),
                ))
            return 0

        analyzer = Analyzer(self.client, self.llm_client) if self.llm_client else None
        restored = 0
        for addr, result in previous.items():
            if result.error or not result.name:
                if not include_failed:
                    continue
            if analyzer is not None and result.name:
                response = LLMResponse(
                    name=result.name,
                    signature=result.signature,
                    confidence=result.confidence,
                    classification=result.classification,
                    comments=result.comments,
                )
                result.signature_applied = analyzer._write_back(addr, response)
            self._store_result(addr, result)
            self.stats.record_result(result)
            restored += 1

            if (
                self.llm_client is not None
                and not result.refined
                and result.model != self._primary_model
                and refinement_reason(
                    result, threshold=self.config.llm.refine_below,
                )
            ):
                self._pending_refinement.add(addr)

        logger.info(
            "Resumed from %d saved results; %d reusable, the rest will be redone.",
            len(previous), restored,
        )
        if self._pending_refinement:
            logger.info(
                "%d of them were drafted and will be re-analyzed by %s.",
                len(self._pending_refinement), self._primary_model,
            )
        return restored

    def _store_result(self, addr: int, result: FunctionResult) -> None:
        """Record one result. Every write to `results` goes through here.

        The lock is not about consistency between fields, it is about the
        dict itself: `checkpoint()` can be called from the main thread while
        the analysis thread is still inserting.
        """
        with self._results_lock:
            self.results[addr] = result
            self._result_version += 1

    def _identity(self) -> BinaryIdentity | None:
        """Fingerprint of the binary under analysis, computed once per run."""
        if not self._identity_resolved:
            self._identity_resolved = True
            path = getattr(self.client, "binary_path", "")
            if isinstance(path, str) and path:
                self._binary_identity = BinaryIdentity.of(path)
        return self._binary_identity

    def checkpoint(self) -> None:
        """Write the results so far to disk.

        Public because the callers that matter are the ones that are about to
        stop: a Ctrl-C in the terminal, a closed window, a quit key. Safe to
        call at any point, including from another thread and before anything
        has been analyzed.
        """
        self._save_state()

    def _save_state(self) -> None:
        """Persist results so a later run can skip what this one paid for."""
        with self._results_lock:
            snapshot = dict(self.results)
        edges = (
            self.triage_result.call_graph.edges()
            if self.triage_result is not None
            else None
        )
        try:
            save_state(
                snapshot,
                self.config.output.directory,
                self._identity(),
                call_edges=edges,
            )
        except OSError:
            # Losing the checkpoint is not worth losing the run over.
            logger.warning("Could not write the analysis state file", exc_info=True)
        self._results_since_save = 0

    def on_event(self, callback: EventCallback) -> None:
        """Register an event listener."""
        self._listeners.append(callback)

    def _emit(self, event: Event) -> None:
        for cb in self._listeners:
            cb(event)

    @property
    def is_paused(self) -> bool:
        return self._paused

    def pause(self) -> None:
        self._paused = True
        self._resume_event.clear()

    def resume(self) -> None:
        self._paused = False
        self._resume_event.set()

    def _wait_if_paused(self) -> None:
        """Block until resumed. No-op if not paused."""
        self._resume_event.wait()

    def export(self) -> None:
        """Trigger export manually (e.g. from TUI keybind)."""
        self._run_export()

    # --------------------------------------------------------------- finishing

    def run_finishing_pass(self) -> int:
        """Re-read what the draft left unfinished. Returns what it improved.

        The other half of a ``--stage draft`` run, and the one worth deciding
        about: it re-analyzes every function that failed, came back under the
        confidence threshold, or was held back for the per-function path, with
        the primary model, one call each. Then it redoes cleanup, synthesis and
        export, so what is on disk describes the finished analysis rather than
        the draft it was built from.

        Kept out of ``run()`` for a draft stage on purpose: the draft is the
        cheap, long, unsupervised half, and this is the expensive half somebody
        should be able to look at the draft before paying for.
        """
        if not self._finish_lock.acquire(blocking=False):
            logger.info("A finishing pass is already running; ignoring this one.")
            return 0
        try:
            improved = self._run_refinement(manual=True)
            self._run_cleanup()
            self._run_synthesis()
            self._run_export()
            self.checkpoint()
            return improved
        finally:
            self._finish_lock.release()

    # ------------------------------------------------------------- on demand

    def analyze_function(self, address: int) -> None:
        """Re-run one function through the primary model, picked by hand.

        For a caller outside the run loop — a graph node clicked in the
        window — asking for one function regardless of its confidence or
        refinement state. Uses the same per-function path as the second pass:
        callers, callees, cross-references, strings, one call. Only Ghidra and
        the in-memory result are touched; unlike the finishing pass this does
        not re-export, since redoing cleanup and synthesis for every function
        someone points at would cost far more than the call itself.
        """
        if not self._function_lock.acquire(blocking=False):
            logger.info(
                "A requested function analysis is already running; ignoring "
                "this one."
            )
            return
        try:
            self._analyze_one(address)
        finally:
            self._function_lock.release()

    def _analyze_one(self, address: int) -> None:
        if self.llm_client is None:
            self._emit(Event(
                type=EventType.FUNCTION_ERROR,
                phase=Phase.ANALYSIS,
                message=f"Cannot analyze 0x{address:08x}: this run has no model.",
                data={"address": address, "error": "no model"},
            ))
            return
        if not self._program_is_open():
            self._emit(Event(
                type=EventType.FUNCTION_ERROR,
                phase=Phase.ANALYSIS,
                message=(
                    f"Cannot analyze 0x{address:08x}: the binary is closed in "
                    f"Ghidra. Analyze it again to reopen the program."
                ),
                data={"address": address, "error": "program closed"},
            ))
            return

        item = self.queue.get_by_address(address)
        if item is None:
            self._emit(Event(
                type=EventType.FUNCTION_ERROR,
                phase=Phase.ANALYSIS,
                message=f"0x{address:08x} is not a function this run knows about.",
                data={"address": address, "error": "unknown address"},
            ))
            return

        self._wait_if_paused()
        func = item.function
        previous = self.results.get(func.address)
        best_name = (previous.name if previous else "") or func.name
        self._invalidate_decompilation([address])

        self._emit(Event(
            type=EventType.FUNCTION_START,
            phase=Phase.ANALYSIS,
            message=(
                f"Analyzing {best_name} ({func.address_hex}), requested by "
                f"hand..."
            ),
            data={
                "address": func.address,
                "name": best_name,
                "size": func.size,
                "depth": item.depth,
                "model": self._primary_model,
            },
        ))

        try:
            result = self._analyze_function_sequential(item)
        except Exception as e:
            err_msg = _clean_api_error(e)
            logger.exception(
                "Requested analysis of %s (%s) failed", func.name, func.address_hex,
            )
            self._emit(Event(
                type=EventType.FUNCTION_ERROR,
                phase=Phase.ANALYSIS,
                message=f"Analysis of {func.name} failed: {err_msg}",
                data={"address": func.address, "error": err_msg},
            ))
            return

        if result.error or not result.name:
            kept = "; keeping the previous answer." if previous else "."
            self._emit(Event(
                type=EventType.FUNCTION_ERROR,
                phase=Phase.ANALYSIS,
                message=f"Analysis of {func.name} produced nothing usable{kept}",
                data={
                    "address": func.address,
                    "error": result.error or "Empty name in response",
                },
            ))
            if previous is None:
                self._record_analysis_result(func, result)
            return

        result.refined = True
        result.llm_calls += previous.llm_calls if previous else 0
        self._replace_analysis_result(func, result)
        self._pending_refinement.discard(func.address)
        self._deferred.pop(func.address, None)
        self.checkpoint()

    # -------------------------------------------------------------- coherence

    def review_coherence(self, limit: int = DEFAULT_REVIEW_LIMIT) -> CoherenceReport:
        """Cross-check the finished analysis, and resolve what contradicts.

        Kept out of `run()` on purpose. Every other phase answers a question
        about one function; this one asks whether the answers hold together,
        which is only worth asking once the functions have been decompiled and
        named — and which the person reading the results, rather than the
        pipeline, is the one to ask for.

        Detection is free and exhaustive; only the contradictions it finds are
        sent to a model. Returns what it found, and leaves the same thing on
        disk as `coherence.json`.
        """
        if not self._coherence_lock.acquire(blocking=False):
            logger.info("A coherence review is already running; ignoring this one.")
            return CoherenceReport()
        try:
            return self._review_coherence(limit)
        finally:
            self._coherence_lock.release()

    def _review_coherence(self, limit: int) -> CoherenceReport:
        if not self._program_is_open():
            self._emit(Event(
                type=EventType.PHASE_COMPLETE,
                phase=Phase.COHERENCE,
                message=(
                    "Coherence review needs the binary open in Ghidra, and it "
                    "has been closed. Analyze it again to reopen the program."
                ),
            ))
            return CoherenceReport()

        self._emit(Event(
            type=EventType.PHASE_START,
            phase=Phase.COHERENCE,
            message="Cross-checking the analysis for contradictions...",
        ))

        with self._results_lock:
            snapshot = dict(self.results)

        decompilations = self._coherence_decompilations(snapshot)
        callers = self._callers_map(decompilations)

        # Triage loads it; a review asked for before triage has run has to.
        if self.sig_db.size == 0:
            self.sig_db.load_directory()

        conflicts = detect_conflicts(snapshot, decompilations, callers, self.sig_db)
        report = CoherenceReport(
            functions_checked=len(decompilations), conflicts=conflicts,
        )
        self.coherence_report = report

        self._emit(Event(
            type=EventType.COHERENCE_CHECKED,
            phase=Phase.COHERENCE,
            message=(
                f"Cross-checked {len(decompilations)} functions: "
                f"{len(conflicts)} contradictions."
            ),
            data={"checked": len(decompilations), "conflicts": len(conflicts)},
        ))

        for conflict in conflicts:
            self._emit(Event(
                type=EventType.COHERENCE_CONFLICT,
                phase=Phase.COHERENCE,
                message=f"{conflict.kind.value}: {conflict.summary}",
                data={
                    "id": conflict.id,
                    "kind": conflict.kind.value,
                    "addresses": list(conflict.addresses),
                    "summary": conflict.summary,
                    "detail": conflict.detail,
                },
            ))

        if conflicts and self.llm_client is not None:
            self._resolve_conflicts(report, decompilations, limit)

        self._write_coherence_report(report)

        if report.applied:
            # The names and signatures just written change the text of every
            # function that uses them, and a second review has to read the new
            # text rather than the one these conflicts were found in.
            self._invalidate_decompilation([
                change.address
                for resolution in report.resolutions.values()
                for change in resolution.changes
            ])
            self._save_state()

        self._emit(Event(
            type=EventType.PHASE_COMPLETE,
            phase=Phase.COHERENCE,
            message=self._coherence_summary(report),
            data={
                "checked": report.functions_checked,
                "conflicts": len(report.conflicts),
                "reviewed": report.reviewed,
                "applied": len(report.applied),
                "unreviewed": report.unreviewed,
            },
        ))
        return report

    def _coherence_summary(self, report: CoherenceReport) -> str:
        """The one line the log, the window and the terminal all end on."""
        if not report.conflicts:
            return (
                f"Coherence check complete. {report.functions_checked} functions "
                f"agree with each other."
            )
        if self.llm_client is None:
            return (
                f"Coherence check complete. {len(report.conflicts)} contradictions "
                f"found; no model is configured to resolve them."
            )

        summary = (
            f"Coherence check complete. {len(report.applied)} changes over "
            f"{report.reviewed} of {len(report.conflicts)} contradictions."
        )
        if report.unreviewed:
            summary += (
                f" {report.unreviewed} were left for a later pass; run the check "
                f"again to take the next {DEFAULT_REVIEW_LIMIT}."
            )
        return summary

    def _coherence_decompilations(
        self, results: dict[int, FunctionResult],
    ) -> dict[int, str]:
        """The current decompilation of every function worth cross-checking."""
        decompilations: dict[int, str] = {}
        for addr, result in sorted(results.items()):
            if result.skipped or result.error or not result.name:
                continue
            try:
                code = self._get_decompilation(addr)
            except Exception:
                logger.debug("No decompilation for 0x%08x", addr, exc_info=True)
                continue
            if code:
                decompilations[addr] = normalize(code)
        return decompilations

    def _callers_map(self, addresses: Iterable[int]) -> dict[int, list[int]]:
        """addr -> its callers. From triage where it ran, from Ghidra where not.

        A review can be asked for on a supervisor that only restored results
        from a state file, in which case there is no work queue to read the
        call graph off.
        """
        callers: dict[int, list[int]] = {}
        for addr in addresses:
            item = self.queue.get_by_address(addr)
            if item is not None:
                callers[addr] = list(item.callers)
                continue
            try:
                callers[addr] = list(self.client.get_callers(addr))
            except Exception:
                logger.debug("Could not list callers of 0x%08x", addr, exc_info=True)
                callers[addr] = []
        return callers

    def _resolve_conflicts(
        self,
        report: CoherenceReport,
        decompilations: dict[int, str],
        limit: int,
    ) -> None:
        """Hand the contradictions to the model, batch by batch, and apply."""
        assert self.llm_client is not None

        conflicts = report.conflicts[:limit] if limit > 0 else list(report.conflicts)
        report.unreviewed = len(report.conflicts) - len(conflicts)

        reviewer = CoherenceReviewer(
            self.llm_client,
            max_prompt_chars=self._get_effective_limits().max_prompt_chars,
        )
        batches = reviewer.batches(conflicts, decompilations)

        for number, batch in enumerate(batches, start=1):
            self._emit(Event(
                type=EventType.PHASE_START,
                phase=Phase.COHERENCE,
                message=(
                    f"Resolving {len(batch)} contradictions with "
                    f"{self._primary_model or 'the model'} "
                    f"({number}/{len(batches)})..."
                ),
                data={
                    "batch": number,
                    "batches": len(batches),
                    "conflicts": len(batch),
                    "model": self._primary_model,
                },
            ))

            # Re-read the results each time: an earlier batch may have renamed
            # a function this one is about.
            with self._results_lock:
                snapshot = dict(self.results)

            try:
                resolutions = reviewer.review(
                    batch,
                    snapshot,
                    decompilations,
                    model=self._primary_model or None,
                )
            except Exception as e:
                err_msg = _clean_api_error(e)
                logger.exception("Coherence review %d/%d failed", number, len(batches))
                self._emit(Event(
                    type=EventType.PHASE_COMPLETE,
                    phase=Phase.COHERENCE,
                    message=f"Could not resolve this group: {err_msg}",
                    data={"error": err_msg},
                ))
                continue

            report.llm_calls += 1
            # The window reads its LLM call counter off the stats, and this
            # call was paid for like any other.
            self.stats.llm_calls += 1

            if not resolutions:
                # The call succeeded and said nothing usable. Outside a run
                # there is no events.log to find that in, so it goes where the
                # person who asked for the pass is looking.
                self._emit(Event(
                    type=EventType.PHASE_COMPLETE,
                    phase=Phase.COHERENCE,
                    message=(
                        f"The answer for this group could not be read; its "
                        f"{len(batch)} contradictions are left as they are."
                    ),
                    data={"conflicts": len(batch)},
                ))
                continue

            for resolution in resolutions:
                report.resolutions[resolution.conflict_id] = resolution
                applied = self._apply_resolution(resolution)
                report.applied.extend(applied)
                self._emit(Event(
                    type=EventType.COHERENCE_RESOLVED,
                    phase=Phase.COHERENCE,
                    message=(
                        f"{resolution.conflict_id}: " + (
                            "; ".join(applied) if applied
                            else "left alone — "
                                 f"{resolution.explanation or 'nothing to change'}"
                        )
                    ),
                    data={
                        "id": resolution.conflict_id,
                        "verdict": resolution.verdict,
                        "explanation": resolution.explanation,
                        "applied": applied,
                    },
                ))

    def _apply_resolution(self, resolution: Resolution) -> list[str]:
        """Write one arbitration back to Ghidra and to the results.

        Returns a line per function it actually changed. A rename Ghidra
        refuses is reported as not done rather than recorded as done: the
        results file and the program database have to say the same thing.
        """
        applied: list[str] = []

        for change in resolution.changes:
            with self._results_lock:
                previous = self.results.get(change.address)
            if previous is None:
                continue

            updated = replace(previous)
            notes: list[str] = []

            if change.name and change.name != previous.name:
                try:
                    self.client.rename_function(change.address, change.name)
                except Exception as e:
                    logger.warning(
                        "Coherence rename of 0x%08x to %s failed: %s",
                        change.address, change.name, e,
                    )
                else:
                    updated.name = change.name
                    notes.append(f"{previous.name} to {change.name}")

            if change.signature and change.signature != previous.signature:
                updated.signature = change.signature
                try:
                    self.client.set_function_signature(change.address, change.signature)
                except Exception as e:
                    # Same deal as the analysis pass: a signature Ghidra will
                    # not take yet stays on the result for cleanup to retry.
                    logger.debug(
                        "Coherence signature deferred for 0x%08x: %s",
                        change.address, e,
                    )
                    updated.signature_applied = False
                else:
                    updated.signature_applied = True
                notes.append(f"signature {change.signature}")

            if change.confidence is not None and change.confidence != previous.confidence:
                updated.confidence = change.confidence
                notes.append(
                    f"confidence {previous.confidence} to {change.confidence}"
                )

            if change.comment and change.comment != previous.comments:
                updated.comments = change.comment
                try:
                    self.client.add_comment(change.address, change.comment)
                except Exception as e:
                    logger.warning(
                        "Coherence comment at 0x%08x failed: %s", change.address, e,
                    )
                notes.append("description")

            if not notes:
                continue

            self._store_result(change.address, updated)
            # Through replace_result so the confidence buckets the window shows
            # follow the score this pass just changed.
            self.stats.replace_result(previous, updated)
            applied.append(f"0x{change.address:08x}: " + ", ".join(notes))

        return applied

    def _write_coherence_report(self, report: CoherenceReport) -> None:
        """Leave the findings on disk, next to the rest of the output."""
        output_dir = self.config.output.directory
        path = output_dir / REPORT_NAME
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
        except OSError:
            logger.warning(
                "Could not write the coherence report to %s", path, exc_info=True,
            )
            return

        self._emit(Event(
            type=EventType.EXPORT_FILE,
            phase=Phase.COHERENCE,
            message=f"Exported {path}",
            data={"path": str(path), "format": "coherence"},
        ))

    def run(self) -> dict[int, FunctionResult]:
        """Run the full analysis pipeline. Returns addr -> FunctionResult."""
        self.stats.start_time = time.time()

        # Registered first so it sees every event, and removed in the finally
        # so a second run() on the same supervisor does not log twice.
        run_log = RunLog(self.config.output.directory, verbose=self.config.verbose)
        log_path = run_log.open()
        self._listeners.insert(0, run_log.record)
        try:
            self._emit(Event(
                type=EventType.RUN_START,
                message=f"Kong analysis starting. Trace: {log_path}",
                data={"log_path": str(log_path)},
            ))

            stage = self.config.stage
            try:
                self._run_triage()
                if stage is RunStage.FINISH:
                    self._run_finish_stage()
                    full_resume = False
                else:
                    restored, fresh = self._run_analysis()
                    # Restored something, and did not need to touch any of
                    # it: this run has nothing cleanup, synthesis or export
                    # would not just redo byte-for-byte. Skipping them is
                    # what makes reopening a finished run inside the window
                    # (rather than the read-only viewer) cheap enough to
                    # offer at all — synthesis alone is the run's single
                    # most expensive phase.
                    full_resume = restored > 0 and fresh == 0

                if full_resume:
                    self._emit(Event(
                        type=EventType.PHASE_COMPLETE,
                        phase=Phase.CLEANUP,
                        message=(
                            f"Nothing new: all {restored} functions resumed "
                            f"from the last run's checkpoint. Skipping "
                            f"cleanup, synthesis and export — pass --fresh "
                            f"to redo them."
                        ),
                        data={"restored": restored},
                    ))
                else:
                    self._run_cleanup()
                    # Synthesis unifies naming across the whole binary in one
                    # call. Spending it on draft output would only buy an
                    # answer the finishing pass invalidates, so a draft run
                    # leaves it.
                    if stage is not RunStage.DRAFT:
                        self._run_synthesis()
                    self._run_export()
            except Exception as e:
                logger.exception("Run failed")
                self._emit(Event(
                    type=EventType.RUN_ERROR,
                    message=f"Fatal error: {e}",
                    data={"error": str(e)},
                ))
                raise

            self.stats.end_time = time.time()
            pending = len(self._refinement_candidates(manual=True))
            headline = (
                f"Draft complete. {self.stats.named}/{self.stats.total_functions} "
                f"functions named in {self.stats.duration_seconds:.1f}s, "
                f"{pending} waiting for the finishing pass."
                if stage is RunStage.DRAFT else
                f"Analysis complete. {self.stats.named}/{self.stats.total_functions} "
                f"functions named in {self.stats.duration_seconds:.1f}s."
            )
            self._emit(Event(
                type=EventType.RUN_COMPLETE,
                message=headline,
                data={"stats": self._stats_dict(), "pending_finish": pending},
            ))
            return self.results
        finally:
            # Whatever happened — finished, crashed, or a Ctrl-C on its way
            # out — what has been paid for is on disk before this returns.
            # It also captures the renames synthesis made after the analysis
            # phase wrote its last checkpoint.
            if self.results:
                self.checkpoint()
            self._listeners.remove(run_log.record)
            run_log.close()

    def _run_triage(self) -> None:
        """Enumerate functions, match signatures, build work queue."""
        self._emit(Event(
            type=EventType.PHASE_START,
            phase=Phase.TRIAGE,
            message="Starting triage...",
        ))

        triage = TriageAgent(self.client, signature_db=self.sig_db)
        result = triage.run()
        self.triage_result = result

        self.binary_info = result.binary_info
        self.functions = result.functions
        self._functions_by_addr = {f.address: f for f in self.functions}
        self.strings = result.strings
        self.queue = result.queue
        self.stats.total_functions = result.queue_size
        self.stats.signature_matches = result.matched_count

        self._emit(Event(
            type=EventType.TRIAGE_FUNCTIONS_ENUMERATED,
            phase=Phase.TRIAGE,
            message=f"Enumerated {len(self.functions)} functions.",
            data={
                "total": len(self.functions),
                "binary_info": {
                    "arch": self.binary_info.arch,
                    "format": self.binary_info.format,
                    "compiler": self.binary_info.compiler,
                },
            },
        ))

        self._emit(Event(
            type=EventType.TRIAGE_SIGNATURES_MATCHED,
            phase=Phase.TRIAGE,
            message=f"Matched {self.stats.signature_matches} functions against signature DB.",
            data={"matched": self.stats.signature_matches},
        ))

        self._emit(Event(
            type=EventType.TRIAGE_QUEUE_BUILT,
            phase=Phase.TRIAGE,
            message=(
                f"Work queue built: {self.queue.total} functions to analyze "
                f"(bottom-up order)."
            ),
            data={"queue_size": self.queue.total},
        ))

        if result.language_hints.language != "C":
            self._emit(Event(
                type=EventType.PHASE_START,
                phase=Phase.TRIAGE,
                message=f"Detected language: {result.language_hints.language}",
                data={"language": result.language_hints.language},
            ))

        self._emit(Event(
            type=EventType.PHASE_COMPLETE,
            phase=Phase.TRIAGE,
            message=(
                f"Triage complete. {len(self.functions)} functions found, "
                f"{self.stats.signature_matches} pre-labeled."
            ),
        ))

    def _call_graph(self) -> CallGraph | None:
        """This run's call graph, or the one a previous run saved.

        An export triggered without a triage of its own — from the TUI, or a
        finishing pass over a saved draft — would otherwise write a document
        with no edges in it at all.
        """
        if self.triage_result is not None:
            return self.triage_result.call_graph
        edges = load_call_edges(self.config.output.directory, self._identity())
        if not edges:
            return None
        graph = CallGraph()
        for caller, callee in edges:
            graph.callees.setdefault(caller, []).append(callee)
            graph.callers.setdefault(callee, []).append(caller)
        logger.info("Loaded %d call edges from the saved state.", len(edges))
        return graph

    def _signature_matched_addresses(self) -> frozenset[int]:
        """Addresses the signature database identified during triage.

        Deliberately not "Ghidra gave it a name": on a resumed run that name is
        Kong's own from the run before, so every function analysed once would
        look like a documented library function.
        """
        if self.triage_result is None:
            return frozenset()
        return frozenset(
            match.function_address for match in self.triage_result.signature_matches
        )

    def _run_analysis(self) -> tuple[int, int]:
        """Analyze functions in large chunks via sequential LLM calls.

        Returns ``(restored, fresh)``: how many functions this call reused
        unchanged from a previous checkpoint, and how many it did new work
        on — analyzed for the first time or improved by the automatic
        refinement pass. ``restored and not fresh`` is a full resume: nothing
        is different from what a previous run already wrote to disk, which is
        what tells the caller cleanup, synthesis and export are safe to skip.
        """
        self._emit(Event(
            type=EventType.PHASE_START,
            phase=Phase.ANALYSIS,
            message="Starting analysis...",
        ))

        restored = self._restore_previous_results() if self.config.resume else 0
        if restored:
            self._emit(Event(
                type=EventType.PHASE_START,
                phase=Phase.ANALYSIS,
                message=(
                    f"Resumed {restored} functions from a previous run; "
                    f"pass --fresh to analyze them again."
                ),
                data={"restored": restored},
            ))

        all_items = self.queue.all_items()
        draft_items: list[tuple[WorkItem, str]] = []
        chunk_items: list[tuple[WorkItem, str]] = []
        sequential_items: list[tuple[WorkItem, list]] = []
        completed_count = 0
        draft_model = self._draft_model
        # A draft run owes the primary model nothing: every function it cannot
        # handle itself is held for the finishing pass instead of being sent
        # there now, which is what makes the two stages separable at all.
        drafting = self.config.stage is RunStage.DRAFT

        # Read every pending function before dispatching any of it. The
        # obfuscation decision cannot be taken one function at a time: the
        # heuristics read structure, and a while(1) around a switch is both
        # control-flow flattening and every hand-written state machine, so
        # whether a hit means anything depends on how many of them there are.
        pending: list[tuple[WorkItem, str, list]] = []
        matched = self._signature_matched_addresses()
        skip_matched = self.config.analysis.skip_matched_signatures

        for item in all_items:
            func = item.function

            if func.address in self.results:
                # Restored from a previous run; already named and written back.
                completed_count += 1
                continue

            if func.classification and func.classification.value == "trivial":
                result = FunctionResult(
                    address=func.address,
                    original_name=func.name,
                    skipped=True,
                    skip_reason="trivial",
                )
                self._store_result(func.address, result)
                self.stats.record_result(result)
                completed_count += 1
                continue

            if skip_matched and is_known_library_code(func.address, matched):
                # A documented library function, already named by the symbol
                # it matched. Paying a model to describe it again is the one
                # cost in a run that buys nothing.
                result = FunctionResult(
                    address=func.address,
                    original_name=func.name,
                    name=func.name,
                    skipped=True,
                    skip_reason="identified by signature",
                )
                self._store_result(func.address, result)
                self.stats.record_result(result)
                completed_count += 1
                continue

            if self.llm_client is None:
                result = FunctionResult(
                    address=func.address,
                    original_name=func.name,
                    name=func.name,
                    confidence=0,
                )
                self._store_result(func.address, result)
                self.stats.record_result(result)
                completed_count += 1
                continue

            decompilation = self._get_decompilation(func.address)
            # Library code is where the heuristics misfire hardest and where
            # the agentic loop has least to offer: the CRT is full of large
            # dispatch loops, and they are already named and documented.
            techniques = (
                []
                if is_known_library_code(func.address, matched)
                else classify_obfuscation(decompilation)
            )
            pending.append((item, decompilation, techniques))

        verdict = obfuscation_verdict(
            flagged=sum(1 for _, _, t in pending if t),
            total=len(pending),
            threshold=self.config.analysis.obfuscation_threshold,
        )
        self._obfuscation_believed = verdict.believed
        if verdict.flagged:
            logger.info("Obfuscation: %s", verdict.describe())
            self._emit(Event(
                type=EventType.PHASE_START,
                phase=Phase.ANALYSIS,
                message=verdict.describe(),
                data={
                    "flagged": verdict.flagged,
                    "total": verdict.total,
                    "believed": verdict.believed,
                },
            ))

        for item, decompilation, techniques in pending:
            func = item.function

            if techniques and verdict.believed:
                # Obfuscated code goes to the strong model directly: the draft
                # would be re-analyzed whatever it answered, and the agentic
                # deobfuscation loop is the part a small model handles worst.
                if drafting:
                    self._deferred[func.address] = (
                        "obfuscated: "
                        + ", ".join(t.value for t in techniques)
                    )
                    self._result_version += 1
                    continue
                sequential_items.append((item, techniques))
                continue

            entry = (item, normalize(decompilation))
            # Outside a draft run, a function too large to draft is worth the
            # primary model's time straight away. Inside one, the draft takes
            # it too: chunking sizes the prompt to the draft model's own
            # window, and whatever does not fit comes back as a failure the
            # finishing pass picks up.
            if draft_model and (drafting or should_draft(func)):
                draft_items.append(entry)
            else:
                chunk_items.append(entry)

        if draft_items:
            self._emit(Event(
                type=EventType.PHASE_START,
                phase=Phase.ANALYSIS,
                message=(
                    f"Draft pass: {len(draft_items)} functions on {draft_model}."
                ),
                data={"model": draft_model, "count": len(draft_items)},
            ))
            self._analyze_chunks(draft_items, completed_count, model=draft_model)
            completed_count += len(draft_items)

        if chunk_items:
            if draft_model:
                self._emit(Event(
                    type=EventType.PHASE_START,
                    phase=Phase.ANALYSIS,
                    message=(
                        f"{len(chunk_items)} functions are too large to draft; "
                        f"sending them straight to {self._primary_model}."
                    ),
                    data={
                        "model": self._primary_model,
                        "count": len(chunk_items),
                    },
                ))
            self._analyze_chunks(chunk_items, completed_count)
            completed_count += len(chunk_items)

        for item, techniques in sequential_items:
            self._wait_if_paused()
            completed_count += 1
            func = item.function
            self._emit(Event(
                type=EventType.FUNCTION_START,
                phase=Phase.ANALYSIS,
                message=f"Analyzing {func.name} ({func.address_hex})...",
                data={
                    "address": func.address,
                    "name": func.name,
                    "size": func.size,
                    "depth": item.depth,
                    "progress": f"{completed_count}/{self.queue.total}",
                },
            ))
            try:
                result = self._analyze_function_sequential(item, techniques)
            except Exception as e:
                err_msg = _clean_api_error(e)
                logger.exception(
                    "Analysis of %s (%s) failed", func.name, func.address_hex,
                )
                result = FunctionResult(
                    address=func.address,
                    original_name=func.name,
                    error=err_msg,
                    model=self._primary_model,
                )
                self._emit(Event(
                    type=EventType.FUNCTION_ERROR,
                    phase=Phase.ANALYSIS,
                    message=f"Error analyzing {func.name}: {err_msg}",
                    data={"address": func.address, "error": err_msg},
                ))
            self._record_analysis_result(func, result)

        refined = 0
        if drafting:
            self._announce_pending_finish()
        else:
            refined = self._run_refinement()

        self._save_state()

        self._emit(Event(
            type=EventType.PHASE_COMPLETE,
            phase=Phase.ANALYSIS,
            message=(
                f"Analysis complete. {self.stats.named}/{self.stats.total_functions} "
                f"functions named."
            ),
        ))
        return restored, len(all_items) - restored + refined

    def _limits_for(self, model: str) -> ModelLimits:
        """Chunking limits for one model of the run.

        Two-pass runs send chunks to two different models, and a draft model
        with a smaller window has to be chunked for its own window, not for the
        one the final pass enjoys. The explicit overrides describe the endpoint
        rather than a model, so they apply to both — and to every provider,
        not only a local one. The model table says what a window holds; it
        cannot know that a hosted account is being rate limited today, or that
        this particular model starts dropping functions halfway through a batch
        of a hundred. That is the caller's to say, whoever serves the model.
        """
        cfg = self.config.llm
        base = get_model_limits(model)
        return ModelLimits(
            max_prompt_chars=cfg.max_prompt_chars if cfg.max_prompt_chars is not None else base.max_prompt_chars,
            max_chunk_functions=cfg.max_chunk_functions if cfg.max_chunk_functions is not None else base.max_chunk_functions,
            max_output_tokens=cfg.max_output_tokens if cfg.max_output_tokens is not None else base.max_output_tokens,
        )

    def _get_effective_limits(self) -> ModelLimits:
        return self._limits_for(self._primary_model)

    def _known_functions_budget(self, limits: ModelLimits | None = None) -> int:
        """Char budget for the "already identified" preamble of a chunk prompt.

        A tenth of the prompt budget: enough to carry recent context on a large
        window, small enough to stay predictable on a local model.
        """
        limits = limits or self._get_effective_limits()
        return max(0, limits.max_prompt_chars // 10)

    @staticmethod
    def _chunk_entry(item: WorkItem, decompilation: str) -> str:
        """One function as it appears in a chunk prompt."""
        func = item.function
        return (
            f"### 0x{func.address:08x}: {func.name} ({func.size} bytes)\n"
            f"```c\n{decompilation}\n```\n\n"
        )

    def _split_into_chunks(
        self,
        items: list[tuple[WorkItem, str]],
        limits: ModelLimits | None = None,
    ) -> tuple[list[Chunk], list[tuple[WorkItem, str]]]:
        """Split items into chunks by building the actual prompt and measuring size.

        A function too large to share a call is not therefore too large to
        analyze. It goes in a chunk of its own, which needs no room reserved
        for the names preamble — that list is a convenience, and on a solo call
        it is worth more as body. Only when a function does not fit even then
        is it truncated, or, if the run forbids that, reported unanalyzed.

        Returns the chunks, and the functions nothing could be sent for, as
        ``(item, reason)`` pairs.
        """
        assert self.binary_info is not None

        limits = limits or self._get_effective_limits()
        preamble_budget = self._known_functions_budget(limits)

        overhead_len = len(self._build_chunk_prompt([], limits=limits, preamble=False))
        shared = limits.max_prompt_chars - overhead_len - preamble_budget
        solo = limits.max_prompt_chars - overhead_len
        if solo <= 0:
            raise ValueError(
                f"max_prompt_chars={limits.max_prompt_chars} is too small: the "
                f"prompt scaffolding alone needs {overhead_len} chars."
            )

        chunks: list[Chunk] = []
        current: list[tuple[WorkItem, str]] = []
        unanalyzable: list[tuple[WorkItem, str]] = []
        current_chars = 0

        def flush() -> None:
            nonlocal current, current_chars
            if current:
                chunks.append(Chunk(entries=current))
                current = []
                current_chars = 0

        for item, decomp in items:
            entry_len = len(self._chunk_entry(item, decomp))

            if entry_len > shared:
                # Its own call, without the preamble reserve.
                if entry_len <= solo:
                    flush()
                    chunks.append(Chunk(entries=[(item, decomp)], preamble=False))
                    continue

                cut = self._truncate_to_fit(item, decomp, solo)
                if cut is None:
                    unanalyzable.append(
                        (item, self._too_large_reason(entry_len, solo, limits))
                    )
                    continue
                flush()
                chunks.append(Chunk(
                    entries=[(item, cut)],
                    preamble=False,
                    truncated=frozenset({item.function.address}),
                ))
                continue

            chunk_full = (
                current_chars + entry_len > shared
                or len(current) >= limits.max_chunk_functions
            )
            if current and chunk_full:
                flush()
            current.append((item, decomp))
            current_chars += entry_len

        flush()
        return chunks, unanalyzable

    def _truncate_to_fit(
        self, item: WorkItem, decompilation: str, budget: int
    ) -> str | None:
        """*decompilation* cut down to fit *budget*, or None if not allowed.

        A body the model never sees is a function the analysis has nothing at
        all to say about, and on the FA18 binary the one that did not fit was
        the program's main dispatch loop — its most connected function. Most of
        a body names it well enough, so long as the model is told that is what
        it is looking at, and the result says so afterwards.
        """
        if not self.config.analysis.truncate_oversized:
            return None

        marker_template = (
            "\n/* --- TRUNCATED: {shown} of {total} characters shown. The rest of "
            "this function was over the run's prompt budget. --- */"
        )
        marker = marker_template.format(total=len(decompilation), shown=len(decompilation))
        room = budget - (len(self._chunk_entry(item, "")) + len(marker))
        if room <= 0:
            return None

        kept = decompilation[:room]
        return kept + marker_template.format(shown=len(kept), total=len(decompilation))

    @staticmethod
    def _too_large_reason(entry_len: int, solo: int, limits: ModelLimits) -> str:
        """Why a function could not be sent, in numbers a reader can act on.

        Naming the configured budget alone was misleading: the scaffolding and
        the names preamble come out of it first, so a reader who raised the
        budget to just over the figure in the message failed again.
        """
        return (
            f"Decompilation needs {entry_len} chars and only {solo} are left for "
            f"a function body by the {limits.max_prompt_chars} char prompt budget. "
            f"Raise --max-prompt-chars to at least "
            f"{limits.max_prompt_chars + entry_len - solo}, use a model with a "
            f"larger context, or allow truncation."
        )

    #: Chunk calls in flight when the provider is a hosted API. Local
    #: endpoints keep to one: they are already saturating the machine Kong
    #: runs on, and a second call in flight only makes both slower.
    HOSTED_CHUNK_CONCURRENCY = 4

    def _chunk_concurrency(self) -> int:
        """How many chunk calls may be in flight at once.

        Safe because of how the queue is built: it is ordered bottom-up by
        call-graph depth, and functions at the same depth cannot be each
        other's callees, so nothing in one chunk is waiting on a name another
        chunk is about to produce. Only the sending is concurrent — results are
        written back on this thread, in chunk order.
        """
        configured = self.config.analysis.chunk_concurrency
        if configured is not None:
            return max(1, configured)
        if self.config.llm.provider is LLMProvider.CUSTOM:
            return 1
        return self.HOSTED_CHUNK_CONCURRENCY

    def _send_one(
        self, prompt: str, model: str | None, limits: ModelLimits,
    ) -> tuple[list[LLMResponse], str | None]:
        """One batch call, retried. Returns its responses, or the last error.

        The retry is for the transient half of the failures — a 500, a reset
        connection — which used to fail every function in the chunk on the
        first try. What the endpoint genuinely objects to fails again here and
        is left to `_recover_chunk` to isolate.
        """
        assert self.llm_client is not None
        attempts = max(1, self.config.analysis.chunk_attempts)
        error = "no attempt made"
        for attempt in range(1, attempts + 1):
            try:
                responses = self.llm_client.analyze_function_batch(
                    prompt, model=model, max_tokens=limits.max_output_tokens,
                )
            except Exception as e:
                error = _clean_api_error(e)
                if attempt < attempts:
                    logger.warning(
                        "Chunk call failed (attempt %d/%d): %s. Retrying.",
                        attempt, attempts, error,
                    )
                continue
            return responses, None
        return [], error

    def _send_prompts(
        self, prompts: list[str], model: str | None, limits: ModelLimits,
    ) -> list[tuple[list[LLMResponse], str | None]]:
        """Send a wave of prompts, concurrently when more than one is allowed.

        Results come back in the order the prompts were given, whatever order
        they actually finished in, so a run's events and its state file do not
        depend on which call the endpoint answered first.
        """
        if len(prompts) == 1:
            return [self._send_one(prompts[0], model, limits)]

        with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
            futures = [
                pool.submit(self._send_one, prompt, model, limits)
                for prompt in prompts
            ]
            return [future.result() for future in futures]

    def _recover_chunk(
        self,
        chunk: Chunk,
        model: str | None,
        limits: ModelLimits,
        error: str,
    ) -> tuple[list[LLMResponse], dict[int, str]]:
        """Salvage a failed chunk by halving it, down to single functions.

        A chunk call is one request for many functions, so one transient error
        used to fail every function in it at once — which is why failures
        arrived in bursts of eight or sixteen. Halving finds the function the
        endpoint actually objects to, usually an oversized one, and keeps the
        rest of the chunk.

        Returns the responses recovered, and the addresses that could not be,
        each with the error that stopped it.
        """
        entries = chunk.entries
        if len(entries) == 1:
            return [], {entries[0][0].function.address: f"Chunk call failed: {error}"}

        middle = len(entries) // 2
        responses: list[LLMResponse] = []
        failures: dict[int, str] = {}
        for half in (entries[:middle], entries[middle:]):
            prompt = self._build_chunk_prompt(
                half, limits=limits, preamble=chunk.preamble,
            )
            got, half_error = self._send_one(prompt, model, limits)
            if half_error is None:
                responses.extend(got)
                continue
            logger.warning(
                "Half of %d functions also failed: %s. Splitting again.",
                len(half), half_error,
            )
            deeper, deeper_failures = self._recover_chunk(
                Chunk(entries=half, preamble=chunk.preamble,
                      truncated=chunk.truncated),
                model, limits, half_error,
            )
            responses.extend(deeper)
            failures.update(deeper_failures)

        if failures:
            logger.warning(
                "Recovered %d of %d functions from the failed chunk.",
                len(entries) - len(failures), len(entries),
            )
        else:
            logger.info(
                "Recovered all %d functions from the failed chunk.", len(entries)
            )
        return responses, failures

    def _analyze_chunks(
        self,
        items: list[tuple[WorkItem, str]],
        completed_base: int,
        model: str | None = None,
    ) -> None:
        """Analyze functions in large chunks, streaming results per chunk."""
        assert self.llm_client is not None
        assert self.binary_info is not None

        items.sort(key=lambda x: len(x[1]))

        model = model or self._primary_model
        analyzer = Analyzer(self.client, self.llm_client)
        limits = self._limits_for(model)
        chunks, unanalyzable = self._split_into_chunks(items, limits=limits)
        total_chunks = len(chunks)
        processed = 0

        for item, reason in unanalyzable:
            func = item.function
            logger.warning("Skipping %s: %s", func.name, reason)
            self._emit(Event(
                type=EventType.FUNCTION_ERROR,
                phase=Phase.ANALYSIS,
                message=f"Skipping {func.name}: decompilation exceeds the prompt budget",
                data={"address": func.address, "error": reason},
            ))
            self._record_analysis_result(
                func,
                FunctionResult(
                    address=func.address,
                    original_name=func.name,
                    error=reason,
                    model=model,
                ),
            )

        concurrency = self._chunk_concurrency()
        if concurrency > 1 and total_chunks > 1:
            logger.info(
                "Sending %d chunks %d at a time. The queue is ordered bottom-up "
                "and functions at the same depth do not depend on each other.",
                total_chunks, concurrency,
            )

        for wave_start in range(0, total_chunks, concurrency):
            wave = chunks[wave_start:wave_start + concurrency]

            offset = 0
            for chunk in wave:
                for idx, (item, _) in enumerate(chunk):
                    func = item.function
                    self._emit(Event(
                        type=EventType.FUNCTION_START,
                        phase=Phase.ANALYSIS,
                        message=f"Analyzing {func.name} ({func.address_hex})...",
                        data={
                            "address": func.address,
                            "name": func.name,
                            "size": func.size,
                            "depth": item.depth,
                            "model": model,
                            "progress": (
                                f"{completed_base + processed + offset + idx + 1}"
                                f"/{self.queue.total}"
                            ),
                        },
                    ))
                offset += len(chunk)

            self._wait_if_paused()

            # Prompts are built here, on the one thread that owns self.results:
            # the preamble names functions analyzed so far, and a worker
            # reading it while this loop writes to it is a race for nothing.
            prompts = [
                self._build_chunk_prompt(
                    chunk.entries, limits=limits, preamble=chunk.preamble,
                )
                for chunk in wave
            ]
            for offset_in_wave, (chunk, prompt) in enumerate(zip(wave, prompts)):
                logger.info(
                    "Chunk %d/%d prompt: %d chars (%d functions).",
                    wave_start + offset_in_wave + 1, total_chunks, len(prompt), len(chunk),
                )

            sent = self._send_prompts(prompts, model, limits)

            for offset_in_wave, (chunk, outcome) in enumerate(zip(wave, sent)):
                chunk_num = wave_start + offset_in_wave + 1
                responses, error = outcome
                failures: dict[int, str] = {}

                if error is not None:
                    logger.warning(
                        "Chunk %d/%d failed: %s. Retrying it in halves rather than "
                        "failing all %d functions on one call.",
                        chunk_num, total_chunks, error, len(chunk),
                    )
                    responses, failures = self._recover_chunk(chunk, model, limits, error)

                by_addr = {r.address: r for r in responses if r.address}

                logger.info(
                    "Chunk %d/%d: LLM returned %d responses, %d with valid addresses "
                    "(chunk has %d functions).",
                    chunk_num, total_chunks, len(responses), len(by_addr), len(chunk),
                )

                missing = [
                    item.function.address
                    for item, _ in chunk
                    if item.function.address not in by_addr
                ]
                if missing:
                    logger.warning(
                        "Chunk %d/%d: no response for %s; the model answered for %s.",
                        chunk_num, total_chunks,
                        ", ".join(f"0x{a:08x}" for a in missing),
                        ", ".join(f"0x{a:08x}" for a in sorted(by_addr))
                        or "no usable address",
                    )

                matched = 0
                for item, _ in chunk:
                    func = item.function
                    response = by_addr.get(func.address)

                    if response and response.name:
                        sig_applied = analyzer._write_back(func.address, response)
                        result = FunctionResult(
                            address=func.address,
                            original_name=func.name,
                            name=response.name,
                            signature=response.signature,
                            confidence=response.confidence,
                            classification=response.classification,
                            comments=response.comments,
                            reasoning=response.reasoning,
                            model=model,
                            llm_calls=1,
                            signature_applied=sig_applied,
                            struct_proposals=response.struct_proposals,
                            truncated=func.address in chunk.truncated,
                        )
                        matched += 1
                    else:
                        reason = failures.get(
                            func.address, "No matching response from LLM"
                        )
                        if func.address not in failures and response:
                            reason = response.reasoning or "Empty name in response"
                        result = FunctionResult(
                            address=func.address,
                            original_name=func.name,
                            error=reason,
                            model=model,
                        )
                        self._emit(Event(
                            type=EventType.FUNCTION_ERROR,
                            phase=Phase.ANALYSIS,
                            message=f"No analysis for {func.name}",
                            data={"address": func.address, "error": reason},
                        ))

                    self._record_analysis_result(func, result)

                logger.info(
                    "Chunk %d/%d complete: %d/%d functions matched.",
                    chunk_num, total_chunks, matched, len(chunk),
                )
                processed += len(chunk)

    def _build_chunk_prompt(
        self,
        items: list[tuple[WorkItem, str]],
        limits: ModelLimits | None = None,
        preamble: bool = True,
    ) -> str:
        """Build a prompt with all decompilations for this chunk.

        *preamble* carries the list of functions named so far. A call sending
        one oversized function leaves it out: the list is a convenience, and
        the tenth of the budget it reserves is worth more as body.
        """
        assert self.binary_info is not None

        parts = [
            f"Binary: {self.binary_info.arch} {self.binary_info.format} "
            f"({self.binary_info.compiler})",
            "",
            f"Analyze the following {len(items)} functions.",
            "",
        ]

        # A name the draft model is not trusted on is left out: this preamble
        # is what the next chunks reason from, and a wrong callee name spreads
        # further than the function it was invented for. It comes back once the
        # second pass has confirmed or replaced it.
        threshold = self.config.llm.refine_below
        known = [] if not preamble else [
            (addr, r.name)
            for addr, r in self.results.items()
            if r.name and not r.skipped and not r.error
            and not refinement_reason(r, threshold=threshold)
        ]
        if known:
            # This preamble grows with every function analyzed, so it has to be
            # bounded: on a small context window an unbounded list eventually
            # crowds out the decompilation the prompt is actually about.
            # Newest first — analysis runs bottom-up, so the most recent names
            # are the nearest callees of what we are about to send.
            kept = take_within_budget(
                reversed(known),
                lambda pair: f"- 0x{pair[0]:08x}: {pair[1]}",
                self._known_functions_budget(limits),
            )
            lines = [f"- 0x{addr:08x}: {name}" for addr, name in kept]
            if lines:
                parts.append("### Already Identified Functions")
                parts.extend(sorted(lines))
                parts.append("")

        for item, decompilation in items:
            func = item.function
            parts.append(f"### 0x{func.address:08x}: {func.name} ({func.size} bytes)")
            parts.append("```c")
            parts.append(decompilation)
            parts.append("```")
            parts.append("")

        return "\n".join(parts)

    def _analyze_function_sequential(
        self, item: WorkItem, techniques: list | None = None,
    ) -> FunctionResult:
        """Analyze a single function sequentially (used for obfuscated functions).

        *techniques* is what the analysis phase already decided about this
        function. Left out — the second pass re-reads functions for reasons
        that have nothing to do with obfuscation — the binary-level verdict
        stands in, so a run that dismissed the heuristics does not quietly
        re-enter the agentic loop one function at a time.
        """
        assert self.llm_client is not None
        func = item.function
        if techniques is None and not self._obfuscation_believed:
            techniques = []
        deobfuscator = Deobfuscator(self.client, self.llm_client)
        analyzer = Analyzer(
            self.client,
            self.llm_client,
            deobfuscator=deobfuscator,
            max_prompt_chars=self._get_effective_limits().max_prompt_chars,
            deobfuscation_time_budget=self.config.analysis.deobfuscation_time_budget,
        )
        result = analyzer.analyze(
            item,
            binary_info=self.binary_info,
            known_results=self.results,
            strings=self.strings,
            model=self._primary_model,
            techniques=techniques,
        )
        result.model = self._primary_model

        if result.obfuscation_techniques:
            self._emit(Event(
                type=EventType.DEOBFUSCATION_DETECTED,
                phase=Phase.ANALYSIS,
                message=(
                    f"Obfuscation detected in {func.name}: "
                    f"{', '.join(result.obfuscation_techniques)}"
                ),
                data={
                    "address": func.address,
                    "techniques": result.obfuscation_techniques,
                    "tool_calls": result.deobfuscation_tool_calls,
                },
            ))

        return result

    def _refinement_candidates(
        self, *, manual: bool = False,
    ) -> list[tuple[WorkItem, str]]:
        """Functions the strong model should redo, with the reason for each.

        Run automatically at the end of an analysis, a result the primary
        model already produced is never a candidate: asking the same model the
        same question twice buys nothing. What qualifies is draft output —
        from this run, or from an earlier one by way of the state file.

        Asked for by hand, that restriction is lifted. Someone who starts a
        finishing pass is asking for everything that failed or came back under
        the threshold to be re-read, whoever wrote it; the per-function path it
        uses — callers, callees, strings, the deobfuscation loop — is a
        different question from the batch one, even on the same model.

        Functions with no result at all count too, but only in a staged run:
        there they are the ones the draft held back, or never reached before it
        was stopped, and nothing else is going to come back for them. In a full
        run an unanalyzed function is simply one the analysis has not got to
        yet, and stealing it would cost a single call apiece for work the batch
        pass is about to do far more cheaply.
        """
        threshold = self.config.llm.refine_below
        with self._results_lock:
            results = dict(self.results)

        reasons: dict[int, str] = {}
        for addr, result in results.items():
            if result.refined:
                continue
            drafted = addr in self._pending_refinement or result.model not in (
                "", self._primary_model,
            )
            if not manual and not drafted:
                continue
            reason = refinement_reason(result, threshold=threshold)
            if reason:
                reasons[addr] = reason

        if manual and self.config.stage is not RunStage.FULL:
            reasons.update(self._deferred)
            for item in self.queue.all_items():
                if item.address not in results and item.address not in reasons:
                    reasons[item.address] = "never analyzed"

        candidates: list[tuple[WorkItem, str]] = []
        for addr in sorted(reasons):
            item = self.queue.get_by_address(addr)
            if item is not None:
                candidates.append((item, reasons[addr]))
        return candidates

    @property
    def pending_finish(self) -> int:
        """How many functions a finishing pass would re-read right now.

        A window polls this several times a second to label a button, so the
        answer is cached until a result is written: recounting a few thousand
        results every frame, to watch a number that only moves when one of them
        does, is work nobody asked for. The version is read without the lock —
        worst case the count is one write out of date, which is a number on
        screen, not a decision.
        """
        version = self._result_version
        cached = self._pending_finish_cache
        if cached is not None and cached[0] == version:
            return cached[1]
        count = len(self._refinement_candidates(manual=True))
        self._pending_finish_cache = (version, count)
        return count

    def _announce_pending_finish(self) -> None:
        """Close a draft run by saying what is left, and for whom."""
        pending = self.pending_finish
        held = len(self._deferred)
        detail = f" {held} of them were never sent to a model." if held else ""
        self._emit(Event(
            type=EventType.PHASE_START,
            phase=Phase.ANALYSIS,
            message=(
                f"Draft pass done; {pending} functions are waiting for the "
                f"finishing pass.{detail} Start it when you have looked at "
                f"what the draft produced."
            ),
            data={"pending_finish": pending, "deferred": held},
        ))

    def _run_finish_stage(self) -> None:
        """Pick up a draft left on disk and finish it.

        Triage has already rebuilt the queue, so all this needs is the saved
        results — including the failed ones, which are exactly what it came
        for. Without them there is nothing to finish, and saying so beats
        exporting an empty analysis over the draft.
        """
        self._emit(Event(
            type=EventType.PHASE_START,
            phase=Phase.ANALYSIS,
            message="Loading the draft saved in the output directory...",
        ))

        restored = self._restore_previous_results(include_failed=True)
        if not restored:
            raise ValueError(
                f"No saved analysis in {self.config.output.directory}: run the "
                f"draft stage over this binary before finishing it."
            )

        self._emit(Event(
            type=EventType.PHASE_START,
            phase=Phase.ANALYSIS,
            message=f"Restored {restored} drafted functions.",
            data={"restored": restored},
        ))

        self._run_refinement(manual=True)
        self._save_state()

        self._emit(Event(
            type=EventType.PHASE_COMPLETE,
            phase=Phase.ANALYSIS,
            message=(
                f"Analysis complete. {self.stats.named}/{self.stats.total_functions} "
                f"functions named."
            ),
        ))

    def _run_refinement(self, *, manual: bool = False) -> int:
        """Second pass: hand the draft's weak answers to the strong model.

        One call per function rather than a chunk: this is where the rich
        single-function context — callers, callees, cross-references, strings —
        earns its tokens, and there are few enough functions left for it.

        Returns how many functions came back with something better than they
        had. ``manual`` is the hand-started pass, which casts a wider net (see
        ``_refinement_candidates``) and says so when it finds nothing, since
        somebody is waiting on an answer either way.
        """
        label = "Finishing pass" if manual else "Second pass"
        if self.llm_client is None:
            if manual:
                self._emit(Event(
                    type=EventType.PHASE_COMPLETE,
                    phase=Phase.ANALYSIS,
                    message=f"{label} needs a model to run, and this run has none.",
                    data={"refined": 0, "candidates": 0},
                ))
            return 0

        if not self._program_is_open():
            # Every call below decompiles, so without a program this is 61
            # identical failures and a rewritten export built from none of
            # them. Stop before the first one.
            self._emit(Event(
                type=EventType.PHASE_COMPLETE,
                phase=Phase.ANALYSIS,
                message=(
                    f"{label} needs the binary open in Ghidra, and it has been "
                    f"closed. Analyze it again to reopen the program."
                ),
                data={"refined": 0, "candidates": 0},
            ))
            return 0

        candidates = self._refinement_candidates(manual=manual)
        if not candidates:
            if manual:
                self._emit(Event(
                    type=EventType.PHASE_COMPLETE,
                    phase=Phase.ANALYSIS,
                    message=(
                        f"{label}: nothing to redo. No function failed or came "
                        f"back under {self.config.llm.refine_below}% confidence."
                    ),
                    data={"refined": 0, "candidates": 0},
                ))
            return 0

        self._emit(Event(
            type=EventType.PHASE_START,
            phase=Phase.ANALYSIS,
            message=(
                f"{label}: re-analyzing {len(candidates)} functions with "
                f"{self._primary_model}."
            ),
            data={"model": self._primary_model, "count": len(candidates)},
        ))

        # The draft renamed and retyped as it went, so the cached decompilation
        # is the text the draft saw, not the text a reader would see now.
        # Dropping it is what lets the second pass read its predecessor's work.
        self._invalidate_decompilation([item.address for item, _ in candidates])

        improved = 0
        for item, reason in candidates:
            self._wait_if_paused()
            func = item.function
            # A function the draft held back has no result to improve on, only
            # the reason it was held.
            previous = self.results.get(func.address)
            best_name = (previous.name if previous else "") or func.name
            self._emit(Event(
                type=EventType.FUNCTION_START,
                phase=Phase.ANALYSIS,
                message=(
                    f"Re-analyzing {best_name} "
                    f"({func.address_hex}): {reason}"
                ),
                data={
                    "address": func.address,
                    "name": best_name,
                    "size": func.size,
                    "depth": item.depth,
                    "model": self._primary_model,
                    "reason": reason,
                },
            ))

            try:
                result = self._analyze_function_sequential(item)
            except Exception as e:
                err_msg = _clean_api_error(e)
                logger.exception(
                    "Second pass on %s (%s) failed", func.name, func.address_hex,
                )
                self._emit(Event(
                    type=EventType.FUNCTION_ERROR,
                    phase=Phase.ANALYSIS,
                    message=f"Second pass on {func.name} failed: {err_msg}",
                    data={"address": func.address, "error": err_msg},
                ))
                continue

            if result.error or not result.name:
                # The draft's answer, whatever it is worth, beats no answer —
                # unless there was none, in which case the failure is the
                # result and has to be visible rather than dropped.
                logger.info(
                    "%s on %s produced nothing usable; keeping what was there.",
                    label, func.address_hex,
                )
                kept = "; keeping the draft answer." if previous else "."
                self._emit(Event(
                    type=EventType.FUNCTION_ERROR,
                    phase=Phase.ANALYSIS,
                    message=(
                        f"{label} on {func.name} produced nothing usable{kept}"
                    ),
                    data={
                        "address": func.address,
                        "error": result.error or "Empty name in response",
                    },
                ))
                if previous is None:
                    self._record_analysis_result(func, result)
                continue

            result.refined = True
            # The draft call was paid for too, so the report counts both.
            result.llm_calls += previous.llm_calls if previous else 0
            self._replace_analysis_result(func, result)
            self._pending_refinement.discard(func.address)
            self._deferred.pop(func.address, None)
            improved += 1

        self._emit(Event(
            type=EventType.PHASE_COMPLETE,
            phase=Phase.ANALYSIS,
            message=(
                f"{label} complete. {improved}/{len(candidates)} functions "
                f"re-analyzed by {self._primary_model}."
            ),
            data={"refined": improved, "candidates": len(candidates)},
        ))
        return improved

    #: Results between checkpoints. A chunk holds up to 15 functions, so this
    #: lands roughly every other chunk.
    SAVE_INTERVAL = 25

    def _record_analysis_result(self, func: FunctionInfo, result: FunctionResult) -> None:
        """Record a function result into the results dict and stats."""
        self._store_result(func.address, result)
        self.stats.record_result(result)
        self._after_result(func, result)

    def _replace_analysis_result(
        self, func: FunctionInfo, result: FunctionResult,
    ) -> None:
        """Record a result over one this run already recorded for the function."""
        previous = self.results.get(func.address)
        if previous is None:
            self._record_analysis_result(func, result)
            return

        self._store_result(func.address, result)
        self.stats.replace_result(previous, result)
        # The struct proposals the replaced result argued for go with it:
        # leaving them in would let a draft answer that was thrown away keep
        # voting on field names in the unifier.
        self.struct_accumulator.drop_proposals(func.address)
        self._after_result(func, result)

    def _after_result(self, func: FunctionInfo, result: FunctionResult) -> None:
        self._results_since_save += 1
        if self._results_since_save >= self.SAVE_INTERVAL:
            self._save_state()

        if result.struct_proposals:
            self.struct_accumulator.add_proposals(func.address, result.struct_proposals)

        if not result.error and not result.skipped:
            self._emit(Event(
                type=EventType.FUNCTION_COMPLETE,
                phase=Phase.ANALYSIS,
                message=(
                    f"{func.address_hex} → {result.name} "
                    f"(confidence: {result.confidence}%)"
                ),
                data={
                    "address": func.address,
                    "original_name": result.original_name,
                    "name": result.name,
                    "confidence": result.confidence,
                    "classification": result.classification,
                },
            ))

    def _run_cleanup(self) -> None:
        """Unify struct types and retry failed signatures."""
        self._emit(Event(
            type=EventType.PHASE_START,
            phase=Phase.CLEANUP,
            message="Starting cleanup pass...",
        ))

        structs_created = 0
        retyped: list[int] = []
        resigned: list[int] = []

        if self.struct_accumulator.proposal_count > 0:
            unified = self.struct_accumulator.unify()
            self._emit(Event(
                type=EventType.CLEANUP_TYPES_UNIFIED,
                phase=Phase.CLEANUP,
                message=(
                    f"Unified {self.struct_accumulator.proposal_count} struct proposals "
                    f"into {len(unified)} types."
                ),
                data={
                    "proposals": self.struct_accumulator.proposal_count,
                    "unified": len(unified),
                },
            ))

            retyped = apply_unified_structs(self.client, unified)
            structs_created = len(unified)

            for us in unified:
                self._emit(Event(
                    type=EventType.CLEANUP_TYPE_CREATED,
                    phase=Phase.CLEANUP,
                    message=(
                        f"Created struct '{us.definition.name}' "
                        f"({us.definition.size} bytes, {us.definition.field_count} fields)"
                    ),
                    data={
                        "name": us.definition.name,
                        "size": us.definition.size,
                        "fields": us.definition.field_count,
                    },
                ))

        pending_signatures = [
            r for r in self.results.values()
            if r.signature and not r.signature_applied and not r.skipped and not r.error
        ]
        if pending_signatures:
            sigs_retried = 0
            for r in pending_signatures:
                try:
                    self.client.set_function_signature(r.address, r.signature)
                    r.signature_applied = True
                    resigned.append(r.address)
                    sigs_retried += 1
                except Exception:
                    logger.debug(
                        "Signature retry still failed for 0x%08x: %s",
                        r.address, r.signature,
                    )
            self._emit(Event(
                type=EventType.CLEANUP_SIGNATURES_RETRIED,
                phase=Phase.CLEANUP,
                message=(
                    f"Retried {len(pending_signatures)} pending signatures, "
                    f"{sigs_retried} succeeded."
                ),
                data={
                    "pending": len(pending_signatures),
                    "succeeded": sigs_retried,
                },
            ))

        # Synthesis and export read decompilation through the cache, which was
        # filled during analysis. Without this the types and signatures just
        # applied would never reach either of them.
        self._invalidate_decompilation(retyped + resigned)

        self._emit(Event(
            type=EventType.PHASE_COMPLETE,
            phase=Phase.CLEANUP,
            message=f"Cleanup complete. {structs_created} structs created.",
            data={"structs_created": structs_created},
        ))

    def _record_phase_failure(
        self, phase: Phase, error: Exception, *, detail: str = "",
    ) -> None:
        """Note a phase that failed but let the run carry on.

        The event this accompanies reaches whoever was watching live; this is
        what reaches whoever reads analysis.json afterwards.
        """
        self.phase_failures.append(
            PhaseFailure(phase=phase.value, error=str(error), detail=detail)
        )

    def _run_synthesis(self) -> None:
        """Cross-function synthesis: unify globals, synthesize structs, refine names."""
        self._emit(Event(
            type=EventType.PHASE_START,
            phase=Phase.SYNTHESIS,
            message="Starting synthesis...",
        ))

        if self.llm_client is None:
            self._emit(Event(
                type=EventType.PHASE_COMPLETE,
                phase=Phase.SYNTHESIS,
                message="Synthesis skipped (no LLM client).",
            ))
            return

        decompilations: dict[int, str] = {}
        for addr, result in self.results.items():
            if result.skipped or result.error:
                continue
            decomp = self._get_decompilation(addr)
            if decomp:
                decompilations[addr] = normalize(decomp)

        if not decompilations:
            self._emit(Event(
                type=EventType.PHASE_COMPLETE,
                phase=Phase.SYNTHESIS,
                message="Synthesis skipped (no decompilations).",
            ))
            return

        synthesizer = SemanticSynthesizer(
            self.llm_client,
            max_prompt_chars=self._get_effective_limits().max_prompt_chars,
        )
        try:
            synthesis_result = synthesizer.synthesize(
                list(self.results.values()), decompilations, model=self.llm_client.model,
            )
        except Exception as e:
            logger.warning("Synthesis failed: %s", e)
            self._record_phase_failure(Phase.SYNTHESIS, e)
            self._emit(Event(
                type=EventType.PHASE_COMPLETE,
                phase=Phase.SYNTHESIS,
                message=f"Synthesis failed: {e}",
                data={"error": str(e)},
            ))
            return

        if synthesis_result.partial:
            # Some groups came back and some did not. The result is real and
            # worth applying; the document should still say it is incomplete.
            message = (
                f"Synthesis incomplete: {synthesis_result.failed_passes} of "
                f"{synthesis_result.passes + synthesis_result.failed_passes} "
                f"passes failed."
            )
            logger.warning(message)
            self._record_phase_failure(Phase.SYNTHESIS, RuntimeError(message))

        if synthesis_result.globals:
            self._emit(Event(
                type=EventType.SYNTHESIS_GLOBALS_UNIFIED,
                phase=Phase.SYNTHESIS,
                message=f"Unified {len(synthesis_result.globals)} global variables.",
                data={"count": len(synthesis_result.globals), "globals": synthesis_result.globals},
            ))

        if synthesis_result.structs:
            self._emit(Event(
                type=EventType.SYNTHESIS_STRUCTS_SYNTHESIZED,
                phase=Phase.SYNTHESIS,
                message=f"Synthesized {len(synthesis_result.structs)} structs.",
                data={"count": len(synthesis_result.structs)},
            ))

        if synthesis_result.name_refinements:
            for addr_str, new_name in synthesis_result.name_refinements.items():
                addr = int(addr_str, 16) if addr_str.startswith("0x") else int(addr_str)
                if addr in self.results:
                    self.results[addr].name = new_name
                    try:
                        self.client.rename_function(addr, new_name)
                    except Exception as e:
                        logger.warning(
                            "Failed to rename 0x%08x to %s during synthesis: %s",
                            addr, new_name, e,
                        )
            self._emit(Event(
                type=EventType.SYNTHESIS_NAMES_REFINED,
                phase=Phase.SYNTHESIS,
                message=f"Refined {len(synthesis_result.name_refinements)} function names.",
                data={"count": len(synthesis_result.name_refinements)},
            ))

        self._synthesis_result = synthesis_result

        self._emit(Event(
            type=EventType.PHASE_COMPLETE,
            phase=Phase.SYNTHESIS,
            message="Synthesis complete.",
        ))

    def _transpile_selection(self) -> set[int] | None:
        """The addresses the transpiling exporters may translate, or None for all.

        A translation is a second full LLM pass over the binary, so it is
        normally wanted for one subsystem rather than for 1500 functions of
        runtime. The configured addresses are the ones somebody asked to read;
        what those call comes along by default, because a function translated
        without its callees names things that were never produced.
        """
        wanted = self.config.output.transpile_addresses
        if wanted is None:
            return None

        selection = set(wanted)
        graph = self.triage_result.call_graph if self.triage_result else None
        if self.config.output.transpile_follow_callees and graph is not None:
            selection = graph.closure(selection)

        logger.info(
            "Transpile selection: %d requested, %d with their callees.",
            len(wanted), len(selection),
        )
        return selection

    def _run_export(self) -> None:
        """Generate output files."""
        self._emit(Event(
            type=EventType.PHASE_START,
            phase=Phase.EXPORT,
            message="Starting export...",
        ))

        output_dir = self.config.output.directory
        output_dir.mkdir(parents=True, exist_ok=True)

        exportable_addrs = [
            addr for addr, r in self.results.items()
            if not r.skipped and not r.error
        ]
        decompilations: dict[int, str] = {}
        for addr in exportable_addrs:
            decomp = self._get_decompilation(addr)
            if decomp:
                decompilations[addr] = normalize(decomp)

        if self._synthesis_result:
            synthesizer = SemanticSynthesizer(self.llm_client)
            decompilations = synthesizer.apply_to_decompilations(
                self._synthesis_result, decompilations,
            )

        raw_usage = getattr(self.llm_client, "usage", None) if self.llm_client else None
        token_usage = raw_usage if isinstance(raw_usage, TokenUsage) else TokenUsage()

        export_data = ExportData(
            binary_info=self.binary_info or BinaryInfo(
                arch="unknown", format="unknown", endianness="unknown",
                word_size=0, compiler="unknown", name="unknown",
            ),
            stats=self.stats,
            results=self.results,
            decompilations=decompilations,
            token_usage=token_usage,
            duration_seconds=self.stats.duration_seconds,
            provider=self.config.llm.provider,
            phase_failures=list(self.phase_failures),
            call_graph=self._call_graph(),
        )

        formats = self.config.output.formats
        selection = self._transpile_selection()

        if "source" in formats:
            path = export_source(export_data, output_dir / "decompiled.c")
            self._emit(Event(
                type=EventType.EXPORT_FILE,
                phase=Phase.EXPORT,
                message=f"Exported {path}",
                data={"path": str(path), "format": "source"},
            ))

        for language in TargetLanguage:
            if language.value not in formats:
                continue
            if self.llm_client is None:
                self._emit(Event(
                    type=EventType.PHASE_COMPLETE,
                    phase=Phase.EXPORT,
                    message=(
                        f"{language.display_name} export skipped (no LLM client)."
                    ),
                ))
                continue
            limits = self._get_effective_limits()
            try:
                path = transpile_and_export(
                    export_data,
                    output_dir,
                    language,
                    self.llm_client,
                    max_prompt_chars=limits.max_prompt_chars,
                    max_output_tokens=limits.max_output_tokens,
                    selection=selection,
                )
            except Exception as e:
                logger.warning("%s export failed: %s", language.display_name, e)
                self._record_phase_failure(Phase.EXPORT, e, detail=language.value)
                self._emit(Event(
                    type=EventType.PHASE_COMPLETE,
                    phase=Phase.EXPORT,
                    message=f"{language.display_name} export failed: {e}",
                    data={"error": str(e)},
                ))
                continue
            self._emit(Event(
                type=EventType.EXPORT_FILE,
                phase=Phase.EXPORT,
                message=f"Exported {path}",
                data={"path": str(path), "format": language.value},
            ))

        if "json" in formats:
            # Refreshed rather than left as built: a transpiling export that
            # failed just above is a phase failure too, and json is written
            # last precisely so it can still report one.
            export_data.phase_failures = list(self.phase_failures)
            path = export_json(export_data, output_dir / "analysis.json")
            self._emit(Event(
                type=EventType.EXPORT_FILE,
                phase=Phase.EXPORT,
                message=f"Exported {path}",
                data={"path": str(path), "format": "json"},
            ))

        self._emit(Event(
            type=EventType.PHASE_COMPLETE,
            phase=Phase.EXPORT,
            message=f"Export complete. Files saved to {output_dir}/",
            data={"output_dir": str(output_dir)},
        ))

    def _find_function(self, addr: int) -> FunctionInfo | None:
        return self._functions_by_addr.get(addr)

    def _stats_dict(self) -> dict[str, int | float]:
        s = self.stats
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
            "signature_matches": s.signature_matches,
            "duration_seconds": round(s.duration_seconds, 1),
        }
