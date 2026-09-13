"""Glue between the Supervisor and a GUI main loop.

Deliberately free of any tkinter import: the supervisor runs on a worker
thread and publishes events into a queue, and everything the interface needs
to render is derived here. That keeps this layer testable without a display,
and keeps widget access on the main thread, where Tk requires it.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kong.agent.events import Event, EventType
from kong.agent.refinement import DEFAULT_REFINE_BELOW
from kong.banner import _ENV_VARS, check_api_key
from kong.config import (
    GhidraConfig,
    KongConfig,
    LLMConfig,
    LLMProvider,
    OutputConfig,
    RunStage,
)
from kong.llm.activity import ActivitySnapshot, LLMActivity, TrackedLLMClient

if TYPE_CHECKING:
    from kong.agent.models import AnalysisStats

logger = logging.getLogger(__name__)

#: Refusal shared by both manual passes. Ghidra is released when the window
#: closes, so a controller kept alive past that has results to show but no
#: program to re-read them from.
_GHIDRA_RELEASED = (
    "Ghidra has been released for this binary. Analyze it again to reopen the "
    "program, then run this pass."
)


@dataclass
class RunSettings:
    """Everything the user picks in the window before starting a run."""

    binary_path: str = ""
    output_dir: str = ""
    formats: list[str] = field(default_factory=lambda: ["source", "json"])
    provider: LLMProvider = LLMProvider.ANTHROPIC
    model: str = ""
    draft_model: str = ""
    refine_below: int = DEFAULT_REFINE_BELOW
    base_url: str = ""
    api_key: str = ""
    max_prompt_chars: int | None = None
    max_chunk_functions: int | None = None
    max_output_tokens: int | None = None
    ghidra_dir: str = ""
    #: Matches the CLI, where resuming is the default and --fresh opts out.
    resume: bool = True
    #: FULL drafts and finishes in one go; DRAFT stops after the first pass
    #: and leaves the second to the Finish pass button.
    stage: RunStage = RunStage.FULL

    def validate(self) -> list[str]:
        """Return the reasons this configuration cannot be run, if any."""
        problems: list[str] = []

        if not self.binary_path:
            problems.append("Select a binary to analyze.")
        elif not Path(self.binary_path).is_file():
            problems.append(f"Not a file: {self.binary_path}")

        if not self.output_dir:
            problems.append("Select an output directory.")

        if not self.formats:
            problems.append("Select at least one output format.")

        if self.provider is LLMProvider.CUSTOM:
            if not self.base_url:
                problems.append("A custom endpoint needs a base URL.")
            elif not self.base_url.startswith(("http://", "https://")):
                problems.append("The base URL must start with http:// or https://.")
            if not self.model:
                problems.append("A custom endpoint needs a model name.")
        elif self.base_url and not self.base_url.startswith(("http://", "https://")):
            problems.append("The base URL must start with http:// or https://.")

        # Checked here rather than when the client is built: by then Ghidra has
        # spent a minute opening the binary, and a missing key is a question
        # the dialog can answer immediately.
        if not self.api_key:
            env_var = _ENV_VARS.get(self.provider)
            if env_var is not None and not check_api_key(self.provider):
                problems.append(
                    f"{env_var} is not set. Export it and start Kong again."
                )

        if not 0 <= self.refine_below <= 100:
            problems.append("Refine below must be a percentage between 0 and 100.")

        if self.draft_model and not self.model:
            problems.append("A draft model needs a second model to refine with.")

        for label, value in (
            ("Max prompt chars", self.max_prompt_chars),
            ("Max functions per batch", self.max_chunk_functions),
            ("Max output tokens", self.max_output_tokens),
        ):
            if value is not None and value <= 0:
                problems.append(f"{label} must be greater than zero.")

        return problems

    def to_config(self) -> KongConfig:
        return KongConfig(
            ghidra=GhidraConfig(install_dir=self.ghidra_dir or None),
            llm=LLMConfig(
                provider=self.provider,
                model=self.model or None,
                draft_model=self.draft_model or None,
                refine_below=self.refine_below,
                api_key=self.api_key or None,
                base_url=self.base_url or None,
                max_prompt_chars=self.max_prompt_chars,
                max_chunk_functions=self.max_chunk_functions,
                max_output_tokens=self.max_output_tokens,
            ),
            output=OutputConfig(
                directory=Path(self.output_dir), formats=list(self.formats)
            ),
            headless=True,
            stage=self.stage,
            resume=self.resume,
        )


@dataclass
class RunState:
    """Snapshot the interface renders. Owned by the main thread."""

    phase: str = "idle"
    completed: int = 0
    total: int = 0
    high_confidence: int = 0
    medium_confidence: int = 0
    low_confidence: int = 0
    llm_calls: int = 0
    cost_usd: float = 0.0
    elapsed_seconds: float = 0.0
    running: bool = False
    paused: bool = False
    finished: bool = False
    #: True while the manual coherence pass is working. It runs on its own
    #: thread, so the window has to know not to offer it twice.
    checking_coherence: bool = False
    #: Same for the manual finishing pass.
    finishing: bool = False
    #: How many functions a finishing pass would re-read right now. What makes
    #: the button worth pressing, so it belongs in the rendered state.
    pending_finish: int = 0
    error: str = ""
    binary_label: str = ""
    #: True while at least one request has been sent and not yet answered.
    #: Minutes can pass there with nothing else to show, which is exactly the
    #: stretch where a window that says nothing looks like a hung one.
    llm_waiting: bool = False
    #: Requests in flight, and what the oldest one is waiting for.
    llm_in_flight: int = 0
    llm_wait_label: str = ""
    llm_wait_seconds: float = 0.0
    #: Every answered request, and the time the run has spent waiting.
    llm_answered: int = 0
    llm_wait_total_seconds: float = 0.0
    llm_last_wait_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def progress_fraction(self) -> float:
        if self.total <= 0:
            return 0.0
        return min(1.0, self.completed / self.total)


def _default_ghidra_client(binary_path: str, install_dir: str | None) -> Any:
    from kong.ghidra.client import GhidraClient

    client = GhidraClient(binary_path=binary_path, install_dir=install_dir)
    client.open()
    return client


def _default_llm_client(llm_config: LLMConfig) -> Any:
    from kong.__main__ import create_llm_client

    return create_llm_client(llm_config)


class AnalysisController:
    """Runs one analysis on a worker thread and reports progress to the GUI."""

    def __init__(
        self,
        settings: RunSettings,
        *,
        ghidra_client_factory: Callable[[str, str | None], Any] = _default_ghidra_client,
        llm_client_factory: Callable[[LLMConfig], Any] = _default_llm_client,
        supervisor_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.settings = settings
        self.state = RunState(binary_label=Path(settings.binary_path).name)
        self._events: queue.Queue[Event] = queue.Queue()
        self._ghidra_client_factory = ghidra_client_factory
        self._llm_client_factory = llm_client_factory
        self._supervisor_factory = supervisor_factory or self._build_supervisor
        #: Shared with the wrapper around the LLM client, so every request
        #: the run makes is timed whichever thread sends it.
        self.activity = LLMActivity()
        self._thread: threading.Thread | None = None
        self._coherence_thread: threading.Thread | None = None
        self._finish_thread: threading.Thread | None = None
        self._supervisor: Any = None
        self._llm_client: Any = None
        self._ghidra_client: Any = None
        self._lock = threading.Lock()
        self._start_time: float = 0.0
        self._paused_total: float = 0.0
        self._pause_start: float | None = None

    # ---------------------------------------------------------------- lifecycle

    @staticmethod
    def _build_supervisor(
        ghidra_client: Any, config: KongConfig, llm_client: Any
    ) -> Any:
        from kong.agent.supervisor import Supervisor

        return Supervisor(ghidra_client, config, llm_client=llm_client)

    def start(self) -> None:
        """Launch the run. Raises ValueError if the settings are unusable."""
        problems = self.settings.validate()
        if problems:
            raise ValueError("\n".join(problems))
        if self.state.running:
            raise RuntimeError("An analysis is already running.")
        if self._manual_pass_running():
            raise RuntimeError(
                "A finishing pass or coherence review is still running. "
                "Wait for it: starting another analysis would close the "
                "program it is writing to."
            )

        self.state = RunState(
            running=True,
            phase="opening binary",
            binary_label=Path(self.settings.binary_path).name,
        )
        # A second run starts from zero calls waited on, like it starts from
        # zero functions analyzed.
        self.activity = LLMActivity()
        self._start_time = time.time()
        self._paused_total = 0.0
        self._pause_start = None
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        config = self.settings.to_config()
        try:
            # A second run in the same window opens a second program; the one
            # the previous run left behind is released here rather than leaked.
            self._close_ghidra_client()
            client = self._ghidra_client_factory(
                self.settings.binary_path, self.settings.ghidra_dir or None
            )
            with self._lock:
                self._ghidra_client = client

            llm_client = TrackedLLMClient(
                self._llm_client_factory(config.llm), self.activity
            )
            with self._lock:
                self._llm_client = llm_client

            supervisor = self._supervisor_factory(client, config, llm_client)
            supervisor.on_event(self._events.put)
            with self._lock:
                self._supervisor = supervisor

            supervisor.run()
        except Exception as exc:
            # The window is the only place the user can see this, so it has to
            # travel back as an event rather than die with the worker thread.
            logger.exception("Analysis failed")
            self._events.put(
                Event(type=EventType.RUN_ERROR, message=f"{type(exc).__name__}: {exc}")
            )
        # Ghidra is deliberately left open here. The finishing pass and the
        # coherence review are started from the window *after* this thread has
        # ended, and both read and write the same program database: closing it
        # on the way out left them re-analyzing every function against a shut
        # program, one "Not open. Call open() first." apiece. `shutdown` is
        # what releases it, when the window closes.

    def _ghidra_is_open(self) -> bool:
        """True while there is a program for a manual pass to work on.

        Duck-typed like the rest of the client: a factory that hands back
        something without `is_open` is taken at its word.
        """
        with self._lock:
            client = self._ghidra_client
        return client is not None and bool(getattr(client, "is_open", True))

    def _close_ghidra_client(self) -> None:
        with self._lock:
            client = self._ghidra_client
            self._ghidra_client = None
        if client is None:
            return
        try:
            client.close()
        except Exception:
            logger.exception("Failed to close the Ghidra client")

    def shutdown(self) -> None:
        """Pause the run, save what it has done, and release Ghidra.

        Called when the window closes, which is the one moment where the
        analysis thread is about to be abandoned mid-function: the checkpoint
        has to be taken here, from the main thread, rather than left to a
        worker that will not get to run again.
        """
        supervisor = self._get_supervisor()
        if supervisor is not None:
            if not supervisor.is_paused:
                supervisor.pause()
            try:
                supervisor.checkpoint()
            except Exception:
                logger.exception("Could not save the analysis state on close")
        self._close_ghidra_client()

    # ------------------------------------------------------------------ polling

    def _get_supervisor(self) -> Any:
        with self._lock:
            return self._supervisor

    def _manual_pass_running(self) -> bool:
        """True while the finishing pass or the coherence review is working."""
        return any(
            thread is not None and thread.is_alive()
            for thread in (self._coherence_thread, self._finish_thread)
        )

    def poll(self) -> list[Event]:
        """Drain pending events, fold them into the state, and return them."""
        drained: list[Event] = []
        while True:
            try:
                drained.append(self._events.get_nowait())
            except queue.Empty:
                break

        for event in drained:
            self._apply(event)

        self._refresh_live_counters()
        return drained

    def _apply(self, event: Event) -> None:
        state = self.state

        if event.type is EventType.TRIAGE_QUEUE_BUILT:
            state.total = int(event.data.get("queue_size", 0))
        elif event.type is EventType.PHASE_START and event.phase is not None:
            state.phase = event.phase.value
        elif event.type in (EventType.FUNCTION_COMPLETE, EventType.FUNCTION_SKIPPED):
            state.completed += 1
        elif event.type is EventType.RUN_ERROR:
            state.error = event.message
            state.running = False
            state.finished = True
            state.phase = "failed"
        elif event.type is EventType.RUN_COMPLETE:
            state.running = False
            state.finished = True
            state.phase = "complete"

    def _refresh_live_counters(self) -> None:
        state = self.state
        supervisor = self._get_supervisor()
        if supervisor is not None:
            stats: AnalysisStats = supervisor.stats
            state.high_confidence = stats.high_confidence
            state.medium_confidence = stats.medium_confidence
            state.low_confidence = stats.low_confidence
            state.llm_calls = stats.llm_calls
            state.paused = supervisor.is_paused

        with self._lock:
            llm_client = self._llm_client
        if llm_client is not None:
            state.cost_usd = getattr(llm_client, "total_cost_usd", 0.0)
            usage = getattr(llm_client, "usage", None)
            if usage is not None:
                state.input_tokens = getattr(usage, "input_tokens", 0)
                state.output_tokens = getattr(usage, "output_tokens", 0)

        self._apply_activity(self.activity.snapshot())

        state.checking_coherence = (
            self._coherence_thread is not None and self._coherence_thread.is_alive()
        )
        state.finishing = (
            self._finish_thread is not None and self._finish_thread.is_alive()
        )
        if supervisor is not None:
            state.pending_finish = supervisor.pending_finish

        # The clock stops once the run is over, and does not count paused time.
        if self._start_time and not state.finished:
            paused_now = time.time() - self._pause_start if self._pause_start else 0.0
            state.elapsed_seconds = (
                time.time() - self._start_time - self._paused_total - paused_now
            )

    def _apply_activity(self, snapshot: ActivitySnapshot) -> None:
        """Render what is in flight right now, and what it cost in waiting."""
        state = self.state
        state.llm_waiting = snapshot.waiting
        state.llm_in_flight = snapshot.in_flight
        state.llm_wait_label = snapshot.label
        state.llm_wait_seconds = snapshot.waiting_seconds
        state.llm_answered = snapshot.completed
        state.llm_wait_total_seconds = snapshot.total_wait_seconds
        state.llm_last_wait_seconds = snapshot.last_wait_seconds

    # ------------------------------------------------------------------ controls

    def toggle_pause(self) -> bool:
        """Pause or resume. Returns True if the run is now paused."""
        supervisor = self._get_supervisor()
        if supervisor is None:
            return False

        if supervisor.is_paused:
            if self._pause_start is not None:
                self._paused_total += time.time() - self._pause_start
                self._pause_start = None
            supervisor.resume()
        else:
            self._pause_start = time.time()
            supervisor.pause()

        self.state.paused = supervisor.is_paused
        return self.state.paused

    def request_export(self) -> bool:
        """Export the results collected so far, without ending the run."""
        supervisor = self._get_supervisor()
        if supervisor is None:
            return False
        threading.Thread(target=supervisor.export, daemon=True).start()
        return True

    def request_coherence_review(self) -> str:
        """Start the coherence pass. Returns why it did not, or "".

        A reason rather than a bare False: unlike an export, there are several
        ways for this to be the wrong moment, and the window has to be able to
        say which one. The pass renames functions and rewrites signatures in
        the program database, so it cannot share Ghidra with an analysis that
        is still writing to it — pausing is what makes the two safe to
        interleave.
        """
        supervisor = self._get_supervisor()
        if supervisor is None:
            return "Analyze a binary first: there is nothing to cross-check yet."
        # The thread and the supervisor, not the rendered state: a run that
        # ended without saying so is not a reason to refuse.
        analyzing = self._thread is not None and self._thread.is_alive()
        if analyzing and not supervisor.is_paused:
            return (
                "Pause the analysis first. The coherence pass writes names and "
                "signatures back into Ghidra, and cannot do that while the run "
                "is still writing its own."
            )
        if self._coherence_thread is not None and self._coherence_thread.is_alive():
            return "The coherence pass is already running."
        if self._finish_thread is not None and self._finish_thread.is_alive():
            return "Wait for the finishing pass: it is still renaming functions."
        if not self._ghidra_is_open():
            return _GHIDRA_RELEASED

        self.state.checking_coherence = True
        self._coherence_thread = threading.Thread(
            target=supervisor.review_coherence, daemon=True,
        )
        self._coherence_thread.start()
        return ""

    def request_finishing_pass(self) -> str:
        """Start the finishing pass. Returns why it did not, or "".

        The second half of a draft run, on the same terms as the coherence
        pass: it writes names and signatures back into Ghidra, so it cannot
        share the program database with a run that is still writing its own,
        and it re-exports, so nothing else may be exporting either.
        """
        supervisor = self._get_supervisor()
        if supervisor is None:
            return "Analyze a binary first: there is no draft to finish yet."
        analyzing = self._thread is not None and self._thread.is_alive()
        if analyzing and not supervisor.is_paused:
            return (
                "Pause the analysis first. The finishing pass rewrites names "
                "and signatures in Ghidra, and cannot do that while the run is "
                "still writing its own."
            )
        if self._finish_thread is not None and self._finish_thread.is_alive():
            return "The finishing pass is already running."
        if self._coherence_thread is not None and self._coherence_thread.is_alive():
            return "Wait for the coherence pass: it is still renaming functions."
        if not self._ghidra_is_open():
            return _GHIDRA_RELEASED
        if supervisor.pending_finish == 0:
            return (
                "Nothing to finish: no function failed or came back under the "
                "confidence threshold."
            )

        self.state.finishing = True
        self._finish_thread = threading.Thread(
            target=supervisor.run_finishing_pass, daemon=True,
        )
        self._finish_thread.start()
        return ""
