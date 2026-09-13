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

if TYPE_CHECKING:
    from kong.agent.models import AnalysisStats

logger = logging.getLogger(__name__)


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

        self.state = RunState(
            running=True,
            phase="opening binary",
            binary_label=Path(self.settings.binary_path).name,
        )
        self._start_time = time.time()
        self._paused_total = 0.0
        self._pause_start = None
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        config = self.settings.to_config()
        try:
            client = self._ghidra_client_factory(
                self.settings.binary_path, self.settings.ghidra_dir or None
            )
            with self._lock:
                self._ghidra_client = client

            llm_client = self._llm_client_factory(config.llm)
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
        finally:
            self._close_ghidra_client()

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
