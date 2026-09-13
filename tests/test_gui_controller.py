"""Tests for the GUI controller (no display required)."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from kong.agent.events import Event, EventType, Phase
from kong.agent.models import AnalysisStats
from kong.config import LLMProvider, RunStage
from kong.gui.controller import AnalysisController, RunSettings, RunState


@pytest.fixture(autouse=True)
def _provider_keys(monkeypatch, tmp_path):
    """A machine with the API keys configured, and its own config store.

    Without the keys every settings check would fail on the missing key rather
    than on what the test is about; the tests that care delete them again. The
    config directory is redirected so saving a key never touches the one
    belonging to whoever is running the tests.
    """
    monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path / "kong-config"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test1234")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test1234")
    monkeypatch.setenv("ZAI_API_KEY", "zai-test1234")


def _settings(tmp_path, **overrides) -> RunSettings:
    binary = tmp_path / "target.bin"
    binary.write_bytes(b"\x7fELF")
    params = dict(
        binary_path=str(binary),
        output_dir=str(tmp_path / "out"),
        provider=LLMProvider.ANTHROPIC,
    )
    params.update(overrides)
    return RunSettings(**params)


class _FakeSupervisor:
    """Stands in for Supervisor: emits a scripted event sequence."""

    def __init__(self, events: list[Event] | None = None) -> None:
        self.stats = AnalysisStats()
        self.events = events or []
        self.listeners: list = []
        self.is_paused = False
        self.exported = False
        self.pending_finish = 0
        self.reviewed = threading.Event()
        self.finished_pass = threading.Event()
        self.finish_run: threading.Event | None = None
        self.ran = False
        self.checkpoints = 0
        self.checkpoint_error: Exception | None = None

    def on_event(self, callback) -> None:
        self.listeners.append(callback)

    def run(self) -> None:
        self.ran = True
        for event in self.events:
            for callback in self.listeners:
                callback(event)
        if self.finish_run is not None:
            # Hold the analysis thread open, the way a real run does while it
            # is still writing to Ghidra.
            self.finish_run.wait(timeout=5)

    def pause(self) -> None:
        self.is_paused = True

    def resume(self) -> None:
        self.is_paused = False

    def export(self) -> None:
        self.exported = True

    def review_coherence(self) -> None:
        self.reviewed.set()

    def run_finishing_pass(self) -> int:
        self.finished_pass.set()
        return 0

    def checkpoint(self) -> None:
        if self.checkpoint_error is not None:
            raise self.checkpoint_error
        self.checkpoints += 1


def _controller(settings, supervisor, ghidra_client=None):
    ghidra_client = ghidra_client or MagicMock()
    return AnalysisController(
        settings,
        ghidra_client_factory=lambda path, install_dir: ghidra_client,
        llm_client_factory=lambda config: MagicMock(total_cost_usd=1.25),
        supervisor_factory=lambda client, config, llm: supervisor,
    )


def _run_to_completion(controller: AnalysisController) -> None:
    controller.start()
    deadline = time.time() + 5
    while time.time() < deadline:
        controller.poll()
        if controller.state.finished or not (
            controller._thread and controller._thread.is_alive()
        ):
            break
        time.sleep(0.01)
    controller.poll()


class TestRunSettings:
    def test_valid_settings_have_no_problems(self, tmp_path):
        assert _settings(tmp_path).validate() == []

    def test_missing_binary_is_reported(self, tmp_path):
        settings = _settings(tmp_path, binary_path="")
        assert any("Select a binary" in p for p in settings.validate())

    def test_nonexistent_binary_is_reported(self, tmp_path):
        settings = _settings(tmp_path, binary_path=str(tmp_path / "nope.bin"))
        assert any("Not a file" in p for p in settings.validate())

    def test_custom_provider_needs_url_and_model(self, tmp_path):
        settings = _settings(tmp_path, provider=LLMProvider.CUSTOM)
        problems = settings.validate()
        assert any("base URL" in p for p in problems)
        assert any("model name" in p for p in problems)

    def test_custom_provider_rejects_a_bare_host(self, tmp_path):
        settings = _settings(
            tmp_path,
            provider=LLMProvider.CUSTOM,
            base_url="127.0.0.1:8080/v1",
            model="local",
        )
        assert any("http://" in p for p in settings.validate())

    def test_custom_provider_accepts_a_local_endpoint(self, tmp_path):
        settings = _settings(
            tmp_path,
            provider=LLMProvider.CUSTOM,
            base_url="http://127.0.0.1:8080/v1",
            model="kong-local",
            max_prompt_chars=32000,
        )
        assert settings.validate() == []

    def test_non_positive_limits_are_rejected(self, tmp_path):
        settings = _settings(
            tmp_path,
            provider=LLMProvider.CUSTOM,
            base_url="http://127.0.0.1:8080/v1",
            model="kong-local",
            max_output_tokens=0,
        )
        assert any("greater than zero" in p for p in settings.validate())

    def test_formats_default_to_the_cli_defaults(self, tmp_path):
        assert _settings(tmp_path).formats == ["source", "json"]

    def test_no_format_selected_is_rejected(self, tmp_path):
        settings = _settings(tmp_path, formats=[])
        assert any("at least one output format" in p for p in settings.validate())

    def test_to_config_carries_the_formats(self, tmp_path):
        settings = _settings(tmp_path, formats=["json", "python"])
        assert settings.to_config().output.formats == ["json", "python"]

    def test_to_config_carries_the_custom_limits(self, tmp_path):
        settings = _settings(
            tmp_path,
            provider=LLMProvider.CUSTOM,
            base_url="http://127.0.0.1:8080/v1",
            model="kong-local",
            max_prompt_chars=32000,
            max_chunk_functions=8,
            max_output_tokens=2048,
        )
        config = settings.to_config()
        assert config.llm.provider is LLMProvider.CUSTOM
        assert config.llm.base_url == "http://127.0.0.1:8080/v1"
        assert config.llm.max_prompt_chars == 32000
        assert config.llm.max_chunk_functions == 8
        assert config.llm.max_output_tokens == 2048
        assert config.output.directory == Path(settings.output_dir)

    def test_to_config_carries_the_two_pass_settings(self, tmp_path):
        settings = _settings(
            tmp_path,
            model="qwen2.5-coder-32b",
            draft_model="qwen2.5-coder-7b",
            refine_below=65,
        )
        config = settings.to_config()

        assert config.llm.draft_model == "qwen2.5-coder-7b"
        assert config.llm.refine_below == 65

    def test_no_draft_model_means_a_single_pass(self, tmp_path):
        assert _settings(tmp_path).to_config().llm.draft_model is None

    def test_a_threshold_outside_the_scale_is_rejected(self, tmp_path):
        problems = _settings(tmp_path, refine_below=150).validate()

        assert any("between 0 and 100" in p for p in problems)

    def test_a_draft_model_without_a_second_model_is_rejected(self, tmp_path):
        problems = _settings(tmp_path, model="", draft_model="fast-local").validate()

        assert any("needs a second model" in p for p in problems)


class TestProviderKeys:
    def test_a_missing_key_is_caught_before_ghidra_opens_anything(
        self, tmp_path, monkeypatch,
    ):
        monkeypatch.delenv("ZAI_API_KEY", raising=False)
        settings = _settings(tmp_path, provider=LLMProvider.ZAI)

        assert any("ZAI_API_KEY" in p for p in settings.validate())

    def test_a_key_given_in_the_settings_is_enough(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ZAI_API_KEY", raising=False)
        settings = _settings(
            tmp_path, provider=LLMProvider.ZAI, api_key="zai-key",
        )

        assert settings.validate() == []

    def test_a_custom_endpoint_needs_no_key(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ZAI_API_KEY", raising=False)
        settings = _settings(
            tmp_path,
            provider=LLMProvider.CUSTOM,
            base_url="http://127.0.0.1:8080/v1",
            model="kong-local",
        )

        assert settings.validate() == []

    def test_zai_carries_its_base_url_to_the_config(self, tmp_path):
        settings = _settings(
            tmp_path,
            provider=LLMProvider.ZAI,
            base_url="https://api.z.ai/api/coding/paas/v4",
        )

        assert settings.validate() == []
        assert settings.to_config().llm.base_url == (
            "https://api.z.ai/api/coding/paas/v4"
        )

    def test_a_malformed_base_url_is_rejected_for_zai_too(self, tmp_path):
        settings = _settings(
            tmp_path, provider=LLMProvider.ZAI, base_url="api.z.ai/api/paas/v4",
        )

        assert any("http://" in p for p in settings.validate())


class TestRunState:
    def test_progress_is_zero_before_the_queue_is_known(self):
        assert RunState().progress_fraction == 0.0

    def test_progress_is_capped_at_one(self):
        assert RunState(completed=12, total=10).progress_fraction == 1.0

    def test_progress_is_a_ratio(self):
        assert RunState(completed=5, total=10).progress_fraction == 0.5


class TestAnalysisController:
    def test_start_rejects_invalid_settings(self, tmp_path):
        controller = _controller(
            _settings(tmp_path, binary_path=""), _FakeSupervisor()
        )
        with pytest.raises(ValueError, match="Select a binary"):
            controller.start()

    def test_events_reach_the_caller_and_update_the_state(self, tmp_path):
        supervisor = _FakeSupervisor([
            Event(type=EventType.RUN_START, message="starting"),
            Event(
                type=EventType.TRIAGE_QUEUE_BUILT,
                phase=Phase.TRIAGE,
                message="queue built",
                data={"queue_size": 3},
            ),
            Event(
                type=EventType.PHASE_START, phase=Phase.ANALYSIS, message="analysis"
            ),
            Event(type=EventType.FUNCTION_COMPLETE, message="one", data={}),
            Event(type=EventType.FUNCTION_SKIPPED, message="two", data={}),
            Event(type=EventType.RUN_COMPLETE, message="done"),
        ])
        controller = _controller(_settings(tmp_path), supervisor)

        _run_to_completion(controller)

        assert supervisor.ran
        assert controller.state.total == 3
        assert controller.state.completed == 2
        assert controller.state.phase == "complete"
        assert controller.state.finished
        assert not controller.state.running
        assert controller.state.progress_fraction == pytest.approx(2 / 3)

    def test_a_worker_failure_becomes_a_run_error(self, tmp_path):
        def explode(path, install_dir):
            raise RuntimeError("Ghidra is not installed")

        controller = AnalysisController(
            _settings(tmp_path),
            ghidra_client_factory=explode,
            llm_client_factory=lambda config: MagicMock(),
            supervisor_factory=lambda *a: _FakeSupervisor(),
        )

        _run_to_completion(controller)

        assert "Ghidra is not installed" in controller.state.error
        assert controller.state.finished
        assert not controller.state.running

    def test_the_ghidra_client_is_closed_when_the_run_ends(self, tmp_path):
        ghidra_client = MagicMock()
        controller = _controller(
            _settings(tmp_path),
            _FakeSupervisor([Event(type=EventType.RUN_COMPLETE, message="done")]),
            ghidra_client=ghidra_client,
        )

        _run_to_completion(controller)

        ghidra_client.close.assert_called_once()

    def test_cost_and_counters_come_from_the_live_objects(self, tmp_path):
        supervisor = _FakeSupervisor()
        supervisor.stats.high_confidence = 7
        supervisor.stats.llm_calls = 4
        controller = _controller(_settings(tmp_path), supervisor)

        _run_to_completion(controller)

        assert controller.state.high_confidence == 7
        assert controller.state.llm_calls == 4
        assert controller.state.cost_usd == 1.25

    def test_toggle_pause_drives_the_supervisor(self, tmp_path):
        supervisor = _FakeSupervisor()
        controller = _controller(_settings(tmp_path), supervisor)
        _run_to_completion(controller)

        assert controller.toggle_pause() is True
        assert supervisor.is_paused
        assert controller.toggle_pause() is False
        assert not supervisor.is_paused

    def test_paused_time_is_not_counted_as_elapsed(self, tmp_path):
        supervisor = _FakeSupervisor()
        controller = _controller(_settings(tmp_path), supervisor)
        _run_to_completion(controller)
        controller.state.finished = False  # keep the clock running

        controller.toggle_pause()
        time.sleep(0.05)
        controller.toggle_pause()
        controller.poll()

        assert controller._paused_total >= 0.05

    def test_controls_are_inert_before_a_run_starts(self, tmp_path):
        controller = _controller(_settings(tmp_path), _FakeSupervisor())
        assert controller.toggle_pause() is False
        assert controller.request_export() is False

    def test_export_is_requested_on_a_worker_thread(self, tmp_path):
        supervisor = _FakeSupervisor()
        controller = _controller(_settings(tmp_path), supervisor)
        _run_to_completion(controller)

        assert controller.request_export() is True
        deadline = time.time() + 2
        while time.time() < deadline and not supervisor.exported:
            time.sleep(0.01)
        assert supervisor.exported


class TestCoherenceReview:
    """The manual cross-check, which the window offers once a run exists."""

    def test_there_is_nothing_to_cross_check_before_a_run(self, tmp_path):
        controller = _controller(_settings(tmp_path), _FakeSupervisor())
        assert "Analyze a binary first" in controller.request_coherence_review()

    @staticmethod
    def _started(tmp_path, supervisor):
        """A controller whose analysis thread is still inside the run."""
        supervisor.finish_run = threading.Event()
        controller = _controller(_settings(tmp_path), supervisor)
        controller.start()
        deadline = time.time() + 2
        while time.time() < deadline and not supervisor.ran:
            controller.poll()
            time.sleep(0.01)
        controller.poll()
        return controller

    def test_a_running_analysis_is_asked_to_pause_first(self, tmp_path):
        """Both write names into Ghidra; only one of them may be doing it."""
        supervisor = _FakeSupervisor()
        controller = self._started(tmp_path, supervisor)
        try:
            assert "Pause the analysis first" in controller.request_coherence_review()
            assert not supervisor.reviewed.is_set()
        finally:
            supervisor.finish_run.set()

    def test_a_paused_run_can_be_cross_checked(self, tmp_path):
        supervisor = _FakeSupervisor()
        controller = self._started(tmp_path, supervisor)
        try:
            controller.toggle_pause()
            assert controller.request_coherence_review() == ""
            assert supervisor.reviewed.wait(timeout=2)
        finally:
            supervisor.finish_run.set()

    def test_it_runs_on_a_worker_thread_and_says_so(self, tmp_path):
        supervisor = _FakeSupervisor()
        controller = _controller(_settings(tmp_path), supervisor)
        _run_to_completion(controller)

        assert controller.request_coherence_review() == ""
        assert supervisor.reviewed.wait(timeout=2)
        assert controller._coherence_thread is not None

    def test_the_state_reports_the_pass_while_it_lasts(self, tmp_path):
        finish = threading.Event()
        supervisor = _FakeSupervisor()
        supervisor.review_coherence = lambda: finish.wait(timeout=2)
        controller = _controller(_settings(tmp_path), supervisor)
        _run_to_completion(controller)

        controller.request_coherence_review()
        controller.poll()
        assert controller.state.checking_coherence

        finish.set()
        controller._coherence_thread.join(timeout=2)
        controller.poll()
        assert not controller.state.checking_coherence

    def test_a_second_request_waits_for_the_first(self, tmp_path):
        finish = threading.Event()
        supervisor = _FakeSupervisor()
        supervisor.review_coherence = lambda: finish.wait(timeout=2)
        controller = _controller(_settings(tmp_path), supervisor)
        _run_to_completion(controller)

        assert controller.request_coherence_review() == ""
        try:
            assert "already running" in controller.request_coherence_review()
        finally:
            finish.set()
            controller._coherence_thread.join(timeout=2)

    def test_shutdown_pauses_and_releases_ghidra(self, tmp_path):
        ghidra_client = MagicMock()
        supervisor = _FakeSupervisor()
        controller = _controller(
            _settings(tmp_path), supervisor, ghidra_client=ghidra_client
        )
        controller._supervisor = supervisor
        controller._ghidra_client = ghidra_client

        controller.shutdown()

        assert supervisor.is_paused
        ghidra_client.close.assert_called_once()

    def test_shutdown_saves_the_state_before_letting_go_of_ghidra(self, tmp_path):
        """Closing the window abandons the analysis thread mid-function."""
        ghidra_client = MagicMock()
        supervisor = _FakeSupervisor()
        controller = _controller(
            _settings(tmp_path), supervisor, ghidra_client=ghidra_client
        )
        controller._supervisor = supervisor
        controller._ghidra_client = ghidra_client

        controller.shutdown()

        assert supervisor.checkpoints == 1

    def test_a_checkpoint_failure_does_not_block_the_close(self, tmp_path):
        ghidra_client = MagicMock()
        supervisor = _FakeSupervisor()
        supervisor.checkpoint_error = OSError("read-only file system")
        controller = _controller(
            _settings(tmp_path), supervisor, ghidra_client=ghidra_client
        )
        controller._supervisor = supervisor
        controller._ghidra_client = ghidra_client

        controller.shutdown()

        ghidra_client.close.assert_called_once()

    def test_shutdown_is_safe_when_idle(self, tmp_path):
        controller = _controller(_settings(tmp_path), _FakeSupervisor())
        controller.shutdown()  # must not raise


class TestFinishingPass:
    """The second half of a draft run, started from the window."""

    def test_it_needs_a_run_first(self, tmp_path):
        controller = _controller(_settings(tmp_path), _FakeSupervisor())
        assert "Analyze a binary first" in controller.request_finishing_pass()

    def test_a_running_analysis_is_asked_to_pause_first(self, tmp_path):
        supervisor = _FakeSupervisor()
        supervisor.pending_finish = 3
        supervisor.finish_run = threading.Event()
        controller = _controller(_settings(tmp_path), supervisor)
        controller.start()
        try:
            deadline = time.time() + 5
            while time.time() < deadline and controller._get_supervisor() is None:
                time.sleep(0.01)
            assert "Pause the analysis first" in controller.request_finishing_pass()
        finally:
            supervisor.finish_run.set()

    def test_a_paused_run_can_be_finished(self, tmp_path):
        supervisor = _FakeSupervisor()
        supervisor.pending_finish = 3
        supervisor.finish_run = threading.Event()
        controller = _controller(_settings(tmp_path), supervisor)
        controller.start()
        try:
            deadline = time.time() + 5
            while time.time() < deadline and controller._get_supervisor() is None:
                time.sleep(0.01)
            controller.toggle_pause()
            assert controller.request_finishing_pass() == ""
            assert supervisor.finished_pass.wait(timeout=2)
        finally:
            supervisor.finish_run.set()

    def test_nothing_to_redo_is_said_rather_than_started(self, tmp_path):
        supervisor = _FakeSupervisor()
        supervisor.pending_finish = 0
        controller = _controller(_settings(tmp_path), supervisor)
        _run_to_completion(controller)

        assert "Nothing to finish" in controller.request_finishing_pass()
        assert not supervisor.finished_pass.is_set()

    def test_the_state_reports_the_pass_while_it_lasts(self, tmp_path):
        supervisor = _FakeSupervisor()
        supervisor.pending_finish = 2
        release = threading.Event()
        supervisor.run_finishing_pass = lambda: release.wait(timeout=2)
        controller = _controller(_settings(tmp_path), supervisor)
        _run_to_completion(controller)

        controller.request_finishing_pass()
        controller.poll()
        assert controller.state.finishing

        release.set()
        controller._finish_thread.join(timeout=2)
        controller.poll()
        assert not controller.state.finishing

    def test_the_pending_count_reaches_the_window(self, tmp_path):
        supervisor = _FakeSupervisor()
        supervisor.pending_finish = 7
        controller = _controller(_settings(tmp_path), supervisor)
        _run_to_completion(controller)

        assert controller.state.pending_finish == 7

    def test_the_two_manual_passes_do_not_overlap(self, tmp_path):
        supervisor = _FakeSupervisor()
        supervisor.pending_finish = 2
        release = threading.Event()
        supervisor.run_finishing_pass = lambda: release.wait(timeout=2)
        controller = _controller(_settings(tmp_path), supervisor)
        _run_to_completion(controller)

        assert controller.request_finishing_pass() == ""
        try:
            assert "finishing pass" in controller.request_coherence_review()
            assert "already running" in controller.request_finishing_pass()
        finally:
            release.set()
            controller._finish_thread.join(timeout=2)


class TestStageSetting:
    def test_a_full_run_is_the_default(self, tmp_path):
        assert _settings(tmp_path).to_config().stage is RunStage.FULL

    def test_draft_only_reaches_the_config(self, tmp_path):
        settings = _settings(tmp_path, stage=RunStage.DRAFT)
        assert settings.to_config().stage is RunStage.DRAFT

    def test_a_hosted_batch_size_reaches_the_config(self, tmp_path):
        settings = _settings(tmp_path, max_chunk_functions=30)
        config = settings.to_config()

        assert config.llm.provider is LLMProvider.ANTHROPIC
        assert config.llm.max_chunk_functions == 30
