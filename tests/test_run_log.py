"""Tests for the on-disk run log."""

from __future__ import annotations

import logging

import pytest

from kong.agent.events import Event, EventType, Phase
from kong.agent.run_log import LOG_NAME, RunLog


@pytest.fixture
def run_log(tmp_path):
    log = RunLog(tmp_path / "out")
    log.open()
    try:
        yield log
    finally:
        log.close()


class TestOpenAndClose:
    def test_open_creates_the_file_and_its_directory(self, tmp_path):
        log = RunLog(tmp_path / "out")
        path = log.open()
        try:
            assert path == tmp_path / "out" / LOG_NAME
            assert path.exists()
        finally:
            log.close()

    def test_close_detaches_the_handler(self, tmp_path):
        kong_logger = logging.getLogger("kong")
        before = list(kong_logger.handlers)

        log = RunLog(tmp_path / "out")
        log.open()
        assert len(kong_logger.handlers) == len(before) + 1
        log.close()

        assert kong_logger.handlers == before

    def test_close_restores_the_logger_level(self, tmp_path):
        kong_logger = logging.getLogger("kong")
        before = kong_logger.level

        log = RunLog(tmp_path / "out")
        log.open()
        log.close()

        assert kong_logger.level == before

    def test_closing_twice_is_harmless(self, tmp_path):
        log = RunLog(tmp_path / "out")
        log.open()
        log.close()
        log.close()

    def test_a_second_run_starts_a_fresh_trace(self, tmp_path):
        first = RunLog(tmp_path / "out")
        first.open()
        first.record(Event(type=EventType.RUN_START, message="first run"))
        first.close()

        second = RunLog(tmp_path / "out")
        path = second.open()
        second.record(Event(type=EventType.RUN_START, message="second run"))
        second.close()

        contents = path.read_text(encoding="utf-8")
        assert "second run" in contents
        assert "first run" not in contents

    def test_recording_without_opening_is_a_no_op(self, tmp_path):
        log = RunLog(tmp_path / "out")
        log.record(Event(type=EventType.RUN_START, message="dropped"))

        assert not (tmp_path / "out" / LOG_NAME).exists()


class TestRecordedEvents:
    def test_the_message_and_phase_are_written(self, run_log):
        run_log.record(Event(
            type=EventType.PHASE_START,
            phase=Phase.ANALYSIS,
            message="Starting analysis...",
        ))

        line = run_log.path.read_text(encoding="utf-8")
        assert "phase_start" in line
        assert "[analysis]" in line
        assert "Starting analysis..." in line

    def test_a_failure_is_written_at_warning(self, run_log):
        run_log.record(Event(
            type=EventType.FUNCTION_ERROR,
            phase=Phase.ANALYSIS,
            message="Error analyzing FUN_00401a30: HTTP 429: Too Many Requests",
            data={"address": 0x401A30, "error": "HTTP 429: Too Many Requests"},
        ))

        line = run_log.path.read_text(encoding="utf-8")
        assert "WARNING" in line
        # Here it arrives inside the FUN_ name; either way the line is
        # greppable by the bare address.
        assert "00401a30" in line

    def test_an_address_missing_from_the_message_is_added(self, run_log):
        run_log.record(Event(
            type=EventType.FUNCTION_ERROR,
            message="No analysis for parse_header",
            data={"address": 0x401A30, "error": "No matching response from LLM"},
        ))

        assert "0x00401a30" in run_log.path.read_text(encoding="utf-8")

    def test_an_address_already_in_the_message_is_not_repeated(self, run_log):
        run_log.record(Event(
            type=EventType.FUNCTION_START,
            message="Analyzing FUN_00401a30 (0x00401a30)...",
            data={"address": 0x401A30},
        ))

        assert run_log.path.read_text(encoding="utf-8").count("0x00401a30") == 1

    def test_an_ordinary_event_is_not_a_warning(self, run_log):
        run_log.record(Event(type=EventType.FUNCTION_COMPLETE, message="named"))

        assert "WARNING" not in run_log.path.read_text(encoding="utf-8")

    def test_a_reason_kept_only_in_the_payload_is_written(self, run_log):
        """"No analysis for X" carries its reason in data, not the message."""
        run_log.record(Event(
            type=EventType.FUNCTION_ERROR,
            message="No analysis for FUN_00401a30",
            data={"address": 0x401A30, "error": "No matching response from LLM"},
        ))

        assert "No matching response from LLM" in run_log.path.read_text(
            encoding="utf-8"
        )

    def test_a_reason_already_in_the_message_is_not_repeated(self, run_log):
        run_log.record(Event(
            type=EventType.FUNCTION_ERROR,
            message="Error analyzing f: boom",
            data={"error": "boom"},
        ))

        assert run_log.path.read_text(encoding="utf-8").count("boom") == 1


class TestPackageLogging:
    """The point of the file: records the package already emits land in it."""

    def test_module_records_are_captured(self, run_log):
        logging.getLogger("kong.agent.supervisor").info(
            "Chunk 3/12 prompt: 68000 chars (9 functions)."
        )

        assert "68000 chars" in run_log.path.read_text(encoding="utf-8")

    def test_debug_records_need_verbose(self, tmp_path):
        log = RunLog(tmp_path / "quiet")
        log.open()
        logging.getLogger("kong.agent.analyzer").debug("recovered via json_repair")
        log.close()

        assert "json_repair" not in log.path.read_text(encoding="utf-8")

    def test_verbose_captures_debug_records(self, tmp_path):
        log = RunLog(tmp_path / "loud", verbose=True)
        log.open()
        logging.getLogger("kong.agent.analyzer").debug("recovered via json_repair")
        log.close()

        assert "json_repair" in log.path.read_text(encoding="utf-8")

    def test_a_traceback_is_kept(self, run_log):
        try:
            raise ValueError("decompilation is empty")
        except ValueError:
            logging.getLogger("kong.agent.supervisor").exception("Run failed")

        contents = run_log.path.read_text(encoding="utf-8")
        assert "Traceback (most recent call last)" in contents
        assert "decompilation is empty" in contents

    def test_other_packages_are_left_alone(self, run_log):
        logging.getLogger("httpx").info("POST /v1/messages 200")

        assert "httpx" not in run_log.path.read_text(encoding="utf-8")
