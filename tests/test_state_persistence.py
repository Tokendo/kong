"""Tests for the analysis state file and resuming from it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kong.agent.models import FunctionResult
from kong.state.persistence import (
    STATE_NAME,
    STATE_VERSION,
    BinaryIdentity,
    load_state,
    save_state,
    state_path,
)


def _result(addr, **kwargs):
    kwargs.setdefault("original_name", f"FUN_{addr:08x}")
    return FunctionResult(address=addr, **kwargs)


class TestSaveState:
    def test_it_writes_next_to_the_output(self, tmp_path):
        path = save_state({0x1000: _result(0x1000, name="parse")}, tmp_path / "out")

        assert path == tmp_path / "out" / STATE_NAME
        assert json.loads(path.read_text())["version"] == STATE_VERSION

    def test_it_creates_the_directory(self, tmp_path):
        save_state({}, tmp_path / "does" / "not" / "exist")

        assert state_path(tmp_path / "does" / "not" / "exist").exists()

    def test_entries_are_ordered_by_address(self, tmp_path):
        results = {0x2000: _result(0x2000), 0x1000: _result(0x1000)}

        path = save_state(results, tmp_path)

        addresses = [f["address"] for f in json.loads(path.read_text())["functions"]]
        assert addresses == [0x1000, 0x2000]

    def test_a_later_save_replaces_the_earlier_one(self, tmp_path):
        save_state({0x1000: _result(0x1000, name="first")}, tmp_path)
        save_state({0x1000: _result(0x1000, name="second")}, tmp_path)

        assert load_state(tmp_path)[0x1000].name == "second"


class TestLoadState:
    def test_a_round_trip_keeps_what_the_call_cost(self, tmp_path):
        original = _result(
            0x401A30,
            name="parse_http_header",
            signature="int parse_http_header(char *)",
            confidence=92,
            classification="parser",
            comments="Parses a request line.",
            reasoning="String references name the header fields.",
            llm_calls=1,
            signature_applied=True,
            obfuscation_techniques=["control_flow_flattening"],
            deobfuscation_tool_calls=3,
        )
        save_state({0x401A30: original}, tmp_path)

        restored = load_state(tmp_path)[0x401A30]

        for attribute in (
            "address", "original_name", "name", "signature", "confidence",
            "classification", "comments", "reasoning", "llm_calls",
            "signature_applied", "obfuscation_techniques",
            "deobfuscation_tool_calls",
        ):
            assert getattr(restored, attribute) == getattr(original, attribute)

    def test_failures_survive_so_they_can_be_retried(self, tmp_path):
        save_state({0x1000: _result(0x1000, error="HTTP 429")}, tmp_path)

        assert load_state(tmp_path)[0x1000].error == "HTTP 429"

    def test_no_file_is_not_an_error(self, tmp_path):
        assert load_state(tmp_path) == {}

    def test_a_corrupt_file_is_ignored(self, tmp_path):
        state_path(tmp_path).write_text("{ this is not json")

        assert load_state(tmp_path) == {}

    def test_a_file_from_another_version_is_ignored(self, tmp_path):
        state_path(tmp_path).write_text(json.dumps({
            "version": STATE_VERSION + 1,
            "functions": [{"address": 4096, "name": "parse"}],
        }))

        assert load_state(tmp_path) == {}

    def test_an_entry_without_an_address_is_skipped(self, tmp_path):
        state_path(tmp_path).write_text(json.dumps({
            "version": STATE_VERSION,
            "functions": [{"name": "nameless"}, {"address": 4096, "name": "parse"}],
        }))

        assert list(load_state(tmp_path)) == [4096]

    def test_an_entry_missing_fields_still_loads(self, tmp_path):
        """A file written by an older build of the same version."""
        state_path(tmp_path).write_text(json.dumps({
            "version": STATE_VERSION,
            "functions": [{"address": 4096, "name": "parse"}],
        }))

        result = load_state(tmp_path)[4096]

        assert result.name == "parse"
        assert result.confidence == 0


class TestTwoPassFields:
    def test_the_model_and_the_pass_survive_the_round_trip(self, tmp_path):
        save_state(
            {0x1000: _result(
                0x1000, name="parse", model="fast-model", refined=False,
            )},
            tmp_path,
        )

        result = load_state(tmp_path)[0x1000]

        assert result.model == "fast-model"
        assert not result.refined

    def test_a_refined_result_is_marked_as_such(self, tmp_path):
        save_state(
            {0x1000: _result(
                0x1000, name="parse", model="strong-model", refined=True,
            )},
            tmp_path,
        )

        assert load_state(tmp_path)[0x1000].refined

    def test_a_file_from_before_the_second_pass_is_ignored(self, tmp_path):
        """Version 1 entries carry no model, so nothing knows what wrote them."""
        state_path(tmp_path).write_text(json.dumps({
            "version": 1,
            "functions": [{"address": 4096, "name": "parse"}],
        }))

        assert load_state(tmp_path) == {}


class TestBinaryIdentity:
    def _binary(self, tmp_path, content=b"\x7fELF hello"):
        path = tmp_path / "target.bin"
        path.write_bytes(content)
        return path

    def test_a_binary_identifies_itself(self, tmp_path):
        identity = BinaryIdentity.of(self._binary(tmp_path))

        assert identity is not None
        assert identity.size == 10
        assert len(identity.sha256) == 64

    def test_two_copies_of_the_same_bytes_are_the_same_binary(self, tmp_path):
        first = BinaryIdentity.of(self._binary(tmp_path))
        other = tmp_path / "copy.bin"
        other.write_bytes((tmp_path / "target.bin").read_bytes())

        assert BinaryIdentity.of(other).sha256 == first.sha256

    def test_a_missing_file_has_no_identity(self, tmp_path):
        assert BinaryIdentity.of(tmp_path / "gone.bin") is None

    def test_state_is_restored_for_the_binary_it_was_written_for(self, tmp_path):
        identity = BinaryIdentity.of(self._binary(tmp_path))
        save_state({0x1000: _result(0x1000, name="parse")}, tmp_path, identity)

        assert list(load_state(tmp_path, identity)) == [0x1000]

    def test_state_from_another_binary_is_refused(self, tmp_path):
        written_for = BinaryIdentity.of(self._binary(tmp_path))
        save_state({0x1000: _result(0x1000, name="parse")}, tmp_path, written_for)

        other = tmp_path / "other.bin"
        other.write_bytes(b"\x7fELF something else entirely")

        assert load_state(tmp_path, BinaryIdentity.of(other)) == {}

    def test_a_rebuilt_binary_at_the_same_path_is_refused(self, tmp_path):
        """Same path, different bytes: the addresses moved."""
        binary = self._binary(tmp_path)
        save_state(
            {0x1000: _result(0x1000, name="parse")},
            tmp_path,
            BinaryIdentity.of(binary),
        )

        binary.write_bytes(b"\x7fELF rebuilt with another compiler")

        assert load_state(tmp_path, BinaryIdentity.of(binary)) == {}

    def test_a_state_file_without_an_identity_is_still_usable(self, tmp_path):
        """Written by a build that could not read the binary."""
        save_state({0x1000: _result(0x1000, name="parse")}, tmp_path)

        identity = BinaryIdentity.of(self._binary(tmp_path))

        assert list(load_state(tmp_path, identity)) == [0x1000]

    def test_reading_without_an_identity_checks_nothing(self, tmp_path):
        save_state(
            {0x1000: _result(0x1000, name="parse")},
            tmp_path,
            BinaryIdentity.of(self._binary(tmp_path)),
        )

        assert list(load_state(tmp_path)) == [0x1000]


class TestAtomicWrite:
    def test_no_temporary_file_is_left_behind(self, tmp_path):
        save_state({0x1000: _result(0x1000, name="parse")}, tmp_path)

        leftovers = list(tmp_path.glob("*.tmp"))
        assert leftovers == []

    def test_a_failed_write_leaves_the_previous_checkpoint_intact(
        self, tmp_path, monkeypatch,
    ):
        save_state({0x1000: _result(0x1000, name="first")}, tmp_path)

        def explode(self, *args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(Path, "write_text", explode)

        with pytest.raises(OSError):
            save_state({0x1000: _result(0x1000, name="second")}, tmp_path)

        monkeypatch.undo()
        assert load_state(tmp_path)[0x1000].name == "first"
