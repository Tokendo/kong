"""Tests for the browser front-end: its session, and the server around it."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from kong.agent.events import Event, EventType
from kong.config import LLMProvider, RunStage
from kong.webui.server import (
    KongSession,
    bootstrap_payload,
    browse,
    serve,
    settings_from_payload,
)


@pytest.fixture(autouse=True)
def _own_config(monkeypatch, tmp_path):
    """Keys and the remembered form go to a store of this test's own."""
    monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path / "kong-config"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test1234")


def _payload(tmp_path, **overrides):
    binary = tmp_path / "target.bin"
    binary.write_bytes(b"\x7fELF")
    form = {
        "binary_path": str(binary),
        "output_dir": str(tmp_path / "out"),
        "formats": ["source", "json"],
        "provider": "anthropic",
        "resume": True,
    }
    form.update(overrides)
    return form


class TestSettingsFromPayload:
    def test_the_form_becomes_settings(self, tmp_path):
        settings = settings_from_payload(_payload(tmp_path, model="claude-opus-5"))

        assert settings.provider is LLMProvider.ANTHROPIC
        assert settings.model == "claude-opus-5"
        assert settings.formats == ["source", "json"]
        assert settings.validate() == []

    def test_an_empty_budget_means_the_model_default(self, tmp_path):
        settings = settings_from_payload(
            _payload(tmp_path, max_output_tokens="", max_prompt_chars=" ")
        )

        assert settings.max_output_tokens is None
        assert settings.max_prompt_chars is None

    def test_the_budget_is_honoured_on_a_hosted_provider_too(self, tmp_path):
        """The desktop window only offered this on a custom endpoint."""
        settings = settings_from_payload(
            _payload(tmp_path, provider="anthropic", max_output_tokens="8000")
        )

        config = settings.to_config()
        assert config.llm.provider is LLMProvider.ANTHROPIC
        assert config.llm.max_output_tokens == 8000

    def test_a_budget_that_is_not_a_number_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="Token budget"):
            settings_from_payload(_payload(tmp_path, max_output_tokens="lots"))

    def test_a_negative_budget_is_refused_by_validation(self, tmp_path):
        settings = settings_from_payload(_payload(tmp_path, max_output_tokens="-1"))

        assert "Max output tokens must be greater than zero." in settings.validate()

    def test_an_unknown_provider_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown provider"):
            settings_from_payload(_payload(tmp_path, provider="mystery"))

    def test_draft_only_becomes_the_draft_stage(self, tmp_path):
        settings = settings_from_payload(_payload(tmp_path, draft_only=True))

        assert settings.stage is RunStage.DRAFT


class TestSession:
    def test_an_idle_session_renders_without_a_run(self):
        snapshot = KongSession().snapshot()

        assert snapshot["has_controller"] is False
        assert snapshot["state"]["phase"] == "idle"
        assert snapshot["state"]["llm_waiting"] is False
        assert snapshot["log"] == []

    def test_the_log_is_only_sent_once(self):
        session = KongSession()
        session.note("first")
        session.note("second")

        first = session.snapshot(log_cursor=0)
        assert [entry["message"] for entry in first["log"]] == ["first", "second"]

        session.note("third")
        second = session.snapshot(log_cursor=first["log_cursor"])
        assert [entry["message"] for entry in second["log"]] == ["third"]

    def test_the_log_is_capped(self, monkeypatch):
        monkeypatch.setattr("kong.webui.server.LOG_LIMIT", 5)
        session = KongSession()
        for index in range(20):
            session.note(f"line {index}")

        snapshot = session.snapshot()
        assert len(snapshot["log"]) == 5
        assert snapshot["log"][-1]["message"] == "line 19"
        # The cursor keeps counting, so the page never re-reads a dropped line.
        assert snapshot["log_cursor"] == 20

    def test_a_finished_function_becomes_a_row(self):
        session = KongSession()
        session._record(Event(
            type=EventType.FUNCTION_COMPLETE,
            message="analyzed FUN_00401a30",
            data={
                "address": 0x401A30,
                "original_name": "FUN_00401a30",
                "name": "parse_http_header",
                "confidence": 92,
                "classification": "parser",
            },
        ))

        snapshot = session.snapshot()
        assert snapshot["results"] == [{
            "i": 0,
            "address": "0x00401a30",
            "original": "FUN_00401a30",
            "name": "parse_http_header",
            "confidence": 92,
            "classification": "parser",
        }]
        assert session.snapshot(results_cursor=snapshot["results_cursor"])["results"] == []

    def test_a_contradiction_is_listed_then_resolved(self):
        session = KongSession()
        session._record(Event(type=EventType.COHERENCE_CHECKED, message="checked"))
        session._record(Event(
            type=EventType.COHERENCE_CONFLICT,
            message="two functions named the same",
            data={
                "id": "c1",
                "kind": "duplicate_name",
                "addresses": [0x401000, 0x402000],
                "summary": "both called parse_header",
            },
        ))
        conflicts = session.snapshot()["conflicts"]
        assert conflicts[0]["kind"] == "duplicate name"
        assert conflicts[0]["functions"] == "0x00401000, 0x00402000"
        assert conflicts[0]["resolution"] == ""

        session._record(Event(
            type=EventType.COHERENCE_RESOLVED,
            message="renamed",
            data={"id": "c1", "applied": ["renamed 0x402000 to parse_body"]},
        ))
        assert session.snapshot()["conflicts"][0]["resolution"] == (
            "renamed 0x402000 to parse_body"
        )

    def test_conflicts_are_only_resent_when_they_change(self):
        session = KongSession()
        session._record(Event(type=EventType.COHERENCE_CHECKED, message="checked"))
        first = session.snapshot(conflicts_version=-1)

        assert first["conflicts"] is not None
        again = session.snapshot(conflicts_version=first["conflicts_version"])
        assert again["conflicts"] is None

    def test_the_controls_refuse_politely_before_a_run(self):
        session = KongSession()

        assert session.toggle_pause()["ok"] is False
        assert session.export()["ok"] is False
        assert "no draft to finish" in session.finishing_pass()["message"]
        assert "nothing to cross-check" in session.coherence_review()["message"]

    def test_unusable_settings_never_start_a_run(self, tmp_path):
        session = KongSession()

        with pytest.raises(ValueError, match="Select a binary"):
            session.start(_payload(tmp_path, binary_path=""))
        assert session.controller is None

    def test_the_form_is_remembered_without_the_key(self, tmp_path):
        KongSession._remember(_payload(tmp_path, api_key="sk-ant-secret", model="m"))

        remembered = KongSession.remembered_form()
        assert remembered["model"] == "m"
        assert "api_key" not in remembered


class TestBootstrap:
    def test_every_provider_is_offered(self):
        payload = bootstrap_payload(KongSession())

        assert [p["value"] for p in payload["providers"]] == [
            "anthropic", "openai", "zai", "custom",
        ]
        anthropic = payload["providers"][0]
        assert anthropic["env_key_set"] is True
        assert anthropic["default_model"] == "claude-opus-5"
        assert payload["providers"][3]["detectable"] is True

    def test_only_formats_the_export_writes_are_offered(self):
        """'ghidra' was a checkbox for a file the export never wrote: names
        and types reach the program database during the run, not at export."""
        payload = bootstrap_payload(KongSession())

        assert [f["value"] for f in payload["formats"]] == [
            "source", "json", "python", "csharp",
        ]

    def test_a_binary_on_the_command_line_names_the_output_directory(self, tmp_path):
        payload = bootstrap_payload(KongSession(str(tmp_path / "libfoo.so")))

        assert payload["binary_path"].endswith("libfoo.so")
        assert payload["output_dir"].endswith("kong_output_libfoo")


class TestBrowse:
    def test_a_directory_lists_its_children(self, tmp_path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "target.bin").write_bytes(b"\x7fELF")

        listing = browse(str(tmp_path))
        names = [entry["name"] for entry in listing["entries"]]

        assert names == ["sub", "target.bin"]
        assert listing["entries"][0]["is_dir"] is True
        assert listing["entries"][1]["size"] == 4
        assert listing["error"] == ""

    def test_a_file_lists_the_directory_holding_it(self, tmp_path):
        binary = tmp_path / "target.bin"
        binary.write_bytes(b"\x7fELF")

        assert browse(str(binary))["path"] == str(tmp_path.resolve())

    def test_a_path_that_is_not_there_falls_back_to_the_working_directory(self):
        listing = browse("/no/such/place/anywhere")

        assert listing["entries"] or listing["error"] == ""


@pytest.fixture
def server():
    httpd = serve()
    thread = threading.Thread(
        target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    )
    thread.start()
    yield httpd
    httpd.shutdown()
    thread.join(timeout=5)
    httpd.server_close()


def _request(httpd, path, *, token=None, method="GET", body=None, origin=None):
    headers = {"X-Kong-Token": httpd.token if token is None else token}
    if origin:
        headers["Origin"] = origin
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        f"http://127.0.0.1:{httpd.server_address[1]}{path}",
        data=data,
        headers=headers,
        method=method,
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return response.status, response.read()


def _content_type(httpd, path):
    request = urllib.request.Request(
        f"http://127.0.0.1:{httpd.server_address[1]}{path}",
        headers={"X-Kong-Token": httpd.token},
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return response.headers.get("Content-Type", "")


class TestServer:
    def test_the_page_and_its_assets_are_served(self, server):
        status, body = _request(server, "/")
        assert status == 200
        assert b"<title>Kong" in body

        for asset in ("/app.js", "/style.css"):
            assert _request(server, asset)[0] == 200

    def test_the_script_is_served_as_javascript(self, server):
        """mimetypes reads the Windows registry, where .js is often text/plain.

        Served that way next to the nosniff header, the browser refuses to run
        it: the page draws, and every control built by the script — the
        provider buttons, the formats, the hints — is simply missing.
        """
        assert _content_type(server, "/app.js") == "text/javascript; charset=utf-8"
        assert _content_type(server, "/style.css") == "text/css; charset=utf-8"
        assert _content_type(server, "/") == "text/html; charset=utf-8"

    def test_the_state_is_json(self, server):
        status, body = _request(server, "/api/state")
        payload = json.loads(body)

        assert status == 200
        assert payload["state"]["phase"] == "idle"
        assert payload["has_controller"] is False

    def test_an_api_call_without_the_token_is_refused(self, server):
        with pytest.raises(urllib.error.HTTPError) as caught:
            _request(server, "/api/state", token="not-the-token")

        assert caught.value.code == 403

    def test_a_call_from_another_origin_is_refused(self, server):
        with pytest.raises(urllib.error.HTTPError) as caught:
            _request(server, "/api/state", origin="https://example.com")

        assert caught.value.code == 403

    def test_the_page_itself_carries_no_token(self, server):
        """The token travels in the URL, not in the file on disk."""
        status, body = _request(server, "/", token="anything")

        assert status == 200
        assert server.token.encode() not in body

    def test_a_path_outside_the_static_directory_is_not_served(self, server):
        with pytest.raises(urllib.error.HTTPError) as caught:
            _request(server, "/../../pyproject.toml")

        assert caught.value.code == 404

    def test_a_bad_start_answers_with_the_reason(self, server):
        status, body = _request(
            server, "/api/start", method="POST",
            body={"provider": "anthropic", "binary_path": "", "output_dir": ""},
        )
        payload = json.loads(body)

        assert status == 200
        assert payload["ok"] is False
        assert "Select a binary to analyze." in payload["message"]

    def test_a_malformed_body_is_reported_rather_than_crashing(self, server):
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_address[1]}/api/start",
            data=b"{not json",
            headers={"X-Kong-Token": server.token,
                     "Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request, timeout=5)

        assert caught.value.code == 400

    def test_an_unknown_route_is_a_404(self, server):
        with pytest.raises(urllib.error.HTTPError) as caught:
            _request(server, "/api/nope")

        assert caught.value.code == 404
