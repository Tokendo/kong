"""A browser front-end for Kong, served over loopback.

The same `AnalysisController` as the desktop window, behind a small JSON API
instead of Tk widgets: the browser polls for state and posts the controls.
That buys a front-end that can be styled, and one that runs on a Python build
with no Tk bindings at all — the interface is a page, and only the standard
library is needed to serve it.

The server is bound to 127.0.0.1 and every call carries a token minted at
startup, so a page open in the same browser on some other site cannot drive
an analysis, read the paths being browsed, or spend the API key.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import threading
import webbrowser
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from kong.agent.events import Event, EventType
from kong.agent.refinement import DEFAULT_REFINE_BELOW
from kong.banner import _ENV_VARS
from kong.config import ZAI_BASE_URL, LLMProvider, RunStage
from kong.db import get_saved_api_key, read_config, save_api_key, write_config
from kong.gui.controller import AnalysisController, RunSettings, RunState

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

#: Content types for what the page is made of, spelled out rather than looked
#: up. `mimetypes` reads the Windows registry, where .js is routinely mapped to
#: text/plain — and a script served as text/plain next to the nosniff header
#: below is one the browser refuses to run, leaving a page that draws but does
#: nothing.
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".woff2": "font/woff2",
}

#: The log is a live view, not a record: the run log on disk keeps everything.
#: Old lines are dropped so a binary with thousands of functions does not grow
#: the page until the browser struggles with it.
LOG_LIMIT = 4000

#: Where the form is remembered between sessions. The API key is deliberately
#: not part of it: keys are saved on purpose, through the Save button, into
#: the place `kong setup` already keeps them.
FORM_CONFIG_KEY = "webui_form"

LOCAL_BASE_URL = "http://127.0.0.1:8080/v1"

#: What a provider analyzes with when the model field is left empty.
DEFAULT_MODEL_HINTS = {
    LLMProvider.ANTHROPIC.value: "claude-opus-5",
    LLMProvider.OPENAI.value: "gpt-4o",
    LLMProvider.ZAI.value: "glm-5.3",
}

#: Files the export phase actually writes. Ghidra is not one of them: names
#: and types are written into the program database as each function is
#: analyzed, not at the end and not on request, so offering it as an output
#: format only promised a file that was never going to appear.
OUTPUT_FORMATS: list[tuple[str, str]] = [
    ("source", "C source"),
    ("json", "JSON"),
    ("python", "Python"),
    ("csharp", "C#"),
]

DEFAULT_FORMATS = ("source", "json")

#: How each event reads in the log. The same five roles the desktop window
#: paints its text with, named rather than coloured: the stylesheet decides.
LOG_TAGS: dict[EventType, str] = {
    EventType.PHASE_START: "phase",
    EventType.PHASE_COMPLETE: "success",
    EventType.FUNCTION_COMPLETE: "success",
    EventType.FUNCTION_SKIPPED: "muted",
    EventType.FUNCTION_ERROR: "error",
    EventType.RUN_ERROR: "error",
    EventType.RUN_COMPLETE: "success",
    EventType.DEOBFUSCATION_DETECTED: "accent",
    EventType.DEOBFUSCATION_COMPLETE: "accent",
    EventType.EXPORT_FILE: "accent",
    EventType.COHERENCE_CHECKED: "phase",
    EventType.COHERENCE_CONFLICT: "accent",
    EventType.COHERENCE_RESOLVED: "success",
}

#: What the page renders before anything has been started, so the idle payload
#: carries the same fields as a running one.
_IDLE_STATE = RunState()


def _optional_int(value: Any, label: str) -> int | None:
    """Read a number the form may have left empty."""
    if value is None:
        return None
    text = str(value).strip().rstrip("%").strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f"{label}: {text!r} is not a whole number.") from exc


def settings_from_payload(payload: dict[str, Any]) -> RunSettings:
    """Turn what the form posted into the settings the controller runs."""
    try:
        provider = LLMProvider(str(payload.get("provider", "")).strip())
    except ValueError as exc:
        raise ValueError(f"Unknown provider: {payload.get('provider')!r}") from exc

    refine_below = _optional_int(payload.get("refine_below"), "Refine below")

    return RunSettings(
        binary_path=str(payload.get("binary_path", "")).strip(),
        output_dir=str(payload.get("output_dir", "")).strip(),
        formats=[str(f) for f in payload.get("formats", []) if f],
        resume=bool(payload.get("resume", True)),
        provider=provider,
        model=str(payload.get("model", "")).strip(),
        draft_model=str(payload.get("draft_model", "")).strip(),
        refine_below=DEFAULT_REFINE_BELOW if refine_below is None else refine_below,
        base_url=str(payload.get("base_url", "")).strip(),
        api_key=str(payload.get("api_key", "")).strip(),
        ghidra_dir=str(payload.get("ghidra_dir", "")).strip(),
        # Unlike the desktop window, the budget fields are honoured on every
        # provider: how many tokens one request may spend is a question a
        # hosted endpoint answers too, and the answer is what the bill is
        # made of.
        max_prompt_chars=_optional_int(
            payload.get("max_prompt_chars"), "Max prompt chars"
        ),
        max_chunk_functions=_optional_int(
            payload.get("max_chunk_functions"), "Functions per batch"
        ),
        max_output_tokens=_optional_int(
            payload.get("max_output_tokens"), "Token budget per request"
        ),
        stage=RunStage.DRAFT if payload.get("draft_only") else RunStage.FULL,
    )


def _parse_address(value: Any) -> int | None:
    """An address as the graph and the functions table both spell it: hex."""
    try:
        return int(str(value), 16)
    except (TypeError, ValueError):
        return None


def default_output_dir(binary_path: str) -> str:
    if not binary_path:
        return str(Path.cwd() / "kong_output")
    return str(Path.cwd() / f"kong_output_{Path(binary_path).stem}")


class KongSession:
    """One window's worth of state: the run, and everything it has said.

    The browser is a client that can reload, so the log, the results and the
    contradictions live here rather than in the page: reopening the tab shows
    the run that is still going, not an empty one.
    """

    def __init__(self, initial_binary: str = "") -> None:
        self._lock = threading.Lock()
        self.initial_binary = initial_binary
        self.controller: AnalysisController | None = None
        self._log: list[dict[str, Any]] = []
        self._log_next = 0
        self._results: list[dict[str, Any]] = []
        self._conflicts: dict[str, dict[str, Any]] = {}
        self._conflicts_version = 0

    def output_directory(self) -> str:
        """Where this session's run writes, when it has one."""
        controller = self.controller
        return controller.settings.output_dir if controller is not None else ""

    # ----------------------------------------------------------------- events

    def _append_log(self, tag: str, message: str, kind: str) -> None:
        self._log.append(
            {"i": self._log_next, "tag": tag, "message": message, "type": kind}
        )
        self._log_next += 1
        if len(self._log) > LOG_LIMIT:
            del self._log[: len(self._log) - LOG_LIMIT]

    def _record(self, event: Event) -> None:
        self._append_log(
            LOG_TAGS.get(event.type, ""), event.message, event.type.value
        )

        if event.type is EventType.FUNCTION_COMPLETE:
            data = event.data
            self._results.append({
                "i": len(self._results),
                "address": f"0x{int(data.get('address', 0)):08x}",
                "original": data.get("original_name", ""),
                "name": data.get("name", ""),
                "confidence": int(data.get("confidence", 0) or 0),
                "classification": data.get("classification", ""),
            })
        elif event.type is EventType.COHERENCE_CHECKED:
            # Emitted once, ahead of this pass's conflicts: the table shows
            # the current review rather than every review.
            self._conflicts.clear()
            self._conflicts_version += 1
        elif event.type is EventType.COHERENCE_CONFLICT:
            data = event.data
            key = str(data.get("id") or f"conflict-{len(self._conflicts)}")
            self._conflicts[key] = {
                "id": key,
                "kind": str(data.get("kind", "")).replace("_", " "),
                "functions": ", ".join(
                    f"0x{int(a):08x}" for a in data.get("addresses", [])
                ),
                "summary": data.get("summary", ""),
                "resolution": "",
            }
            self._conflicts_version += 1
        elif event.type is EventType.COHERENCE_RESOLVED:
            data = event.data
            conflict = self._conflicts.get(str(data.get("id", "")))
            if conflict is not None:
                applied = data.get("applied") or []
                conflict["resolution"] = (
                    "; ".join(applied)
                    if applied
                    else f"kept — {data.get('explanation', 'no change')}"
                )
                self._conflicts_version += 1

    def _drain(self) -> None:
        """Fold whatever the run has emitted since the last poll."""
        if self.controller is None:
            return
        for event in self.controller.poll():
            self._record(event)

    def note(self, message: str, tag: str = "muted") -> None:
        """Put a line in the log that did not come from the run itself."""
        with self._lock:
            self._append_log(tag, message, "notice")

    # ------------------------------------------------------------------ state

    def snapshot(
        self,
        log_cursor: int = 0,
        results_cursor: int = 0,
        conflicts_version: int = -1,
    ) -> dict[str, Any]:
        """What the page renders, and whatever it has not been told yet."""
        with self._lock:
            self._drain()

            if self.controller is None:
                state: dict[str, Any] = asdict(_IDLE_STATE)
                state["progress_fraction"] = 0.0
            else:
                state = asdict(self.controller.state)
                state["progress_fraction"] = self.controller.state.progress_fraction

            return {
                "state": state,
                "log": [entry for entry in self._log if entry["i"] >= log_cursor],
                "log_cursor": self._log_next,
                "results": self._results[results_cursor:],
                "results_cursor": len(self._results),
                # None means "unchanged": the table only redraws when a
                # contradiction was found or resolved.
                "conflicts": (
                    None
                    if conflicts_version == self._conflicts_version
                    else list(self._conflicts.values())
                ),
                "conflicts_version": self._conflicts_version,
                "has_controller": self.controller is not None,
            }

    # ---------------------------------------------------------------- controls

    def start(self, payload: dict[str, Any]) -> dict[str, Any]:
        settings = settings_from_payload(payload)
        controller = AnalysisController(settings)
        controller.start()
        with self._lock:
            self.controller = controller
            self._log.clear()
            self._results.clear()
            self._conflicts.clear()
            self._conflicts_version += 1
        self._remember(payload)
        return {
            "ok": True,
            "message": "Opening the binary in Ghidra (30-60s on a cold run)...",
        }

    @staticmethod
    def _remember(payload: dict[str, Any]) -> None:
        """Keep the form for next time, minus the key."""
        keep = {key: value for key, value in payload.items() if key != "api_key"}
        try:
            write_config(FORM_CONFIG_KEY, json.dumps(keep))
        except Exception:  # a read-only home, a locked sqlite file
            logger.debug("Could not remember the form", exc_info=True)

    @staticmethod
    def remembered_form() -> dict[str, Any]:
        try:
            raw = read_config(FORM_CONFIG_KEY)
        except Exception:
            logger.debug("Could not read the remembered form", exc_info=True)
            return {}
        if not raw:
            return {}
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    def toggle_pause(self) -> dict[str, Any]:
        if self.controller is None:
            return {"ok": False, "message": "Nothing is running."}
        paused = self.controller.toggle_pause()
        return {
            "ok": True,
            "paused": paused,
            "message": "Paused." if paused else "Resumed.",
        }

    def export(self) -> dict[str, Any]:
        if self.controller is None or not self.controller.request_export():
            return {"ok": False, "message": "Nothing to export yet."}
        return {"ok": True, "message": "Exporting what has been analyzed so far..."}

    def finishing_pass(self) -> dict[str, Any]:
        if self.controller is None:
            return {
                "ok": False,
                "message": "Analyze a binary first: there is no draft to finish yet.",
            }
        pending = self.controller.state.pending_finish
        refusal = self.controller.request_finishing_pass()
        if refusal:
            return {"ok": False, "message": refusal}
        return {"ok": True, "message": f"Finishing pass on {pending} function(s)..."}

    def open_existing(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Load a finished run's results for browsing, no run involved.

        Reads analysis.json straight off disk: no Ghidra, no model, and
        nothing recomputed — the opposite of Start, which is the only other
        way today to see a directory's results, and which, even in a run that
        resumes every function, still redoes cleanup, synthesis and export
        over the whole binary before showing anything.
        """
        if self.controller is not None and self.controller.state.running:
            return {
                "ok": False,
                "message": (
                    "Pause or wait for the current run before opening another "
                    "analysis to browse."
                ),
            }

        output_dir = str(payload.get("output_dir", "")).strip()
        if not output_dir:
            return {"ok": False, "message": "Choose an output directory first."}

        path = Path(output_dir).expanduser() / "analysis.json"
        if not path.is_file():
            return {
                "ok": False,
                "message": (
                    f"{path} is not there. Nothing has been exported to this "
                    f"directory yet."
                ),
            }
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            return {"ok": False, "message": f"Could not read {path}: {exc}"}

        results: list[dict[str, Any]] = []
        for entry in document.get("functions") or []:
            address = _parse_address(entry.get("address"))
            if address is None:
                continue
            results.append({
                "i": len(results),
                "address": f"0x{address:08x}",
                "original": entry.get("original_name", ""),
                "name": entry.get("name", ""),
                "confidence": int(entry.get("confidence", 0) or 0),
                "classification": entry.get("classification", ""),
            })

        with self._lock:
            self._results = results
            self._log = []
            self._log_next = 0
            self._append_log(
                "phase",
                (
                    f"Loaded {len(results)} functions from {path}. This is a "
                    f"read-only view — start an analysis to change anything."
                ),
                "notice",
            )

        return {
            "ok": True,
            "message": f"Loaded {len(results)} functions from {path}.",
        }

    def analyze_function(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.controller is None:
            return {
                "ok": False,
                "message": "Analyze a binary first: there is no run to add to.",
            }
        address = _parse_address(payload.get("address"))
        if address is None:
            return {"ok": False, "message": "Not a function address."}
        refusal = self.controller.request_function_analysis(address)
        if refusal:
            return {"ok": False, "message": refusal}
        return {"ok": True, "message": f"Analyzing 0x{address:08x}..."}

    def coherence_review(self) -> dict[str, Any]:
        if self.controller is None:
            return {
                "ok": False,
                "message": (
                    "Analyze a binary first: there is nothing to cross-check yet."
                ),
            }
        refusal = self.controller.request_coherence_review()
        if refusal:
            return {"ok": False, "message": refusal}
        return {"ok": True, "message": "Cross-checking the analysis..."}

    def shutdown(self) -> None:
        if self.controller is not None:
            self.controller.shutdown()


def bootstrap_payload(session: KongSession) -> dict[str, Any]:
    """Everything the form needs to draw itself, once, at page load."""
    binary = session.initial_binary
    providers = []
    for provider in LLMProvider:
        env_var = _ENV_VARS.get(provider)
        providers.append({
            "value": provider.value,
            "label": provider.display_name,
            "default_model": DEFAULT_MODEL_HINTS.get(provider.value, ""),
            "env_var": env_var or "",
            "env_key_set": bool(env_var and os.environ.get(env_var)),
            "saved_key": bool(get_saved_api_key(provider)),
            "needs_base_url": provider in (LLMProvider.CUSTOM, LLMProvider.ZAI),
            "detectable": provider is LLMProvider.CUSTOM,
        })

    return {
        "providers": providers,
        "formats": [{"value": key, "label": label} for key, label in OUTPUT_FORMATS],
        "default_formats": list(DEFAULT_FORMATS),
        "default_refine_below": DEFAULT_REFINE_BELOW,
        "base_urls": {
            LLMProvider.CUSTOM.value: LOCAL_BASE_URL,
            LLMProvider.ZAI.value: ZAI_BASE_URL,
        },
        "binary_path": binary,
        "output_dir": default_output_dir(binary),
        "cwd": str(Path.cwd()),
        "home": str(Path.home()),
        "remembered": session.remembered_form(),
    }


#: Nodes the graph view will accept. A binary an order of magnitude past this
#: is not a thing to draw, and the browser should say so rather than freeze.
GRAPH_NODE_LIMIT = 6000


def call_graph(output_dir: str) -> dict[str, Any]:
    """Read a finished run's call graph back out of analysis.json.

    The graph is served from the file rather than from the live run because
    that is where it is complete: it is written at export, and a page reopened
    days later gets the same answer as one watching the run that produced it.

    Edges are index pairs into `nodes`, and an address that only ever appears
    as a callee still gets a node. That is the case the file exists for: a
    function no pass analyzed is invisible in the recovered C, so a graph built
    by parsing that C back loses every call into it.
    """
    if not output_dir:
        return {"ok": False, "message": "No output directory yet. Start a run first."}

    path = Path(output_dir).expanduser() / "analysis.json"
    if not path.is_file():
        return {
            "ok": False,
            "message": (
                f"{path} is not there yet. Export writes it, at the end of a run "
                f"or from Export now."
            ),
        }

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return {"ok": False, "message": f"Could not read {path}: {exc}"}

    raw_edges = (document.get("call_graph") or {}).get("edges") or []
    functions = document.get("functions") or []

    def address_of(text: object) -> int | None:
        try:
            return int(str(text), 16)
        except (TypeError, ValueError):
            return None

    nodes: list[dict[str, Any]] = []
    index: dict[int, int] = {}

    def node_for(address: int, entry: dict[str, Any] | None = None) -> int:
        position = index.get(address)
        if position is not None:
            return position
        index[address] = len(nodes)
        nodes.append({
            "a": f"0x{address:08x}",
            "n": (entry or {}).get("name") or f"FUN_{address:08x}",
            "c": (entry or {}).get("classification") or "",
            "q": (entry or {}).get("confidence", 0) if entry else 0,
            "u": entry is None,  # never analyzed: it is only a callee
        })
        return index[address]

    for entry in functions:
        address = address_of(entry.get("address"))
        if address is not None:
            node_for(address, entry)

    edges: list[list[int]] = []
    for pair in raw_edges:
        if not isinstance(pair, list) or len(pair) != 2:
            continue
        caller, callee = address_of(pair[0]), address_of(pair[1])
        if caller is None or callee is None:
            continue
        if len(nodes) >= GRAPH_NODE_LIMIT and (
            caller not in index or callee not in index
        ):
            continue
        edges.append([node_for(caller), node_for(callee)])

    if not edges:
        return {
            "ok": False,
            "message": (
                "This analysis.json carries no call graph. It was written by a "
                "version that did not save one — re-export to add it."
            ),
        }

    return {
        "ok": True,
        "binary": (document.get("binary") or {}).get("name", ""),
        "path": str(path),
        "nodes": nodes,
        "edges": edges,
    }


def browse(path: str = "") -> dict[str, Any]:
    """List a directory, for the picker a browser cannot open by itself."""
    target = Path(path).expanduser() if path else Path.cwd()
    try:
        target = target.resolve()
    except OSError:
        target = Path.cwd()
    if target.is_file():
        target = target.parent
    if not target.is_dir():
        target = Path.cwd()

    entries: list[dict[str, Any]] = []
    try:
        children = sorted(
            target.iterdir(), key=lambda child: (not child.is_dir(), child.name.lower())
        )
    except (PermissionError, OSError) as exc:
        return {
            "path": str(target),
            "parent": str(target.parent),
            "entries": [],
            "error": f"Could not read {target}: {exc}",
        }

    for child in children:
        try:
            is_dir = child.is_dir()
            size = 0 if is_dir else child.stat().st_size
        except OSError:
            # A dangling link or a file that vanished mid-listing is not worth
            # failing the whole directory over.
            continue
        entries.append(
            {"name": child.name, "path": str(child), "is_dir": is_dir, "size": size}
        )

    parent = str(target.parent) if target.parent != target else ""
    return {"path": str(target), "parent": parent, "entries": entries, "error": ""}


def detect_endpoint(base_url: str) -> dict[str, Any]:
    """Ask an OpenAI-compatible endpoint what it serves, and size the budget."""
    from kong.llm.endpoint import discover, suggest_limits

    if not base_url:
        return {"ok": False, "message": "Enter a base URL first."}
    try:
        info = discover(base_url)
    except Exception as exc:
        return {"ok": False, "message": f"Could not reach {base_url}: {exc}"}

    payload: dict[str, Any] = {
        "ok": True,
        "models": [model.id for model in info.models],
    }
    context = info.effective_context
    if context is None:
        payload["message"] = (
            f"{len(info.models)} model(s) found, but the endpoint does not "
            f"report a context window. Set the budget by hand."
        )
        return payload

    limits = suggest_limits(context)
    payload["limits"] = {
        "max_prompt_chars": limits.max_prompt_chars,
        "max_chunk_functions": limits.max_chunk_functions,
        "max_output_tokens": limits.max_output_tokens,
    }
    payload["message"] = (
        f"{len(info.models)} model(s), {context:,} token context — "
        f"budget sized to fit."
    )
    return payload


def store_api_key(provider_value: str, key: str) -> dict[str, Any]:
    """Save a key for next time, or forget it when the field is empty."""
    try:
        provider = LLMProvider(provider_value)
    except ValueError:
        return {"ok": False, "message": f"Unknown provider: {provider_value!r}"}
    try:
        save_api_key(provider, key)
    except Exception as exc:  # sqlite failures: a read-only home, a lock
        return {"ok": False, "message": f"Could not save the key: {exc}"}
    return {
        "ok": True,
        "message": (
            f"Key saved for {provider.display_name}."
            if key
            else f"Key cleared for {provider.display_name}."
        ),
    }


class _Handler(BaseHTTPRequestHandler):
    """Routes. Everything under /api needs the token minted at startup."""

    server_version = "Kong"
    protocol_version = "HTTP/1.1"

    # --------------------------------------------------------------- plumbing

    def log_message(self, fmt: str, *args: Any) -> None:
        # One line per poll on stdout would bury the URL the user needs.
        logger.debug("%s - %s", self.address_string(), fmt % args)

    @property
    def session(self) -> KongSession:
        return self.server.session  # type: ignore[attr-defined]

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

    def _authorized(self) -> bool:
        expected: str = self.server.token  # type: ignore[attr-defined]
        given = self.headers.get("X-Kong-Token", "")
        if not given:
            given = (parse_qs(urlparse(self.path).query).get("t") or [""])[0]
        if not secrets.compare_digest(given, expected):
            return False
        # A page on another origin cannot read a cross-origin answer, but it
        # can send the request; refusing one that announces a foreign origin
        # keeps a stray tab from starting a run.
        origin = self.headers.get("Origin")
        allowed = self.server.allowed_origins  # type: ignore[attr-defined]
        return not origin or origin in allowed

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0) or 0)
        if not length:
            return {}
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"Malformed request body: {exc}") from exc
        return payload if isinstance(payload, dict) else {}

    # ----------------------------------------------------------------- routes

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's spelling
        parsed = urlparse(self.path)
        route = parsed.path

        if route.startswith("/api/"):
            if not self._authorized():
                self._send_json({"error": "forbidden"}, HTTPStatus.FORBIDDEN)
                return
            self._api_get(route, parse_qs(parsed.query))
            return

        self._send_static("index.html" if route == "/" else route.lstrip("/"))

    do_HEAD = do_GET

    def _api_get(self, route: str, query: dict[str, list[str]]) -> None:
        def number(name: str, default: int = 0) -> int:
            try:
                return int((query.get(name) or [default])[0])
            except (TypeError, ValueError):
                return default

        if route == "/api/bootstrap":
            self._send_json(bootstrap_payload(self.session))
        elif route == "/api/state":
            self._send_json(
                self.session.snapshot(
                    log_cursor=number("log"),
                    results_cursor=number("results"),
                    conflicts_version=number("conflicts", -1),
                )
            )
        elif route == "/api/browse":
            self._send_json(browse((query.get("path") or [""])[0]))
        elif route == "/api/graph":
            asked = (query.get("path") or [""])[0]
            self._send_json(call_graph(asked or self.session.output_directory()))
        else:
            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        if not route.startswith("/api/"):
            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        if not self._authorized():
            self._send_json({"error": "forbidden"}, HTTPStatus.FORBIDDEN)
            return

        try:
            payload = self._body()
        except ValueError as exc:
            self._send_json({"ok": False, "message": str(exc)}, HTTPStatus.BAD_REQUEST)
            return

        try:
            self._api_post(route, payload)
        except (ValueError, RuntimeError) as exc:
            # Everything the user can get wrong arrives here: unusable
            # settings, a run started twice, a pass asked for at a bad moment.
            self._send_json({"ok": False, "message": str(exc)})
        except Exception as exc:
            logger.exception("Request to %s failed", route)
            self._send_json({"ok": False, "message": f"{type(exc).__name__}: {exc}"})

    def _api_post(self, route: str, payload: dict[str, Any]) -> None:
        session = self.session
        if route == "/api/start":
            self._send_json(session.start(payload))
        elif route == "/api/pause":
            self._send_json(session.toggle_pause())
        elif route == "/api/export":
            self._send_json(session.export())
        elif route == "/api/open":
            self._send_json(session.open_existing(payload))
        elif route == "/api/finish":
            self._send_json(session.finishing_pass())
        elif route == "/api/coherence":
            self._send_json(session.coherence_review())
        elif route == "/api/analyze-function":
            self._send_json(session.analyze_function(payload))
        elif route == "/api/detect":
            self._send_json(detect_endpoint(str(payload.get("base_url", "")).strip()))
        elif route == "/api/key":
            self._send_json(
                store_api_key(
                    str(payload.get("provider", "")),
                    str(payload.get("key", "")).strip(),
                )
            )
        elif route == "/api/quit":
            self._send_json({"ok": True, "message": "Kong is closing."})
            # After the answer is written, or the page never sees it.
            threading.Thread(target=self.server.shutdown, daemon=True).start()
        else:
            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    # ----------------------------------------------------------------- static

    def _send_static(self, name: str) -> None:
        root = STATIC_DIR.resolve()
        candidate = (root / name).resolve()
        if root not in candidate.parents or not candidate.is_file():
            self._send(HTTPStatus.NOT_FOUND, b"Not found", "text/plain; charset=utf-8")
            return
        self._send(
            HTTPStatus.OK,
            candidate.read_bytes(),
            CONTENT_TYPES.get(candidate.suffix.lower(), "application/octet-stream"),
        )


class KongHTTPServer(ThreadingHTTPServer):
    """The server, plus the session and the token its handlers read."""

    daemon_threads = True
    #: Off on Windows, where SO_REUSEADDR lets a second process bind a port
    #: another one is already serving: two Kongs then answer the same URL by
    #: turns, and the one you are looking at is not the one you started.
    #: Elsewhere it is what lets a restart reuse a port still in TIME_WAIT.
    allow_reuse_address = os.name != "nt"

    def __init__(self, address: tuple[str, int], session: KongSession) -> None:
        super().__init__(address, _Handler)
        self.session = session
        self.token = secrets.token_urlsafe(24)
        port = self.server_address[1]
        self.allowed_origins = {
            f"http://{self.server_address[0]}:{port}",
            f"http://localhost:{port}",
            f"http://127.0.0.1:{port}",
        }

    @property
    def url(self) -> str:
        host, port = self.server_address[0], self.server_address[1]
        return f"http://{host}:{port}/?t={self.token}"


def serve(
    initial_binary: str = "", host: str = "127.0.0.1", port: int = 0
) -> KongHTTPServer:
    """Bind the server without serving it. Also the tests' way in."""
    return KongHTTPServer((host, port), KongSession(initial_binary))


def launch(
    initial_binary: str = "",
    host: str = "127.0.0.1",
    port: int = 0,
    open_browser: bool = True,
) -> None:
    """Serve the interface until Ctrl-C, or until the page asks to close."""
    server = serve(initial_binary, host, port)
    url = server.url
    # Flushed: redirected to a file or a pipe, stdout is block-buffered, and
    # the URL — the one thing needed to reach the interface — would sit in the
    # buffer until the server stopped.
    print(f"Kong is at {url}", flush=True)
    print("Keep this terminal open; Ctrl-C stops the interface.", flush=True)

    if open_browser:
        # Slightly after serve_forever starts listening, so the first request
        # the browser makes is answered rather than refused.
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        # Whatever the run had done is checkpointed and Ghidra released, the
        # same as closing the desktop window.
        server.session.shutdown()
        server.server_close()
