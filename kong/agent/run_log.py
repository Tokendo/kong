"""The run log: one chronological trace per analysis, written to disk.

Two things go into `<output_dir>/events.log`: the pipeline events the CLI, TUI
and GUI already show, and the logging records the package emits along the way
(chunk prompt sizes, unparseable responses, write-back failures). The second
half is the reason the file exists — the package logs those diligently, but no
handler was ever attached, so they went nowhere and a failed function was left
with a one-line message in a scrollback nobody kept.

Attaching a handler to the `kong` logger also stops logging's last-resort
fallback from printing warnings to stderr, where they used to corrupt the TUI.
"""

from __future__ import annotations

import logging
from pathlib import Path

from kong.agent.events import Event, EventType

LOG_NAME = "events.log"

# Events that report a failure. They are written at WARNING so a whole run can
# be triaged with `grep WARNING events.log`.
FAILURE_EVENTS = frozenset({
    EventType.FUNCTION_ERROR,
    EventType.RUN_ERROR,
})

_FORMAT = "%(asctime)s  %(levelname)-7s  %(name)s  %(message)s"


class RunLog:
    """A file handler on the `kong` logger, fed by the supervisor's events.

    Open it before the run and close it after; `record` is meant to be
    registered with `Supervisor.on_event`.
    """

    def __init__(self, output_dir: Path, verbose: bool = False) -> None:
        self.path = Path(output_dir) / LOG_NAME
        self.verbose = verbose
        self._handler: logging.FileHandler | None = None
        self._logger = logging.getLogger("kong")
        self._events = logging.getLogger("kong.events")
        self._restore_level: int | None = None

    def open(self) -> Path:
        """Start a fresh trace, truncating any log from an earlier run."""
        if self._handler is not None:
            return self.path

        self.path.parent.mkdir(parents=True, exist_ok=True)
        level = logging.DEBUG if self.verbose else logging.INFO
        handler = logging.FileHandler(self.path, mode="w", encoding="utf-8")
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(_FORMAT))

        self._restore_level = self._logger.level
        self._logger.setLevel(level)
        self._logger.addHandler(handler)
        self._handler = handler
        return self.path

    def close(self) -> None:
        """Detach the handler. Safe to call on a log that never opened."""
        if self._handler is None:
            return
        self._logger.removeHandler(self._handler)
        self._handler.close()
        self._handler = None
        if self._restore_level is not None:
            self._logger.setLevel(self._restore_level)
            self._restore_level = None

    def record(self, event: Event) -> None:
        """Write one pipeline event. Registered as a supervisor listener."""
        if self._handler is None:
            return

        parts = [event.type.value]
        if event.phase is not None:
            parts.append(f"[{event.phase.value}]")
        if event.message:
            parts.append(event.message)

        # Most messages name the function, some only the address; add it when
        # it is missing so every line can be grepped by address.
        address = event.data.get("address")
        if isinstance(address, int) and f"{address:08x}" not in event.message:
            parts.append(f"(0x{address:08x})")

        # For "No analysis for X" the reason only exists in the payload, and
        # for an oversized function the message is a summary of it.
        error = event.data.get("error")
        if error and str(error) not in event.message:
            parts.append(f"— {error}")

        level = logging.WARNING if event.type in FAILURE_EVENTS else logging.INFO
        self._events.log(level, " ".join(parts))

    def __enter__(self) -> RunLog:
        self.open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
