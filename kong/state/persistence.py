"""Analysis state that survives the process.

A run is a long chain of paid calls, and any link can break it: a rate limit
that kills a chunk, a Ghidra crash, a laptop that sleeps, an impatient Ctrl-C.
Without a record on disk the next run pays for all of it again.

The state file sits next to the output and holds one entry per analysed
function. The next run over the same binary reads it back, so it only pays for
what the first one did not finish.

The file also records which binary it describes. Output directories are reused —
`./kong_output` is the default for every run — so without that check a second
binary would inherit the first one's names, which is worse than paying for the
analysis again.

Only the fields that cost an LLM call are stored. Struct proposals are not: they
are consumed by the cleanup phase of the run that produced them, and a resumed
run re-derives them for whatever it re-analyses.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from kong.agent.models import FunctionResult

logger = logging.getLogger(__name__)

STATE_NAME = "analysis_state.json"

#: Bumped when the entry shape changes, so an old file is ignored rather than
#: half-read into a shape that no longer matches.
STATE_VERSION = 4

#: Read in chunks: a binary can be hundreds of megabytes and the hash is
#: computed while the user is waiting for the run to start.
_HASH_CHUNK = 1024 * 1024


@dataclass(frozen=True)
class BinaryIdentity:
    """What makes a saved state belong to one binary rather than another.

    The digest is what actually decides: a path is reused across rebuilds, and
    a rebuilt binary has different function addresses, which is exactly the
    case where restoring old names would be silently wrong.
    """

    path: str
    size: int
    sha256: str

    @classmethod
    def of(cls, binary_path: str | Path) -> BinaryIdentity | None:
        """Identify a binary on disk. None when it cannot be read."""
        path = Path(binary_path)
        try:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                while chunk := handle.read(_HASH_CHUNK):
                    digest.update(chunk)
            return cls(
                path=str(path.resolve()),
                size=path.stat().st_size,
                sha256=digest.hexdigest(),
            )
        except OSError:
            logger.warning("Could not identify %s", path, exc_info=True)
            return None

    def as_dict(self) -> dict[str, object]:
        return {"path": self.path, "size": self.size, "sha256": self.sha256}

    def matches(self, stored: dict[str, object] | None) -> bool:
        """True when a state file's recorded binary is this one.

        A file written before identities existed, or by a build that could not
        read the binary, records nothing; it is accepted rather than thrown
        away, since the alternative is re-paying for an analysis that is
        probably the right one.
        """
        if not stored:
            return True
        return stored.get("sha256") == self.sha256


_FIELDS = (
    "original_name",
    "name",
    "signature",
    "confidence",
    "classification",
    "comments",
    "reasoning",
    "error",
    "model",
    "refined",
    "llm_calls",
    "skipped",
    "skip_reason",
    "signature_applied",
    "obfuscation_techniques",
    "deobfuscation_tool_calls",
)


def state_path(output_dir: Path) -> Path:
    return Path(output_dir) / STATE_NAME


def save_state(
    results: dict[int, FunctionResult],
    output_dir: Path,
    binary: BinaryIdentity | None = None,
    call_edges: list[tuple[int, int]] | None = None,
) -> Path:
    """Write the results so far. Overwrites any earlier state.

    *call_edges* is the call graph triage read from Ghidra. It is saved because
    it is the one thing a run learns that it cannot work out again from its own
    output: parsing the exported C back only finds calls between functions that
    were named, and loses every call into one that was skipped.
    """
    entries = [
        {"address": addr, **{f: getattr(result, f) for f in _FIELDS}}
        for addr, result in sorted(results.items())
    ]

    document: dict[str, object] = {"version": STATE_VERSION}
    if binary is not None:
        document["binary"] = binary.as_dict()
    document["functions"] = entries
    if call_edges is not None:
        document["call_edges"] = [[caller, callee] for caller, callee in call_edges]

    path = state_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Written to a neighbouring file and moved into place: a run killed halfway
    # through the write would otherwise leave a truncated file where the whole
    # previous checkpoint used to be.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2), encoding="utf-8")
    temporary.replace(path)
    logger.debug("Saved analysis state: %d results to %s", len(entries), path)
    return path


def _read_document(
    output_dir: Path, binary: BinaryIdentity | None = None,
) -> dict | None:
    """The state file's contents, or None when it is missing or unusable."""
    path = state_path(output_dir)
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Ignoring unreadable state file %s: %s", path, e)
        return None

    version = data.get("version")
    if version != STATE_VERSION:
        logger.warning(
            "Ignoring state file %s: version %r, expected %d.",
            path, version, STATE_VERSION,
        )
        return None

    if binary is not None and not binary.matches(data.get("binary")):
        stored = data.get("binary") or {}
        logger.warning(
            "Ignoring state file %s: it was written for %s, not %s.",
            path, stored.get("path", "another binary"), binary.path,
        )
        return None

    return data


def load_state(
    output_dir: Path, binary: BinaryIdentity | None = None,
) -> dict[int, FunctionResult]:
    """Read a previous run's results. Empty dict when there is nothing usable.

    Pass *binary* to refuse a state file written for a different one.
    """
    data = _read_document(output_dir, binary)
    if data is None:
        return {}
    path = state_path(output_dir)

    results: dict[int, FunctionResult] = {}
    for entry in data.get("functions", []):
        address = entry.get("address")
        if not isinstance(address, int):
            continue
        known = {f: entry[f] for f in _FIELDS if f in entry}
        # The only field without a default of its own.
        known.setdefault("original_name", f"FUN_{address:08x}")
        results[address] = FunctionResult(address=address, **known)

    logger.info("Loaded %d results from %s", len(results), path)
    return results


def load_call_edges(
    output_dir: Path, binary: BinaryIdentity | None = None,
) -> list[tuple[int, int]]:
    """Read the call graph a previous run saved. Empty when there is none.

    Kept separate from `load_state` so a caller that only wants the results
    does not pay to build the edge list, and so a state file written before
    the graph was saved simply has none.
    """
    document = _read_document(output_dir, binary)
    if document is None:
        return []
    edges: list[tuple[int, int]] = []
    for pair in document.get("call_edges", []):
        if isinstance(pair, list) and len(pair) == 2:
            caller, callee = pair
            if isinstance(caller, int) and isinstance(callee, int):
                edges.append((caller, callee))
    return edges
