"""Putting a call graph into an analysis that has none, or a broken one.

A finished run costs what the model charged for it; its call graph costs
nothing, because the edges come from Ghidra and never from the model. So an
export written before Kong saved the graph, or written while
`GhidraClient.get_callees` was asking the wrong question, does not need the
analysis paying for again — only the edges recovering.

There are two ways to recover them and they are not equivalent:

`edges_from_ghidra` reopens the binary and asks Ghidra, which is the same
answer a fresh run would write. It needs the binary and an installed Ghidra,
and takes as long as loading the program.

`edges_from_source` reads them out of `decompiled.c`, which is already sitting
next to the analysis. Nothing else is required and it is free, but it can only
see what the decompiler printed: a call through a function pointer or a vtable
appears as an indirect call with no name in it, and never becomes an edge.
Anything written this way says so in the document.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kong.ghidra.client import GhidraClient

logger = logging.getLogger(__name__)

#: How the edges in a document were obtained. A reader that cares about
#: completeness — anything deciding what to translate, say — needs to know
#: which of the two it is looking at.
SOURCE_GHIDRA = "ghidra"
SOURCE_DECOMPILED_C = "decompiled.c"

#: One exported function: the doc comment Kong writes, then the C body.
_BLOCK = re.compile(r"(?=^/\*\*$)", re.M)
_NAME = re.compile(r"^ \* @name\s+(\S+)", re.M)
#: A leading block comment, of which each entry has two: the tagged one and the
#: brief repeated as prose. Both mention other functions by name, so they are
#: removed before anything is read as a call.
_LEADING_COMMENT = re.compile(r"\A\s*/\*.*?\*/\s*", re.S)
_CALL = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_PLACEHOLDER = re.compile(r"\A(?:thunk_)?FUN_([0-9a-fA-F]{8})\Z")


def _body(block: str) -> str:
    """The statements of one exported function, and nothing else.

    Two things in front of them read as calls and are not. The doc comments
    name other functions in prose. And the signature is an identifier followed
    by a parenthesis — the function's own — so a block scanned whole reports
    every function as calling itself.
    """
    text = _LEADING_COMMENT.sub("", block, count=1)
    text = _LEADING_COMMENT.sub("", text, count=1)
    opening = text.find("{")
    return text[opening + 1:] if opening >= 0 else ""


def edges_from_source(
    decompiled_c: str, addresses_by_name: dict[str, int]
) -> list[tuple[int, int]]:
    """Recover call edges from an exported `decompiled.c`.

    *addresses_by_name* is what the names in the source mean, which is the
    analysis document's own function list. A callee that no pass ever named is
    still written as `FUN_xxxxxxxx`, and is read back from the name itself —
    those are the edges that matter most, because a function nobody analyzed is
    invisible in every other artefact.
    """
    edges: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    for block in _BLOCK.split(decompiled_c):
        match = _NAME.search(block)
        if match is None:
            continue
        caller = addresses_by_name.get(match.group(1))
        if caller is None:
            continue

        for called in dict.fromkeys(_CALL.findall(_body(block))):
            callee = addresses_by_name.get(called)
            if callee is None:
                placeholder = _PLACEHOLDER.match(called)
                if placeholder is None:
                    continue
                callee = int(placeholder.group(1), 16)
            if (caller, callee) not in seen:
                seen.add((caller, callee))
                edges.append((caller, callee))

    edges.sort()
    return edges


def edges_from_ghidra(client: GhidraClient) -> list[tuple[int, int]]:
    """Ask Ghidra, on an already-open client. The same edges a run would write."""
    edges: list[tuple[int, int]] = []
    for function in client.list_functions():
        for callee in client.get_callees(function.address):
            edges.append((function.address, callee))
    edges.sort()
    return edges


def _addresses_by_name(document: dict[str, Any]) -> dict[str, int]:
    names: dict[str, int] = {}
    for entry in document.get("functions") or []:
        name = entry.get("name")
        try:
            address = int(str(entry.get("address")), 16)
        except (TypeError, ValueError):
            continue
        if name:
            names[name] = address
    return names


def describe(document: dict[str, Any]) -> str:
    """What call graph this document already carries, for a message."""
    graph = document.get("call_graph") or {}
    edges = graph.get("edges") or []
    if not edges:
        return "no call graph"
    source = graph.get("source") or "an earlier version"
    return f"{len(edges)} edges from {source}"


def backfill(
    output_dir: Path,
    client: GhidraClient | None = None,
) -> dict[str, Any]:
    """Write a call graph into the analysis.json in *output_dir*.

    Pass an open *client* to read the edges from Ghidra; leave it out to read
    them from the `decompiled.c` next to the document. Returns a summary, and
    never raises for a problem the caller can be told about.
    """
    path = Path(output_dir) / "analysis.json"
    if not path.is_file():
        return {"ok": False, "message": f"{path} does not exist."}

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return {"ok": False, "message": f"Could not read {path}: {exc}"}

    before = describe(document)

    if client is not None:
        edges = edges_from_ghidra(client)
        source = SOURCE_GHIDRA
    else:
        decompiled = Path(output_dir) / "decompiled.c"
        if not decompiled.is_file():
            return {
                "ok": False,
                "message": (
                    f"{decompiled} does not exist, so there is nothing to read the "
                    f"edges out of. Pass the binary to read them from Ghidra instead."
                ),
            }
        edges = edges_from_source(
            decompiled.read_text(encoding="utf-8", errors="replace"),
            _addresses_by_name(document),
        )
        source = SOURCE_DECOMPILED_C

    if not edges:
        return {"ok": False, "message": f"No call edges found ({source})."}

    document["call_graph"] = {
        "source": source,
        "edges": [[f"0x{caller:08x}", f"0x{callee:08x}"] for caller, callee in edges],
    }

    # Written beside the file and moved into place, so an interrupted write
    # cannot leave a truncated document where a complete analysis used to be.
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(document, indent=2), encoding="utf-8")
        temporary.replace(path)
    except OSError as exc:
        return {"ok": False, "message": f"Could not write {path}: {exc}"}

    logger.info("Wrote %d call edges (%s) to %s", len(edges), source, path)
    return {
        "ok": True,
        "edges": len(edges),
        "source": source,
        "path": str(path),
        "before": before,
        "message": (
            f"{path}: replaced {before} with {len(edges)} edges from {source}."
        ),
    }
