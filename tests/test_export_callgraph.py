"""Putting a call graph into an analysis that was paid for without one.

A run's cost is the model's; its call graph costs nothing, because the edges
come from Ghidra. So an export written before Kong saved the graph — or while
the client was asking Ghidra the wrong question and getting fourteen edges —
needs the edges recovering, not the analysis buying again.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import MagicMock

from click.testing import CliRunner

from kong.__main__ import cli
from kong.export.callgraph import (
    SOURCE_DECOMPILED_C,
    SOURCE_GHIDRA,
    backfill,
    describe,
    edges_from_ghidra,
    edges_from_source,
)
from kong.ghidra.types import FunctionClassification, FunctionInfo


def _entry(name, address, decompilation, brief="does something"):
    """One function as Kong writes it into decompiled.c."""
    return (
        "/**\n"
        f" * @name  {name}\n"
        f" * @brief {brief}\n"
        " * @confidence 80%\n"
        " * @classification utility\n"
        f" * @address 0x{address:08x}\n"
        " */\n"
        "\n"
        f"/* {brief} */\n"
        "\n"
        f"{decompilation}\n"
    )


NAMES = {"game_tick": 0x401000, "draw_hud": 0x402000, "heap_alloc": 0x403000}


class TestEdgesFromSource:
    def test_a_call_becomes_an_edge(self):
        source = _entry("game_tick", 0x401000, "void game_tick(void) { draw_hud(); }")

        assert edges_from_source(source, NAMES) == [(0x401000, 0x402000)]

    def test_the_same_callee_twice_is_one_edge(self):
        source = _entry(
            "game_tick", 0x401000,
            "void game_tick(void) { draw_hud(); draw_hud(); }",
        )

        assert edges_from_source(source, NAMES) == [(0x401000, 0x402000)]

    def test_a_callee_nobody_named_is_read_from_the_placeholder(self):
        """These are the edges that matter most: invisible everywhere else."""
        source = _entry(
            "game_tick", 0x401000, "void game_tick(void) { FUN_0045f2c0(1); }",
        )

        assert edges_from_source(source, NAMES) == [(0x401000, 0x45F2C0)]

    def test_a_thunk_placeholder_is_read_too(self):
        source = _entry(
            "game_tick", 0x401000, "void game_tick(void) { thunk_FUN_00456700(); }",
        )

        assert edges_from_source(source, NAMES) == [(0x401000, 0x456700)]

    def test_names_in_the_doc_comment_are_not_calls(self):
        """The brief names other functions in prose; it is not the code."""
        source = _entry(
            "game_tick", 0x401000,
            "void game_tick(void) { return; }",
            brief="dispatches through draw_hud(x) and heap_alloc(y)",
        )

        assert edges_from_source(source, NAMES) == []

    def test_control_flow_keywords_are_not_calls(self):
        source = _entry(
            "game_tick", 0x401000,
            "void game_tick(void) { if (x) { while (y) { switch (z) {} } } }",
        )

        assert edges_from_source(source, NAMES) == []

    def test_recursion_is_an_edge_like_any_other(self):
        source = _entry(
            "game_tick", 0x401000, "void game_tick(void) { game_tick(); }",
        )

        assert edges_from_source(source, NAMES) == [(0x401000, 0x401000)]

    def test_several_functions_each_get_their_own_edges(self):
        source = (
            _entry("game_tick", 0x401000, "void game_tick(void) { draw_hud(); }")
            + _entry("draw_hud", 0x402000, "void draw_hud(void) { heap_alloc(); }")
        )

        assert edges_from_source(source, NAMES) == [
            (0x401000, 0x402000),
            (0x402000, 0x403000),
        ]

    def test_a_function_the_document_does_not_know_is_skipped(self):
        source = _entry("mystery", 0x409000, "void mystery(void) { draw_hud(); }")

        assert edges_from_source(source, NAMES) == []

    def test_an_empty_file_has_no_edges(self):
        assert edges_from_source("", NAMES) == []


class TestEdgesFromGhidra:
    def test_it_asks_the_client_for_every_function(self):
        client = MagicMock()
        client.list_functions.return_value = [
            FunctionInfo(address=0x401000, name="a", size=10,
                         classification=FunctionClassification.SMALL),
            FunctionInfo(address=0x402000, name="b", size=10,
                         classification=FunctionClassification.SMALL),
        ]
        client.get_callees.side_effect = lambda addr: (
            [0x402000, 0x403000] if addr == 0x401000 else []
        )

        assert edges_from_ghidra(client) == [
            (0x401000, 0x402000),
            (0x401000, 0x403000),
        ]


def _analysis(directory: Path, *, edges=None, source=None, decompiled=None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    document = {
        "binary": {"name": "FA18.exe"},
        "stats": {"cost_usd": 29.95},
        "functions": [
            {"address": f"0x{address:08x}", "original_name": f"FUN_{address:08x}",
             "name": name, "confidence": 80, "classification": "utility"}
            for name, address in NAMES.items()
        ],
        "failures": [],
    }
    if edges is not None:
        graph = {"edges": edges}
        if source:
            graph["source"] = source
        document["call_graph"] = graph
    (directory / "analysis.json").write_text(json.dumps(document), encoding="utf-8")
    if decompiled is not None:
        (directory / "decompiled.c").write_text(decompiled, encoding="utf-8")
    return directory


class TestBackfill:
    def test_it_writes_the_edges_it_recovered(self, tmp_path):
        out = _analysis(
            tmp_path / "out",
            decompiled=_entry(
                "game_tick", 0x401000, "void game_tick(void) { draw_hud(); }"
            ),
        )

        result = backfill(out)

        assert result["ok"]
        assert result["edges"] == 1
        document = json.loads((out / "analysis.json").read_text(encoding="utf-8"))
        assert document["call_graph"]["edges"] == [["0x00401000", "0x00402000"]]

    def test_the_provenance_is_written_down(self, tmp_path):
        """An approximation must never pass for Ghidra's own answer."""
        out = _analysis(
            tmp_path / "out",
            decompiled=_entry(
                "game_tick", 0x401000, "void game_tick(void) { draw_hud(); }"
            ),
        )

        backfill(out)

        document = json.loads((out / "analysis.json").read_text(encoding="utf-8"))
        assert document["call_graph"]["source"] == SOURCE_DECOMPILED_C

    def test_ghidra_is_recorded_as_the_source_when_it_is(self, tmp_path):
        out = _analysis(tmp_path / "out")
        client = MagicMock()
        client.list_functions.return_value = [
            FunctionInfo(address=0x401000, name="a", size=10,
                         classification=FunctionClassification.SMALL),
        ]
        client.get_callees.return_value = [0x402000]

        backfill(out, client)

        document = json.loads((out / "analysis.json").read_text(encoding="utf-8"))
        assert document["call_graph"]["source"] == SOURCE_GHIDRA

    def test_a_broken_graph_is_replaced_and_the_old_one_named(self, tmp_path):
        """The FA18 case: fourteen edges for 1351 functions."""
        out = _analysis(
            tmp_path / "out",
            edges=[["0x00401000", "0x0000008f"]],
            decompiled=_entry(
                "game_tick", 0x401000, "void game_tick(void) { draw_hud(); }"
            ),
        )

        result = backfill(out)

        assert "1 edges" in result["before"]
        document = json.loads((out / "analysis.json").read_text(encoding="utf-8"))
        assert document["call_graph"]["edges"] == [["0x00401000", "0x00402000"]]

    def test_everything_else_in_the_document_is_left_alone(self, tmp_path):
        out = _analysis(
            tmp_path / "out",
            decompiled=_entry(
                "game_tick", 0x401000, "void game_tick(void) { draw_hud(); }"
            ),
        )

        backfill(out)

        document = json.loads((out / "analysis.json").read_text(encoding="utf-8"))
        assert document["stats"]["cost_usd"] == 29.95
        assert len(document["functions"]) == len(NAMES)

    def test_a_missing_document_is_reported(self, tmp_path):
        result = backfill(tmp_path)

        assert not result["ok"]
        assert "does not exist" in result["message"]

    def test_a_missing_decompiled_c_says_to_use_the_binary(self, tmp_path):
        out = _analysis(tmp_path / "out")

        result = backfill(out)

        assert not result["ok"]
        assert "decompiled.c" in result["message"]
        assert "--binary" in result["message"] or "Ghidra" in result["message"]

    def test_an_unreadable_document_is_reported_not_raised(self, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        (out / "analysis.json").write_text("{ not json", encoding="utf-8")

        result = backfill(out)

        assert not result["ok"]
        assert "Could not read" in result["message"]

    def test_nothing_recovered_leaves_the_document_alone(self, tmp_path):
        out = _analysis(
            tmp_path / "out",
            decompiled=_entry("game_tick", 0x401000, "void game_tick(void) { }"),
        )
        before = (out / "analysis.json").read_text(encoding="utf-8")

        result = backfill(out)

        assert not result["ok"]
        assert (out / "analysis.json").read_text(encoding="utf-8") == before


class TestDescribe:
    def test_no_graph(self):
        assert describe({}) == "no call graph"

    def test_an_older_graph_without_a_source(self):
        assert "earlier version" in describe({"call_graph": {"edges": [[1, 2]]}})

    def test_a_graph_that_names_its_source(self):
        described = describe(
            {"call_graph": {"source": "ghidra", "edges": [[1, 2], [3, 4]]}}
        )

        assert described == "2 edges from ghidra"


class TestCommand:
    def test_it_backfills_the_directory_it_is_given(self, tmp_path):
        out = _analysis(
            tmp_path / "out",
            decompiled=_entry(
                "game_tick", 0x401000, "void game_tick(void) { draw_hud(); }"
            ),
        )

        result = CliRunner().invoke(cli, ["graph", str(out)])

        assert result.exit_code == 0
        # rich colours the count, so the digits and the words are not adjacent.
        plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        assert "1 call edges" in plain
        assert "decompiled.c" in plain

    def test_it_says_what_it_could_not_do(self, tmp_path):
        out = _analysis(tmp_path / "out")

        result = CliRunner().invoke(cli, ["graph", str(out)])

        assert result.exit_code == 1

    def test_a_directory_that_is_not_there_is_refused_by_click(self, tmp_path):
        result = CliRunner().invoke(cli, ["graph", str(tmp_path / "nope")])

        assert result.exit_code != 0
