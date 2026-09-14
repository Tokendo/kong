"""Configuration management for Kong."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from kong.agent.refinement import DEFAULT_REFINE_BELOW
from kong.ghidra.environment import find_ghidra_install


def _default_project_dir() -> str:
    """Ghidra project directory, under the platform temp dir.

    A hardcoded ``/tmp`` is not a valid path on Windows.
    """
    return str(Path(tempfile.gettempdir()) / "kong_ghidra")


#: Z.ai's OpenAI-compatible endpoint. Keys issued with a Coding Plan are served
#: from https://api.z.ai/api/coding/paas/v4 instead, which --base-url overrides.
ZAI_BASE_URL = "https://api.z.ai/api/paas/v4"


class RunStage(Enum):
    """How much of the two-pass analysis one run is responsible for.

    A full run drafts and finishes in one go, which is what most binaries
    want. Splitting it in two is for the case where the draft is the cheap,
    long part and the second pass is the one worth deciding about: the draft
    reads every function, the run stops, and the finishing pass is started by
    hand once its results have been looked at.
    """

    FULL = "full"
    #: Read every function with the draft model and stop. Nothing is sent to
    #: the primary model, including the functions the draft cannot handle:
    #: they are held for the finishing pass rather than quietly upgraded.
    DRAFT = "draft"
    #: Re-read what a saved draft left failed, under the threshold, or never
    #: analyzed at all, then redo cleanup, synthesis and export.
    FINISH = "finish"


class LLMProvider(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    ZAI = "zai"
    CUSTOM = "custom"

    @property
    def display_name(self) -> str:
        return {
            LLMProvider.ANTHROPIC: "Anthropic",
            LLMProvider.OPENAI: "OpenAI",
            LLMProvider.ZAI: "Z.ai",
            LLMProvider.CUSTOM: "Custom",
        }[self]


@dataclass
class GhidraConfig:
    install_dir: str | None = None
    project_dir: str = field(default_factory=_default_project_dir)
    project_name: str = "kong_project"

    def __post_init__(self) -> None:
        if self.install_dir is None:
            self.install_dir = find_ghidra_install()


@dataclass
class OutputConfig:
    directory: Path = field(default_factory=lambda: Path("./kong_output"))
    formats: list[str] = field(default_factory=lambda: ["source", "json", "ghidra"])
    #: Which functions the transpiling exporters translate, by address. None
    #: means the whole binary, which is what a translation pass used to cost
    #: whatever the reader was actually interested in: it is a second full LLM
    #: pass, and most of a binary is runtime and helpers nobody reads.
    transpile_addresses: set[int] | None = None
    #: Pull in what the selected functions call, transitively. On by default
    #: because a function translated without its callees refers to names that
    #: were never produced. Turn it off to translate literally one function.
    transpile_follow_callees: bool = True


@dataclass
class LLMConfig:
    provider: LLMProvider = LLMProvider.ANTHROPIC
    model: str | None = None
    #: Fast model for the first pass. When set, ``model`` becomes the second
    #: pass: it only sees what the draft got wrong, was unsure about, or was
    #: never given (large and obfuscated functions skip the draft). Both models
    #: are served by the same client, so they share a provider and an endpoint.
    draft_model: str | None = None
    #: Confidence below which a drafted function is re-analyzed by ``model``.
    refine_below: int = DEFAULT_REFINE_BELOW
    api_key: str | None = None
    base_url: str | None = None
    max_prompt_chars: int | None = None
    #: How many functions go into one batch call. None means the figure the
    #: model table carries. Explicit here rather than model-derived because
    #: the right number is a property of the endpoint on the day: a hosted API
    #: that is rate limiting, or a model that starts losing track of a long
    #: batch, both want fewer functions per call than its context window
    #: allows.
    max_chunk_functions: int | None = None
    max_output_tokens: int | None = None
    #: Seconds to wait on one LLM request before giving up on it. None means
    #: the client default. Worth raising for a local endpoint that generates
    #: slowly, and lowering for a hosted one that should answer promptly: it
    #: bounds what a request that will never be answered costs the run.
    request_timeout: float | None = None


@dataclass
class AnalysisConfig:
    """How the analysis phase spends its time, as opposed to where it sends it."""

    #: Share of a binary's functions that must trip the obfuscation heuristics
    #: before the expensive agentic deobfuscation loop is used at all. The
    #: heuristics read structure, and a `while(1)` around a `switch` is both
    #: control-flow flattening and every hand-written state machine, so on a
    #: clean binary they fire on the CRT and cost hours. A protector is applied
    #: wholesale, so its signature is a large share, not a handful. 0 believes
    #: every detection, which is the old behaviour.
    obfuscation_threshold: float = 0.05
    #: Wall clock one function's deobfuscation loop may spend. None lifts the
    #: bound, which is how a single function once cost 1h48 and produced
    #: nothing.
    deobfuscation_time_budget: float | None = 900.0
    #: Chunk calls in flight at once. The work queue is ordered bottom-up and
    #: functions at the same depth are independent, so the calls do not have to
    #: wait for each other. None picks a default from the provider: several for
    #: a hosted API, one for a local endpoint that is already using the machine
    #: it runs on.
    chunk_concurrency: int | None = None
    #: Attempts on a chunk call before its functions are split and retried
    #: individually. A chunk failure used to fail every function in it at once.
    chunk_attempts: int = 2
    #: Skip functions the signature database already identifies, instead of
    #: paying a model to name a documented library function. Off by default:
    #: the descriptions are still worth something to some readers.
    skip_matched_signatures: bool = False


@dataclass
class KongConfig:
    ghidra: GhidraConfig = field(default_factory=GhidraConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    headless: bool = False
    verbose: bool = False
    #: Which half of the analysis this run does. See RunStage.
    stage: RunStage = RunStage.FULL
    #: Pick up the saved results of an earlier run over the same binary
    #: instead of paying for them again. On by default: a run that was closed,
    #: crashed or interrupted is the common case, and the state file is bound
    #: to the binary it describes, so a different one is never resumed.
    resume: bool = True
