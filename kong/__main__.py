"""Kong CLI — world's first AI reverse engineer."""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import click
from rich.console import Console
from rich.table import Table
from rich.markup import escape
from rich.prompt import Prompt

from kong import __version__
from kong.agent.events import Event, EventType
from kong.agent.refinement import DEFAULT_REFINE_BELOW
from kong.agent.supervisor import Supervisor
from kong.banner import (
    _ENV_VARS,
    _KEY_EXAMPLES,
    _KEY_URLS,
    check_api_key,
    print_analyze_header,
    print_banner,
    resolve_api_key,
)
from kong.config import (
    ZAI_BASE_URL,
    AnalysisConfig,
    GhidraConfig,
    KongConfig,
    LLMConfig,
    LLMProvider,
    OutputConfig,
    RunStage,
)
from kong.db import get_custom_config, get_default_provider, get_enabled_providers, is_setup_complete, save_setup
from kong.evals.harness import score as eval_score
from kong.ghidra.client import GhidraClient, GhidraClientError
from kong.llm.usage import TokenUsage, is_known_model
from kong.state.persistence import state_path
from kong.banner import _ENV_VARS, _KEY_EXAMPLES, _KEY_URLS
from kong.llm.openai_client import OpenAIClient
from kong.llm.client import AnthropicClient
from kong.tui.app import KongApp


if TYPE_CHECKING:
    from kong.agent.analyzer import LLMClient

console = Console()


_DEFAULT_MODELS: dict[LLMProvider, str] = {
    LLMProvider.ANTHROPIC: "claude-opus-5",
    LLMProvider.OPENAI: "gpt-4o",
    LLMProvider.ZAI: "glm-5.3",
}

_PROVIDER_LABELS: dict[LLMProvider, str] = {
    LLMProvider.ANTHROPIC: "Anthropic (Claude)",
    LLMProvider.OPENAI: "OpenAI (GPT-4o)",
    LLMProvider.ZAI: "Z.ai (GLM)",
    LLMProvider.CUSTOM: "Custom (OpenAI-compatible)",
}

_NOT_NEEDED_STR = "not-needed"

# Context assumed for a local endpoint that cannot be queried. Same figure the
# GUI opens with.
_ASSUMED_LOCAL_CONTEXT = 32768

def create_llm_client(config: LLMConfig) -> LLMClient:
    """Instantiate the appropriate LLM client based on provider config."""
    from kong.llm.usage import register_custom_model

    model = config.model or _DEFAULT_MODELS.get(config.provider, "gpt-4o")
    # What one request may spend, in tokens and in seconds. Both are the same
    # for every provider and both are overrides: left out, a single-function
    # call opens on the client's default budget (2048), a batch call is sized
    # by the model table, and the deadline is the client's own. They are here
    # for when the user has an opinion — a local endpoint that generates
    # slowly wants a longer deadline, a hosted one a shorter.
    overrides: dict[str, object] = (
        {"max_tokens": config.max_output_tokens}
        if config.max_output_tokens is not None
        else {}
    )
    if config.request_timeout is not None:
        overrides["timeout"] = config.request_timeout
    if config.provider is LLMProvider.CUSTOM:
        # The draft model is billed and reported separately, so it needs a
        # pricing entry of its own or the usage table drops what it spent.
        if config.draft_model:
            register_custom_model(config.draft_model)
        # Local servers don't need auth, but the OpenAI SDK rejects None/empty
        # api_key by falling back to OPENAI_API_KEY env var or raising an error.
        # A dummy value satisfies the SDK while local servers ignore it.
        api_key = config.api_key if config.api_key else _NOT_NEEDED_STR
        register_custom_model(model)
        return OpenAIClient(
            model=model,
            base_url=config.base_url,
            api_key=api_key,
            **overrides,
        )
    if config.provider is LLMProvider.ZAI:
        # OpenAI-compatible surface, but the key lives under its own name: the
        # SDK would otherwise fall back to OPENAI_API_KEY and send an
        # unrelated key to Z.ai.
        api_key = resolve_api_key(LLMProvider.ZAI, config.api_key)
        if not api_key:
            raise ValueError(
                "ZAI_API_KEY is not set. Create a key at "
                "https://z.ai/manage-apikey/apikey-list and export it."
            )
        return OpenAIClient(
            model=model,
            base_url=config.base_url or ZAI_BASE_URL,
            api_key=api_key,
            **overrides,
        )
    if config.provider is LLMProvider.OPENAI:
        return OpenAIClient(
            model=model,
            api_key=resolve_api_key(LLMProvider.OPENAI, config.api_key),
            **overrides,
        )
    return AnthropicClient(
        model=model,
        api_key=resolve_api_key(LLMProvider.ANTHROPIC, config.api_key),
        **overrides,
    )


def _discover_endpoint(base_url: str, api_key: str | None):
    """Best-effort endpoint introspection. Returns None if it is not available."""
    from kong.llm.endpoint import discover

    try:
        return discover(base_url, api_key=api_key)
    except Exception as e:
        console.print(
            f"  [dim]Could not read {escape(base_url)} ({escape(str(e)[:80])}); "
            f"enter the values by hand.[/dim]"
        )
        return None


def _warn_if_model_missing(config: LLMConfig, model: str) -> None:
    """Say so early when a local endpoint does not serve the draft model.

    A mistyped draft model is not visible until the first chunk comes back
    empty, by which point Ghidra has loaded the binary and the draft pass is
    already walking through the queue failing one call at a time.
    """
    if config.provider is not LLMProvider.CUSTOM or not config.base_url:
        return

    info = _discover_endpoint(config.base_url, config.api_key)
    if info is None or not info.models:
        return

    served = {m.id for m in info.models}
    if model not in served:
        console.print(
            f"  [yellow]{escape(config.base_url)} does not list "
            f"[bold]{escape(model)}[/bold]. Served: "
            f"{escape(', '.join(sorted(served)[:6]))}[/yellow]"
        )


def _parse_address_selection(value: str) -> set[int]:
    """Read a set of function addresses from a file path or an inline list.

    Accepts what a reader actually has to hand: a path to a file, or the
    addresses typed straight on the command line, separated by commas,
    whitespace or newlines. Anything after the address on a line is ignored,
    so a two-column "address  name" listing pasted from a call-graph tool
    works unchanged, as do `#` comments.
    """
    path = Path(value)
    try:
        text = path.read_text(encoding="utf-8") if path.is_file() else value
    except OSError as exc:
        raise click.BadParameter(f"cannot read {value}: {exc}") from exc

    addresses: set[int] = set()
    for raw_line in text.splitlines():
        # Comment first: a comma inside one is prose, not a separator.
        for field in raw_line.split("#", 1)[0].split(","):
            # First token of the field, so "<address>  <name>" needs no editing.
            token = field.split(maxsplit=1)[0] if field.split() else ""
            if not token:
                continue
            try:
                addresses.add(
                    int(token, 16) if token.lower().startswith("0x") else int(token, 0)
                )
            except ValueError:
                raise click.BadParameter(f"{token!r} is not an address") from None

    if not addresses:
        raise click.BadParameter(f"no addresses found in {value!r}")
    return addresses


def _int_or_none(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def validate_base_url(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        raise click.BadParameter(
            f"base-url must start with http:// or https:// (got '{url}')"
        )
    return url.rstrip("/")


def resolve_provider(cli_override: str | None = None, base_url: str | None = None) -> LLMProvider:
    """Pick the best available provider: CLI flag > DB default > any enabled key."""
    if base_url and not cli_override:
        return LLMProvider.CUSTOM

    if cli_override:
        provider = LLMProvider(cli_override)
        if provider is LLMProvider.CUSTOM:
            return provider
        if check_api_key(provider):
            return provider
        env_var = _ENV_VARS.get(provider, "unknown")
        console.print(
            f"[yellow]Warning:[/yellow] --provider {provider.value} specified "
            f"but {env_var} is not set."
        )
        raise SystemExit(1)

    default = get_default_provider()
    if default is LLMProvider.CUSTOM:
        return default
    if default and check_api_key(default):
        return default

    for provider in get_enabled_providers():
        if provider is LLMProvider.CUSTOM:
            continue
        if check_api_key(provider):
            return provider

    console.print("[red]No API keys found for any configured provider.[/red]")
    console.print("Run [bold cyan]kong setup[/bold cyan] to configure providers.")
    raise SystemExit(1)


@click.group(invoke_without_command=True)
@click.version_option(version=__version__, prog_name="kong")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output.")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """Kong — world's first AI reverse engineer.

    Point it at a stripped binary, get back clean decompiled source,
    annotated Ghidra project, and structured JSON.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose

    if ctx.invoked_subcommand is None:
        print_banner(console)
        console.print()
        console.print("Usage: [bold]kong analyze <binary>[/bold]")
        console.print("       [bold]kong info <binary>[/bold]")
        console.print("       [bold]kong setup[/bold]")
        console.print()
        console.print("Run [bold]kong --help[/bold] for all options.")


def _print_final_stats(supervisor: Supervisor, llm_client: LLMClient) -> None:
    stats = supervisor.stats
    is_custom = supervisor.config.llm.provider is LLMProvider.CUSTOM
    console.print()
    console.print(
        f"[bold]Results:[/bold] {stats.named}/{stats.total_functions} functions named "
        f"({stats.renamed} renamed, {stats.confirmed} confirmed)"
    )
    console.print(
        f"[bold]Confidence:[/bold] {stats.high_confidence} high, "
        f"{stats.medium_confidence} med, {stats.low_confidence} low"
    )
    draft_model = supervisor.config.llm.draft_model
    if draft_model:
        drafted = sum(
            1 for r in supervisor.results.values() if r.model == draft_model
        )
        refined = sum(1 for r in supervisor.results.values() if r.refined)
        console.print(
            f"[bold]Passes:[/bold] {drafted} left to {draft_model}, "
            f"{refined} re-analyzed by {supervisor.config.llm.model or 'the main model'}"
        )
    console.print(f"[bold]LLM calls:[/bold] {stats.llm_calls}")
    usage = getattr(llm_client, "usage", None)
    if isinstance(usage, TokenUsage):
        console.print(
            f"[bold]Tokens:[/bold] {usage.input_tokens:,} in / "
            f"{usage.output_tokens:,} out / "
            f"{usage.total_tokens:,} total"
        )
        for model_name, mu in usage.by_model.items():
            if is_custom:
                console.print(
                    f"  {model_name}: {mu.calls} calls, "
                    f"{mu.input_tokens:,} in / {mu.output_tokens:,} out"
                )
            else:
                estimated = "" if is_known_model(model_name) else " [dim](estimated)[/dim]"
                console.print(
                    f"  {model_name}: {mu.calls} calls, "
                    f"${mu.cost_usd(model_name):.4f}{estimated}"
                )
    if is_custom:
        console.print("[dim]Cost tracking disabled for custom provider[/dim]")
    else:
        total_cost = getattr(llm_client, "total_cost_usd", 0.0)
        console.print(f"[bold]Cost:[/bold] ${total_cost:.4f}")
        unpriced = [
            name for name in (usage.by_model if isinstance(usage, TokenUsage) else {})
            if not is_known_model(name)
        ]
        if unpriced:
            console.print(
                "[dim]No published pricing for "
                f"{', '.join(unpriced)} — cost shown at default rates.[/dim]"
            )
    console.print(f"[bold]Duration:[/bold] {stats.duration_seconds:.1f}s")
    if supervisor.config.stage is RunStage.DRAFT:
        console.print(
            f"[bold]Waiting for the finishing pass:[/bold] "
            f"{supervisor.pending_finish} functions — "
            f"[cyan]kong analyze <binary> --stage finish[/cyan]"
        )


@cli.command()
@click.argument("binary", type=click.Path(exists=True, dir_okay=False))
@click.option("--headless", is_flag=True, help="Run without TUI (for CI/Docker).")
@click.option(
    "--output", "-o",
    type=click.Path(),
    default="./kong_output",
    help="Output directory.",
)
@click.option(
    "--format", "-f",
    "formats",
    type=click.Choice(
        ["source", "json", "ghidra", "python", "csharp"], case_sensitive=False
    ),
    multiple=True,
    default=["source", "json"],
    help=(
        "Output formats. 'python' and 'csharp' add a reconstruction of the "
        "recovered C in that language, which costs a second LLM pass over the "
        "whole binary. 'ghidra' writes no file and is kept only so older "
        "command lines still run: names and types reach the program database "
        "during the analysis, whatever is asked for here."
    ),
)
@click.option(
    "--resume",
    is_flag=True,
    help=(
        "Accepted for compatibility: reusing the saved results of an earlier "
        "run over the same binary is the default. Use --fresh to opt out."
    ),
)
@click.option(
    "--fresh",
    is_flag=True,
    help=(
        "Ignore the saved state in the output directory and analyze every "
        "function again, e.g. after changing model, prompt or Ghidra version."
    ),
)
@click.option("--ghidra-dir", default=None, help="Ghidra installation directory.")
@click.option(
    "--provider", "-p",
    type=click.Choice([p.value for p in LLMProvider], case_sensitive=False),
    default=None,
    help="LLM provider (anthropic, openai, zai, or custom).",
)
@click.option("--model", "-m", default=None, help="Override the LLM model name.")
@click.option(
    "--draft-model",
    default=None,
    help=(
        "Draft the easy functions with this model first, then re-analyze the "
        "weak answers with --model. Same provider and endpoint for both."
    ),
)
@click.option(
    "--refine-below",
    type=click.IntRange(0, 100),
    default=None,
    help=(
        "Confidence under which a drafted function is re-analyzed by --model "
        f"(default {DEFAULT_REFINE_BELOW}). Only used with --draft-model."
    ),
)
@click.option(
    "--stage",
    type=click.Choice([s.value for s in RunStage], case_sensitive=False),
    default=RunStage.FULL.value,
    help=(
        "full (default) drafts and finishes in one run. draft reads every "
        "function with --draft-model and stops. finish re-reads what a saved "
        "draft left failed, under --refine-below, or never analyzed, then "
        "exports."
    ),
)
@click.option("--base-url", default=None, help="Custom OpenAI-compatible endpoint URL.")
@click.option("--max-prompt-chars", type=int, default=None, help="Override prompt size limit.")
@click.option(
    "--max-chunk-functions",
    type=int,
    default=None,
    help=(
        "How many functions go into one batch call, on any provider. Lower it "
        "when a hosted endpoint rate-limits, or when long batches come back "
        "missing functions."
    ),
)
@click.option(
    "--max-output-tokens",
    type=int,
    default=None,
    help=(
        "Token budget for one request's answer, on any provider. Leave it out "
        "to use the model's own figure."
    ),
)
@click.option(
    "--concurrency",
    type=click.IntRange(1, 32),
    default=None,
    help=(
        "Chunk calls in flight at once. The work queue is ordered bottom-up "
        "and functions at the same depth do not depend on each other, so the "
        "calls need not wait for one another. Defaults to 4 on a hosted API "
        "and 1 on a local endpoint, which is already busy."
    ),
)
@click.option(
    "--obfuscation-threshold",
    type=click.FloatRange(0.0, 1.0),
    default=None,
    metavar="SHARE",
    help=(
        "Share of functions that must look obfuscated before the agentic "
        "deobfuscation loop is used at all (default 0.05). The heuristics read "
        "structure, and a while(1) around a switch is both control-flow "
        "flattening and every hand-written state machine, so on a clean binary "
        "they fire on the C runtime. A protector is applied wholesale. Use 0 "
        "to act on every detection."
    ),
)
@click.option(
    "--deobfuscation-budget",
    type=click.IntRange(0, 86400),
    default=None,
    metavar="SECONDS",
    help=(
        "Wall clock one function's deobfuscation loop may spend before it "
        "answers with what it has (default 900). 0 lifts the bound."
    ),
)
@click.option(
    "--no-truncate",
    is_flag=True,
    help=(
        "Leave a function whose body is over the prompt budget unanalyzed, "
        "instead of sending it cut down with the fact marked. Keeps the hole "
        "rather than a reading taken from part of a body."
    ),
)
@click.option(
    "--skip-known-library",
    is_flag=True,
    help=(
        "Do not analyze functions the signature database already identifies. "
        "They are documented library code, already named by the symbol they "
        "matched, and the most expensive thing in a run to rediscover."
    ),
)
@click.option(
    "--transpile-only",
    default=None,
    metavar="ADDRESSES|FILE",
    help=(
        "Translate only these functions with --format python/csharp, instead "
        "of the whole binary. Takes a file of addresses or an inline list; "
        "anything after the address on a line is ignored, so a two-column "
        "listing from a call-graph tool pastes in unchanged. What they call "
        "comes with them unless --no-follow-callees."
    ),
)
@click.option(
    "--no-follow-callees",
    is_flag=True,
    help=(
        "Translate exactly the functions named by --transpile-only. The "
        "result will refer to functions it does not contain."
    ),
)
@click.pass_context
def analyze(
    ctx: click.Context,
    binary: str,
    headless: bool,
    output: str,
    formats: tuple[str, ...],
    resume: bool,
    fresh: bool,
    ghidra_dir: str | None,
    provider: str | None,
    model: str | None,
    draft_model: str | None,
    refine_below: int | None,
    stage: str,
    base_url: str | None,
    max_prompt_chars: int | None,
    max_chunk_functions: int | None,
    max_output_tokens: int | None,
    concurrency: int | None,
    obfuscation_threshold: float | None,
    deobfuscation_budget: int | None,
    no_truncate: bool,
    skip_known_library: bool,
    transpile_only: str | None,
    no_follow_callees: bool,
) -> None:
    """Analyze a binary with Kong's autonomous agent."""
    if not is_setup_complete():
        console.print("[yellow]Kong hasn't been set up yet.[/yellow]")
        console.print("Run [bold cyan]kong setup[/bold cyan] first.")
        raise SystemExit(1)

    if base_url:
        base_url = validate_base_url(base_url)
        if provider and provider not in ("custom", "zai"):
            console.print(
                "[red]--base-url can only be used with --provider custom "
                "or --provider zai[/red]"
            )
            raise SystemExit(1)

    transpile_addresses = (
        _parse_address_selection(transpile_only) if transpile_only else None
    )
    translating = {"python", "csharp"} & {f.lower() for f in formats}
    if transpile_addresses and not translating:
        console.print(
            "[red]--transpile-only selects what a translation covers, but no "
            "translation was asked for. Add --format python or "
            "--format csharp.[/red]"
        )
        raise SystemExit(1)
    if no_follow_callees and not transpile_addresses:
        console.print(
            "[yellow]--no-follow-callees only means something with "
            "--transpile-only; ignoring it.[/yellow]"
        )

    run_stage = RunStage(stage.lower())
    if run_stage is RunStage.FINISH and fresh:
        console.print(
            "[red]--stage finish reads the draft saved in the output "
            "directory, which --fresh throws away.[/red]"
        )
        raise SystemExit(1)
    if run_stage is RunStage.DRAFT and not draft_model:
        console.print(
            "[yellow]--stage draft without --draft-model: the first pass will "
            "run on --model, and the finishing pass re-reads its weak answers "
            "one function at a time.[/yellow]"
        )

    llm_provider = resolve_provider(provider, base_url=base_url)

    api_key = resolve_api_key(llm_provider)
    if llm_provider is LLMProvider.CUSTOM:
        custom_db = get_custom_config()
        base_url = base_url or custom_db.get("custom_base_url")
        model = model or custom_db.get("custom_model")
        if not model:
            console.print("[red]--model is required for custom provider[/red]")
            raise SystemExit(1)
        if not base_url:
            console.print("[red]--base-url is required for custom provider[/red]")
            raise SystemExit(1)
        if max_prompt_chars is None:
            max_prompt_chars = _int_or_none(custom_db.get("custom_max_prompt_chars"))
        if max_chunk_functions is None:
            max_chunk_functions = _int_or_none(custom_db.get("custom_max_chunk_functions"))
        if max_output_tokens is None:
            max_output_tokens = _int_or_none(custom_db.get("custom_max_output_tokens"))

    config = KongConfig(
        ghidra=GhidraConfig(install_dir=ghidra_dir),
        llm=LLMConfig(
            provider=llm_provider,
            model=model,
            draft_model=draft_model,
            refine_below=(
                DEFAULT_REFINE_BELOW if refine_below is None else refine_below
            ),
            api_key=api_key,
            base_url=base_url,
            max_prompt_chars=max_prompt_chars,
            max_chunk_functions=max_chunk_functions,
            max_output_tokens=max_output_tokens,
        ),
        output=OutputConfig(
            directory=Path(output),
            formats=list(formats),
            transpile_addresses=transpile_addresses,
            transpile_follow_callees=not no_follow_callees,
        ),
        analysis=AnalysisConfig(
            **(
                {"obfuscation_threshold": obfuscation_threshold}
                if obfuscation_threshold is not None
                else {}
            ),
            **(
                {"deobfuscation_time_budget": deobfuscation_budget or None}
                if deobfuscation_budget is not None
                else {}
            ),
            chunk_concurrency=concurrency,
            truncate_oversized=not no_truncate,
            skip_matched_signatures=skip_known_library,
        ),
        headless=headless,
        verbose=ctx.obj["verbose"],
        stage=run_stage,
        resume=not fresh,
    )

    # Checked before Ghidra spends a minute opening the binary: a finishing
    # pass with no draft to read is a mistyped --output far more often than it
    # is anything else.
    if run_stage is RunStage.FINISH and not state_path(config.output.directory).exists():
        console.print(
            f"[red]No saved analysis in {config.output.directory}.[/red]"
        )
        console.print("Run the draft over this binary first, into the same --output:")
        console.print("  [bold cyan]kong analyze <binary> --stage draft[/bold cyan]")
        raise SystemExit(1)

    from kong.llm.probe import probe_endpoint

    if not probe_endpoint(config.llm):
        console.print("[red]Could not connect to LLM endpoint.[/red]")
        if llm_provider is LLMProvider.CUSTOM:
            console.print(f"Ensure your server is running at {config.llm.base_url}")
        raise SystemExit(1)

    if llm_provider is LLMProvider.CUSTOM:
        console.print("[dim]Cost tracking disabled for custom provider (token counts still recorded)[/dim]")

    if run_stage is RunStage.FINISH:
        console.print(
            f"[cyan]Finishing pass:[/cyan] re-reading what the draft in "
            f"[bold]{config.output.directory}[/bold] left failed, under "
            f"{config.llm.refine_below}% confidence, or never analyzed."
        )
    elif draft_model:
        if draft_model == config.llm.model:
            console.print(
                "[yellow]--draft-model is the same model as --model; "
                "running a single pass.[/yellow]"
            )
        elif run_stage is RunStage.DRAFT:
            console.print(
                f"[cyan]Draft stage:[/cyan] every function goes to "
                f"[bold]{draft_model}[/bold] and nothing goes to "
                f"[bold]{config.llm.model}[/bold]. Run "
                f"[bold]--stage finish[/bold] over the same output directory "
                f"when you want the second pass."
            )
            _warn_if_model_missing(config.llm, draft_model)
        else:
            console.print(
                f"[cyan]Two passes:[/cyan] draft with [bold]{draft_model}[/bold], "
                f"then re-analyze with [bold]{config.llm.model}[/bold] whatever "
                f"comes back under {config.llm.refine_below}% confidence."
            )
            _warn_if_model_missing(config.llm, draft_model)

    binary_path = Path(binary).resolve()

    print_analyze_header(
        console,
        binary_path=str(binary_path),
        output_dir=str(config.output.directory),
        formats=config.output.formats,
    )

    if not config.ghidra.install_dir:
        console.print("[red]Ghidra is not installed or not found.[/red]")
        console.print(
            "\nInstall Ghidra and try again:\n"
            "  [bold]brew install ghidra[/bold]\n"
            "\nOr set [bold]GHIDRA_INSTALL_DIR[/bold] to your Ghidra installation path."
        )
        raise SystemExit(1)

    try:
        with console.status(
            "[bold green]Opening binary in Ghidra (this may take 30-60s on first run) ...",
        ):
            client = GhidraClient(
                binary_path=str(binary_path),
                install_dir=config.ghidra.install_dir,
            )
            client.open()
    except GhidraClientError as e:
        console.print(f"[red]Failed to open binary:[/red] {escape(str(e))}")
        raise SystemExit(1)

    info = client.get_binary_info()
    console.print(f"[green]Loaded.[/green] {info.arch} {info.format} ({info.compiler})")

    llm_client = create_llm_client(config.llm)

    def print_event(event: Event) -> None:
        style = {
            EventType.PHASE_START: "bold cyan",
            EventType.PHASE_COMPLETE: "bold green",
            EventType.FUNCTION_COMPLETE: "green",
            EventType.FUNCTION_SKIPPED: "dim",
            EventType.FUNCTION_ERROR: "red",
            EventType.RUN_ERROR: "bold red",
            EventType.RUN_COMPLETE: "bold green",
        }.get(event.type, "")
        if style:
            console.print(f"[{style}]{escape(event.message)}[/{style}]")
        else:
            console.print(escape(event.message))

    supervisor = Supervisor(client, config, llm_client=llm_client)

    def _report_interrupted() -> None:
        supervisor.checkpoint()
        console.print("\n[yellow]Interrupted.[/yellow]")
        console.print(
            f"[dim]{len(supervisor.results)} functions saved in "
            f"{state_path(config.output.directory)}. Run the same command "
            f"again to pick up where this stopped.[/dim]"
        )

    if headless:
        supervisor.on_event(print_event)
        try:
            supervisor.run()
        except KeyboardInterrupt:
            _report_interrupted()
        finally:
            _print_final_stats(supervisor, llm_client)
            client.close()
    else:
        app = KongApp(supervisor)
        try:
            app.run()
        except KeyboardInterrupt:
            _report_interrupted()
        finally:
            # The TUI runs the supervisor on a worker thread, so quitting the
            # interface leaves it mid-function: this checkpoint is what makes
            # the next run start from here rather than from nothing.
            supervisor.checkpoint()
            _print_final_stats(supervisor, llm_client)
            client.close()


@cli.command()
@click.option(
    "--base-url",
    default=None,
    help="Endpoint to query (defaults to the one saved by `kong setup`).",
)
def models(base_url: str | None) -> None:
    """List the models a custom endpoint serves, and its context window."""
    from kong.llm.endpoint import discover, suggest_limits

    saved = get_custom_config()
    base_url = base_url or saved.get("custom_base_url")
    if not base_url:
        console.print("[red]No endpoint configured.[/red]")
        console.print(
            "Pass [bold]--base-url[/bold] or run [bold cyan]kong setup[/bold cyan]."
        )
        raise SystemExit(1)

    base_url = validate_base_url(base_url)
    try:
        info = discover(base_url, api_key=saved.get("custom_api_key") or None)
    except Exception as e:
        console.print(f"[red]Could not reach {escape(base_url)}:[/red] {escape(str(e))}")
        raise SystemExit(1) from e

    if not info.models:
        console.print(f"[yellow]{escape(base_url)} reports no models.[/yellow]")
        raise SystemExit(1)

    table = Table(title=f"Models at {base_url}", box=None, header_style="bold")
    table.add_column("Model")
    table.add_column("Params", justify="right")
    table.add_column("Trained ctx", justify="right")
    for model in info.models:
        table.add_row(
            model.id,
            model.parameters_label,
            f"{model.train_context:,}" if model.train_context else "?",
        )
    console.print()
    console.print(table)
    console.print()

    if info.build_info:
        console.print(f"[dim]Server build: {escape(info.build_info)}[/dim]")
    if info.total_slots:
        console.print(f"[dim]Slots: {info.total_slots}[/dim]")

    context = info.effective_context
    if context is None:
        console.print(
            "[yellow]The endpoint does not report its context window.[/yellow] "
            "Set the limits by hand with --max-prompt-chars and friends."
        )
        return

    source = (
        "runtime context reported by the server"
        if info.context_tokens
        else "training context of the first model (the server did not report a "
        "runtime value, so this may be larger than what a request really gets)"
    )
    limits = suggest_limits(context)
    console.print(f"[bold]Context:[/bold] {context:,} tokens — {source}")
    console.print()
    console.print("[bold]Suggested limits:[/bold]")
    console.print(f"  --max-prompt-chars    {limits.max_prompt_chars}")
    console.print(f"  --max-chunk-functions {limits.max_chunk_functions}")
    console.print(f"  --max-output-tokens   {limits.max_output_tokens}")


@cli.command()
@click.argument("binary", type=click.Path(exists=True, dir_okay=False), required=False)
@click.option(
    "--tk",
    "use_tk",
    is_flag=True,
    help="Open the old desktop window instead of the browser interface.",
)
@click.option(
    "--port",
    type=click.IntRange(0, 65535),
    default=0,
    help="Port to serve the interface on. 0 picks a free one.",
)
@click.option(
    "--no-browser",
    is_flag=True,
    help="Print the URL instead of opening a browser (remote or headless box).",
)
@click.option(
    "--scale",
    type=click.FloatRange(0.5, 4.0),
    default=None,
    help=(
        "Size of the --tk window, for when Tk misreads the screen: 1 pins it "
        "to no scaling, 1.5 enlarges it by half. Also read from KONG_UI_SCALE."
    ),
)
def gui(
    binary: str | None,
    use_tk: bool,
    port: int,
    no_browser: bool,
    scale: float | None,
) -> None:
    """Open the interface: a page in your browser, or --tk for the window."""
    initial_binary = str(Path(binary).resolve()) if binary else ""

    if not use_tk:
        # Nothing but the standard library, so this is also the interface that
        # works on a Python build without Tk bindings.
        from kong.webui import launch as launch_web

        launch_web(
            initial_binary=initial_binary,
            port=port,
            open_browser=not no_browser,
        )
        return

    try:
        from kong.gui.app import UI_SCALE_ENV, launch
    except ImportError as e:  # Tk is optional in some Python builds
        console.print(f"[red]The desktop window is not available:[/red] "
                      f"{escape(str(e))}")
        console.print(
            "It needs customtkinter and the Tk bindings for your Python "
            "(Debian/Ubuntu: [bold]apt install python3-tk[/bold]). "
            "Drop [bold]--tk[/bold] for the browser interface, which needs "
            "neither."
        )
        raise SystemExit(1) from e

    if scale is not None:
        # launch() reads the environment, so the flag and the variable cannot
        # disagree about which one wins.
        os.environ[UI_SCALE_ENV] = str(scale)

    launch(initial_binary=initial_binary)


@cli.command()
@click.argument("binary", type=click.Path(exists=True, dir_okay=False))
@click.option("--ghidra-dir", default=None, help="Ghidra installation directory.")
def info(binary: str, ghidra_dir: str | None) -> None:
    """Show info about a binary."""
    
    ghidra_config = GhidraConfig(install_dir=ghidra_dir)
    if not ghidra_config.install_dir:
        console.print("[red]Ghidra is not installed or not found.[/red]")
        raise SystemExit(1)

    try:
        client = GhidraClient(
            binary_path=str(Path(binary).resolve()),
            install_dir=ghidra_config.install_dir,
        )
        client.open()
    except GhidraClientError as e:
        console.print(f"[red]Failed to open binary:[/red] {e}")
        raise SystemExit(1)

    bi = client.get_binary_info()
    functions = client.list_functions()

    console.print(f"[bold]Binary:[/bold] {bi.name}")
    console.print(f"[bold]Path:[/bold] {bi.path}")
    console.print(f"[bold]Arch:[/bold] {bi.arch}")
    console.print(f"[bold]Format:[/bold] {bi.format}")
    console.print(f"[bold]Endianness:[/bold] {bi.endianness}")
    console.print(f"[bold]Word Size:[/bold] {bi.word_size * 8}-bit")
    console.print(f"[bold]Compiler:[/bold] {bi.compiler}")
    console.print(f"[bold]Functions:[/bold] {len(functions)}")

    # Classification breakdown
    counts = Counter(f.classification.value for f in functions if f.classification)
    for cls, count in sorted(counts.items()):
        console.print(f"  {cls}: {count}")

    client.close()


@cli.command()
def setup() -> None:
    """Interactive setup wizard for Kong."""
    print_banner(console)
    console.print()
    console.print("[bold]Welcome to Kong setup![/bold]")
    console.print()

    from kong.llm.probe import probe_endpoint

    current_default = get_default_provider()
    current_enabled = get_enabled_providers()
    saved_custom = get_custom_config()

    def _current_marker(provider: LLMProvider) -> str:
        if provider == current_default:
            return " [green](current default)[/green]"
        if provider in current_enabled:
            return " [cyan](enabled)[/cyan]"
        return ""

    console.print("[bold]Step 1:[/bold] Which LLM providers would you like to use?")
    console.print()
    console.print(f"  [bold]1[/bold]) Anthropic (Claude){_current_marker(LLMProvider.ANTHROPIC)}")
    console.print(f"  [bold]2[/bold]) OpenAI (GPT-4o){_current_marker(LLMProvider.OPENAI)}")
    console.print(f"  [bold]3[/bold]) Custom endpoint (OpenAI-compatible){_current_marker(LLMProvider.CUSTOM)}")
    console.print("  [bold]4[/bold]) Anthropic + OpenAI")
    console.print(f"  [bold]5[/bold]) Z.ai (GLM){_current_marker(LLMProvider.ZAI)}")
    console.print()

    choice = Prompt.ask(
        "Choice", choices=["1", "2", "3", "4", "5"], console=console,
    )
    choice_int = int(choice)

    custom_config: dict[str, str] | None = None
    if choice_int == 1:
        enabled: list[LLMProvider] = [LLMProvider.ANTHROPIC]
    elif choice_int == 2:
        enabled = [LLMProvider.OPENAI]
    elif choice_int == 3:
        enabled = [LLMProvider.CUSTOM]
    elif choice_int == 5:
        enabled = [LLMProvider.ZAI]
    else:
        enabled = [LLMProvider.ANTHROPIC, LLMProvider.OPENAI]

    if LLMProvider.CUSTOM in enabled:
        saved_url = saved_custom.get("custom_base_url", "")
        saved_model = saved_custom.get("custom_model", "")
        saved_key = saved_custom.get("custom_api_key", "")
        # _DEFAULT_LIMITS is sized for a hosted model; a custom endpoint is
        # usually a local server, so fall back to something that fits one.
        from kong.llm.endpoint import suggest_limits

        local = suggest_limits(_ASSUMED_LOCAL_CONTEXT)
        saved_max_pc = saved_custom.get("custom_max_prompt_chars", str(local.max_prompt_chars))
        saved_max_cf = saved_custom.get("custom_max_chunk_functions", str(local.max_chunk_functions))
        saved_max_ot = saved_custom.get("custom_max_output_tokens", str(local.max_output_tokens))

        console.print()
        console.print("[bold]Step 2:[/bold] Configure custom endpoint")
        console.print()
        if not saved_url:
            console.print("  Examples:", style="dim")
            console.print("    http://localhost:8080/v1       (llama.cpp)", style="dim")
            console.print("    http://localhost:11434/v1      (Ollama)", style="dim")
            console.print("    http://localhost:8000/v1       (vLLM)", style="dim")
            console.print("    https://openrouter.ai/api/v1  (OpenRouter)", style="dim")
            console.print()
        custom_base_url = Prompt.ask("  Endpoint URL", default=saved_url or None, console=console)
        custom_base_url = validate_base_url(custom_base_url)

        # Ask the server what it serves, so the next four answers can be
        # offered instead of guessed.
        discovered = _discover_endpoint(custom_base_url, saved_key or None)
        if discovered is not None:
            if discovered.models:
                console.print()
                console.print("  Models available:", style="dim")
                for index, model in enumerate(discovered.models, start=1):
                    context = (
                        f", {model.train_context:,} ctx" if model.train_context else ""
                    )
                    console.print(
                        f"    {index}. {model.id} "
                        f"({model.parameters_label}{context})",
                        style="dim",
                    )
                console.print()
                saved_model = saved_model or discovered.models[0].id
            context = discovered.effective_context
            if context:
                from kong.llm.endpoint import suggest_limits

                suggested = suggest_limits(context)
                saved_max_pc = str(suggested.max_prompt_chars)
                saved_max_cf = str(suggested.max_chunk_functions)
                saved_max_ot = str(suggested.max_output_tokens)
                console.print(
                    f"  Context window: {context:,} tokens — "
                    f"limits below are sized for it.",
                    style="dim",
                )
                console.print()

        custom_model = Prompt.ask("  Model name", default=saved_model or None, console=console)
        custom_api_key = Prompt.ask("  API key (leave blank for none)", default=saved_key, console=console)
        custom_max_pc = Prompt.ask(
            "  Max prompt size (chars)",
            default=saved_max_pc,
            console=console,
        )
        custom_max_cf = Prompt.ask(
            "  Max functions per batch",
            default=saved_max_cf,
            console=console,
        )
        custom_max_ot = Prompt.ask(
            "  Max output tokens",
            default=saved_max_ot,
            console=console,
        )
        custom_config = {
            "custom_base_url": custom_base_url,
            "custom_model": custom_model,
            "custom_api_key": custom_api_key,
            "custom_max_prompt_chars": custom_max_pc,
            "custom_max_chunk_functions": custom_max_cf,
            "custom_max_output_tokens": custom_max_ot,
        }

        console.print()
        console.print("  Probing endpoint...")
        probe_cfg = LLMConfig(
            provider=LLMProvider.CUSTOM,
            base_url=custom_base_url,
            api_key=custom_api_key or None,
        )
        if probe_endpoint(probe_cfg):
            console.print("  [green]Connected successfully.[/green]")
        else:
            console.print("  [yellow]Could not connect (server may not be running). Config saved anyway.[/yellow]")

    console.print()
    console.print("[bold]Step 2:[/bold] Checking API keys..." if LLMProvider.CUSTOM not in enabled else "[bold]Step 3:[/bold] Checking API keys...")
    console.print()

    any_key_found = False
    for p in enabled:
        if p is LLMProvider.CUSTOM:
            any_key_found = True
            continue
        env_var = _ENV_VARS[p]
        if check_api_key(p):
            key = os.environ.get(env_var, "")
            masked = key[:7] + "..." + key[-4:] if len(key) > 11 else "***"
            console.print(f"  {_PROVIDER_LABELS[p]:25s} [green]Found[/green] ({masked})")
            any_key_found = True
        else:
            console.print(f"  {_PROVIDER_LABELS[p]:25s} [yellow]Not set[/yellow]")
            console.print(f"    Get your key at: [bold]{_KEY_URLS[p]}[/bold]")
            console.print(f"    [bold]export {env_var}={_KEY_EXAMPLES[p]}[/bold]")
        console.print()

    non_custom = [p for p in enabled if p is not LLMProvider.CUSTOM]
    if len(non_custom) > 1:
        console.print("[bold]Step 3:[/bold] Which provider should be the default?")
        console.print()
        for i, p in enumerate(non_custom, 1):
            console.print(f"  [bold]{i}[/bold]) {_PROVIDER_LABELS[p]}")
        console.print()
        default_choice = Prompt.ask(
            "Default",
            choices=[str(i) for i in range(1, len(non_custom) + 1)],
            console=console,
        )
        default_provider = non_custom[int(default_choice) - 1]
    else:
        default_provider = enabled[0]

    save_setup(enabled=enabled, default=default_provider, custom_config=custom_config)

    console.print()
    ghidra_config = GhidraConfig()
    console.print("[bold]Ghidra[/bold]")
    console.print()
    if ghidra_config.install_dir:
        console.print(f"  [green]Found:[/green] {ghidra_config.install_dir}")
    else:
        console.print("  [yellow]Not found.[/yellow]")
        console.print()
        console.print("  Install Ghidra:")
        console.print("    [bold]brew install ghidra[/bold]  (macOS)")
        console.print("    Or download from [bold]https://ghidra-sre.org[/bold]")
        console.print()
        console.print("  Then set [bold]GHIDRA_INSTALL_DIR[/bold] to the install path.")

    console.print()
    if any_key_found and ghidra_config.install_dir:
        console.print(
            f"[bold green]All set![/bold green] Default provider: "
            f"[bold]{_PROVIDER_LABELS[default_provider]}[/bold]"
        )
        console.print("Run [bold cyan]kong analyze <binary>[/bold cyan] to get started.")
    else:
        console.print(
            "[yellow]Setup saved, but some dependencies are missing. "
            "See above for instructions.[/yellow]"
        )


@cli.command(name="graph")
@click.argument("output_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--binary",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help=(
        "Read the edges from Ghidra instead of from decompiled.c. Exact, and "
        "what a fresh run would write, but it needs the binary and takes as "
        "long as loading the program."
    ),
)
@click.option("--ghidra-dir", default=None, help="Ghidra installation directory.")
def graph_cmd(output_dir: str, binary: str | None, ghidra_dir: str | None) -> None:
    """Put a call graph into a finished analysis, without re-analyzing it.

    The analysis is what a run paid the model for; the call graph is not — the
    edges come from Ghidra and never from the model. An export written before
    Kong saved the graph therefore needs only this, not the run again.

    Without --binary the edges are read out of the decompiled.c beside the
    document. That is free and needs nothing installed, but it sees only what
    the decompiler printed: a call through a function pointer or a vtable has
    no name in it and never becomes an edge. The document records which of the
    two it was given.
    """
    from kong.export.callgraph import backfill

    client = None
    if binary:
        install_dir = GhidraConfig(install_dir=ghidra_dir).install_dir
        if not install_dir:
            console.print("[red]Ghidra is not installed or not found.[/red]")
            raise SystemExit(1)
        try:
            client = GhidraClient(
                binary_path=str(Path(binary).resolve()), install_dir=install_dir
            )
            client.open()
        except GhidraClientError as e:
            console.print(f"[red]Failed to open binary:[/red] {e}")
            raise SystemExit(1)

    try:
        result = backfill(Path(output_dir), client)
    finally:
        if client is not None:
            client.close()

    if not result["ok"]:
        console.print(f"[red]{escape(result['message'])}[/red]")
        raise SystemExit(1)

    console.print(
        f"[green]{result['edges']} call edges[/green] from "
        f"[bold]{result['source']}[/bold] written to {escape(result['path'])}"
    )
    console.print(f"  [dim]it had {escape(result['before'])}[/dim]")
    if result["source"] != "ghidra":
        console.print(
            "  [dim]Indirect calls are not in it: pass --binary to read the "
            "edges from Ghidra instead.[/dim]"
        )


@cli.command(name="eval")
@click.argument("analysis_json", type=click.Path(exists=True, dir_okay=False))
@click.argument("source_file", type=click.Path(exists=True, dir_okay=False))
def eval_cmd(analysis_json: str, source_file: str) -> None:
    """Score a Kong analysis against ground truth source code."""
    scorecard = eval_score(
        analysis_path=Path(analysis_json),
        source_path=Path(source_file),
    )

    console.print(f"[bold]Binary:[/bold] {scorecard.binary}")
    console.print(f"[bold]Functions:[/bold] {scorecard.functions_analyzed} analyzed / {scorecard.total_functions} in source")
    console.print(f"[bold]Symbol Accuracy:[/bold] {scorecard.symbol_accuracy:.1%}")
    console.print(f"[bold]Type Accuracy:[/bold] {scorecard.type_accuracy:.1%}")
    console.print()

    console.print("[bold]Per-Function Scores:[/bold]")
    for pf in scorecard.per_function:
        pred = pf["predicted_name"]
        truth = pf["truth_name"]
        sym = pf["symbol_accuracy"]
        typ = pf["type_accuracy"]
        match_indicator = "[green]OK[/green]" if sym >= 0.8 else "[yellow]~~[/yellow]" if sym > 0 else "[red]NO[/red]"
        console.print(f"  {match_indicator} {pred:30s} -> {truth:20s}  sym={sym:.2f}  type={typ:.2f}")

    console.print()
    console.print(f"[bold]LLM Calls:[/bold] {scorecard.llm_calls}")
    console.print(f"[bold]Duration:[/bold] {scorecard.duration_seconds:.1f}s")
    console.print(f"[bold]Cost:[/bold] ${scorecard.cost_usd:.4f}")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
