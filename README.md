<div align="left">


# Kong: The Agentic Reverse Engineer

![PyPI - Version](https://img.shields.io/pypi/v/kong-re)
![X (formerly Twitter) URL](https://img.shields.io/twitter/url?url=https%3A%2F%2Fx.com%2F0xamruth)


<img src="./assets/kong-logo.png" alt="Kong: World's first AI reverse engineer" width="50%">

**LLM orchestration for reverse engineering binaries** <br />

</div>

## What is Kong?

Reverse engineering a stripped binary is rarely hard. It is long: a few hundred
functions is hours of recognizing standard library code, inferring types from
usage, and propagating names up the call graph.

Kong automates that mechanical layer. It takes a stripped — or fully
obfuscated — binary and runs the whole pipeline: triaging functions, building
call-graph context, recovering types and symbols through LLM-guided
decompilation, and writing the results back into Ghidra's program database. The
output is a binary where `FUN_00401a30` is now `parse_http_header`, with
recovered structs, parameter names and calling conventions.

Pointing an LLM at raw decompiler output and asking "what does this do?" gives
mediocre answers: the model has no calling context, no cross-references, no
picture of the binary around the function. Kong builds that context from
Ghidra's own analysis first, then orchestrates the work in dependency order so
each function is read with its callees already named.

## In Action

<img src="./assets/github-banner.png" alt="Kong: World's first AI reverse engineer" width="100%">

<img src="./assets/kong-demo.gif" alt="Kong: World's first AI reverse engineer" width="100%">

## Features

- **Fully autonomous pipeline** — one command runs triage, analysis, cleanup, synthesis and export.
- **In-process Ghidra** — PyGhidra and JPype, no server, no RPC, direct access to the program database.
- **Call-graph-ordered analysis** — leaves first, so callers are read with their callees already named.
- **Rich context windows** — decompilation plus cross-references, strings, caller and callee signatures.
- **Signature matching** — known libc and crypto functions are identified by pattern, before any LLM call.
- **Syntactic normalization** — modulo recovery, negative literals, dead assignments removed before the prompt.
- **Agentic deobfuscation** — control-flow flattening, bogus control flow, instruction substitution, string encryption, VM protection.
- **Semantic synthesis** — one pass unifies naming across the binary and builds structs from field access patterns.
- **Coherence review** — reads the finished results against each other and pays a model only to arbitrate the contradictions it found.
- **Two-model passes** — a fast model drafts, a stronger one re-reads only what came back wrong or unsure; `--stage` splits the two so the expensive half is a decision, not a consequence.
- **Browser interface** — `kong gui` serves a page instead of a Tk window, and says when a request is out with the model and for how long.
- **Editable token budget** — what one request may spend, on every provider, in the interface and on the command line.
- **Multi-provider** — Anthropic, OpenAI, Z.ai, and any OpenAI-compatible endpoint, with per-model cost tracking.
- **Eval framework** — scores output against ground-truth source: symbol accuracy (word-based Jaccard) and type accuracy.

## Supported Architectures

Kong works with most Ghidra-decompilable binaries.

| | C | C++ | Go | Rust |
|---|---|---|---|---|
| x86 | High | High | Medium | Medium |
| x86-64 | High | High | Medium | Medium |
| ARM (32-bit) | High | High | Medium | Low |
| AArch64 | High | High | Medium | Low |
| MIPS | Medium | Medium | Low | Low |
| PowerPC | Medium | Medium | Low | Low |

**High**: names, types and structure come back reliably. **Medium**:
decompilation is usable but noisy, expect partial recovery. **Low**: significant
gaps, results stay incomplete.

Size cuts both ways: more functions means more cost and more time, and lower
confidence per function.

## Architecture

```
                    ┌──────────────────────┐
                    │       Triage         │
                    │  enumerate, classify,│
                    │  build call graph,   │
                    │  match signatures    │
                    └──────────┬───────────┘
                               │
                               ▼
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
     ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
     │   Analyze    │ │   Analyze    │ │     ...      │
     │  (leaf fns)  │ │ (next tier)  │ │              │
     └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
            │                │                │
            └────────┬───────┴────────────────┘
                     │
                     ▼
            ┌──────────────────────┐
            │      Cleanup         │
            │  normalize, dedupe   │
            └──────────┬───────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │     Synthesis        │
            │  unify names, build  │
            │  structs, deobfuscate│
            └──────────┬───────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │       Export         │
            │  decompiled.c +      │
            │  analysis.json       │
            └──────────────────────┘
```

**Triage** enumerates every function, classifies it by size, builds the call
graph, detects the source language and matches known library signatures.
Matched functions skip LLM analysis entirely.

**Analysis** works bottom-up from the call graph. Each function is sent with its
decompilation, cross-references, strings and the signatures of callees already
analyzed; obfuscated ones go through an agentic deobfuscation pass with
symbolic tool access first. Results are written back to Ghidra immediately, so
callers see the new names.

**Cleanup** unifies the struct types proposed during analysis and retries the
signatures that failed to apply.

**Synthesis** takes a global view in a single call: naming conventions unified,
structs synthesized from field access patterns, inconsistent names refined.

**Export** writes `decompiled.c` and `analysis.json`. The writeback into Ghidra
is not part of it — names and types land in the program database as each
function is analyzed.

**Coherence review** sits outside the five phases: it is asked for by hand once
the pipeline has finished. See [Cross-checking the
results](#cross-checking-the-results).

## Install

Kong is managed with [uv](https://docs.astral.sh/uv/getting-started/installation/).

You also need **Ghidra** ([install
guide](https://ghidra-sre.org/InstallationGuide.html)), **JDK 21+**
([Adoptium](https://adoptium.net/)) and at least one API key —
[Anthropic](https://console.anthropic.com/settings/keys),
[OpenAI](https://platform.openai.com/api-keys) or
[Z.ai](https://z.ai/manage-apikey/apikey-list). Ghidra and the JDK are
auto-detected, and uv fetches a Python 3.11+ for you if you have none.

```bash
# As a tool, available everywhere
uv tool install kong-re

# Or from a clone
git clone https://github.com/amruth-sn/kong.git
cd kong
uv sync
```

From a clone, every `kong` below is `uv run kong`.

## Quick Start

```bash
export ANTHROPIC_API_KEY="sk-ant-..."   # PowerShell: $env:ANTHROPIC_API_KEY = "sk-ant-..."

kong setup                  # pick providers and a default, once
kong gui                    # the interface, in your browser
kong analyze ./binary       # or straight from the command line
```

`kong setup` writes your choices to `~/.config/kong/config.db`. Ghidra opens the
binary in-process on the first run, which takes 30-60 seconds before anything
else happens.

## The interface

`kong gui` serves the interface on 127.0.0.1 and opens a browser at it:

```bash
kong gui                             # a free port, browser opened for you
kong gui ./binary                    # pre-fill the target
kong gui --port 8765 --no-browser    # print the URL instead (remote box, over an SSH tunnel)
kong gui --tk                        # the old desktop window
```

The URL carries a token minted for that process, and everything under `/api`
needs it: another page open in the same browser cannot start a run, read the
paths you browse, or spend your key. The terminal has to stay open. Closing the
tab leaves the run going; **Quit** stops it, checkpoints, and releases Ghidra.

Configuration is on the left, the run on the right: progress, live log,
functions as they come back, contradictions found by the coherence pass, and
the counters — cost, tokens, elapsed, and **Waiting on model**.

**Waiting on the model.** A chunk of forty functions can be out with the model
for minutes with nothing else moving, which looks exactly like a hung program.
The banner under the buttons names what is in flight — model, kind of call,
prompt size, budget given — and counts the seconds; the badge in the title bar
says the same when the page is scrolled. Idle, it reads *Model idle* with how
long the last answer took.

**Token budget.** What one request may spend: *Output tokens per request* caps a
single answer, batch calls included; *Prompt chars* caps what is sent; and
*Functions per batch* how much is asked for at once. Empty means the model's own
figure. All three apply to every provider.

### The desktop window

`kong gui --tk` opens the original customtkinter window. It needs the Tk
bindings, packaged separately on Debian and Ubuntu (`apt install python3-tk`);
the browser interface needs neither those nor customtkinter.

It picks a font the system actually has — `Ubuntu`, `Cantarell`, `Noto Sans`,
`DejaVu Sans` — rather than the `Roboto` customtkinter asks for and that Tk
silently replaces with something else. What it cannot fix is a desktop whose
DPI Tk reads wrong, fractional scaling under Wayland especially:

```bash
kong gui --tk --scale 1        # no scaling at all
kong gui --tk --scale 1.5      # half again as large
KONG_UI_SCALE=1.25 kong gui --tk
```

The flag turns Tk's own DPI guess off, so the number given is the one applied.

## Usage

```bash
# Analyze with your configured default provider
kong analyze ./binary

# Pick a provider and a model
kong analyze ./binary --provider openai --model gpt-4o-mini

# Draft with a fast model, re-read the weak answers with a stronger one
kong analyze ./binary --draft-model claude-haiku-4-5 --model claude-opus-5

# Run only the draft, then only the pass that finishes it
kong analyze ./binary --draft-model claude-haiku-4-5 --stage draft
kong analyze ./binary --draft-model claude-haiku-4-5 --stage finish

# Binary metadata, without analyzing
kong info ./binary

# Ask a custom endpoint what it serves, and how to size Kong for it
kong models --base-url http://127.0.0.1:8080/v1

# Score an analysis against ground-truth source
kong eval ./analysis.json ./source.c
```

## Providers

### Z.ai (GLM)

A provider in its own right, not a hand-configured endpoint:

```bash
export ZAI_API_KEY="..."
kong analyze ./binary --provider zai                      # glm-5.3 by default
kong analyze ./binary --provider zai --model glm-5.3-flash

# The two flagship models pair naturally for a two-pass run
kong analyze ./binary --provider zai --draft-model glm-5.3-flash --model glm-5.3

# A Coding Plan key is served from a different host
kong analyze ./binary --provider zai --base-url https://api.z.ai/api/coding/paas/v4
```

GLM-5.3 carries a 1M token context, so Kong chunks up to 900k characters of
decompilation per call, and its published rates are in the cost table — the run
report gives a real figure rather than an estimate. `glm-5.3-flash` has no
published rate and is reported at default rates, marked as such. Z.ai documents
no model-listing endpoint, so `kong models` may come back empty against it;
analysis works regardless.

### Local and OpenAI-compatible endpoints

`kong models` asks an endpoint what it serves and, on llama.cpp, the context it
was started with:

```
 Model                             Params  Trained ctx
 qwen2.5-coder-7b-instruct-q5_k_m    7.6B       32,768

Context: 16,384 tokens — runtime context reported by the server

Suggested limits:
  --max-prompt-chars    41208
  --max-chunk-functions 7
  --max-output-tokens   2048
```

The runtime figure is the one that matters: a server started with `-c 32768
-np 4` gives each request 8192 tokens, not 32768, and the training context says
nothing about either. `kong setup` runs the same query and offers these values
as defaults; the interface has a **Detect** button beside the base URL.

### API keys

A key can come from the environment, or from the interface: type it in the **API
key** field, press **Save**, and it is kept for that provider — masked, one per
provider, cleared by saving an empty field. Saved keys work for `kong analyze`
too.

When several sources have one, the order is: what you typed, then the
environment variable, then the saved key. An exported variable therefore stays a
local override for that shell.

Saved keys land in `~/.config/kong/config.db` in clear text, readable by
anything running as you. It is a convenience store, not a secret manager: on a
shared or backed-up machine, prefer the environment variable.

### Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | At least one | Anthropic API key (Claude) |
| `OPENAI_API_KEY` | At least one | OpenAI API key (GPT-4o) |
| `ZAI_API_KEY` | At least one | Z.ai API key (GLM) |
| `GHIDRA_INSTALL_DIR` | No | Ghidra installation (auto-detected) |
| `JAVA_HOME` | No | JDK (auto-detected) |
| `KONG_CONFIG_DIR` | No | Config directory (default `~/.config/kong`) |
| `KONG_UI_SCALE` | No | Scaling for `--tk`, when Tk misreads the screen |

## Running an analysis

### Output

Written to `./kong_output_{binary_name}/` unless `--output` says otherwise:

```
├── analysis.json         # All recovered function names, types, parameters
├── decompiled.c          # Recovered C, grouped by classification
├── analysis_state.json   # Checkpoint, read back on the next run
├── coherence.json        # Contradictions found by the coherence review
└── events.log            # Pipeline execution trace
```

### When functions fail

A run rarely names every function. `analysis.json` lists what failed under
`failures` with a reason per address, and `events.log` holds the full trace —
rewritten on each run, so keep a copy before re-running into the same directory.
`kong -v analyze ...` adds DEBUG records to it.

| Line in `events.log` | What happened |
|---|---|
| `decompilation exceeds the prompt budget` | Larger than `--max-prompt-chars`; it never reached the model. Raise the budget or use a larger context. |
| `Chunk N/M failed: HTTP ...` | The call for a whole chunk failed, so every function in it fails at once. This is why failures arrive in bursts. |
| `Chunk N/M: no response for 0x...` | The model answered, but not for those addresses. Common with small local models; lower `--max-chunk-functions`. |
| `Failed to parse batch LLM response` | The reply was not valid JSON even after repair, so the chunk is lost. The line carries the start of the raw reply. |
| `spent its N-token budget without answering` | The model reasoned until the output budget ran out and returned nothing. Kong retries once at double the budget; the line appears either way. |

### Batch size and token budget

Three knobs decide what one request costs. All are optional, and each provider
keeps its own:

| Flag | Interface field | What it caps |
|---|---|---|
| `--max-chunk-functions` | Functions per batch | How many functions go into one call |
| `--max-output-tokens` | Output tokens per request | What one answer may spend, batch calls included |
| `--max-prompt-chars` | Prompt chars | How much decompilation is sent |

Leave them out and each model's own figures are used — and those are not only
about context windows. A model that reasons before it answers charges the
reasoning to the same completion budget as the answer, so its batch has to
leave room for both: `glm-5.3` sends 40 functions against a 32k output budget
where a Claude model sends 120 against 16k. Too large for the budget and the
call comes back empty, paid for in full, with the whole chunk marked failed.

```bash
kong analyze ./binary --model claude-opus-5 --max-chunk-functions 25
```

Lower the batch when `events.log` shows `no response for 0x...` lines — a model
losing track of a long batch — or when a chunk fails with an HTTP 429.

### Two models in one run

`--draft-model` splits the analysis in two: a fast model reads everything cheap
to get right, and `--model` re-reads what the draft got wrong.

```bash
kong analyze ./binary \
  --base-url http://127.0.0.1:8080/v1 \
  --draft-model qwen2.5-coder-7b \
  --model qwen2.5-coder-32b \
  --refine-below 80
```

A function goes to the second pass when the draft errored or returned no name,
when the name says nothing (`FUN_`, `sub_`, `helper`, `handler`), or when its
self-reported confidence is under `--refine-below` (80 by default). The score
alone is a weak filter — it is not calibrated, and a small model is
*confidently* wrong more often than a large one — which is why the objective
signals sit beside it. Anything triage classified `LARGE`, and anything the
obfuscation detector flags, skips the draft entirely.

The second pass sends one function per call with the full context and re-reads
the decompilation first, so it sees what the draft already wrote into the
program. Draft names below the threshold are kept out of the "already
identified" preamble of later prompts: a wrong callee name spreads further than
the function it was invented for.

Both models share one provider and endpoint, and the budget flags cap both
passes — size them for the smaller model. The run report names how many
functions each model answered for, and the token table breaks down per model.

### Drafting and finishing separately

Back to back is what you want on a binary you can leave alone. On a large one
they are two kinds of spend: the draft is long, cheap and unattended, the
finishing pass short, expensive and worth a look first.

```bash
# Read every function with the fast model, then stop.
kong analyze ./binary --draft-model claude-haiku-4-5 --model claude-opus-5 --stage draft

# Later, over the same output directory: re-read what the draft left behind.
kong analyze ./binary --draft-model claude-haiku-4-5 --model claude-opus-5 --stage finish
```

A draft run sends nothing to `--model`: the functions a full run would hand
straight to it are drafted or held back, so the first stage is billed at the
draft model's rate alone. It also skips semantic synthesis, which would only be
unifying names the finishing pass is about to change. Cleanup and export still
run, so `analysis.json` and `decompiled.c` describe the draft.

The finishing pass takes everything that failed, came back under
`--refine-below`, was named something that says nothing, or was never analyzed —
including the obfuscated functions the draft held back and anything a stopped
run never reached. Each goes one per call with full context, then cleanup,
synthesis and export run again. Unlike the automatic second pass it does not
care which model wrote the answer it is redoing: asked for by hand, a weak
answer from the main model is re-read too.

`--stage finish` reads `analysis_state.json` from the output directory, so it
needs the same `--output` the draft used, and refuses to run with `--fresh`.

In the interface this is the **Draft only** checkbox and the **Finish pass**
button, with an `N to finish` count once a draft is done. Like the coherence
pass it writes to Ghidra, so pause a running analysis first — one writer at a
time.

### Cross-checking the results

Every function is named on its own, in a chunk that knows nothing about what the
other chunks answered. That is what makes the pipeline cheap, and what lets it
contradict itself. **Check coherence** reads the finished analysis back against
itself and resolves what does not hold together.

It is manual on purpose: it rewrites names and signatures and costs LLM calls.
Pause a running analysis first — both write to Ghidra, and only one may.

Detection is free and runs over the whole binary, no model involved:

| Contradiction | What it means |
|---|---|
| `duplicate_name` | Two functions given the same name. One is wrong, and the export cannot keep both. |
| `argument_count` | The signature declares *n* parameters and every call site passes *m*. |
| `return_value` | Declared `void`, and a caller uses the value it returns. |
| `known_signature` | The name claims a known library or crypto function whose real signature has a different arity or return. |
| `signature_name` | The signature declares a different name than the one recovered. |
| `unsupported_confidence` | A high confidence score on an explanation that says it could not tell what the code does. |

Only the contradictions found are sent to the model, with the decompilation of
the functions involved. It answers per conflict and may answer `no_change`: the
decompiler is wrong often enough that any of these can fire on an analysis that
is right, and a false alarm kept beats a fix invented. Real changes are written
into Ghidra, into the results and into `analysis_state.json`, so a later run
resumes from the corrected names.

The **Coherence** tab lists every contradiction next to what was done about it,
and `coherence.json` holds the same on disk. One pass arbitrates at most 50; if
there are more, the report says how many were left and running it again takes
the next batch. Running it twice helps in one other case too: while two
functions answer to the same name, no call site can say which it meant, so the
call-site checks stay quiet until the duplicate is resolved.

### Stopping and picking up again

Closing Kong does not throw the run away. Analysis checkpoints to
`analysis_state.json` every 25 functions, at each phase boundary, and on the way
out — Ctrl-C, `q` in the TUI, the interface closed, or a crash. Re-running the
same command resumes:

```bash
kong analyze ./binary       # stopped after 200 of 400 functions
kong analyze ./binary       # picks up at 201
```

Named functions are restored and written back to Ghidra without a second LLM
call. Everything else, failures included, is analyzed again — so a run cut short
by a rate limit, a crash or an impatient Ctrl-C costs only what it had left to
do.

The state file records which binary it describes, by SHA-256, which is what
keeps a second binary from inheriting the first one's names in a reused output
directory. A rebuilt binary at the same path counts as a different one: its
addresses have moved, and restoring old names against them would be silently
wrong.

A draft name that was never re-read is restored like any other — its callers
benefit from it — but goes back into the second pass rather than being promoted
to a final answer: the state file records which model answered for each
function.

Resuming is the default. `--fresh` starts over, after a model change, a prompt
edit or a Ghidra upgrade (`--resume` is still accepted and does nothing):

```bash
kong analyze ./binary --fresh
```

### Reconstruction in another language

`--format python` or `--format csharp` adds a translation of the recovered C, as
`decompiled.py` or `Decompiled.cs`:

```bash
kong analyze ./binary --format json --format python
```

This is a **reading aid, not a port**. Raw memory access, pointer casts and
calling-convention artifacts have no faithful equivalent in a managed language,
so any function that loses behaviour is marked in place with a `NOT FAITHFUL`
comment naming what could not be expressed, and the header counts how many
survived intact. Nothing in the output is expected to run.

It costs a second LLM pass over the whole binary — budget roughly the analysis
again — and honours the same prompt budget, so it works against a local
endpoint.

## Benchmarks

Kong autonomously reconstructed the full [XZ
backdoor](https://en.wikipedia.org/wiki/XZ_Utils_backdoor) (CVE-2024-3094) kill
chain from a stripped `liblzma.so.5.4.1` — all five core implant functions at
90-95% confidence, in 15 minutes, for $6.63.

See **[BENCHMARKS.md](BENCHMARKS.md)** for the case study and reproduction
instructions.

## Project Layout

```
kong/
├── __main__.py     # CLI (click)
├── config.py       # KongConfig, LLMProvider, LLMConfig
├── db.py           # SQLite config store (~/.config/kong/)
├── agent/          # Supervisor, triage, analyzer, deobfuscator, coherence,
│                   # work queue, prompts, events, run log
├── ghidra/         # In-process client (PyGhidra/JPype), types, auto-detection
├── llm/            # Anthropic and OpenAI clients, endpoint probing, limits,
│                   # in-flight request tracking, usage and pricing
├── normalizer/     # Decompiler output normalization
├── symbolic/       # z3-backed simplification, dead code, state machines
├── synthesis/      # Global name unification + struct synthesis
├── export/         # analysis.json, decompiled.c, transpilation
├── state/          # analysis_state.json checkpointing
├── evals/          # Ground-truth scoring harness
├── webui/          # Browser interface (stdlib HTTP server + static page)
├── gui/            # customtkinter window and its toolkit-free controller
├── tui/            # Textual TUI
├── signatures/     # Known libc and crypto signatures (JSON)
└── patterns/       # Obfuscation pattern notes fed to the deobfuscator
```

## Stack

Python 3.11+ managed with [uv](https://github.com/astral-sh/uv) ·
[Ghidra](https://ghidra-sre.org/) via
[PyGhidra](https://github.com/NationalSecurityAgency/ghidra/tree/master/Ghidra/Features/PyGhidra)
(in-process, JPype) ·
[Anthropic](https://github.com/anthropics/anthropic-sdk-python) and
[OpenAI](https://github.com/openai/openai-python) SDKs ·
[z3-solver](https://github.com/Z3Prover/z3) ·
[Click](https://click.palletsprojects.com/) ·
[Rich](https://rich.readthedocs.io/) ·
[Textual](https://textual.textualize.io/) ·
[customtkinter](https://github.com/TomSchimansky/CustomTkinter) ·
[hatchling](https://hatch.pypa.io/) · [pytest](https://pytest.org/)

## License

[APACHE](LICENSE)

Kong is licensed under the Apache License 2.0. Kong is a free and open source project.

This license is compatible with the Ghidra license, and allows for commercial use.

## Contributing

Issues and feature requests are welcome via [GitHub Issues](https://github.com/amruth-sn/kong/issues).

Also, don't hesitate to reach out to me on [X](https://x.com/0xamruth) or [LinkedIn](https://www.linkedin.com/in/amruthn/)!

## Acknowledgments

- [Ghidra](https://ghidra-sre.org/)
- [PyGhidra](https://github.com/NationalSecurityAgency/ghidra/tree/master/Ghidra/Features/PyGhidra)
- [JPype](https://github.com/jpype-project/jpype)
- [Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python)
- [OpenAI SDK](https://github.com/openai/openai-python)
- [Z3](https://github.com/Z3Prover/z3)
- [Textual](https://textual.textualize.io/)
- [Rich](https://rich.readthedocs.io/)

A big shoutout to [KeygraphHQ](https://keygraph.io/)'s [Shannon](https://github.com/KeygraphHQ/shannon) project, which provided the inspiration for this project. My motivation was driven by replicating the same kind of pipeline that Shannon uses for its web-based pentesting tool, and adapting it for binary analysis and decompilation.

---

Fear the monkey.

---

<p align="center">
  <img src="./assets/kong-logo.png" width="40" alt="Kong"> <br />
  <b>Kong</b>: The world's first AI reverse engineer 
</p>
