<div align="left">


# Kong: The Agentic Reverse Engineer

![PyPI - Version](https://img.shields.io/pypi/v/kong-re)
![X (formerly Twitter) URL](https://img.shields.io/twitter/url?url=https%3A%2F%2Fx.com%2F0xamruth)


<img src="./assets/kong-logo.png" alt="Kong: World's first AI reverse engineer" width="50%">

**LLM orchestration for reverse engineering binaries** <br />

</div>

## What is Kong?
Most tasks follow a linear relationship: the more difficult a task, the longer it usually takes. Reverse engineering (and binary analysis) is a task in which the actual difficulty is somewhat trivial, but the time-to-execute can be on the order of hours (and days!), even for a binary with a couple hundred functions.   

Kong automates the mechanical layer, using an NSA-grade reverse engineering framework. Kong can take a fully obfuscated, stripped binary and run a full analysis pipeline: triaging functions, building call-graph context, recovering types and symbols through LLM-guided decompilation, and writing the results back into Ghidra's program database. The output is a binary where some `FUN_00401a30` is now `parse_http_header`, with recovered structs, parameter names, and calling conventions.

**Why this exists**

Stripped binaries lose all the context that makes code readable: function names, type information, variable names, struct layouts. Recovering that context is the bulk of the work in most RE tasks, and it's largely pattern matching: recognizing standard library functions, inferring types from usage, propagating names through call graphs.

LLMs are good at exactly this kind of pattern matching. But pointing an LLM at raw decompiler output and asking "what does this do?" gives you mediocre results. The model lacks calling context, cross-reference information, and the broader picture of how the binary is structured. In addition, most obfuscated binaries introduce extreme techniques in order to prevent reverse engineering.

Kong solves this by building rich context windows from Ghidra's program analysis (call graphs, cross-references, string references, data flow) before ever touching the LLM, then orchestrating the analysis in dependency order so each function benefits from its callees already being named. Additionally, Kong introduces its own, first-of-its-kind, agentic deobfuscation pipeline.

## In Action

<img src="./assets/github-banner.png" alt="Kong: World's first AI reverse engineer" width="100%">

<img src="./assets/kong-demo.gif" alt="Kong: World's first AI reverse engineer" width="100%">

## Features

- **Fully Autonomous Pipeline**: A single command runs the complete analysis. Triage, function analysis, cleanup, semantic synthesis, and export. No manual intervention required.
- **In-Process Ghidra Integration**: Runs Ghidra's analysis engine in-process via PyGhidra and JPype. No server, no RPC, no subprocess overhead. Direct access to the program database.
- **Call-Graph-Ordered Analysis**: Functions are analyzed bottom-up from the call graph. Leaf functions are named first, so callers benefit from already-resolved context in their decompilation.
- **Rich Context Windows**: Each LLM prompt includes the target function's decompilation plus cross-references, string references, caller/callee signatures, and neighboring data; not just raw decompiler output in isolation.
- **Semantic Synthesis**: A post-analysis pass that unifies naming conventions across the binary, synthesizes struct definitions from field access patterns, and resolves inconsistencies between independently analyzed functions.
- **Coherence Review**: A cross-check launched by hand from the window once the functions are decompiled. It reads the finished results against each other — two functions given one name, a signature that disagrees with every call site, a `void` whose value the caller uses, certainty in an explanation that admits it understood nothing — and pays a model only to arbitrate the contradictions it actually found.
- **Signature Matching**: Known standard library and cryptographic functions are identified by pattern before LLM analysis, skipping expensive inference for functions with known identities.
- **Syntactic Normalization**: Decompiler output is cleaned up (modulo recovery, negative literal reconstruction, dead assignment removal) before reaching the LLM, reducing noise and token waste. 
- **Agentic Deobfuscation**: Kong uses an agentic deobfuscation pipeline which can identify and remove obfuscation techniques (Control flow flattening, bogus control flow, instruction substitution, string encryption, VM protection, etc.) from the decompiler output.
- **Eval Framework**: Built-in evaluation harness that scores analysis output against ground-truth source code, measuring symbol accuracy (word-based Jaccard) and type accuracy (signature component scoring).
- **Two-Model Passes**: A fast model can draft the easy functions and a stronger one re-read only what it got wrong, was unsure about, or was too large to be handed to it in the first place. One endpoint, one run, one report.
- **Staged Passes**: The two passes also split apart. `--stage draft` reads every function with the cheap model and stops; the finishing pass — `--stage finish`, or the **Finish pass** button in the window — re-reads only what failed or came back under the threshold, once you have looked at what the draft produced. The expensive half becomes a decision rather than a consequence.
- **Multi-Provider LLM Support**: Works with Anthropic (Claude), OpenAI (GPT-4o) and Z.ai (GLM) out of the box, plus any OpenAI-compatible endpoint. An interactive setup wizard configures providers and smart routing auto-selects whichever has a valid key.
- **Cost-Tracking**: Tracks token usage and costs per model across providers, with provider-aware pricing.

## Supported Architectures

Kong works with most Ghidra-decompilable binaries (for now, more to come).

#### Confidence

| | C | C++ | Go | Rust |
|---|---|---|---|---|
| x86 | High | High | Medium | Medium |
| x86-64 | High | High | Medium | Medium |
| ARM (32-bit) | High | High | Medium | Low |
| AArch64 | High | High | Medium | Low |
| MIPS | Medium | Medium | Low | Low |
| PowerPC | Medium | Medium | Low | Low |

**High**: Kong reliably decompiles, deobfuscates, and recovers names, types, and structure.

**Medium**: Decompilation is usable but noisier. Expect partial recovery and lower confidence scores.

**Low**: Decompilation has significant gaps and results will stay incomplete, noisy, or unreadable.

**Note**: Binary size scales positively with function count, LLM cost, and time to completion. However, binary size also scales negatively with confidence, so keep this in mind when analyzing larger binaries.

## Architecture

Kong uses a five-phase pipeline orchestrated by a supervisor that coordinates triage, parallel analysis, and post-processing:

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
            │  analysis.json +     │
            │  Ghidra writeback    │
            └──────────────────────┘
```

### How it works

**Triage** enumerates all functions in the binary, classifies them by size (trivial / small / medium / large), builds the call graph, detects the source language, and runs signature matching against known standard library and crypto functions. Functions matched by signature are marked as resolved and skip LLM analysis entirely.

**Analysis** processes functions in bottom-up order from the call graph using a work queue. For each function, Kong builds a context window from Ghidra's program database — decompilation, cross-references, string references, and the signatures of already-analyzed callees — normalizes the decompiler output, and sends it to the LLM for name, type, and parameter recovery. If obfuscation is detected in a function's decompilation, Kong runs an agentic deobfuscation pass with symbolic tool access before producing the analysis. Results are written back to Ghidra immediately so downstream callers see updated names.

**Cleanup** unifies struct types from proposals accumulated during analysis and retries any function signatures that failed to apply during the analysis pass.

**Synthesis** takes a global view across all analyzed functions. A single LLM call reviews the most-connected functions, unifies naming conventions, synthesizes struct definitions from field access patterns, and refines names that look inconsistent in the broader context.

**Export** writes the final `analysis.json` and applies all recovered names, types, and signatures back to the Ghidra program database.

**Coherence review** sits outside those five phases: it is asked for by hand, from the window, once the pipeline has finished. See [Cross-checking the results](#cross-checking-the-results).

## Stack

- **Runtime**: Python 3.11+, managed with [uv](https://github.com/astral-sh/uv)
- **Binary analysis**: [Ghidra](https://ghidra-sre.org/) via [PyGhidra](https://github.com/NationalSecurityAgency/ghidra/tree/master/Ghidra/Features/PyGhidra) (in-process, JPype)
- **LLM**: [Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python) (Claude) and [OpenAI SDK](https://github.com/openai/openai-python) (GPT-4o)
- **Symbolic analysis**: [z3-solver](https://github.com/Z3Prover/z3)
- **CLI**: [Click](https://click.palletsprojects.com/)
- **TUI**: [Textual](https://textual.textualize.io/)
- **GUI**: [customtkinter](https://github.com/TomSchimansky/CustomTkinter)
- **Display**: [Rich](https://rich.readthedocs.io/)
- **Build**: [hatchling](https://hatch.pypa.io/)
- **Testing**: [pytest](https://pytest.org/)

## Setup

### Prerequisites

- **Python 3.11+** — ([python.org](https://www.python.org/downloads/) or your system package manager)
- **uv** — Python package manager ([Install uv](https://docs.astral.sh/uv/getting-started/installation/))
- **Ghidra** — The National Security Agency's reverse engineering framework ([Install Ghidra](https://ghidra-sre.org/InstallationGuide.html))
- **JDK 21+** — Required by Ghidra ([Adoptium](https://adoptium.net/))
- **LLM API key** — At least one of:
  - [Anthropic](https://console.anthropic.com/settings/keys) (Claude)
  - [OpenAI](https://platform.openai.com/api-keys) (GPT-4o)

### Quick Start

```bash
# 1. Install Kong
uv pip install kong-re

# 2. Set your API key(s)
export ANTHROPIC_API_KEY="sk-ant-..."
# and/or
export OPENAI_API_KEY="sk-..."
# and/or
export ZAI_API_KEY="..."

# 3. Run the setup wizard (first time only)
kong setup

# 4. Analyze a binary
kong analyze ./path/to/stripped_binary
```

The setup wizard lets you pick which LLM providers to use and sets a default. Kong auto-detects your Ghidra and JDK installations, loads the binary into an in-process Ghidra instance, and runs the full pipeline.

#### From source

```bash
git clone https://github.com/amruth-sn/kong.git
cd kong
uv sync
uv run kong setup
uv run kong analyze ./path/to/stripped_binary
```

### Z.ai (GLM)

Z.ai is a provider in its own right, not a hand-configured endpoint:

```bash
export ZAI_API_KEY="..."
kong analyze ./binary --provider zai                      # glm-5.3 by default
kong analyze ./binary --provider zai --model glm-5.3-flash
```

Requests go to `https://api.z.ai/api/paas/v4`, Z.ai's OpenAI-compatible
endpoint. A Coding Plan key is served from a different host, which `--base-url`
selects:

```bash
kong analyze ./binary --provider zai \
  --base-url https://api.z.ai/api/coding/paas/v4
```

GLM-5.3 carries a 1M token context, so Kong chunks for it the way it does for
the other million-token models: up to 900k characters of decompilation per
call. Published rates for `glm-5.3` ($1.40 in / $4.40 out per million tokens,
$0.26 cached input) are in the cost table, so the run report gives a real
figure rather than an estimate; `glm-5.3-flash` has no published rate in Kong
and is reported at default rates, marked as such.

The two flagship models pair naturally for a two-pass run:

```bash
kong analyze ./binary --provider zai \
  --draft-model glm-5.3-flash --model glm-5.3
```

Z.ai does not document a model-listing endpoint, so `kong models` may come back
empty against it — that is the listing, not the API, and analysis works
regardless.

In the GUI, Z.ai is one of the provider buttons: picking it fills the base URL
field with the standard endpoint (editable, for a Coding Plan key) and the model
field shows which model an empty box runs. The key still comes from
`ZAI_API_KEY`, read when Kong starts, and a missing one is reported in the
dialog rather than a minute later once Ghidra has opened the binary.

### Local endpoints

`kong models` queries an OpenAI-compatible endpoint for the models it serves
and, on llama.cpp, the context window it was started with:

```
 Model                             Params  Trained ctx
 qwen2.5-coder-7b-instruct-q5_k_m    7.6B       32,768

Server build: b4321
Slots: 1
Context: 16,384 tokens — runtime context reported by the server

Suggested limits:
  --max-prompt-chars    41208
  --max-chunk-functions 7
  --max-output-tokens   2048
```

The runtime figure is the one that matters: a server started with `-c 32768
-np 4` gives each request 8192 tokens, not 32768, and the model's training
context says nothing about either. `kong setup` runs the same query and offers
these values as defaults, and the GUI has a **Detect** button beside the base
URL. A server that reports nothing usable leaves the fields for you to fill.

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | At least one | Anthropic API key (Claude) |
| `OPENAI_API_KEY` | At least one | OpenAI API key (GPT-4o) |
| `ZAI_API_KEY` | At least one | Z.ai API key (GLM) |
| `GHIDRA_INSTALL_DIR` | No | Path to Ghidra installation (auto-detected if not set) |
| `JAVA_HOME` | No | Path to JDK (auto-detected if not set) |
| `KONG_CONFIG_DIR` | No | Override config directory (default: `~/.config/kong`) |

### The window on Linux

The interface picks a font the system actually has — `Ubuntu`, then
`Cantarell`, `Noto Sans`, `DejaVu Sans` — instead of the `Roboto`
customtkinter asks for everywhere and that a stock Ubuntu does not install; a
missing family is not an error in Tk, it is silently replaced, which is what
made the window look like a different program. The functions table follows the
same font and sizes its rows from the text rather than from a fixed 24 pixels,
and the window opens no larger than the screen it is on.

What none of that fixes is a desktop whose DPI Tk reads wrong — fractional
scaling under Wayland especially, where the result is either blurry or twice
the intended size. Override it:

```bash
kong gui --scale 1        # no scaling at all
kong gui --scale 1.5      # half again as large
KONG_UI_SCALE=1.25 kong gui
```

The flag turns Tk's own DPI guess off, so the number given is the one applied.

If the GUI does not start at all, the Tk bindings are separate from Python on
Debian and Ubuntu: `apt install python3-tk`.

### API keys

A key does not have to come from the environment. The GUI has an **API key**
field next to the provider buttons: type the key, press **Save**, and it is
kept for that provider. Each provider holds its own, the field is masked, and
saving an empty field forgets the key again. Saved keys are used by `kong
analyze` too, so a key entered once in the window works on the command line.

When several sources have a key, the order is: what you typed in the field (or
`--base-url`-style explicit configuration), then the environment variable, then
the saved key. An exported variable therefore stays a local override for that
shell rather than something a saved key silently replaces.

Saved keys land in the same local config database as the rest of `kong setup`
— `~/.config/kong/config.db`, or wherever `KONG_CONFIG_DIR` points — in clear
text, readable by anything running as you. It is a convenience store, not a
secret manager: on a shared or backed-up machine, prefer the environment
variable, and remember that deleting the key in the GUI is what removes it from
the file.

### Usage

```bash
# Run the setup wizard
kong setup

# Open the graphical interface
kong gui
kong gui ./binary          # pre-fill the target

# Analyze a stripped binary (uses your configured default provider)
kong analyze ./binary

# Analyze with a specific provider
kong analyze ./binary --provider openai

# Override the model
kong analyze ./binary --provider openai --model gpt-4o-mini

# Draft with a fast model, re-analyze the weak answers with a stronger one
kong analyze ./binary --draft-model claude-haiku-4-5 --model claude-opus-5

# Run only the draft, then only the pass that finishes it
kong analyze ./binary --draft-model claude-haiku-4-5 --stage draft
kong analyze ./binary --draft-model claude-haiku-4-5 --stage finish

# Show binary metadata without running analysis
kong info ./binary

# Ask a custom endpoint what it serves, and how to size Kong for it
kong models --base-url http://127.0.0.1:8080/v1

# Evaluate analysis output against ground-truth source
kong eval ./analysis.json ./source.c
```

### Output

Results are written to the output directory (default: `./kong_output_{binary_name}/`):

```
kong_output_{binary_name}/
├── analysis.json         # All recovered function names, types, parameters
├── decompiled.c          # Recovered C, grouped by classification
├── analysis_state.json   # Checkpoint, read back by --resume
├── coherence.json        # Contradictions found by the coherence review
└── events.log            # Pipeline execution trace
```

### When functions fail

A run rarely names every function. `analysis.json` lists what failed under
`failures`, with the reason per address, and `events.log` holds the full
chronological trace of the run — pipeline events plus the diagnostics the
analysis emits along the way. The file is rewritten on each run, so keep a
copy before re-running into the same output directory.

Failures come in four shapes, and the log names which one you hit:

| Line in `events.log` | What happened |
|---|---|
| `decompilation exceeds the prompt budget` | The function is larger than `--max-prompt-chars`. It never reached the model. Raise the budget or use a larger context. |
| `Chunk N/M failed: HTTP ...` | The API call for a whole chunk failed, so every function in it is marked failed at once. This is why failures arrive in bursts. |
| `Chunk N/M: no response for 0x...` | The model answered, but not for those addresses — it returned fewer entries than asked, or mangled an address. Common with small local models; lower `--max-chunk-functions`. |
| `Failed to parse batch LLM response` | The reply was not valid JSON even after repair, so the whole chunk is lost. The line carries the start of the raw reply. |
| `spent its N-token budget without answering` | The model reasoned until the output budget ran out and returned nothing. Kong retries the call once at double the budget; the line appears either way. |

`kong -v analyze ...` adds DEBUG records to the same file.

### Two models in one run

`--draft-model` splits the analysis in two. A fast model reads everything that
is cheap to get right, then `--model` re-reads what the draft got wrong:

```bash
kong analyze ./binary \
  --base-url http://127.0.0.1:8080/v1 \
  --draft-model qwen2.5-coder-7b \
  --model qwen2.5-coder-32b \
  --refine-below 80
```

A function goes to the second pass when the draft returned an error or no name,
when the name it returned says nothing (`FUN_`, `sub_`, `helper`, `handler`), or
when its self-reported confidence is under `--refine-below` (80 by default, the
same bar the run report calls high confidence). The score alone is a weak
filter — it is not calibrated, and a small model is *confidently* wrong more
often than a large one — which is why the objective signals sit next to it.

Two kinds of function skip the draft entirely, because a second pass on them is
a near-certainty anyway: anything triage classified `LARGE`, and anything the
obfuscation detector flags.

The second pass sends one function per call, with the full context — callers,
callees, cross-references, strings — rather than the batched chunks the draft
pass uses, and it re-reads the decompilation first, so it sees the names and
types the draft already wrote into the program. Draft names below the threshold
are also kept out of the "already identified" preamble of later prompts: a
wrong callee name spreads further than the function it was invented for.

Both models are served by the same provider and endpoint. Chunking follows each
model's own context window, and `--max-prompt-chars` and its siblings cap both
passes, so size them for the smaller of the two models. The run report ends with
a line naming how many functions each model answered for, and the token table
breaks down per model.

### How many functions per call

`--max-chunk-functions` is how many functions go into one batch call. It
applies to every provider, hosted ones included — the model table knows what a
context window holds, but not that your account is being rate limited today, or
that this model starts dropping functions halfway through a batch of a hundred:

```bash
kong analyze ./binary --model claude-opus-5 --max-chunk-functions 25
```

Leave it out and each model's own figure is used, and those figures are not
only about context windows. A model that reasons before it answers charges the
reasoning to the same completion budget as the answer, so its batch has to be
small enough to leave room for both: `glm-5.3` sends 40 functions against a 32k
output budget where a Claude model sends 120 against 16k. When a batch is too
large for that budget the call comes back empty — paid for in full — and the
whole chunk is marked failed at once, which is what `spent its N-token budget
without answering` in `events.log` means.

In the window the field is **Functions per batch**, next to the two limits that
only a local endpoint sets; it is blank on a hosted provider, meaning the
model's own figure, and each provider keeps whatever you last typed for it.
Lower it when `events.log` shows `no response for 0x...` lines, which is a model
losing track of a long batch, or when a chunk failure comes back as an HTTP 429.

### Drafting and finishing separately

The two passes above run back to back, which is what you want on a binary you
are willing to leave alone. On a large one they are two different kinds of
spend: the draft is long, cheap and unattended, and the finishing pass is short,
expensive and worth a look first. `--stage` separates them.

```bash
# Read every function with the fast model, then stop.
kong analyze ./binary \
  --draft-model claude-haiku-4-5 --model claude-opus-5 --stage draft

# Later, over the same output directory: re-read what the draft left behind.
kong analyze ./binary \
  --draft-model claude-haiku-4-5 --model claude-opus-5 --stage finish
```

A draft run sends nothing to `--model`. The functions a full run would hand
straight to it — anything triage classified `LARGE`, anything the obfuscation
detector flags — are drafted or held back instead, so the bill for the first
stage is the draft model's alone. It also skips semantic synthesis, which
unifies naming across the whole binary in one call and would only be answering
about names the finishing pass is about to change. Cleanup and export still run,
so `analysis.json` and `decompiled.c` describe the draft.

The finishing pass takes every function that failed, came back under
`--refine-below`, was named something that says nothing, or was never analyzed
at all — including the obfuscated ones the draft held back, and anything a
stopped run never reached. Each goes one per call with the full context, then
cleanup, synthesis and export run again over the finished results. Unlike the
automatic second pass, it does not care which model wrote the answer it is
redoing: asked for by hand, a weak answer from the main model is re-read too,
because the per-function path is a different question from the batched one even
on the same model.

`--stage finish` reads the draft from `analysis_state.json` in the output
directory, so it needs the same `--output` the draft used, and refuses to run
with `--fresh`, which would throw that file away.

In the window the same split is the **Draft only** checkbox and the **Finish
pass** button. The status line carries a `N to finish` count once a draft is
done, and the button is disabled while it reads zero. Like the coherence pass,
it writes names and signatures back into Ghidra, so pause the analysis first if
one is still running — only one writer at a time.

### Cross-checking the results

Every function is named and typed on its own, in a chunk that knows nothing
about what the other chunks answered. That is what makes the pipeline cheap,
and it is also what lets it contradict itself. The **Check coherence** button in
the window reads the finished analysis back against itself and resolves what
does not hold together.

It is manual on purpose: the pass rewrites names and signatures in the program
database and costs LLM calls, so it happens when the person reading the results
asks for it — typically once a run has finished, or on a run paused midway.
Pause first if the analysis is still going; both write to Ghidra, and only one
of them may be doing it.

Detection is free and runs over the whole binary, no model involved:

| Contradiction | What it means |
|---|---|
| `duplicate_name` | Two functions were given the same name. One of them is wrong, and the export cannot keep both. |
| `argument_count` | The recovered signature declares *n* parameters and every call site passes *m*. |
| `return_value` | The function is declared `void` and a caller uses the value it returns. |
| `known_signature` | The name claims a known library or crypto function whose real signature has a different arity or return. |
| `signature_name` | The signature declares a different function name than the one recovered. |
| `unsupported_confidence` | A high confidence score attached to an explanation that says it could not tell what the code does. |

Only the contradictions that turn up are sent to the model, one batch at a
time, with the decompilation of the functions involved. It answers per conflict
and may answer `no_change`: the decompiler is wrong often enough that any of
these checks can fire on an analysis that is right, and a false alarm kept is
better than a fix invented. Changes it does ask for — a rename, a signature, a
lowered confidence, a rewritten description — are written back into Ghidra, into
the results, and into `analysis_state.json`, so a later run resumes from the
corrected names.

The **Coherence** tab lists every contradiction next to what was done about it,
and the same thing is left on disk as `coherence.json`. A pass arbitrates at
most 50 contradictions; if a binary holds more, the report says how many were
left and running the check again takes the next batch.

Running it twice is worth it in one other case: while two functions answer to
the same name, no call site can say which of them it meant, so the call-site
checks stay quiet on both and only the duplicate is reported. Resolving the
duplicate makes those call sites legible, and the next pass reads them.

### Stopping and picking up again

Closing Kong does not throw the run away. Analysis checkpoints itself to
`analysis_state.json` every 25 functions, when a phase ends, and on the way
out — a Ctrl-C in the terminal, `q` in the TUI, the window closed in the GUI,
or a crash. Re-running the same command resumes from there:

```bash
kong analyze ./binary       # stopped after 200 of 400 functions
kong analyze ./binary       # picks up at 201
```

Functions that were named successfully are restored and written back into the
Ghidra database without a second LLM call. Everything else — the failures above
included — is analyzed again, so a run cut short by a rate limit, a crash or an
impatient Ctrl-C costs only what it had left to do.

The state file records which binary it describes, by SHA-256. Output
directories get reused, and `./kong_output` is the default for every run, so
this is what stops a second binary from inheriting the first one's names. A
rebuilt binary at the same path counts as a different one: its addresses have
moved, and restoring old names against them would be silently wrong.

Resuming is the default. To redo an analysis from scratch — after changing
model, editing prompts, or upgrading Ghidra — pass `--fresh`:

```bash
kong analyze ./binary --fresh
```

`--resume` is still accepted and does nothing; it is the default behaviour now.

A name a draft model produced and that was never re-read is restored like any
other — its callers benefit from it — but it goes back into the second pass
rather than being promoted to a final answer. The state file records which
model answered for each function, which is what makes that distinction
survive the process.

### Reconstruction in another language

`--format python` or `--format csharp` adds a translation of the recovered C,
written as `decompiled.py` or `Decompiled.cs`:

```bash
kong analyze ./binary --format json --format python
```

This is a **reading aid, not a port**. Decompiler output contains raw memory
access, pointer casts and calling-convention artifacts that have no faithful
equivalent in a managed language, so any function that loses behaviour in
translation is marked in place with a `NOT FAITHFUL` comment naming what could
not be expressed, and the file header counts how many survived intact. Nothing
in the output is expected to run.

It costs a second LLM pass over the whole binary, so budget roughly the same
again as the analysis itself. The pass reuses the same context budget as the
analysis, so it works with a local endpoint under `--max-prompt-chars`.

## Benchmarks

Kong autonomously reconstructed the full [XZ backdoor](https://en.wikipedia.org/wiki/XZ_Utils_backdoor) (CVE-2024-3094) kill chain from a stripped `liblzma.so.5.4.1` — identifying all five core implant functions at 90-95% confidence in 15 minutes for $6.63.

See **[BENCHMARKS.md](BENCHMARKS.md)** for the full case study and reproduction instructions.

## Project Layout

```
kong/
├── __main__.py           # CLI entry point (click)
├── config.py             # KongConfig, LLMProvider, LLMConfig
├── db.py                 # SQLite config store (~/.config/kong/)
├── banner.py             # ASCII banner, API key helpers
├── agent/
│   ├── supervisor.py     # Pipeline orchestrator
│   ├── run_log.py        # events.log writer + logging handler
│   ├── triage.py         # Function enumeration + classification
│   ├── analyzer.py       # LLM-guided function analysis
│   ├── coherence.py      # Cross-checks the results, arbitrates conflicts
│   ├── queue.py          # BFS work queue from call graph
│   ├── signatures.py     # Known function signature matching
│   ├── prompts.py        # System prompt + output schema
│   ├── events.py         # Phase/event types for pipeline tracing
│   └── models.py         # FunctionResult dataclass
├── ghidra/
│   ├── client.py         # In-process GhidraClient (PyGhidra/JPype)
│   ├── types.py          # FunctionInfo, BinaryInfo, XRef, etc.
│   └── environment.py    # Ghidra/JDK auto-detection
├── llm/
│   ├── client.py         # AnthropicClient
│   ├── openai_client.py  # OpenAIClient
│   ├── truncation.py     # Retry for a response that spent its budget
│   ├── usage.py          # TokenUsage, cost tracking, pricing registry
│   └── limits.py         # Model-specific limits + rate limiter
├── normalizer/
│   └── syntactic.py      # Decompiler output normalization
├── synthesis/
│   └── semantic.py       # Global name unification + struct synthesis
├── evals/
│   ├── harness.py        # Ground-truth extraction + scoring
│   └── metrics.py        # symbol_accuracy, type_accuracy
├── state/
│   └── persistence.py    # analysis_state.json (--resume)
├── export/
│   └── source.py         # analysis.json + Ghidra writeback
├── signatures/
│   ├── stdlib.json       # C standard library signatures
│   └── crypto.json       # Cryptographic function signatures
└── tui/
    └── app.py            # Textual TUI
```

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
