"""Cross-checking a finished analysis against itself.

Functions are named and typed one chunk at a time, and a chunk knows almost
nothing about what the other chunks answered. That is what makes the pipeline
cheap, and it is also what lets it contradict itself: two functions end up with
the same name, a signature says a function takes two parameters while every
call site passes four, a function is declared ``void`` and its callers use the
value anyway, a model claims 95% confidence in a comment that admits it could
not work out what the code does.

None of that is visible from inside a single function's analysis. It only
shows up when the results are read together, which is what this module does.

The split is deliberate: **detection is free, resolution is paid for**.
``detect_conflicts`` is text and arithmetic over results that already exist, so
it can be exhaustive — every function, every call site, no LLM.
``CoherenceReviewer`` then sends only the contradictions to a model and asks it
to arbitrate, so the bill is proportional to the number of real problems rather
than to the size of the binary.

A conflict is a question, not a verdict. The decompiler is wrong often enough
that any of these checks can fire on an analysis that is right; that is why the
reviewer is allowed to answer "no_change", and why nothing here rewrites a
result on its own.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import json_repair

# Same coercion as the analysis parse boundary: a model that answers "85%"
# there answers "85%" here too.
from kong.agent.analyzer import _confidence as coerce_confidence
from kong.agent.analyzer import strip_markdown_fences
from kong.agent.models import FunctionResult
from kong.agent.signatures import SignatureDB

if TYPE_CHECKING:
    from kong.agent.analyzer import LLMClient

logger = logging.getLogger(__name__)

#: Where a pass leaves its findings, next to the rest of the output.
REPORT_NAME = "coherence.json"

#: Confidence at or above which a hedged explanation is a contradiction rather
#: than honest uncertainty. Same boundary as the high-confidence bucket in
#: AnalysisStats, so what the run reports as confident is what gets checked.
CONFIDENT_AT = 80

#: Conflicts handed to the model in one call. Small on purpose: every conflict
#: in a batch carries the decompilation of the functions it is about, and the
#: answer has to fit the same output budget a single-function analysis gets.
CONFLICTS_PER_CALL = 10

#: Conflicts arbitrated in one pass. Detection stays exhaustive — the report
#: lists everything it found — but a binary that contradicts itself a thousand
#: times should not turn one click into a thousand paid calls. What is left
#: over is picked up by running the pass again.
DEFAULT_REVIEW_LIMIT = 50

# Phrases a model reaches for when it has not understood the code. Paired with
# a high confidence score they contradict each other, and one of the two is
# wrong.
_HEDGES = (
    "cannot determine",
    "could not determine",
    "unable to determine",
    "cannot tell",
    "hard to tell",
    "unclear",
    "not clear",
    "no clear",
    "unknown purpose",
    "purpose is unknown",
    "insufficient evidence",
    "no evidence",
    "too obfuscated",
    "speculative",
    "wild guess",
    "just a guess",
)


class ConflictKind(Enum):
    """What was found. Declaration order is review order.

    A duplicated name breaks the export outright — two functions cannot share
    an identifier — so it goes first; a hedged confidence score costs nothing
    but a wrong number, so it goes last.
    """

    DUPLICATE_NAME = "duplicate_name"
    ARGUMENT_COUNT = "argument_count"
    RETURN_VALUE = "return_value"
    KNOWN_SIGNATURE = "known_signature"
    SIGNATURE_NAME = "signature_name"
    UNSUPPORTED_CONFIDENCE = "unsupported_confidence"

    @property
    def order(self) -> int:
        return list(ConflictKind).index(self)


@dataclass(frozen=True)
class Conflict:
    """One contradiction between results, and the evidence for it."""

    kind: ConflictKind
    addresses: tuple[int, ...]
    summary: str
    detail: str = ""

    @property
    def id(self) -> str:
        """Stable handle: the model answers with it, the window keys rows on it."""
        return f"{self.kind.value}:" + "-".join(f"{a:08x}" for a in self.addresses)

    @property
    def address_list(self) -> str:
        return ", ".join(f"0x{a:08x}" for a in self.addresses)

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "addresses": [f"0x{a:08x}" for a in self.addresses],
            "summary": self.summary,
            "detail": self.detail,
        }


@dataclass
class Change:
    """What a resolution wants done to one function."""

    address: int
    name: str = ""
    signature: str = ""
    confidence: int | None = None
    comment: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "address": f"0x{self.address:08x}",
            "name": self.name,
            "signature": self.signature,
            "confidence": self.confidence,
            "comment": self.comment,
        }


@dataclass
class Resolution:
    """The model's answer to one conflict."""

    conflict_id: str
    verdict: str = "no_change"
    explanation: str = ""
    changes: list[Change] = field(default_factory=list)

    @property
    def resolved(self) -> bool:
        return bool(self.changes)

    def as_dict(self) -> dict[str, object]:
        return {
            "conflict_id": self.conflict_id,
            "verdict": self.verdict,
            "explanation": self.explanation,
            "changes": [c.as_dict() for c in self.changes],
        }


@dataclass
class CoherenceReport:
    """What one pass looked at, found, and changed."""

    functions_checked: int = 0
    conflicts: list[Conflict] = field(default_factory=list)
    resolutions: dict[str, Resolution] = field(default_factory=dict)
    #: Human-readable record of every write that reached Ghidra.
    applied: list[str] = field(default_factory=list)
    #: Conflicts detected but never sent to the model, because of the cap.
    unreviewed: int = 0
    llm_calls: int = 0

    @property
    def reviewed(self) -> int:
        return len(self.conflicts) - self.unreviewed

    def as_dict(self) -> dict[str, object]:
        return {
            "functions_checked": self.functions_checked,
            "conflicts_found": len(self.conflicts),
            "conflicts_reviewed": self.reviewed,
            "changes_applied": len(self.applied),
            "llm_calls": self.llm_calls,
            "conflicts": [
                {
                    **conflict.as_dict(),
                    "resolution": (
                        self.resolutions[conflict.id].as_dict()
                        if conflict.id in self.resolutions
                        else None
                    ),
                }
                for conflict in self.conflicts
            ],
            "applied": list(self.applied),
        }


# ---------------------------------------------------------------- signatures


@dataclass(frozen=True)
class ParsedSignature:
    """The parts of a C declaration this module actually compares."""

    return_type: str
    name: str
    params: tuple[str, ...]
    variadic: bool = False
    #: True for ``int f()`` — empty parentheses say nothing in C, so an
    #: argument count can neither agree nor disagree with them.
    unspecified: bool = False

    @property
    def param_count(self) -> int:
        return len(self.params)

    @property
    def returns_void(self) -> bool:
        # A void* return is a value like any other; only bare void is not.
        return self.return_type.replace(" ", "") == "void"


_IDENTIFIER_TAIL = re.compile(r"[A-Za-z_][A-Za-z0-9_]*$")


def parse_signature(signature: str) -> ParsedSignature | None:
    """Read ``return_type name(params)``. None when it is not one.

    Deliberately forgiving about types: the point is to compare arity and
    void-ness, not to type-check C.
    """
    text = signature.strip().rstrip(";").strip()
    if not text or "(" not in text:
        return None

    open_paren = text.index("(")
    head = text[:open_paren]
    name_match = _IDENTIFIER_TAIL.search(head.rstrip())
    if name_match is None:
        return None

    parsed = _read_call(text, open_paren)
    if parsed is None:
        return None
    args, _ = parsed

    variadic = any(a.strip() == "..." for a in args)
    params = tuple(a for a in args if a.strip() and a.strip() != "...")
    if len(params) == 1 and params[0].strip() == "void":
        params = ()

    return ParsedSignature(
        return_type=head[: name_match.start()].strip(),
        name=name_match.group(0),
        params=params,
        variadic=variadic,
        unspecified=not args,
    )


# ---------------------------------------------------------------- call sites


@dataclass(frozen=True)
class CallSite:
    """One call to a function, as it appears in a caller's decompilation."""

    name: str
    arg_count: int
    uses_return: bool
    line: str


_COMMENT_TAIL = re.compile(r"/\*.*?\*/\s*$", re.DOTALL)

#: What can stand in front of a call that throws its result away.
_STATEMENT_ENDS = ";{}:"


def _read_call(text: str, open_paren: int) -> tuple[list[str], int] | None:
    """Split the argument list starting at *open_paren*.

    Returns the top-level arguments and the index of the closing parenthesis,
    or None when the call is not closed — a truncated decompilation, or a
    parenthesis inside something this scanner does not understand.
    """
    depth = 0
    args: list[str] = []
    current: list[str] = []
    index = open_paren
    end = len(text)

    while index < end:
        char = text[index]

        if char in "\"'":
            closing = text.find(char, index + 1)
            while closing != -1 and text[closing - 1] == "\\":
                closing = text.find(char, closing + 1)
            if closing == -1:
                return None
            current.append(text[index : closing + 1])
            index = closing + 1
            continue

        if char in "([{":
            depth += 1
            if not (depth == 1 and char == "("):
                current.append(char)
        elif char in ")]}":
            depth -= 1
            if depth == 0:
                argument = "".join(current).strip()
                if argument or args:
                    args.append(argument)
                return args, index
            current.append(char)
        elif char == "," and depth == 1:
            args.append("".join(current).strip())
            current = []
        else:
            current.append(char)

        index += 1

    return None


def _return_is_used(code: str, call_start: int) -> bool:
    """True when the caller does something with the value the call returns.

    Read backwards rather than forwards: a call whose result is dropped is a
    statement of its own, so what precedes it is the end of the previous one.
    """
    before = _COMMENT_TAIL.sub("", code[:call_start]).rstrip()
    if not before:
        return False
    if before[-1] in _STATEMENT_ENDS:
        return False

    word = _IDENTIFIER_TAIL.search(before)
    if word is not None and word.group(0) in ("else", "do"):
        return False
    return True


def find_call_sites(code: str, names: Iterable[str]) -> list[CallSite]:
    """Every call to one of *names* in this decompilation.

    Several names because a result carries two: the recovered one and the
    ``FUN_00401000`` it replaced. Which of the two the text uses depends on
    whether it was decompiled before or after the rename, and the check has to
    work either way.
    """
    if not code:
        return []

    # Ghidra opens a decompilation with the function's own header, which reads
    # exactly like a call to itself. Everything before the first brace is that
    # header, and no call site lives there.
    body_start = code.find("{")
    if body_start == -1:
        return []

    sites: list[CallSite] = []
    for name in dict.fromkeys(n for n in names if n):
        for match in re.finditer(rf"\b{re.escape(name)}\s*\(", code):
            if match.start() < body_start:
                continue
            parsed = _read_call(code, match.end() - 1)
            if parsed is None:
                continue
            args, _ = parsed
            line_start = code.rfind("\n", 0, match.start()) + 1
            line_end = code.find("\n", match.start())
            if line_end == -1:
                line_end = len(code)
            sites.append(CallSite(
                name=name,
                arg_count=len(args),
                uses_return=_return_is_used(code, match.start()),
                line=code[line_start:line_end].strip(),
            ))
    return sites


# ----------------------------------------------------------------- detection


def _usable(
    results: Mapping[int, FunctionResult],
) -> Iterator[tuple[int, FunctionResult]]:
    """The results that carry an answer worth cross-checking."""
    for addr, result in sorted(results.items()):
        if result.skipped or result.error or not result.name:
            continue
        yield addr, result


def _describe(addr: int, result: FunctionResult) -> str:
    parts = [f"0x{addr:08x} `{result.name}`"]
    if result.signature:
        parts.append(f"signature `{result.signature}`")
    parts.append(f"confidence {result.confidence}")
    if result.classification:
        parts.append(result.classification)
    return ", ".join(parts)


def _duplicate_name_conflicts(results: Mapping[int, FunctionResult]) -> list[Conflict]:
    by_name: dict[str, list[int]] = defaultdict(list)
    for addr, result in _usable(results):
        by_name[result.name].append(addr)

    conflicts = []
    for name, addresses in sorted(by_name.items()):
        if len(addresses) < 2:
            continue
        conflicts.append(Conflict(
            kind=ConflictKind.DUPLICATE_NAME,
            addresses=tuple(sorted(addresses)),
            summary=f"{len(addresses)} functions are all named {name}",
            detail="\n".join(
                f"- {_describe(addr, results[addr])}" for addr in sorted(addresses)
            ),
        ))
    return conflicts


def _signature_name_conflicts(results: Mapping[int, FunctionResult]) -> list[Conflict]:
    conflicts = []
    for addr, result in _usable(results):
        parsed = parse_signature(result.signature)
        if parsed is None or not parsed.name or parsed.name == result.name:
            continue
        conflicts.append(Conflict(
            kind=ConflictKind.SIGNATURE_NAME,
            addresses=(addr,),
            summary=(
                f"0x{addr:08x} is named {result.name} but its signature "
                f"declares {parsed.name}"
            ),
            detail=f"- {_describe(addr, result)}",
        ))
    return conflicts


def _known_signature_conflicts(
    results: Mapping[int, FunctionResult],
    signature_db: SignatureDB,
) -> list[Conflict]:
    """Names claiming a known identity the recovered signature contradicts.

    Only arity and void-ness are compared: ``uint8_t *`` against ``void *`` is
    a difference of taste, ``memcpy`` with two parameters is not.
    """
    conflicts = []
    for addr, result in _usable(results):
        entry = signature_db.lookup(result.name)
        if entry is None or not entry.signature:
            continue
        known = parse_signature(entry.signature)
        mine = parse_signature(result.signature)
        if known is None or mine is None:
            continue
        if known.variadic or mine.variadic or mine.unspecified:
            continue

        if known.param_count != mine.param_count:
            disagreement = (
                f"takes {mine.param_count} parameters where {entry.name} takes "
                f"{known.param_count}"
            )
        elif known.returns_void != mine.returns_void:
            disagreement = (
                f"returns {mine.return_type or 'nothing'} where {entry.name} "
                f"returns {known.return_type or 'nothing'}"
            )
        else:
            continue

        conflicts.append(Conflict(
            kind=ConflictKind.KNOWN_SIGNATURE,
            addresses=(addr,),
            summary=f"0x{addr:08x} is named {result.name} but {disagreement}",
            detail="\n".join([
                f"- {_describe(addr, result)}",
                f"- known: `{entry.signature}` — {entry.description}",
            ]),
        ))
    return conflicts


def _shared_names(results: Mapping[int, FunctionResult]) -> set[str]:
    seen: set[str] = set()
    shared: set[str] = set()
    for _, result in _usable(results):
        if result.name in seen:
            shared.add(result.name)
        seen.add(result.name)
    return shared


def _call_site_conflicts(
    results: Mapping[int, FunctionResult],
    decompilations: Mapping[int, str],
    callers: Mapping[int, Sequence[int]],
) -> list[Conflict]:
    """Signatures that disagree with the way the function is actually called."""
    conflicts = []
    shared = _shared_names(results)

    for addr, result in _usable(results):
        parsed = parse_signature(result.signature)
        if parsed is None:
            continue

        # A name two functions answer to cannot say which of them a call site
        # meant, and a signature "fixed" against the wrong function's callers
        # is worse than the duplicate that is already being reported. The
        # decompiler's own name is unambiguous, so that one still counts.
        names = (
            (result.original_name,) if result.name in shared
            else (result.name, result.original_name)
        )

        for caller in sorted(callers.get(addr, ())):
            code = decompilations.get(caller)
            if not code or caller == addr:
                continue
            sites = find_call_sites(code, names)
            if not sites:
                continue

            caller_result = results.get(caller)
            caller_name = (
                caller_result.name
                if caller_result is not None and caller_result.name
                else f"FUN_{caller:08x}"
            )
            counts = {site.arg_count for site in sites}

            # One count only: a caller that calls the same function with two
            # different argument counts is being read badly, not contradicted.
            if (
                not parsed.variadic
                and not parsed.unspecified
                and len(counts) == 1
                and parsed.param_count not in counts
            ):
                passed = next(iter(counts))
                conflicts.append(Conflict(
                    kind=ConflictKind.ARGUMENT_COUNT,
                    addresses=(addr, caller),
                    summary=(
                        f"{result.name} declares {parsed.param_count} parameters "
                        f"but {caller_name} passes {passed}"
                    ),
                    detail="\n".join([
                        f"- callee {_describe(addr, result)}",
                        f"- caller 0x{caller:08x} `{caller_name}`",
                        *(f"    {site.line}" for site in sites[:3]),
                    ]),
                ))

            if parsed.returns_void and any(site.uses_return for site in sites):
                conflicts.append(Conflict(
                    kind=ConflictKind.RETURN_VALUE,
                    addresses=(addr, caller),
                    summary=(
                        f"{result.name} is declared void but {caller_name} uses "
                        f"its return value"
                    ),
                    detail="\n".join([
                        f"- callee {_describe(addr, result)}",
                        f"- caller 0x{caller:08x} `{caller_name}`",
                        *(
                            f"    {site.line}"
                            for site in sites[:3]
                            if site.uses_return
                        ),
                    ]),
                ))
    return conflicts


def _confidence_conflicts(results: Mapping[int, FunctionResult]) -> list[Conflict]:
    """High confidence in an explanation that admits it understood nothing."""
    conflicts = []
    for addr, result in _usable(results):
        if result.confidence < CONFIDENT_AT:
            continue
        prose = f"{result.comments} {result.reasoning}".lower()
        hedge = next((h for h in _HEDGES if h in prose), "")
        if not hedge:
            continue
        lines = [f"- {_describe(addr, result)}"]
        if result.comments:
            lines.append(f"- comments: {result.comments}")
        if result.reasoning:
            lines.append(f"- reasoning: {result.reasoning}")
        conflicts.append(Conflict(
            kind=ConflictKind.UNSUPPORTED_CONFIDENCE,
            addresses=(addr,),
            summary=(
                f"0x{addr:08x} {result.name} is {result.confidence}% confident "
                f"and says \"{hedge}\""
            ),
            detail="\n".join(lines),
        ))
    return conflicts


def detect_conflicts(
    results: Mapping[int, FunctionResult],
    decompilations: Mapping[int, str],
    callers: Mapping[int, Sequence[int]],
    signature_db: SignatureDB | None = None,
) -> list[Conflict]:
    """Every contradiction the results hold, worst first.

    Costs nothing but the decompilations already in hand, so it runs over the
    whole binary rather than over a sample of it.
    """
    conflicts = [
        *_duplicate_name_conflicts(results),
        *_call_site_conflicts(results, decompilations, callers),
        *_signature_name_conflicts(results),
        *_confidence_conflicts(results),
    ]
    if signature_db is not None:
        conflicts += _known_signature_conflicts(results, signature_db)

    conflicts.sort(key=lambda c: (c.kind.order, c.addresses))
    return conflicts


# --------------------------------------------------------------- arbitration


_RULES = """\
Decide what is actually true and return the smallest set of changes that makes \
the analysis consistent.

- Judge from the decompilation shown, not from the names already assigned: a \
name is only evidence of what an earlier pass guessed.
- Change only what the conflict is about. Anything you do not mention is kept.
- A conflict can be a false alarm — the decompiler is wrong often enough. Say \
so with "no_change" and one line of explanation rather than inventing a fix.
- Two functions must never end up with the same name. Tell them apart by what \
they actually do, not by adding _1 and _2.
- When you keep a name you are not certain of, lower its confidence to match.
- A signature has to match the way the function is called at its call sites.
- Names are snake_case unless the surrounding names are clearly camelCase."""

_RESPONSE_SHAPE = """\
Respond with exactly one JSON object, no prose before or after:
```json
{
  "resolutions": [
    {
      "conflict": "<the conflict id, copied exactly>",
      "verdict": "resolved" | "no_change",
      "explanation": "one sentence on what was actually wrong",
      "changes": [
        {
          "address": "0x00401000",
          "name": "<new name, or omit to keep it>",
          "signature": "<new full signature, or omit to keep it>",
          "confidence": <0-100, or omit to keep it>,
          "comment": "<replacement description, or omit to keep it>"
        }
      ]
    }
  ]
}
```
Return one entry per conflict, in the order they are listed, and use \
"no_change" with an empty "changes" list where the existing analysis is right."""


class CoherenceReviewer:
    """Asks a model to arbitrate the conflicts detection found.

    One call per batch of conflicts. Batched rather than one call per conflict
    because conflicts overlap — the same function turns up in several of them —
    and rather than one call for all of them because the answer has to fit an
    output budget sized for a single function's analysis.
    """

    def __init__(
        self,
        llm: LLMClient,
        max_prompt_chars: int | None = None,
        conflicts_per_call: int = CONFLICTS_PER_CALL,
    ) -> None:
        self.llm = llm
        self.max_prompt_chars = max_prompt_chars
        self.conflicts_per_call = max(1, conflicts_per_call)

    def batches(
        self,
        conflicts: Sequence[Conflict],
        decompilations: Mapping[int, str],
    ) -> list[list[Conflict]]:
        """Group conflicts into prompts that fit the context window.

        The cost of a conflict is mostly the decompilation of the functions it
        names, and conflicts that share a function share that cost, so it is
        counted once per batch rather than once per conflict.
        """
        budget = self.max_prompt_chars
        batches: list[list[Conflict]] = []
        current: list[Conflict] = []
        seen: set[int] = set()
        spent = 0

        for conflict in conflicts:
            fresh = [a for a in conflict.addresses if a not in seen]
            cost = len(conflict.summary) + len(conflict.detail) + 200
            cost += sum(len(decompilations.get(a, "")) + 200 for a in fresh)

            too_long = budget is not None and current and spent + cost > budget
            if current and (len(current) >= self.conflicts_per_call or too_long):
                batches.append(current)
                current = []
                seen = set()
                spent = 0
                fresh = list(conflict.addresses)

            current.append(conflict)
            seen.update(fresh)
            spent += cost

        if current:
            batches.append(current)
        return batches

    def build_prompt(
        self,
        conflicts: Sequence[Conflict],
        results: Mapping[int, FunctionResult],
        decompilations: Mapping[int, str],
    ) -> str:
        addresses: list[int] = []
        for conflict in conflicts:
            addresses += [a for a in conflict.addresses if a not in addresses]

        parts = [
            "An automated pass named and typed every function in this binary "
            "one at a time. Read together, some of its answers contradict each "
            "other. Every contradiction below was found mechanically, by "
            "comparing the results against each other and against the code.",
            "",
            _RULES,
            "",
            _RESPONSE_SHAPE,
            "",
            "## Functions involved",
            "",
        ]

        for addr in addresses:
            result = results.get(addr)
            if result is None:
                continue
            parts.append(f"### 0x{addr:08x}: {result.name}")
            parts.append(
                f"Was: {result.original_name}. "
                f"Signature: {result.signature or 'none recovered'}. "
                f"Confidence: {result.confidence}. "
                f"Classification: {result.classification or 'none'}."
            )
            if result.comments:
                parts.append(f"Description: {result.comments}")
            code = decompilations.get(addr, "")
            if code:
                parts.append("```c")
                parts.append(code)
                parts.append("```")
            else:
                parts.append("(decompilation unavailable)")
            parts.append("")

        parts.append("## Conflicts")
        parts.append("")
        for conflict in conflicts:
            parts.append(f"### {conflict.id}")
            parts.append(conflict.summary)
            if conflict.detail:
                parts.append(conflict.detail)
            parts.append("")

        return "\n".join(parts)

    def review(
        self,
        conflicts: Sequence[Conflict],
        results: Mapping[int, FunctionResult],
        decompilations: Mapping[int, str],
        model: str | None = None,
    ) -> list[Resolution]:
        """Arbitrate one batch. Never raises over a badly formed answer.

        Sent through ``analyze_function`` — the single-function entry point —
        because that is the only one every provider client implements, and the
        shape asked for is spelled out in the prompt rather than left to the
        client's own schema. The synthesis pass goes through the same door.
        """
        prompt = self.build_prompt(conflicts, results, decompilations)
        response = self.llm.analyze_function(prompt, model=model)
        return parse_resolutions(response.raw, conflicts)


def _address_of(value: object) -> int | None:
    """Read an address a model wrote as a hex string, or as an int."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        for base in (0, 16):
            try:
                return int(text, base)
            except (ValueError, TypeError):
                continue
    return None


def _confidence_or_none(value: object) -> int | None:
    """The score a change asks for, or None when it does not ask for one."""
    if value is None:
        return None
    coerced = coerce_confidence(value, default=-1)
    return None if coerced < 0 else coerced


def parse_resolutions(
    raw: str,
    conflicts: Sequence[Conflict],
) -> list[Resolution]:
    """Read the model's answer, keeping only what it was asked about.

    Everything here is defensive on purpose. The answer ends up in Ghidra, so
    a hallucinated address — one belonging to a function that was not part of
    the batch — is dropped rather than renamed.
    """
    text = strip_markdown_fences(raw or "")
    if not text:
        return []

    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        # Same fallback as the analysis parse boundary. Worth more here than
        # there: an answer cut off mid-array still carries the resolutions
        # that made it, and each one is independent of the rest.
        try:
            data = json_repair.loads(text)
        except Exception:
            logger.warning("Coherence review returned no usable JSON")
            return []
        logger.debug("Recovered malformed coherence JSON via json_repair")

    if isinstance(data, dict):
        entries = data.get("resolutions", [])
    elif isinstance(data, list):
        entries = data
    else:
        entries = []
    if not isinstance(entries, list):
        return []

    by_id = {conflict.id: conflict for conflict in conflicts}
    allowed = {addr for conflict in conflicts for addr in conflict.addresses}

    resolutions: list[Resolution] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue

        conflict_id = str(
            entry.get("conflict") or entry.get("conflict_id") or ""
        ).strip()
        if conflict_id not in by_id:
            logger.warning(
                "Coherence review answered for unknown conflict %r", conflict_id,
            )
            continue

        changes: list[Change] = []
        for raw_change in entry.get("changes") or []:
            if not isinstance(raw_change, dict):
                continue
            address = _address_of(raw_change.get("address"))
            if address is None or address not in allowed:
                logger.warning(
                    "Dropping a change for %r: not a function in this batch",
                    raw_change.get("address"),
                )
                continue
            change = Change(
                address=address,
                name=str(raw_change.get("name") or "").strip(),
                signature=str(raw_change.get("signature") or "").strip(),
                confidence=_confidence_or_none(raw_change.get("confidence")),
                comment=str(raw_change.get("comment") or "").strip(),
            )
            if (
                change.name
                or change.signature
                or change.confidence is not None
                or change.comment
            ):
                changes.append(change)

        verdict = str(entry.get("verdict") or "").strip().lower()
        resolutions.append(Resolution(
            conflict_id=conflict_id,
            verdict=verdict or ("resolved" if changes else "no_change"),
            explanation=str(entry.get("explanation") or "").strip(),
            changes=changes,
        ))

    return resolutions
