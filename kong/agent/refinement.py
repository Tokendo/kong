"""Routing between the draft pass and the second, stronger pass.

Kong can run the analysis in two passes: a fast model over everything that is
cheap to get right, then a stronger model over what the fast one got wrong or
was unsure about. Both directions of that decision live here.

``should_draft`` keeps work away from the draft model when a second pass is a
near-certainty anyway — a large function is going to be re-read whatever the
draft says about it, so drafting it only costs a round trip.

``refinement_reason`` looks at a drafted result and says whether the strong
model should redo it. The self-reported confidence is one signal among several
on purpose: it is not calibrated (see the TODO in ``AnalysisStats``), and a
small model is *confidently* wrong more often than a large one is. The
objective signals — no answer at all, no name, a name that says nothing —
catch what the score misses.

Deliberately not a signal: ``signature_applied``. A rejected signature usually
means the type it references does not exist yet, which the cleanup phase
retries on its own; treating it as a refinement trigger would re-run most of
the binary through the strong model for a problem that fixes itself.
"""

from __future__ import annotations

from kong.agent.models import FunctionResult
from kong.ghidra.types import FunctionClassification, FunctionInfo

#: Confidence below which a drafted function is re-analyzed. Matches the
#: "high confidence" bucket in AnalysisStats, so the second pass covers exactly
#: what the run report does not call high confidence.
DEFAULT_REFINE_BELOW = 80

#: Size above which a function skips the draft, used only when triage left no
#: classification. Same boundary as FunctionClassification.LARGE.
LARGE_FUNCTION_BYTES = 256

#: Longest a draft's error is shown before it is cut off. The over-budget
#: errors from the analyzer and the supervisor are the longest legitimate ones
#: and run a little over 200 chars because they end in the number to raise
#: --max-prompt-chars to — the actionable part a reader needs. The cap sits
#: above that so those are never the ones cut; it exists only to keep a
#: pathological error (a raw traceback, say) from running into the log.
MAX_ERROR_CHARS = 240

# Decompiler placeholders and the names a model falls back on when it has not
# understood the function. Either way the name carries no information, which is
# the thing worth a second pass.
_GENERIC_PREFIXES = (
    "fun_", "sub_", "loc_", "lab_", "unk_", "func_", "nullsub_", "j_", "case_",
)

_GENERIC_NAMES = frozenset({
    "callback", "do_something", "do_work", "entry", "func", "function",
    "handler", "helper", "main_logic", "process", "process_data", "routine",
    "stub", "subroutine", "unknown", "unnamed", "utility", "utility_function",
    "wrapper",
})


def is_generic_name(name: str) -> bool:
    """True when the name says nothing the original did not already say."""
    lowered = name.strip().lower()
    if not lowered:
        return True
    if lowered.startswith(_GENERIC_PREFIXES):
        return True
    return lowered in _GENERIC_NAMES


def should_draft(
    func: FunctionInfo, *, large_bytes: int = LARGE_FUNCTION_BYTES,
) -> bool:
    """True when the fast model gets first look at this function."""
    if func.classification is FunctionClassification.LARGE:
        return False
    if func.classification is None and func.size > large_bytes:
        return False
    return True


def refinement_reason(
    result: FunctionResult, *, threshold: int = DEFAULT_REFINE_BELOW,
) -> str:
    """Why the strong model should redo this function; empty means leave it.

    The string is user-facing: it goes into the event stream so the run log
    says what each second-pass call is for.
    """
    if result.skipped:
        return ""
    if result.error:
        error = result.error
        if len(error) > MAX_ERROR_CHARS:
            error = error[:MAX_ERROR_CHARS] + "…"
        return f"draft failed: {error}"
    if not result.name:
        return "draft returned no name"
    if is_generic_name(result.name):
        return f"name carries no information: {result.name}"
    if result.confidence < threshold:
        return f"confidence {result.confidence} is below {threshold}"
    return ""


def needs_refinement(
    result: FunctionResult, *, threshold: int = DEFAULT_REFINE_BELOW,
) -> bool:
    return bool(refinement_reason(result, threshold=threshold))
