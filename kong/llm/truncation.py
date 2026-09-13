"""Retrying a response that spent its whole budget without answering.

A model that reasons before it answers charges the reasoning to the same
completion budget as the answer. When the budget runs out first the request
still succeeds — HTTP 200, usage recorded, cost incurred — and hands back an
empty message. Kong then parses "" as JSON, gets nothing back, and marks every
function in the chunk unanalysed without a hint of what went wrong.

Doubling the budget once covers the common case: a budget sized for the answer
but not for the thinking in front of it. Growing further is deliberately not
attempted — a truncated call is paid for in full, so a model that burns 32k
output tokens without reaching its answer is misbehaving, not short of room.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TypeVar

logger = logging.getLogger(__name__)

#: No retry grows a budget past this.
MAX_TOKENS_CAP = 65536

#: Attempts per call, so the default is one retry.
DEFAULT_ATTEMPTS = 2

Response = TypeVar("Response")


def next_budget(current: int) -> int | None:
    """The budget to retry at, or None when there is no room left to grow."""
    if current >= MAX_TOKENS_CAP:
        return None
    return min(current * 2, MAX_TOKENS_CAP)


def call_with_budget(
    send: Callable[[int], Response],
    *,
    budget: int,
    is_truncated: Callable[[Response], bool],
    label: str,
    attempts: int = DEFAULT_ATTEMPTS,
) -> Response:
    """Call `send(budget)`, retrying at a larger budget while it truncates.

    `send` is responsible for recording usage: every attempt is billed, the
    truncated ones included. The last response is returned either way, so the
    caller parses a genuinely empty answer rather than raising.
    """
    response = send(budget)
    for _ in range(attempts - 1):
        if not is_truncated(response):
            return response

        larger = next_budget(budget)
        if larger is None:
            break
        logger.warning(
            "%s spent its %d-token budget without answering; retrying at %d.",
            label, budget, larger,
        )
        budget = larger
        response = send(budget)

    if is_truncated(response):
        logger.warning(
            "%s still had nothing to say within %d tokens; giving up on it.",
            label, budget,
        )
    return response
