"""Ask an OpenAI-compatible endpoint what it is serving.

A local server knows things Kong otherwise has to be told: which models it
has loaded, the context window it was started with, how many slots that
context is split across. llama.cpp exposes the OpenAI `/v1/models` list plus
its own `/props`; other servers expose only the former. Everything here
degrades to "unknown" rather than failing, because the field names differ
between servers and between llama.cpp builds.

Two GETs do not justify a third-party HTTP client: the Anthropic and OpenAI
SDKs are migrating from httpx to httpx2, and depending on either directly
would pin Kong to one side of that split.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from kong.llm.limits import ModelLimits

logger = logging.getLogger(__name__)

# Sizing constants, shared with the guidance in the README.
_SYSTEM_PROMPT_TOKENS = 600
_CHARS_PER_TOKEN = 3  # decompiled C tokenizes worse than prose
_OUTPUT_SHARE = 8  # a model gets an eighth of its window to answer in
_TOKENS_PER_RESULT = 260  # one batch result, with margin

_MIN_OUTPUT_TOKENS = 256
_MAX_OUTPUT_TOKENS = 8192


@dataclass
class EndpointModel:
    """One model an endpoint reports as available."""

    id: str
    train_context: int | None = None
    parameters: int | None = None
    size_bytes: int | None = None

    @property
    def parameters_label(self) -> str:
        if self.parameters is None:
            return "?"
        if self.parameters >= 1_000_000_000:
            return f"{self.parameters / 1_000_000_000:.1f}B"
        return f"{self.parameters / 1_000_000:.0f}M"


@dataclass
class EndpointInfo:
    """What an endpoint says about itself."""

    base_url: str
    models: list[EndpointModel] = field(default_factory=list)
    context_tokens: int | None = None
    total_slots: int | None = None
    model_path: str = ""
    build_info: str = ""

    @property
    def effective_context(self) -> int | None:
        """Context a single request can use, or None if the server won't say.

        llama.cpp reports the per-slot figure in /props, which is what a
        request actually gets. Falling back to the training context of the
        first model is a guess, and a generous one, so it is only used when
        the server exposes nothing better.
        """
        if self.context_tokens:
            return self.context_tokens
        if self.models and self.models[0].train_context:
            return self.models[0].train_context
        return None


def _as_int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _first_int(*candidates: Any) -> int | None:
    for candidate in candidates:
        number = _as_int(candidate)
        if number is not None:
            return number
    return None


def _parse_models(payload: Any) -> list[EndpointModel]:
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return []

    models: list[EndpointModel] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        identifier = entry.get("id")
        if not identifier:
            continue
        meta = entry.get("meta") if isinstance(entry.get("meta"), dict) else {}
        models.append(
            EndpointModel(
                id=str(identifier),
                train_context=_first_int(
                    meta.get("n_ctx_train"), entry.get("n_ctx_train")
                ),
                parameters=_first_int(meta.get("n_params"), entry.get("n_params")),
                size_bytes=_first_int(meta.get("size"), entry.get("size")),
            )
        )
    return models


def _parse_props(payload: Any) -> tuple[int | None, int | None, str, str]:
    """Pull (context, slots, model path, build) out of a llama.cpp /props body."""
    if not isinstance(payload, dict):
        return None, None, "", ""

    generation = payload.get("default_generation_settings")
    generation = generation if isinstance(generation, dict) else {}
    nested = generation.get("params")
    nested = nested if isinstance(nested, dict) else {}

    context = _first_int(
        generation.get("n_ctx"), nested.get("n_ctx"), payload.get("n_ctx")
    )
    slots = _first_int(payload.get("total_slots"), payload.get("n_parallel"))
    model_path = str(payload.get("model_path") or payload.get("model") or "")
    build_info = str(payload.get("build_info") or "")
    return context, slots, model_path, build_info


class EndpointError(RuntimeError):
    """The endpoint could not be read."""


def _get_json(url: str, api_key: str | None, timeout: float) -> Any:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def discover(
    base_url: str,
    api_key: str | None = None,
    timeout: float = 5.0,
) -> EndpointInfo:
    """Query *base_url* for its models and runtime limits.

    Raises EndpointError if the models listing cannot be read. A missing or
    unparseable /props only leaves the runtime fields empty, since that
    endpoint is llama.cpp-specific and other servers do not have it.
    """
    root = base_url.rstrip("/")
    info = EndpointInfo(base_url=root)

    try:
        info.models = _parse_models(_get_json(f"{root}/models", api_key, timeout))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise EndpointError(f"{root}/models: {exc}") from exc

    # /props sits next to the OpenAI surface, not under /v1.
    props_root = root[: -len("/v1")] if root.endswith("/v1") else root
    try:
        payload = _get_json(f"{props_root}/props", api_key, timeout)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        logger.debug("No /props on %s: %s", props_root, exc)
        return info

    context, slots, model_path, build = _parse_props(payload)
    info.context_tokens = context
    info.total_slots = slots
    info.model_path = model_path
    info.build_info = build
    return info


def suggest_limits(context_tokens: int) -> ModelLimits:
    """Derive Kong's three context knobs from a server's window.

    The prompt budget is in characters and excludes the system prompt, so it
    is what is left of the window once the answer and the instructions are
    paid for. Batch size follows the output budget, not the window.
    """
    output_tokens = min(
        _MAX_OUTPUT_TOKENS, max(_MIN_OUTPUT_TOKENS, context_tokens // _OUTPUT_SHARE)
    )
    prompt_tokens = context_tokens - output_tokens - _SYSTEM_PROMPT_TOKENS
    prompt_chars = max(2_000, prompt_tokens * _CHARS_PER_TOKEN)
    chunk_functions = max(1, output_tokens // _TOKENS_PER_RESULT)

    return ModelLimits(
        max_prompt_chars=prompt_chars,
        max_chunk_functions=chunk_functions,
        max_output_tokens=output_tokens,
    )
