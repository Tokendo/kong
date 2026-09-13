"""Endpoint probing for LLM providers.

Validates connectivity and authentication before committing
to expensive operations like Ghidra startup.
"""

from __future__ import annotations

import logging

import anthropic
import openai

from kong.config import ZAI_BASE_URL, LLMConfig, LLMProvider

logger = logging.getLogger(__name__)

_PROBE_DUMMY_KEY = "not-needed"


def _api_key_for(config: LLMConfig) -> str | None:
    """The key the run will actually authenticate with.

    Probing with a different key than the analysis uses would make the check
    worthless, so both go through the same resolution.
    """
    from kong.banner import resolve_api_key

    return resolve_api_key(config.provider, config.api_key)


def probe_endpoint(config: LLMConfig) -> bool:
    if config.provider is LLMProvider.CUSTOM:
        return _probe_custom(config)
    if config.provider is LLMProvider.ZAI:
        return _probe_zai(config)
    if config.provider is LLMProvider.OPENAI:
        return _probe_openai(config)
    return _probe_anthropic(config)


def _probe_zai(config: LLMConfig) -> bool:
    api_key = _api_key_for(config)
    if not api_key:
        logger.warning("ZAI_API_KEY is not set")
        return False
    try:
        client = openai.OpenAI(
            api_key=api_key, base_url=config.base_url or ZAI_BASE_URL,
        )
        client.models.list()
        return True
    except openai.AuthenticationError:
        logger.warning("Z.ai rejected the API key")
        return False
    except openai.APIStatusError as e:
        if e.status_code in (404, 405):
            # Z.ai documents chat/completions, not a model listing. A missing
            # listing says nothing about whether the endpoint answers.
            logger.debug("Z.ai does not serve /models; treating as reachable")
            return True
        logger.warning("Z.ai API error: %s", e.message)
        return False
    except openai.APIConnectionError:
        logger.warning("Could not connect to %s", config.base_url or ZAI_BASE_URL)
        return False
    except openai.APIError as e:
        logger.warning("Z.ai API error: %s", e.message)
        return False


def _probe_custom(config: LLMConfig) -> bool:
    api_key = config.api_key if config.api_key else _PROBE_DUMMY_KEY
    try:
        client = openai.OpenAI(api_key=api_key, base_url=config.base_url)
        client.models.list()
        return True
    except openai.AuthenticationError:
        logger.warning("Custom endpoint rejected API key")
        return False
    except openai.APIConnectionError:
        logger.warning("Could not connect to %s", config.base_url)
        return False
    except openai.APIError as e:
        logger.warning("Custom endpoint error: %s", e.message)
        return False
    except Exception as e:
        logger.warning("Could not validate %s: %s", config.base_url, e)
        return False


def _probe_openai(config: LLMConfig) -> bool:
    try:
        client = openai.OpenAI(
            api_key=_api_key_for(config), base_url=config.base_url,
        )
        client.models.list()
        return True
    except openai.AuthenticationError:
        logger.warning("OpenAI API key is invalid")
        return False
    except openai.APIError as e:
        logger.warning("OpenAI API error: %s", e.message)
        return False


def _probe_anthropic(config: LLMConfig) -> bool:
    try:
        client = anthropic.Anthropic(api_key=_api_key_for(config))
        client.models.list()
        return True
    except anthropic.AuthenticationError:
        logger.warning("Anthropic API key is invalid")
        return False
    except anthropic.APIError as e:
        logger.warning("Anthropic API error: %s", e.message)
        return False
