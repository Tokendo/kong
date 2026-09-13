"""Tests for the Anthropic LLM client wrapper."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from kong.llm.client import DEFAULT_MODEL, AnthropicClient
from kong.llm.usage import ModelTokenUsage, TokenUsage


def _mock_message(text: str, input_tokens: int = 100, output_tokens: int = 50):
    """Create a mock Anthropic message response."""
    block = MagicMock()
    block.type = "text"
    block.text = text

    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    usage.cache_creation_input_tokens = 0
    usage.cache_read_input_tokens = 0

    msg = MagicMock()
    msg.content = [block]
    msg.usage = usage
    return msg


class TestAnthropicClient:
    @patch("kong.llm.client.anthropic.Anthropic")
    def test_analyze_function_parses_json(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _mock_message(
            '{"name": "parse_config", "confidence": 85, "classification": "parser"}'
        )

        client = AnthropicClient(api_key="test-key")
        response = client.analyze_function("analyze this function")

        assert response.name == "parse_config"
        assert response.confidence == 85
        assert response.classification == "parser"

    @patch("kong.llm.client.anthropic.Anthropic")
    def test_tracks_token_usage(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _mock_message(
            '{"name": "f"}', input_tokens=200, output_tokens=80
        )

        client = AnthropicClient(api_key="test-key")
        client.analyze_function("prompt1")

        assert client.usage.input_tokens == 200
        assert client.usage.output_tokens == 80
        assert client.usage.calls == 1

    @patch("kong.llm.client.anthropic.Anthropic")
    def test_accumulates_across_calls(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _mock_message(
            '{"name": "f"}', input_tokens=100, output_tokens=50
        )

        client = AnthropicClient(api_key="test-key")
        client.analyze_function("p1")
        client.analyze_function("p2")

        assert client.usage.input_tokens == 200
        assert client.usage.output_tokens == 100
        assert client.usage.calls == 2

    @patch("kong.llm.client.anthropic.Anthropic")
    def test_passes_system_prompt_with_cache_control(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _mock_message('{"name": "f"}')

        client = AnthropicClient(api_key="test-key")
        client.analyze_function("test prompt")

        call_kwargs = mock_client.messages.create.call_args
        assert "system" in call_kwargs.kwargs
        system_val = call_kwargs.kwargs["system"]
        assert isinstance(system_val, list)
        assert system_val[0]["cache_control"] == {"type": "ephemeral"}
        assert "reverse engineer" in system_val[0]["text"].lower()

    @patch("kong.llm.client.anthropic.Anthropic")
    def test_model_override(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _mock_message('{"name": "f"}')

        client = AnthropicClient(api_key="test-key")
        client.analyze_function("prompt", model="claude-haiku-4-5-20251001")

        call_kwargs = mock_client.messages.create.call_args
        assert call_kwargs.kwargs["model"] == "claude-haiku-4-5-20251001"

    @patch("kong.llm.client.anthropic.Anthropic")
    def test_per_model_cost_tracking(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _mock_message(
            '{"name": "f"}', input_tokens=100, output_tokens=50
        )

        client = AnthropicClient(api_key="test-key")
        client.analyze_function("p1")
        client.analyze_function("p2", model="claude-haiku-4-5-20251001")

        assert len(client.usage.by_model) == 2
        assert DEFAULT_MODEL in client.usage.by_model
        assert "claude-haiku-4-5-20251001" in client.usage.by_model
        assert client.usage.calls == 2

    @patch("kong.llm.client.anthropic.Anthropic")
    def test_handles_malformed_response(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _mock_message(
            "I cannot analyze this function because reasons."
        )

        client = AnthropicClient(api_key="test-key")
        response = client.analyze_function("prompt")

        assert response.name == ""
        assert "Failed to parse" in response.reasoning

class TestModelTokenUsage:
    def test_cost_calculation(self):
        u = ModelTokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
        cost = u.cost_usd("claude-opus-4-6")
        assert cost == 5.0 + 25.0

    def test_cost_with_unknown_model_uses_default(self):
        u = ModelTokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
        cost = u.cost_usd("unknown-model")
        assert cost == 3.0 + 15.0

    def test_cache_token_costs(self):
        u = ModelTokenUsage(
            input_tokens=0, output_tokens=0,
            cache_creation_tokens=1_000_000, cache_read_tokens=1_000_000,
        )
        cost = u.cost_usd("claude-opus-4-6")
        assert cost == (5.0 * 1.25) + (5.0 * 0.10)


class TestTokenUsage:
    def test_aggregate_properties(self):
        u = TokenUsage()
        u._get("model_a").input_tokens = 100
        u._get("model_a").output_tokens = 50
        u._get("model_b").input_tokens = 200
        u._get("model_b").output_tokens = 80
        u._get("model_a").calls = 1
        u._get("model_b").calls = 2

        assert u.input_tokens == 300
        assert u.output_tokens == 130
        assert u.total_tokens == 430
        assert u.calls == 3

    def test_total_cost_across_models(self):
        u = TokenUsage()
        u._get("claude-opus-4-6").input_tokens = 1_000_000
        u._get("claude-opus-4-6").output_tokens = 0
        u._get("claude-haiku-4-5-20251001").input_tokens = 1_000_000
        u._get("claude-haiku-4-5-20251001").output_tokens = 0

        assert u.total_cost_usd == 5.0 + 1.0


class TestAnalyzeFunctionBatch:
    def test_batch_returns_list_of_responses(self) -> None:
        mock_anthropic = MagicMock()
        mock_message = MagicMock()
        mock_message.content = [MagicMock(type="text", text=json.dumps([
            {"address": "0x1000", "name": "foo", "confidence": 80},
            {"address": "0x2000", "name": "bar", "confidence": 70},
        ]))]
        mock_message.usage = MagicMock(
            input_tokens=100, output_tokens=50,
            cache_creation_input_tokens=0, cache_read_input_tokens=0,
        )
        mock_anthropic.messages.create.return_value = mock_message

        client = AnthropicClient(api_key="test")
        client._client = mock_anthropic
        results = client.analyze_function_batch("batch prompt", model="claude-haiku-4-5-20251001")

        assert len(results) == 2
        assert results[0].name == "foo"
        assert results[1].name == "bar"

    def test_batch_uses_batch_system_prompt(self) -> None:
        mock_anthropic = MagicMock()
        mock_message = MagicMock()
        mock_message.content = [MagicMock(type="text", text='[{"name": "f", "confidence": 50}]')]
        mock_message.usage = MagicMock(
            input_tokens=100, output_tokens=50,
            cache_creation_input_tokens=0, cache_read_input_tokens=0,
        )
        mock_anthropic.messages.create.return_value = mock_message

        client = AnthropicClient(api_key="test")
        client._client = mock_anthropic
        client.analyze_function_batch("prompt")

        call_kwargs = mock_anthropic.messages.create.call_args
        system_text = call_kwargs.kwargs["system"][0]["text"]
        assert "multiple" in system_text.lower() or "batch" in system_text.lower()

    def test_batch_records_usage(self) -> None:
        mock_anthropic = MagicMock()
        mock_message = MagicMock()
        mock_message.content = [MagicMock(type="text", text='[{"name": "f", "confidence": 50}]')]
        mock_message.usage = MagicMock(
            input_tokens=500, output_tokens=200,
            cache_creation_input_tokens=100, cache_read_input_tokens=50,
        )
        mock_anthropic.messages.create.return_value = mock_message

        client = AnthropicClient(api_key="test")
        client._client = mock_anthropic
        client.analyze_function_batch("prompt")

        assert client.usage.input_tokens == 500
        assert client.usage.output_tokens == 200
        assert client.usage.calls == 1


class TestOpenAIClientBaseUrl:
    @patch("kong.llm.openai_client.openai.OpenAI")
    def test_base_url_passed_to_sdk(self, mock_openai_cls):
        from kong.llm.openai_client import OpenAIClient

        OpenAIClient(
            model="llama3:8b",
            base_url="http://localhost:11434/v1",
            api_key="test",
        )
        mock_openai_cls.assert_called_once_with(
            api_key="test",
            base_url="http://localhost:11434/v1",
            max_retries=5,
        )

    @patch("kong.llm.openai_client.openai.OpenAI")
    def test_base_url_none_by_default(self, mock_openai_cls):
        from kong.llm.openai_client import OpenAIClient

        OpenAIClient(model="gpt-4o", api_key="sk-test")
        mock_openai_cls.assert_called_once_with(
            api_key="sk-test",
            base_url=None,
            max_retries=5,
        )


class TestProviderAwarePricing:
    def test_custom_provider_returns_zero_pricing(self):
        from kong.config import LLMProvider
        from kong.llm.usage import get_pricing

        tier = get_pricing("any-model", provider=LLMProvider.CUSTOM)
        assert tier.input_rate == 0.0
        assert tier.output_rate == 0.0

    def test_known_model_unaffected(self):
        from kong.llm.usage import PRICING_REGISTRY, get_pricing

        tier = get_pricing("gpt-4o")
        assert tier == PRICING_REGISTRY["gpt-4o"]

    def test_unknown_first_party_model_uses_default(self):
        from kong.config import LLMProvider
        from kong.llm.usage import get_pricing

        tier = get_pricing("claude-future-model", provider=LLMProvider.ANTHROPIC)
        assert tier.input_rate == 3.0
        assert tier.output_rate == 15.0

    def test_register_custom_model_adds_zero_pricing(self):
        from kong.llm.usage import PRICING_REGISTRY, get_pricing, register_custom_model

        register_custom_model("my-local-llama")
        try:
            tier = get_pricing("my-local-llama")
            assert tier.input_rate == 0.0
            assert tier.output_rate == 0.0
        finally:
            PRICING_REGISTRY.pop("my-local-llama", None)

    def test_cost_usd_zero_for_registered_custom_model(self):
        from kong.llm.usage import PRICING_REGISTRY, ModelTokenUsage, register_custom_model

        register_custom_model("test-custom-model")
        try:
            mu = ModelTokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
            assert mu.cost_usd("test-custom-model") == 0.0
        finally:
            PRICING_REGISTRY.pop("test-custom-model", None)


class TestCurrentModelPricing:
    """The registry must cover the models Kong actually defaults to and offers."""

    def test_default_model_is_priced(self):
        from kong.llm.usage import is_known_model

        assert is_known_model(DEFAULT_MODEL)

    @pytest.mark.parametrize(
        "model,input_rate,output_rate",
        [
            ("claude-fable-5", 10.0, 50.0),
            ("claude-opus-5", 5.0, 25.0),
            ("claude-opus-4-8", 5.0, 25.0),
            ("claude-opus-4-7", 5.0, 25.0),
            ("claude-opus-4-6", 5.0, 25.0),
            ("claude-sonnet-5", 2.0, 10.0),
            ("claude-sonnet-4-6", 3.0, 15.0),
            ("claude-haiku-4-5", 1.0, 5.0),
        ],
    )
    def test_published_rates(self, model, input_rate, output_rate):
        from kong.llm.usage import get_pricing

        tier = get_pricing(model)
        assert tier.input_rate == input_rate
        assert tier.output_rate == output_rate

    def test_cache_rates_follow_anthropic_multipliers(self):
        from kong.llm.usage import get_pricing

        tier = get_pricing("claude-opus-5")
        assert tier.cache_write_rate == tier.input_rate * 1.25
        assert tier.cache_read_rate == tier.input_rate * 0.10

    def test_unknown_model_is_not_reported_as_known(self):
        from kong.llm.usage import is_known_model

        assert not is_known_model("claude-not-a-real-model")

    def test_unknown_model_warns_once(self, caplog):
        import logging

        from kong.llm.usage import _warned_unknown_models, get_pricing

        _warned_unknown_models.discard("some-unlisted-model")
        try:
            with caplog.at_level(logging.WARNING, logger="kong.llm.usage"):
                get_pricing("some-unlisted-model")
                get_pricing("some-unlisted-model")
            warnings = [r for r in caplog.records if "some-unlisted-model" in r.message]
            assert len(warnings) == 1
        finally:
            _warned_unknown_models.discard("some-unlisted-model")


class TestModelLimits:
    def test_large_context_models_get_a_larger_prompt_cap(self):
        from kong.llm.limits import _DEFAULT_LIMITS, get_model_limits

        limits = get_model_limits("claude-opus-5")
        assert limits.max_prompt_chars > _DEFAULT_LIMITS.max_prompt_chars

    def test_chunk_size_stays_within_the_output_budget(self):
        """Batch size is bound by output tokens, not by the context window."""
        from kong.llm.limits import get_model_limits

        limits = get_model_limits("claude-opus-5")
        assert limits.max_chunk_functions * 130 <= limits.max_output_tokens

    def test_unknown_model_falls_back_to_conservative_limits(self):
        from kong.llm.limits import _DEFAULT_LIMITS, get_model_limits

        assert get_model_limits("who-knows") == _DEFAULT_LIMITS

class TestAnthropicOutputTokenBudget:
    def _mock_batch_client(self):
        mock_anthropic = MagicMock()
        mock_message = MagicMock()
        mock_message.content = [
            MagicMock(type="text", text='[{"name": "f", "confidence": 50}]')
        ]
        mock_message.usage = MagicMock(
            input_tokens=100, output_tokens=50,
            cache_creation_input_tokens=0, cache_read_input_tokens=0,
        )
        mock_anthropic.messages.create.return_value = mock_message
        return mock_anthropic

    def test_batch_defaults_to_the_full_budget(self):
        from kong.llm.client import DEFAULT_BATCH_MAX_TOKENS

        mock_anthropic = self._mock_batch_client()
        client = AnthropicClient(api_key="test")
        client._client = mock_anthropic
        client.analyze_function_batch("prompt")

        kwargs = mock_anthropic.messages.create.call_args.kwargs
        assert kwargs["max_tokens"] == DEFAULT_BATCH_MAX_TOKENS

    def test_batch_honours_an_explicit_budget(self):
        mock_anthropic = self._mock_batch_client()
        client = AnthropicClient(api_key="test")
        client._client = mock_anthropic
        client.analyze_function_batch("prompt", max_tokens=2048)

        kwargs = mock_anthropic.messages.create.call_args.kwargs
        assert kwargs["max_tokens"] == 2048
