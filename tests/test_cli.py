"""Tests for the Kong CLI."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import os

import pytest
from click.testing import CliRunner

import click

from kong.__main__ import (
    _NOT_NEEDED_STR,
    _parse_address_selection,
    cli,
    create_llm_client,
    resolve_provider,
    validate_base_url,
)
from kong.config import LLMConfig, LLMProvider
from kong.db import get_custom_config, save_setup


def _complete_setup(tmp_path, monkeypatch):
    """Mark setup as complete with Anthropic as default."""
    monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
    save_setup(
        enabled=[LLMProvider.ANTHROPIC],
        default=LLMProvider.ANTHROPIC,
    )


def test_analyze_missing_binary(tmp_path, monkeypatch):
    _complete_setup(tmp_path, monkeypatch)
    runner = CliRunner()
    result = runner.invoke(cli, ["analyze", "/nonexistent/binary"])
    assert result.exit_code != 0


def test_analyze_no_ghidra_installed(tmp_path, monkeypatch):
    """When Ghidra is not installed, show install instructions."""
    _complete_setup(tmp_path, monkeypatch)
    binary = tmp_path / "test_binary"
    binary.write_bytes(b"\x00" * 16)

    runner = CliRunner()
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test"}), \
         patch("kong.config.find_ghidra_install", return_value=None), \
         patch("kong.llm.probe.probe_endpoint", return_value=True):
        result = runner.invoke(cli, ["analyze", str(binary)])

    assert result.exit_code != 0
    assert "not installed" in result.output.lower() or "not found" in result.output.lower()
    assert "brew install ghidra" in result.output


def test_setup_wizard_saves_config(tmp_path, monkeypatch):
    """Setup wizard persists provider selection."""
    monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test1234")

    runner = CliRunner()
    result = runner.invoke(cli, ["setup"], input="1\n")

    assert result.exit_code == 0
    assert "sk-ant-" in result.output
    assert "..." in result.output


def test_eval_with_test_data(tmp_path):
    """Run eval against a simple test case."""
    import json

    analysis = {
        "binary": {"name": "test"},
        "stats": {"llm_calls": 5, "duration_seconds": 10.0, "cost_usd": 0.05},
        "functions": [
            {"name": "hash_string", "signature": "uint hash_string(byte *str)", "confidence": 95, "address": "0x1000"},
        ],
    }
    analysis_path = tmp_path / "analysis.json"
    analysis_path.write_text(json.dumps(analysis))

    source = "unsigned int hash_string(const char *s) {\n    return 0;\n}\n"
    source_path = tmp_path / "test.c"
    source_path.write_text(source)

    runner = CliRunner()
    result = runner.invoke(cli, ["eval", str(analysis_path), str(source_path)])
    assert result.exit_code == 0
    assert "hash_string" in result.output
    assert "Symbol Accuracy" in result.output


class TestBannerCustomProvider:
    def test_env_vars_does_not_have_custom(self):
        from kong.banner import _ENV_VARS

        assert LLMProvider.CUSTOM not in _ENV_VARS

    def test_check_api_key_returns_true_for_custom(self):
        from kong.banner import check_api_key

        assert check_api_key(LLMProvider.CUSTOM) is True


class TestCreateLLMClient:
    @patch("kong.llm.openai_client.openai.OpenAI")
    def test_custom_returns_openai_client_with_base_url(self, mock_openai_cls):
        from kong.llm.openai_client import OpenAIClient

        config = LLMConfig(
            provider=LLMProvider.CUSTOM,
            model="llama3:8b",
            base_url="http://localhost:11434/v1",
            api_key="test-key",
        )
        client = create_llm_client(config)
        assert isinstance(client, OpenAIClient)
        kwargs = mock_openai_cls.call_args.kwargs
        assert kwargs["api_key"] == "test-key"
        assert kwargs["base_url"] == "http://localhost:11434/v1"

    @patch("kong.llm.openai_client.openai.OpenAI")
    def test_custom_no_auth_passes_empty_string(self, mock_openai_cls):
        from kong.llm.openai_client import OpenAIClient

        config = LLMConfig(
            provider=LLMProvider.CUSTOM,
            model="llama3:8b",
            base_url="http://localhost:11434/v1",
        )
        client = create_llm_client(config)
        assert isinstance(client, OpenAIClient)
        kwargs = mock_openai_cls.call_args.kwargs
        assert kwargs["api_key"] == _NOT_NEEDED_STR
        assert kwargs["base_url"] == "http://localhost:11434/v1"

    @patch("kong.llm.openai_client.openai.OpenAI")
    def test_openai_returns_openai_client_no_base_url(self, mock_openai_cls):
        from kong.llm.openai_client import OpenAIClient

        config = LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4o")
        client = create_llm_client(config)
        assert isinstance(client, OpenAIClient)
        kwargs = mock_openai_cls.call_args.kwargs
        assert kwargs["api_key"] is None
        assert kwargs["base_url"] is None

    @patch("kong.llm.openai_client.openai.OpenAI")
    def test_zai_uses_its_own_endpoint_and_key(self, mock_openai_cls, monkeypatch):
        from kong.config import ZAI_BASE_URL
        from kong.llm.openai_client import OpenAIClient

        monkeypatch.setenv("ZAI_API_KEY", "zai-test-key")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-not-this-one")

        client = create_llm_client(
            LLMConfig(provider=LLMProvider.ZAI, model="glm-5.3")
        )

        assert isinstance(client, OpenAIClient)
        kwargs = mock_openai_cls.call_args.kwargs
        assert kwargs["api_key"] == "zai-test-key"
        assert kwargs["base_url"] == ZAI_BASE_URL

    @patch("kong.llm.openai_client.openai.OpenAI")
    def test_zai_base_url_can_be_overridden_for_a_coding_plan_key(
        self, mock_openai_cls, monkeypatch,
    ):
        monkeypatch.setenv("ZAI_API_KEY", "zai-test-key")

        create_llm_client(LLMConfig(
            provider=LLMProvider.ZAI,
            model="glm-5.3",
            base_url="https://api.z.ai/api/coding/paas/v4",
        ))

        assert mock_openai_cls.call_args.kwargs["base_url"] == (
            "https://api.z.ai/api/coding/paas/v4"
        )

    @patch("kong.llm.openai_client.openai.OpenAI")
    def test_zai_defaults_to_glm_5_3(self, mock_openai_cls, monkeypatch):
        monkeypatch.setenv("ZAI_API_KEY", "zai-test-key")

        client = create_llm_client(LLMConfig(provider=LLMProvider.ZAI))

        assert client.model == "glm-5.3"

    def test_zai_without_a_key_says_which_one_is_missing(self, monkeypatch):
        monkeypatch.delenv("ZAI_API_KEY", raising=False)

        with pytest.raises(ValueError, match="ZAI_API_KEY"):
            create_llm_client(LLMConfig(provider=LLMProvider.ZAI))

    @patch("kong.llm.client.anthropic.Anthropic")
    def test_anthropic_returns_anthropic_client(self, mock_anthropic_cls):
        from kong.llm.client import AnthropicClient

        config = LLMConfig(provider=LLMProvider.ANTHROPIC, model="claude-opus-4-6")
        client = create_llm_client(config)
        assert isinstance(client, AnthropicClient)


class TestGuiCommand:
    def test_the_browser_interface_is_what_opens(self, monkeypatch):
        """Tk is the fallback now, so plain `kong gui` serves the page."""
        seen = {}

        def fake_launch(initial_binary="", port=0, open_browser=True):
            seen.update(binary=initial_binary, port=port, browser=open_browser)

        monkeypatch.setattr("kong.webui.launch", fake_launch)

        result = CliRunner().invoke(cli, ["gui", "--port", "8899", "--no-browser"])

        assert result.exit_code == 0
        assert seen == {"binary": "", "port": 8899, "browser": False}

    def test_the_scale_flag_reaches_the_window(self, monkeypatch):
        pytest.importorskip("customtkinter")
        from kong.gui.app import UI_SCALE_ENV

        monkeypatch.delenv(UI_SCALE_ENV, raising=False)
        seen = {}

        def fake_launch(initial_binary=""):
            seen["scale"] = os.environ.get(UI_SCALE_ENV)

        monkeypatch.setattr("kong.gui.app.launch", fake_launch)

        result = CliRunner().invoke(cli, ["gui", "--tk", "--scale", "1.25"])

        assert result.exit_code == 0
        assert seen["scale"] == "1.25"

    def test_an_impossible_scale_is_refused(self):
        result = CliRunner().invoke(cli, ["gui", "--tk", "--scale", "40"])

        assert result.exit_code != 0


class TestApiKeyResolution:
    """A key can live in the environment or in Kong's own config."""

    def test_the_environment_is_used_when_there_is_nothing_saved(
        self, tmp_path, monkeypatch,
    ):
        from kong.banner import resolve_api_key

        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
        monkeypatch.setenv("ZAI_API_KEY", "from-the-environment")

        assert resolve_api_key(LLMProvider.ZAI) == "from-the-environment"

    def test_a_saved_key_is_used_when_the_environment_has_none(
        self, tmp_path, monkeypatch,
    ):
        from kong.banner import resolve_api_key
        from kong.db import save_api_key

        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
        monkeypatch.delenv("ZAI_API_KEY", raising=False)
        save_api_key(LLMProvider.ZAI, "from-the-config")

        assert resolve_api_key(LLMProvider.ZAI) == "from-the-config"

    def test_the_environment_wins_over_a_saved_key(self, tmp_path, monkeypatch):
        from kong.banner import resolve_api_key
        from kong.db import save_api_key

        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
        save_api_key(LLMProvider.ZAI, "from-the-config")
        monkeypatch.setenv("ZAI_API_KEY", "from-the-environment")

        assert resolve_api_key(LLMProvider.ZAI) == "from-the-environment"

    def test_an_explicit_key_wins_over_both(self, tmp_path, monkeypatch):
        from kong.banner import resolve_api_key
        from kong.db import save_api_key

        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
        save_api_key(LLMProvider.ZAI, "from-the-config")
        monkeypatch.setenv("ZAI_API_KEY", "from-the-environment")

        assert resolve_api_key(LLMProvider.ZAI, "typed-in") == "typed-in"

    def test_a_saved_key_counts_as_configured(self, tmp_path, monkeypatch):
        from kong.banner import check_api_key
        from kong.db import save_api_key

        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        save_api_key(LLMProvider.ANTHROPIC, "sk-ant-saved")

        assert check_api_key(LLMProvider.ANTHROPIC)

    @patch("kong.llm.client.anthropic.Anthropic")
    def test_a_key_saved_in_the_gui_reaches_the_anthropic_client(
        self, mock_anthropic_cls, tmp_path, monkeypatch,
    ):
        from kong.db import save_api_key

        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        save_api_key(LLMProvider.ANTHROPIC, "sk-ant-saved")

        create_llm_client(
            LLMConfig(provider=LLMProvider.ANTHROPIC, model="claude-opus-5")
        )

        assert mock_anthropic_cls.call_args.kwargs["api_key"] == "sk-ant-saved"

    @patch("kong.llm.openai_client.openai.OpenAI")
    def test_a_key_saved_in_the_gui_reaches_zai(
        self, mock_openai_cls, tmp_path, monkeypatch,
    ):
        from kong.db import save_api_key

        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
        monkeypatch.delenv("ZAI_API_KEY", raising=False)
        save_api_key(LLMProvider.ZAI, "zai-saved")

        create_llm_client(LLMConfig(provider=LLMProvider.ZAI))

        assert mock_openai_cls.call_args.kwargs["api_key"] == "zai-saved"


class TestResolveProviderCustom:
    def test_base_url_implies_custom(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
        provider = resolve_provider(base_url="http://localhost:8000/v1")
        assert provider is LLMProvider.CUSTOM

    def test_explicit_custom_provider(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
        provider = resolve_provider(cli_override="custom")
        assert provider is LLMProvider.CUSTOM

    def test_custom_skipped_in_fallback(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        save_setup(
            enabled=[LLMProvider.CUSTOM, LLMProvider.ANTHROPIC],
            default=LLMProvider.ANTHROPIC,
        )
        provider = resolve_provider()
        assert provider is LLMProvider.ANTHROPIC


class TestValidateBaseUrl:
    def test_valid_http_url(self):
        assert validate_base_url("http://localhost:8000/v1") == "http://localhost:8000/v1"

    def test_valid_https_url(self):
        assert validate_base_url("https://api.together.xyz/v1") == "https://api.together.xyz/v1"

    def test_strips_trailing_slash(self):
        assert validate_base_url("http://localhost:8000/v1/") == "http://localhost:8000/v1"

    def test_rejects_missing_scheme(self):
        import pytest

        with pytest.raises(click.BadParameter):
            validate_base_url("localhost:8000/v1")


class TestSetupWizardCustom:
    @patch("kong.llm.probe.probe_endpoint", return_value=False)
    def test_setup_custom_provider(self, mock_probe, tmp_path, monkeypatch):
        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["setup"],
            input="3\nhttp://localhost:11434/v1\nllama3:8b\n\n32000\n20\n4096\n",
        )
        assert result.exit_code == 0
        from kong.db import get_default_provider

        assert get_default_provider() == LLMProvider.CUSTOM
        cfg = get_custom_config()
        assert cfg["custom_base_url"] == "http://localhost:11434/v1"
        assert cfg["custom_model"] == "llama3:8b"
        assert cfg["custom_max_prompt_chars"] == "32000"

    def test_setup_option_4_is_anthropic_plus_openai(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test1234")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test1234")
        runner = CliRunner()
        result = runner.invoke(cli, ["setup"], input="4\n1\n")
        assert result.exit_code == 0
        from kong.db import get_enabled_providers

        enabled = get_enabled_providers()
        assert LLMProvider.ANTHROPIC in enabled
        assert LLMProvider.OPENAI in enabled
        assert LLMProvider.CUSTOM not in enabled


class TestCustomProviderIntegration:
    @patch("kong.llm.probe.probe_endpoint", return_value=True)
    @patch("kong.llm.openai_client.openai.OpenAI")
    @patch("kong.config.find_ghidra_install", return_value=None)
    def test_analyze_with_base_url_flag(
        self, mock_ghidra, mock_openai_cls, mock_probe, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
        save_setup(enabled=[LLMProvider.CUSTOM], default=LLMProvider.CUSTOM)

        binary = tmp_path / "test_binary"
        binary.write_bytes(b"\x00" * 16)

        runner = CliRunner()
        result = runner.invoke(cli, [
            "analyze", str(binary),
            "--base-url", "http://localhost:11434/v1",
            "--model", "llama3:8b",
            "--headless",
        ])

        assert "not installed" in result.output.lower() or "not found" in result.output.lower()


def _fake_endpoint(monkeypatch, models_payload, props_payload=None):
    """Serve canned llama.cpp payloads to the endpoint discovery code."""
    import io
    import json as _json
    import urllib.error
    import urllib.request

    class _Body(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.close()
            return False

    def fake_urlopen(request, timeout=None):
        url = request.full_url
        if url.endswith("/models"):
            return _Body(_json.dumps(models_payload).encode())
        if url.endswith("/props") and props_payload is not None:
            return _Body(_json.dumps(props_payload).encode())
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)


_LLAMA_MODELS = {
    "data": [
        {
            "id": "qwen2.5-coder-7b",
            "meta": {"n_ctx_train": 32768, "n_params": 7_615_616_512},
        }
    ]
}
_LLAMA_PROPS = {
    "default_generation_settings": {"n_ctx": 16384},
    "total_slots": 1,
    "build_info": "b4321",
}


def test_models_lists_what_the_endpoint_serves(tmp_path, monkeypatch):
    monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
    _fake_endpoint(monkeypatch, _LLAMA_MODELS, _LLAMA_PROPS)

    result = CliRunner().invoke(
        cli, ["models", "--base-url", "http://127.0.0.1:8080/v1"]
    )

    assert result.exit_code == 0
    assert "qwen2.5-coder-7b" in result.output
    assert "7.6B" in result.output
    assert "16,384" in result.output  # runtime context, not the trained 32768


def test_models_suggests_limits_sized_for_the_window(tmp_path, monkeypatch):
    monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
    _fake_endpoint(monkeypatch, _LLAMA_MODELS, _LLAMA_PROPS)

    result = CliRunner().invoke(
        cli, ["models", "--base-url", "http://127.0.0.1:8080/v1"]
    )

    assert "--max-output-tokens   2048" in result.output
    assert "--max-chunk-functions 7" in result.output
    assert "--max-prompt-chars    41208" in result.output


def test_models_warns_when_the_window_is_unknown(tmp_path, monkeypatch):
    monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
    _fake_endpoint(monkeypatch, {"data": [{"id": "some-model"}]})

    result = CliRunner().invoke(
        cli, ["models", "--base-url", "http://127.0.0.1:8080/v1"]
    )

    assert result.exit_code == 0
    assert "does not report its context window" in result.output


def test_models_reports_an_unreachable_endpoint(tmp_path, monkeypatch):
    import urllib.error
    import urllib.request

    monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        MagicMock(side_effect=urllib.error.URLError("connection refused")),
    )

    result = CliRunner().invoke(
        cli, ["models", "--base-url", "http://127.0.0.1:9999/v1"]
    )

    assert result.exit_code == 1
    assert "Could not reach" in result.output


def test_models_needs_an_endpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))

    result = CliRunner().invoke(cli, ["models"])

    assert result.exit_code == 1
    assert "No endpoint configured" in result.output


class TestZaiProvider:
    """Z.ai is a hosted provider, not a hand-configured custom endpoint."""

    def test_the_model_limits_are_the_ones_of_a_1m_context(self):
        from kong.llm.limits import get_model_limits

        assert get_model_limits("glm-5.3").max_prompt_chars == 900_000
        assert get_model_limits("glm-5.3-flash").max_prompt_chars == 900_000

    def test_the_batch_leaves_room_for_the_reasoning_in_front_of_the_answer(self):
        """GLM charges its thinking to the budget the JSON has to fit in too."""
        from kong.llm.limits import get_model_limits

        for model in ("glm-5.3", "glm-5.3-flash"):
            limits = get_model_limits(model)
            # ~130 output tokens of JSON per function, so the answer alone must
            # not come close to the budget: at 120/16k it was 95% of it, and
            # every chunk truncated before the model reached the answer.
            answer_tokens = limits.max_chunk_functions * 130
            assert answer_tokens < limits.max_output_tokens // 3

    def test_it_does_not_share_the_anthropic_numbers(self):
        """Same context window, different behaviour in the completion budget."""
        from kong.llm.limits import get_model_limits

        glm = get_model_limits("glm-5.3")
        claude = get_model_limits("claude-opus-5")

        assert glm.max_prompt_chars == claude.max_prompt_chars
        assert glm.max_chunk_functions < claude.max_chunk_functions
        assert glm.max_output_tokens > claude.max_output_tokens

    def test_glm_5_3_has_published_pricing(self):
        from kong.llm.usage import is_known_model

        assert is_known_model("glm-5.3")

    def test_a_missing_model_listing_is_not_a_dead_endpoint(self, monkeypatch):
        """Z.ai documents chat/completions, not /models."""
        import httpx
        import openai

        from kong.llm.probe import probe_endpoint

        def not_found(self):
            raise openai.NotFoundError(
                "no such endpoint",
                response=httpx.Response(
                    404, request=httpx.Request("GET", "https://api.z.ai/models")
                ),
                body=None,
            )

        monkeypatch.setattr(openai.resources.models.Models, "list", not_found)

        assert probe_endpoint(
            LLMConfig(provider=LLMProvider.ZAI, api_key="zai-test-key")
        )

    def test_a_run_without_a_key_does_not_start(self, monkeypatch):
        from kong.llm.probe import probe_endpoint

        monkeypatch.delenv("ZAI_API_KEY", raising=False)

        assert not probe_endpoint(LLMConfig(provider=LLMProvider.ZAI))

    def test_the_provider_reaches_the_config(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
        monkeypatch.setenv("ZAI_API_KEY", "zai-test-key")
        save_setup(enabled=[LLMProvider.ZAI], default=LLMProvider.ZAI)

        captured = {}

        def fake_probe(llm_config):
            captured["llm"] = llm_config
            return True

        monkeypatch.setattr("kong.llm.probe.probe_endpoint", fake_probe)
        monkeypatch.setattr("kong.config.find_ghidra_install", lambda: None)

        binary = tmp_path / "target.bin"
        binary.write_bytes(b"\x7fELF")

        CliRunner().invoke(cli, [
            "analyze", str(binary), "--provider", "zai",
            "--model", "glm-5.3", "--headless",
        ])

        assert captured["llm"].provider is LLMProvider.ZAI
        assert captured["llm"].model == "glm-5.3"


class TestTwoPassFlags:
    """The draft flags have to survive the trip from argv into LLMConfig."""

    def _run(self, tmp_path, monkeypatch, *flags):
        _complete_setup(tmp_path, monkeypatch)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test1234")

        captured = {}

        def fake_probe(llm_config):
            captured["llm"] = llm_config
            return True

        monkeypatch.setattr("kong.llm.probe.probe_endpoint", fake_probe)
        monkeypatch.setattr("kong.config.find_ghidra_install", lambda: None)

        binary = tmp_path / "target.bin"
        binary.write_bytes(b"\x7fELF")

        CliRunner().invoke(cli, ["analyze", str(binary), "--headless", *flags])
        return captured["llm"]

    def test_the_draft_model_reaches_the_config(self, tmp_path, monkeypatch):
        llm = self._run(
            tmp_path, monkeypatch,
            "--model", "claude-opus-5",
            "--draft-model", "claude-haiku-4-5",
        )

        assert llm.draft_model == "claude-haiku-4-5"
        assert llm.model == "claude-opus-5"

    def test_the_threshold_reaches_the_config(self, tmp_path, monkeypatch):
        llm = self._run(
            tmp_path, monkeypatch,
            "--draft-model", "claude-haiku-4-5",
            "--refine-below", "60",
        )

        assert llm.refine_below == 60

    def test_the_threshold_defaults_to_the_high_confidence_bar(
        self, tmp_path, monkeypatch,
    ):
        from kong.agent.refinement import DEFAULT_REFINE_BELOW

        llm = self._run(tmp_path, monkeypatch, "--draft-model", "claude-haiku-4-5")

        assert llm.refine_below == DEFAULT_REFINE_BELOW

    def test_a_single_pass_run_carries_no_draft_model(self, tmp_path, monkeypatch):
        llm = self._run(tmp_path, monkeypatch, "--model", "claude-opus-5")

        assert llm.draft_model is None

    def test_a_threshold_outside_the_scale_is_refused(self, tmp_path, monkeypatch):
        _complete_setup(tmp_path, monkeypatch)
        binary = tmp_path / "target.bin"
        binary.write_bytes(b"\x7fELF")

        result = CliRunner().invoke(cli, [
            "analyze", str(binary), "--refine-below", "150", "--headless",
        ])

        assert result.exit_code != 0

    def test_a_batch_size_reaches_the_config_on_a_hosted_provider(
        self, tmp_path, monkeypatch,
    ):
        """Not custom-only any more: a hosted endpoint can be told to go slower."""
        llm = self._run(tmp_path, monkeypatch, "--max-chunk-functions", "20")

        assert llm.provider is LLMProvider.ANTHROPIC
        assert llm.max_chunk_functions == 20


class TestStageFlag:
    """--stage splits the draft from the pass that finishes it."""

    def _invoke(self, tmp_path, monkeypatch, *flags):
        _complete_setup(tmp_path, monkeypatch)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test1234")
        monkeypatch.setattr("kong.llm.probe.probe_endpoint", lambda config: True)
        monkeypatch.setattr("kong.config.find_ghidra_install", lambda: None)

        binary = tmp_path / "target.bin"
        binary.write_bytes(b"\x7fELF")
        return CliRunner().invoke(
            cli, ["analyze", str(binary), "--headless", *flags]
        )

    def test_a_draft_run_says_what_it_will_not_do(self, tmp_path, monkeypatch):
        result = self._invoke(
            tmp_path, monkeypatch,
            "--stage", "draft",
            "--model", "claude-opus-5",
            "--draft-model", "claude-haiku-4-5",
        )

        assert "Draft stage" in result.output
        assert "--stage finish" in result.output

    def test_a_draft_run_without_a_draft_model_is_allowed_and_explained(
        self, tmp_path, monkeypatch,
    ):
        result = self._invoke(tmp_path, monkeypatch, "--stage", "draft")

        assert "--stage draft without --draft-model" in result.output

    def test_a_finish_run_says_where_it_reads_the_draft_from(
        self, tmp_path, monkeypatch,
    ):
        from kong.state.persistence import state_path

        out = tmp_path / "out"
        out.mkdir()
        state_path(out).write_text("{}", encoding="utf-8")

        result = self._invoke(
            tmp_path, monkeypatch, "--stage", "finish", "--output", str(out),
        )

        assert "Finishing pass" in result.output

    def test_finishing_without_a_draft_names_the_directory_it_looked_in(
        self, tmp_path, monkeypatch,
    ):
        """Checked before Ghidra opens the binary, not a minute later."""
        result = self._invoke(
            tmp_path, monkeypatch,
            "--stage", "finish", "--output", str(tmp_path / "nowhere"),
        )

        assert result.exit_code != 0
        assert "No saved analysis" in result.output

    def test_finishing_a_draft_that_fresh_would_delete_is_refused(
        self, tmp_path, monkeypatch,
    ):
        result = self._invoke(tmp_path, monkeypatch, "--stage", "finish", "--fresh")

        assert result.exit_code != 0
        assert "throws away" in result.output

    def test_an_unknown_stage_is_refused(self, tmp_path, monkeypatch):
        result = self._invoke(tmp_path, monkeypatch, "--stage", "halfway")

        assert result.exit_code != 0


class TestTranspileSelectionParsing:
    """--transpile-only takes what a reader has to hand.

    The point is that the address list a call-graph tool prints can be pasted
    in without being reformatted first.
    """

    def test_an_inline_list(self):
        assert _parse_address_selection("0x401000,0x401040") == {0x401000, 0x401040}

    def test_bare_hex_needs_no_prefix_when_it_carries_one(self):
        assert _parse_address_selection("0x00401000") == {0x401000}

    def test_a_two_column_listing_pastes_in_unchanged(self, tmp_path):
        listing = tmp_path / "lot.txt"
        listing.write_text(
            "0x0043a0d0  save_config_to_fa18_cfg\n"
            "0x00439fc0  load_config_from_fa18_cfg\n",
            encoding="utf-8",
        )

        assert _parse_address_selection(str(listing)) == {0x43A0D0, 0x439FC0}

    def test_comments_and_blank_lines_are_skipped(self, tmp_path):
        listing = tmp_path / "lot.txt"
        listing.write_text("# le sous-systeme pack\n\n0x00436430\n", encoding="utf-8")

        assert _parse_address_selection(str(listing)) == {0x436430}

    def test_a_non_address_is_rejected_by_name(self):
        with pytest.raises(click.BadParameter, match="not an address"):
            _parse_address_selection("save_config_to_fa18_cfg")

    def test_an_empty_file_is_rejected(self, tmp_path):
        empty = tmp_path / "empty.txt"
        empty.write_text("# rien\n", encoding="utf-8")

        with pytest.raises(click.BadParameter, match="no addresses"):
            _parse_address_selection(str(empty))

    def test_a_comma_inside_a_comment_is_prose_not_a_separator(self, tmp_path):
        listing = tmp_path / "lot.txt"
        listing.write_text(
            "# pack subsystem, exported from the call graph\n0x00436430\n",
            encoding="utf-8",
        )

        assert _parse_address_selection(str(listing)) == {0x436430}

    def test_a_trailing_comma_is_not_an_empty_address(self):
        assert _parse_address_selection("0x401000, 0x401040,") == {0x401000, 0x401040}


class TestTranspileSelectionGuards:
    @patch("kong.__main__.is_setup_complete", return_value=True)
    def test_selecting_without_asking_for_a_translation_is_an_error(
        self, _setup, tmp_path,
    ):
        binary = tmp_path / "t.bin"
        binary.write_bytes(b"MZ")

        result = CliRunner().invoke(cli, [
            "analyze", str(binary), "--transpile-only", "0x401000",
            "--format", "source",
        ])

        assert result.exit_code == 1
        # rich hard-wraps the console, so compare on normalised whitespace.
        assert "no translation was asked for" in " ".join(result.output.split())
