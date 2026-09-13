"""Tests for the SQLite configuration store."""

from __future__ import annotations

from kong.config import LLMProvider
from kong.db import (
    api_key_config_key,
    get_custom_config,
    get_default_provider,
    get_enabled_providers,
    is_setup_complete,
    get_saved_api_key,
    read_config,
    save_api_key,
    save_setup,
    write_config,
)


class TestConfigReadWrite:
    def test_read_missing_key_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
        assert read_config("nonexistent") is None

    def test_write_then_read(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
        write_config("foo", "bar")
        assert read_config("foo") == "bar"

    def test_upsert_overwrites(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
        write_config("key", "v1")
        write_config("key", "v2")
        assert read_config("key") == "v2"

    def test_creates_config_dir(self, tmp_path, monkeypatch):
        nested = tmp_path / "deep" / "nested"
        monkeypatch.setenv("KONG_CONFIG_DIR", str(nested))
        write_config("x", "y")
        assert nested.exists()
        assert read_config("x") == "y"


class TestSetupHelpers:
    def test_not_setup_by_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
        assert not is_setup_complete()

    def test_save_setup_marks_complete(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
        save_setup(
            enabled=[LLMProvider.ANTHROPIC],
            default=LLMProvider.ANTHROPIC,
        )
        assert is_setup_complete()

    def test_get_default_provider(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
        save_setup(
            enabled=[LLMProvider.ANTHROPIC, LLMProvider.OPENAI],
            default=LLMProvider.OPENAI,
        )
        assert get_default_provider() is LLMProvider.OPENAI

    def test_get_default_provider_returns_none_before_setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
        assert get_default_provider() is None

    def test_get_enabled_providers(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
        save_setup(
            enabled=[LLMProvider.ANTHROPIC, LLMProvider.OPENAI],
            default=LLMProvider.ANTHROPIC,
        )
        assert get_enabled_providers() == [LLMProvider.ANTHROPIC, LLMProvider.OPENAI]

    def test_get_enabled_providers_empty_before_setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
        assert get_enabled_providers() == []

    def test_save_setup_overwrites_previous(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
        save_setup(
            enabled=[LLMProvider.ANTHROPIC],
            default=LLMProvider.ANTHROPIC,
        )
        save_setup(
            enabled=[LLMProvider.OPENAI],
            default=LLMProvider.OPENAI,
        )
        assert get_default_provider() is LLMProvider.OPENAI
        assert get_enabled_providers() == [LLMProvider.OPENAI]


class TestCustomConfig:
    def test_save_with_custom_config(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
        save_setup(
            enabled=[LLMProvider.CUSTOM],
            default=LLMProvider.CUSTOM,
            custom_config={
                "custom_base_url": "http://localhost:11434/v1",
                "custom_model": "llama3:8b",
                "custom_api_key": "",
                "custom_max_prompt_chars": "32000",
                "custom_max_chunk_functions": "20",
                "custom_max_output_tokens": "4096",
            },
        )
        assert get_default_provider() == LLMProvider.CUSTOM
        assert LLMProvider.CUSTOM in get_enabled_providers()

    def test_get_custom_config(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
        save_setup(
            enabled=[LLMProvider.CUSTOM],
            default=LLMProvider.CUSTOM,
            custom_config={
                "custom_base_url": "http://localhost:8000/v1",
                "custom_model": "mistral",
                "custom_api_key": "sk-test",
                "custom_max_prompt_chars": "50000",
                "custom_max_chunk_functions": "30",
                "custom_max_output_tokens": "8192",
            },
        )
        cfg = get_custom_config()
        assert cfg["custom_base_url"] == "http://localhost:8000/v1"
        assert cfg["custom_model"] == "mistral"
        assert cfg["custom_api_key"] == "sk-test"
        assert cfg["custom_max_prompt_chars"] == "50000"

    def test_save_without_custom_config(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
        save_setup(
            enabled=[LLMProvider.ANTHROPIC],
            default=LLMProvider.ANTHROPIC,
        )
        assert get_default_provider() == LLMProvider.ANTHROPIC
        cfg = get_custom_config()
        assert cfg == {}


class TestApiKeys:
    def test_no_key_saved_is_none(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))

        assert get_saved_api_key(LLMProvider.ZAI) is None

    def test_a_saved_key_comes_back(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))

        save_api_key(LLMProvider.ZAI, "zai-secret")

        assert get_saved_api_key(LLMProvider.ZAI) == "zai-secret"

    def test_keys_are_kept_per_provider(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))

        save_api_key(LLMProvider.ZAI, "zai-secret")
        save_api_key(LLMProvider.ANTHROPIC, "sk-ant-secret")

        assert get_saved_api_key(LLMProvider.ZAI) == "zai-secret"
        assert get_saved_api_key(LLMProvider.ANTHROPIC) == "sk-ant-secret"

    def test_saving_an_empty_key_forgets_it(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
        save_api_key(LLMProvider.ZAI, "zai-secret")

        save_api_key(LLMProvider.ZAI, "   ")

        assert get_saved_api_key(LLMProvider.ZAI) is None

    def test_surrounding_spaces_are_not_part_of_the_key(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))

        save_api_key(LLMProvider.ZAI, "  zai-secret\n")

        assert get_saved_api_key(LLMProvider.ZAI) == "zai-secret"

    def test_the_custom_key_keeps_the_name_setup_already_writes(
        self, tmp_path, monkeypatch,
    ):
        """kong setup has written custom_api_key since before this existed."""
        monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path))
        write_config("custom_api_key", "from-the-wizard")

        assert api_key_config_key(LLMProvider.CUSTOM) == "custom_api_key"
        assert get_saved_api_key(LLMProvider.CUSTOM) == "from-the-wizard"
        assert get_custom_config()["custom_api_key"] == "from-the-wizard"
