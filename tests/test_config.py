from __future__ import annotations

import tempfile
from dataclasses import MISSING, fields
from pathlib import Path

from kong.config import GhidraConfig, LLMConfig, LLMProvider


class TestLLMProviderCustom:
    def test_custom_variant_exists(self):
        assert LLMProvider.CUSTOM.value == "custom"

    def test_custom_display_name(self):
        assert LLMProvider.CUSTOM.display_name == "Custom"

    def test_existing_display_names_unchanged(self):
        assert LLMProvider.ANTHROPIC.display_name == "Anthropic"
        assert LLMProvider.OPENAI.display_name == "OpenAI"


class TestLLMConfigCustomFields:
    def test_defaults_are_none(self):
        cfg = LLMConfig()
        assert cfg.base_url is None
        assert cfg.max_prompt_chars is None
        assert cfg.max_chunk_functions is None
        assert cfg.max_output_tokens is None

    def test_custom_config_with_all_fields(self):
        cfg = LLMConfig(
            provider=LLMProvider.CUSTOM,
            model="llama3:8b",
            base_url="http://localhost:11434/v1",
            max_prompt_chars=32000,
            max_chunk_functions=20,
            max_output_tokens=4096,
        )
        assert cfg.provider is LLMProvider.CUSTOM
        assert cfg.base_url == "http://localhost:11434/v1"
        assert cfg.max_prompt_chars == 32000


class TestGhidraConfigPaths:
    def test_project_dir_is_under_the_platform_temp_dir(self):
        cfg = GhidraConfig(install_dir="/somewhere")
        assert Path(cfg.project_dir).parent == Path(tempfile.gettempdir())

    def test_project_dir_default_is_computed_not_hardcoded(self):
        """A literal /tmp default is not a usable path on Windows."""
        field_def = fields(GhidraConfig)[1]
        assert field_def.name == "project_dir"
        assert field_def.default is MISSING
        assert field_def.default_factory is not MISSING

    def test_default_is_stable_across_instances(self):
        assert (
            GhidraConfig(install_dir="/a").project_dir
            == GhidraConfig(install_dir="/b").project_dir
        )
