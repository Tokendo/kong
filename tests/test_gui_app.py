"""Tests for the customtkinter window.

Skipped wherever Tk cannot open a display (headless CI, a Python built
without Tk bindings). The logic worth testing without a display lives in
kong.gui.controller and is covered by tests/test_gui_controller.py.

customtkinter widgets answer cget() but not widget["option"]: tkinter binds
__getitem__ to its own cget at class definition, so it never reaches the
override. Read widget options with cget() below.
"""

from __future__ import annotations

import gc

import pytest

tk = pytest.importorskip("tkinter")
ctk = pytest.importorskip("customtkinter")

from kong.agent.events import Event, EventType  # noqa: E402
from kong.config import LLMProvider  # noqa: E402


@pytest.fixture(autouse=True)
def _provider_keys(monkeypatch, tmp_path):
    """A machine with the API keys configured, and its own config store.

    Without the keys every settings check would fail on the missing key rather
    than on what the test is about; the tests that care delete them again. The
    config directory is redirected so saving a key never touches the one
    belonging to whoever is running the tests.
    """
    monkeypatch.setenv("KONG_CONFIG_DIR", str(tmp_path / "kong-config"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test1234")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test1234")
    monkeypatch.setenv("ZAI_API_KEY", "zai-test1234")


@pytest.fixture(scope="session")
def tk_root():
    """One interpreter for the whole session: repeated Tk() is fragile."""
    try:
        root = ctk.CTk()
    except tk.TclError as exc:
        pytest.skip(f"no display available: {exc}")
    root.withdraw()
    try:
        yield root
    finally:
        root.destroy()


@pytest.fixture
def window(tk_root):
    from kong.gui.app import KongWindow

    top = ctk.CTkToplevel(tk_root)
    top.withdraw()
    win = KongWindow(top)
    try:
        yield win
    finally:
        # Drop the Tk variables here, on the main thread, while the
        # interpreter is still alive. Left to the garbage collector they may
        # be finalized from a worker thread of another test, where tkinter
        # raises "main thread is not in main loop". Variables held in a
        # container count too, so clear those as well.
        for name, value in list(vars(win).items()):
            if isinstance(value, tk.Variable):
                delattr(win, name)
            elif isinstance(value, dict) and any(
                isinstance(v, tk.Variable) for v in value.values()
            ):
                value.clear()
            elif isinstance(value, list) and any(
                isinstance(v, tk.Variable) for v in value
            ):
                value.clear()
        top.destroy()
        # The widgets still hold their variables, and customtkinter's trackers
        # still hold the widgets. Collecting here runs those finalizers on the
        # main thread, rather than in whichever later test happens to allocate.
        gc.collect()


class TestRenderingOnEveryDesktop:
    """customtkinter asks for fonts and sizes that a stock Ubuntu does not have."""

    def _families(self, monkeypatch, platform, families):
        from kong.gui import app

        monkeypatch.setattr(app.sys, "platform", platform)
        monkeypatch.setattr(app.tkfont, "families", lambda widget=None: families)

    def test_an_installed_family_is_preferred_over_roboto(
        self, tk_root, monkeypatch,
    ):
        from kong.gui import app

        self._families(monkeypatch, "linux", ("DejaVu Sans", "Liberation Serif"))

        assert app._font_family(tk_root) == "DejaVu Sans"

    def test_the_desktop_font_wins_when_there_is_one(self, tk_root, monkeypatch):
        from kong.gui import app

        self._families(monkeypatch, "linux", ("DejaVu Sans", "Ubuntu", "Roboto"))

        assert app._font_family(tk_root) == "Ubuntu"

    def test_roboto_is_still_used_where_it_exists(self, tk_root, monkeypatch):
        from kong.gui import app

        self._families(monkeypatch, "win32", ("Roboto", "Courier"))

        assert app._font_family(tk_root) == "Roboto"

    def test_an_unknown_platform_falls_back_to_what_tk_uses(
        self, tk_root, monkeypatch,
    ):
        from kong.gui import app

        self._families(monkeypatch, "sunos5", ("Some Font",))

        assert app._font_family(tk_root)  # never empty, never a missing family

    def test_the_window_is_built_with_a_font_this_system_has(self, window):
        available = {name.lower() for name in tk.font.families(window)}

        assert window.font_family.lower() in available

    def test_the_table_is_drawn_in_the_same_font(self, window):
        from tkinter import ttk

        style = ttk.Style()
        spec = str(style.lookup("Kong.Treeview", "font"))

        assert window.font_family in spec

    def test_row_height_follows_the_font_rather_than_a_constant(self, window):
        from tkinter import font as tkfont
        from tkinter import ttk

        style = ttk.Style()
        row_height = int(style.configure("Kong.Treeview", "rowheight"))
        line = tkfont.Font(
            family=window.font_family, size=ctk.ThemeManager.theme["CTkFont"]["size"],
        ).metrics("linespace")

        assert row_height > line

    def test_a_small_screen_gets_a_window_that_fits_on_it(self):
        from kong.gui.app import fit_to_screen

        width, height = fit_to_screen(1366, 768)

        assert width <= 1366 * 0.92
        assert height <= 768 * 0.92

    def test_a_large_screen_keeps_the_intended_size(self):
        from kong.gui.app import fit_to_screen

        assert fit_to_screen(2560, 1440) == (1080, 760)

    def test_a_tiny_screen_still_gets_a_usable_window(self):
        from kong.gui.app import fit_to_screen

        width, height = fit_to_screen(200, 100)

        assert width >= 320
        assert height >= 240


class TestScaleOverride:
    """KONG_UI_SCALE is the way out of a desktop Tk misreads."""

    def test_nothing_set_means_nothing_forced(self, monkeypatch):
        from kong.gui.app import UI_SCALE_ENV, ui_scale_override

        monkeypatch.delenv(UI_SCALE_ENV, raising=False)

        assert ui_scale_override() is None

    def test_a_factor_is_read(self, monkeypatch):
        from kong.gui.app import UI_SCALE_ENV, ui_scale_override

        monkeypatch.setenv(UI_SCALE_ENV, "1.5")

        assert ui_scale_override() == 1.5

    def test_a_pinned_scale_of_one_is_honoured(self, monkeypatch):
        from kong.gui.app import UI_SCALE_ENV, ui_scale_override

        monkeypatch.setenv(UI_SCALE_ENV, " 1 ")

        assert ui_scale_override() == 1.0

    def test_nonsense_is_ignored_rather_than_crashing_the_window(
        self, monkeypatch,
    ):
        from kong.gui.app import UI_SCALE_ENV, ui_scale_override

        monkeypatch.setenv(UI_SCALE_ENV, "big")

        assert ui_scale_override() is None

    def test_an_absurd_factor_is_ignored(self, monkeypatch):
        from kong.gui.app import UI_SCALE_ENV, ui_scale_override

        monkeypatch.setenv(UI_SCALE_ENV, "40")

        assert ui_scale_override() is None


class TestWindowSetup:
    def test_output_dir_defaults_from_the_binary_name(self, window):
        from kong.gui.app import KongWindow

        default = KongWindow._default_output_dir("/tmp/liblzma.so.5.4.1")
        assert default.endswith("kong_output_liblzma.so.5.4")

    def test_custom_fields_start_disabled(self, window):
        assert str(window.base_url_entry.cget("state")) == "disabled"
        assert all(str(e.cget("state")) == "disabled" for e in window.limit_entries)

    def test_choosing_the_custom_provider_enables_its_fields(self, window):
        window.provider_var.set(LLMProvider.CUSTOM.value)
        window._sync_provider_fields()
        assert str(window.base_url_entry.cget("state")) == "normal"
        assert all(str(e.cget("state")) == "normal" for e in window.limit_entries)


class TestOutputFormats:
    def test_defaults_match_the_cli(self, window):
        assert window.selected_formats() == ["source", "json"]

    def test_every_format_has_a_checkbox(self, window):
        from kong.gui.app import OUTPUT_FORMATS

        assert [key for key, _ in OUTPUT_FORMATS] == [
            "source", "json", "python", "csharp",
        ]

    def test_no_format_is_offered_that_the_export_ignores(self, window):
        """'ghidra' was a checkbox for a file the export never wrote."""
        from kong.gui.app import OUTPUT_FORMATS

        assert "ghidra" not in {key for key, _ in OUTPUT_FORMATS}

    def test_the_writeback_is_still_announced(self, window):
        """Dropping the checkbox must not read as a lost feature."""
        assert "back into Ghidra" in window.format_hint.cget("text")

    def test_selection_is_read_in_catalogue_order(self, window):
        window.format_vars["csharp"].set(True)
        window.format_vars["source"].set(False)

        assert window.selected_formats() == ["json", "csharp"]

    def test_the_second_pass_is_flagged_when_reconstructing(self, window):
        assert "second LLM pass" not in window.format_hint.cget("text")

        window.format_vars["python"].set(True)
        window._sync_format_hint()

        assert "second LLM pass" in window.format_hint.cget("text")

    def test_the_flag_clears_when_reconstruction_is_turned_off(self, window):
        window.format_vars["python"].set(True)
        window._sync_format_hint()
        window.format_vars["python"].set(False)
        window._sync_format_hint()

        assert "second LLM pass" not in window.format_hint.cget("text")

    def test_formats_reach_the_settings(self, window, tmp_path):
        binary = tmp_path / "target.bin"
        binary.write_bytes(b"ELF")
        window.binary_var.set(str(binary))
        window.output_var.set(str(tmp_path / "out"))
        window.format_vars["python"].set(True)

        settings = window.collect_settings()

        assert settings.formats == ["source", "json", "python"]
        assert settings.validate() == []

    def test_clearing_every_format_is_rejected(self, window, tmp_path):
        binary = tmp_path / "target.bin"
        binary.write_bytes(b"ELF")
        window.binary_var.set(str(binary))
        window.output_var.set(str(tmp_path / "out"))
        for var in window.format_vars.values():
            var.set(False)

        problems = window.collect_settings().validate()

        assert any("at least one output format" in p for p in problems)


class TestLimitDefaults:
    """The limit fields open sized for a typical local server, not empty."""

    @staticmethod
    def _select_custom(window):
        window.provider_var.set(LLMProvider.CUSTOM.value)
        window._sync_provider_fields()

    def test_fields_are_pre_filled_for_a_32k_context(self, window):
        from kong.llm.endpoint import suggest_limits

        expected = suggest_limits(32768)
        assert window.prompt_chars_var.get() == str(expected.max_prompt_chars)

        self._select_custom(window)
        assert window.output_tokens_var.get() == str(expected.max_output_tokens)
        assert window.chunk_functions_var.get() == str(expected.max_chunk_functions)

    def test_the_defaults_are_the_documented_numbers(self, window):
        assert window.prompt_chars_var.get() == "84216"

        self._select_custom(window)
        assert window.output_tokens_var.get() == "4096"
        assert window.chunk_functions_var.get() == "15"

    def test_a_hosted_provider_opens_on_the_model_s_own_batch_size(self, window):
        """Blank, not the figure a 32k local server was sized for."""
        assert window.chunk_functions_var.get() == ""
        assert window.collect_settings().max_chunk_functions is None

    def test_a_hosted_provider_opens_on_the_model_s_own_token_budget(self, window):
        assert window.output_tokens_var.get() == ""
        assert window.collect_settings().max_output_tokens is None

    def test_a_token_budget_reaches_a_hosted_provider_too(self, window):
        """The budget used to be dropped on everything but a local endpoint."""
        window.output_tokens_var.set("8000")

        settings = window.collect_settings()
        assert settings.provider is LLMProvider.ANTHROPIC
        assert settings.max_output_tokens == 8000
        assert settings.to_config().llm.max_output_tokens == 8000

    def test_each_provider_keeps_its_own_token_budget(self, window):
        self._select_custom(window)
        window.output_tokens_var.set("4096")

        window.provider_var.set(LLMProvider.ANTHROPIC.value)
        window._sync_provider_fields()
        assert window.output_tokens_var.get() == ""

        window.provider_var.set(LLMProvider.CUSTOM.value)
        window._sync_provider_fields()
        assert window.output_tokens_var.get() == "4096"

    def test_each_provider_keeps_its_own_batch_size(self, window):
        self._select_custom(window)
        window.chunk_functions_var.set("6")

        window.provider_var.set(LLMProvider.ANTHROPIC.value)
        window._sync_provider_fields()
        window.chunk_functions_var.set("40")
        assert window.collect_settings().max_chunk_functions == 40

        self._select_custom(window)
        assert window.chunk_functions_var.get() == "6"

    def test_a_hosted_provider_can_cap_its_batch_size(self, window, tmp_path):
        """The one limit a hosted endpoint has a reason to set by hand."""
        binary = tmp_path / "target.bin"
        binary.write_bytes(b"ELF")
        window.binary_var.set(str(binary))
        window.output_var.set(str(tmp_path / "out"))
        window.chunk_functions_var.set("25")

        config = window.collect_settings().to_config()

        assert config.llm.provider is LLMProvider.ANTHROPIC
        assert config.llm.max_chunk_functions == 25
        assert config.llm.max_prompt_chars is None

    def test_the_hint_names_the_assumed_window(self, window):
        hint = window.limits_hint.cget("text")
        assert "sized for" in hint and "32k" in hint

    def test_the_defaults_survive_a_round_trip_to_the_config(self, window, tmp_path):
        binary = tmp_path / "target.bin"
        binary.write_bytes(b"ELF")
        window.binary_var.set(str(binary))
        window.output_var.set(str(tmp_path / "out"))
        window.provider_var.set(LLMProvider.CUSTOM.value)
        window._sync_provider_fields()
        window.model_var.set("kong-local")

        config = window.collect_settings().to_config()

        assert config.llm.max_prompt_chars == 84216
        assert config.llm.max_chunk_functions == 15
        assert config.llm.max_output_tokens == 4096

    def test_a_hosted_provider_still_ignores_them(self, window, tmp_path):
        binary = tmp_path / "target.bin"
        binary.write_bytes(b"ELF")
        window.binary_var.set(str(binary))
        window.output_var.set(str(tmp_path / "out"))

        settings = window.collect_settings()

        assert settings.max_prompt_chars is None
        assert settings.max_output_tokens is None

    def test_clearing_a_field_means_provider_defaults(self, window, tmp_path):
        window.provider_var.set(LLMProvider.CUSTOM.value)
        window.prompt_chars_var.set("")

        assert window.collect_settings().max_prompt_chars is None


class TestTwoPassFields:
    def test_the_draft_model_is_empty_until_asked_for(self, window):
        settings = window.collect_settings()

        assert settings.draft_model == ""

    def test_the_draft_model_reaches_the_settings(self, window):
        window.draft_model_var.set("qwen2.5-coder-7b")

        assert window.collect_settings().draft_model == "qwen2.5-coder-7b"

    def test_the_threshold_opens_at_the_high_confidence_bar(self, window):
        from kong.agent.refinement import DEFAULT_REFINE_BELOW

        assert window.refine_below_var.get() == str(DEFAULT_REFINE_BELOW)
        assert window.collect_settings().refine_below == DEFAULT_REFINE_BELOW

    def test_a_percent_sign_is_tolerated(self, window):
        window.refine_below_var.set("65%")

        assert window.collect_settings().refine_below == 65

    def test_clearing_the_threshold_means_the_default(self, window):
        from kong.agent.refinement import DEFAULT_REFINE_BELOW

        window.refine_below_var.set("")

        assert window.collect_settings().refine_below == DEFAULT_REFINE_BELOW

    def test_a_non_numeric_threshold_is_rejected(self, window):
        window.refine_below_var.set("high")

        with pytest.raises(ValueError):
            window.collect_settings()


class TestZaiInTheWindow:
    """GLM is picked like any hosted provider, not typed in as an endpoint."""

    def _select_zai(self, window):
        window.provider_var.set(LLMProvider.ZAI.value)
        window._sync_provider_fields()

    def test_zai_is_one_of_the_provider_choices(self, window):
        window.provider_var.set(LLMProvider.ZAI.value)

        assert window.collect_settings().provider is LLMProvider.ZAI

    def test_choosing_zai_fills_in_its_endpoint(self, window):
        from kong.config import ZAI_BASE_URL

        self._select_zai(window)

        assert window.base_url_var.get() == ZAI_BASE_URL
        assert window.collect_settings().base_url == ZAI_BASE_URL

    def test_the_endpoint_stays_editable_for_a_coding_plan_key(self, window):
        self._select_zai(window)

        assert str(window.base_url_entry.cget("state")) == "normal"

        window.base_url_var.set("https://api.z.ai/api/coding/paas/v4")
        assert window.collect_settings().base_url == (
            "https://api.z.ai/api/coding/paas/v4"
        )

    def test_each_provider_keeps_its_own_endpoint(self, window):
        from kong.config import ZAI_BASE_URL

        window.provider_var.set(LLMProvider.CUSTOM.value)
        window._sync_provider_fields()
        window.base_url_var.set("http://127.0.0.1:9999/v1")

        self._select_zai(window)
        assert window.base_url_var.get() == ZAI_BASE_URL

        window.provider_var.set(LLMProvider.CUSTOM.value)
        window._sync_provider_fields()
        assert window.base_url_var.get() == "http://127.0.0.1:9999/v1"

    def test_an_empty_model_field_names_what_will_run(self, window):
        self._select_zai(window)

        assert "glm-5.3" in window.model_hint.cget("text")

    def test_the_hint_follows_the_provider(self, window):
        window.provider_var.set(LLMProvider.ANTHROPIC.value)
        window._sync_provider_fields()
        assert "claude" in window.model_hint.cget("text")

        window.provider_var.set(LLMProvider.CUSTOM.value)
        window._sync_provider_fields()
        assert window.model_hint.cget("text") == ""

    def test_the_local_limit_fields_stay_out_of_the_way(self, window):
        """Z.ai is chunked from the model table, like the other hosted ones."""
        self._select_zai(window)

        assert all(str(e.cget("state")) == "disabled" for e in window.limit_entries)
        settings = window.collect_settings()
        assert settings.max_prompt_chars is None

    def test_a_run_without_a_key_is_refused_in_the_dialog(self, window, tmp_path, monkeypatch):
        monkeypatch.delenv("ZAI_API_KEY", raising=False)
        binary = tmp_path / "target.bin"
        binary.write_bytes(b"ELF")
        window.binary_var.set(str(binary))
        window.output_var.set(str(tmp_path / "out"))
        self._select_zai(window)

        problems = window.collect_settings().validate()

        assert any("ZAI_API_KEY" in p for p in problems)


class TestApiKeyField:
    """Keys can be typed, kept, and forgotten without leaving the window."""

    def _select(self, window, provider):
        window.provider_var.set(provider.value)
        window._sync_provider_fields()

    def test_the_key_is_masked(self, window):
        assert window.api_key_entry.cget("show") == "*"

    def test_a_typed_key_reaches_the_settings(self, window):
        window.api_key_var.set("sk-ant-typed")

        assert window.collect_settings().api_key == "sk-ant-typed"

    def test_saving_keeps_it_for_next_time(self, window):
        from kong.db import get_saved_api_key

        window.api_key_var.set("sk-ant-typed")
        window.on_save_api_key()

        assert get_saved_api_key(LLMProvider.ANTHROPIC) == "sk-ant-typed"
        assert "saved" in window.key_status_var.get()

    def test_saving_an_empty_field_forgets_the_key(self, window):
        from kong.db import get_saved_api_key

        window.api_key_var.set("sk-ant-typed")
        window.on_save_api_key()

        window.api_key_var.set("")
        window.on_save_api_key()

        assert get_saved_api_key(LLMProvider.ANTHROPIC) is None
        assert "cleared" in window.key_status_var.get()

    def test_each_provider_has_its_own_key(self, window):
        from kong.db import get_saved_api_key

        window.api_key_var.set("sk-ant-typed")
        window.on_save_api_key()

        self._select(window, LLMProvider.ZAI)
        assert window.api_key_var.get() == ""
        window.api_key_var.set("zai-typed")
        window.on_save_api_key()

        assert get_saved_api_key(LLMProvider.ANTHROPIC) == "sk-ant-typed"
        assert get_saved_api_key(LLMProvider.ZAI) == "zai-typed"

    def test_switching_back_shows_the_key_of_that_provider(self, window):
        window.api_key_var.set("sk-ant-typed")
        window.on_save_api_key()

        self._select(window, LLMProvider.ZAI)
        self._select(window, LLMProvider.ANTHROPIC)

        assert window.api_key_var.get() == "sk-ant-typed"

    def test_an_unsaved_key_is_not_lost_by_looking_at_another_provider(
        self, window,
    ):
        window.api_key_var.set("half-typed")

        self._select(window, LLMProvider.ZAI)
        self._select(window, LLMProvider.ANTHROPIC)

        assert window.api_key_var.get() == "half-typed"

    def test_an_empty_field_says_the_environment_is_being_used(self, window):
        self._select(window, LLMProvider.ZAI)

        assert "ZAI_API_KEY" in window.key_status_var.get()

    def test_an_empty_field_with_nothing_anywhere_says_so(
        self, window, monkeypatch,
    ):
        monkeypatch.delenv("ZAI_API_KEY", raising=False)
        self._select(window, LLMProvider.ZAI)

        assert "no key" in window.key_status_var.get()

    def test_a_local_endpoint_is_not_nagged_about_a_key(self, window):
        self._select(window, LLMProvider.CUSTOM)

        assert "none needed" in window.key_status_var.get()

    def test_typing_clears_the_status(self, window):
        self._select(window, LLMProvider.ZAI)
        window.api_key_var.set("zai-typed")

        assert window.key_status_var.get() == ""

    def test_a_saved_key_is_enough_to_start(self, window, tmp_path, monkeypatch):
        from kong.db import save_api_key

        monkeypatch.delenv("ZAI_API_KEY", raising=False)
        save_api_key(LLMProvider.ZAI, "zai-saved")

        binary = tmp_path / "target.bin"
        binary.write_bytes(b"ELF")
        window.binary_var.set(str(binary))
        window.output_var.set(str(tmp_path / "out"))
        self._select(window, LLMProvider.ZAI)

        assert window.api_key_var.get() == "zai-saved"
        assert window.collect_settings().validate() == []


class TestDetectEndpoint:
    def _fake_discovery(self, monkeypatch, info):
        import kong.llm.endpoint as endpoint

        monkeypatch.setattr(endpoint, "discover", lambda *a, **kw: info)

    def test_detect_is_disabled_for_a_hosted_provider(self, window):
        assert str(window.detect_button.cget("state")) == "disabled"

    def test_detect_is_enabled_for_a_custom_endpoint(self, window):
        window.provider_var.set(LLMProvider.CUSTOM.value)
        window._sync_provider_fields()

        assert str(window.detect_button.cget("state")) == "normal"

    def test_detection_fills_the_model_and_the_limits(self, window, monkeypatch):
        from kong.llm.endpoint import EndpointInfo, EndpointModel

        self._fake_discovery(
            monkeypatch,
            EndpointInfo(
                base_url="http://127.0.0.1:8080/v1",
                models=[EndpointModel("kong-local", train_context=32768)],
                context_tokens=16384,
            ),
        )

        window.on_detect_endpoint()

        assert window.model_var.get() == "kong-local"
        assert window.output_tokens_var.get() == "2048"
        assert window.chunk_functions_var.get() == "7"
        assert window.prompt_chars_var.get() == "41208"
        assert "16,384 token context" in window.status_var.get()

    def test_detection_does_not_overwrite_a_chosen_model(self, window, monkeypatch):
        from kong.llm.endpoint import EndpointInfo, EndpointModel

        window.model_var.set("my-own-alias")
        self._fake_discovery(
            monkeypatch,
            EndpointInfo(
                base_url="http://127.0.0.1:8080/v1",
                models=[EndpointModel("kong-local")],
                context_tokens=8192,
            ),
        )

        window.on_detect_endpoint()

        assert window.model_var.get() == "my-own-alias"

    def test_detection_without_a_base_url_is_refused(self, window, monkeypatch):
        shown = []
        monkeypatch.setattr(
            "tkinter.messagebox.showerror", lambda *a: shown.append(a)
        )
        window.base_url_var.set("")

        window.on_detect_endpoint()

        assert shown and "base URL" in shown[0][1]

    def test_an_unreachable_endpoint_is_reported(self, window, monkeypatch):
        import kong.llm.endpoint as endpoint

        def explode(*a, **kw):
            raise endpoint.EndpointError("connection refused")

        monkeypatch.setattr(endpoint, "discover", explode)
        shown = []
        monkeypatch.setattr(
            "tkinter.messagebox.showerror", lambda *a: shown.append(a)
        )

        window.on_detect_endpoint()

        assert shown and "connection refused" in shown[0][1]


class TestCollectSettings:
    def test_reads_the_form(self, window, tmp_path):
        binary = tmp_path / "target.bin"
        binary.write_bytes(b"\x7fELF")
        window.binary_var.set(str(binary))
        window.output_var.set(str(tmp_path / "out"))
        window.model_var.set("claude-opus-5")

        settings = window.collect_settings()

        assert settings.binary_path == str(binary)
        assert settings.model == "claude-opus-5"
        assert settings.provider is LLMProvider.ANTHROPIC
        assert settings.validate() == []

    def test_custom_limits_are_ignored_for_a_hosted_provider(self, window):
        window.prompt_chars_var.set("32000")
        window.base_url_var.set("http://127.0.0.1:8080/v1")

        settings = window.collect_settings()

        assert settings.base_url == ""
        assert settings.max_prompt_chars is None

    def test_custom_limits_are_parsed(self, window):
        window.provider_var.set(LLMProvider.CUSTOM.value)
        window._sync_provider_fields()
        window.model_var.set("kong-local")
        window.prompt_chars_var.set("32000")
        window.chunk_functions_var.set("8")
        window.output_tokens_var.set("2048")

        settings = window.collect_settings()

        assert settings.max_prompt_chars == 32000
        assert settings.max_chunk_functions == 8
        assert settings.max_output_tokens == 2048

    def test_a_non_numeric_limit_is_rejected(self, window):
        window.provider_var.set(LLMProvider.CUSTOM.value)
        window.prompt_chars_var.set("beaucoup")

        with pytest.raises(ValueError, match="whole number"):
            window.collect_settings()


class TestRendering:
    def test_log_lines_are_appended(self, window):
        window._append_log(Event(type=EventType.PHASE_START, message="Starting triage"))
        assert "Starting triage" in window.log.get("1.0", "end")

    def test_the_log_stays_read_only(self, window):
        window._append_log(Event(type=EventType.RUN_START, message="go"))
        assert str(window.log.cget("state")) == "disabled"

    def test_a_completed_function_becomes_a_table_row(self, window):
        window._append_result(
            Event(
                type=EventType.FUNCTION_COMPLETE,
                message="named",
                data={
                    "address": 0x401A30,
                    "original_name": "FUN_00401a30",
                    "name": "parse_http_header",
                    "confidence": 92,
                    "classification": "parser",
                },
            )
        )

        rows = window.results.get_children()
        assert len(rows) == 1
        values = window.results.item(rows[0])["values"]
        assert values[0] == "0x00401a30"
        assert values[2] == "parse_http_header"
        assert values[3] == "92%"


class TestCoherenceInTheWindow:
    """The manual cross-check: its button, and the table it fills in."""

    CONFLICT = Event(
        type=EventType.COHERENCE_CONFLICT,
        message="duplicate_name: 2 functions are all named parse_header",
        data={
            "id": "duplicate_name:00401a30-00402b40",
            "kind": "duplicate_name",
            "addresses": [0x401A30, 0x402B40],
            "summary": "2 functions are all named parse_header",
            "detail": "",
        },
    )

    def test_the_button_waits_for_an_analysis(self, window):
        assert str(window.coherence_button.cget("state")) == "disabled"

    def test_a_contradiction_becomes_a_row(self, window):
        window._append_conflict(self.CONFLICT)

        rows = window.conflicts.get_children()
        assert len(rows) == 1
        values = window.conflicts.item(rows[0])["values"]
        assert values[0] == "duplicate name"
        assert values[1] == "0x00401a30, 0x00402b40"
        assert values[2] == "2 functions are all named parse_header"
        assert values[3] == "pending"

    def test_the_resolution_lands_on_the_row_it_answers(self, window):
        window._append_conflict(self.CONFLICT)
        window._resolve_conflict(Event(
            type=EventType.COHERENCE_RESOLVED,
            message="resolved",
            data={
                "id": self.CONFLICT.data["id"],
                "verdict": "resolved",
                "explanation": "the second one writes",
                "applied": ["0x00402b40: parse_header to write_header"],
            },
        ))

        row = window.conflicts.get_children()[0]
        assert window.conflicts.item(row)["values"][3] == (
            "0x00402b40: parse_header to write_header"
        )

    def test_a_conflict_left_alone_says_why(self, window):
        window._append_conflict(self.CONFLICT)
        window._resolve_conflict(Event(
            type=EventType.COHERENCE_RESOLVED,
            message="kept",
            data={
                "id": self.CONFLICT.data["id"],
                "verdict": "no_change",
                "explanation": "both really are called that",
                "applied": [],
            },
        ))

        row = window.conflicts.get_children()[0]
        assert "both really are called that" in window.conflicts.item(row)["values"][3]

    def test_an_answer_to_a_conflict_from_another_pass_is_ignored(self, window):
        window._resolve_conflict(Event(
            type=EventType.COHERENCE_RESOLVED,
            message="resolved",
            data={"id": "duplicate_name:deadbeef", "applied": []},
        ))
        assert window.conflicts.get_children() == ()

    def test_a_new_pass_starts_from_an_empty_table(self, window):
        window._append_conflict(self.CONFLICT)
        window._clear_conflicts()
        assert window.conflicts.get_children() == ()


class TestFinishingPassControls:
    """The Finish pass button is only worth offering when it has work."""

    def _render(self, window, **fields):
        from types import SimpleNamespace

        from kong.gui.controller import RunState

        state = RunState(**fields)
        window.controller = SimpleNamespace(state=state)
        window._render_state()
        return state

    def test_it_starts_disabled(self, window):
        assert str(window.finish_button.cget("state")) == "disabled"

    def test_nothing_pending_keeps_it_disabled(self, window):
        self._render(window, finished=True, pending_finish=0)
        assert str(window.finish_button.cget("state")) == "disabled"

    def test_a_finished_draft_offers_it(self, window):
        self._render(window, finished=True, pending_finish=12)
        assert str(window.finish_button.cget("state")) == "normal"
        assert "12 to finish" in window.status_var.get()

    def test_it_is_not_offered_twice_at_once(self, window):
        self._render(window, finished=True, pending_finish=12, finishing=True)
        assert str(window.finish_button.cget("state")) == "disabled"
        assert "FINISHING" in window.status_var.get()

    def test_the_two_manual_passes_lock_each_other_out(self, window):
        self._render(window, finished=True, pending_finish=12, finishing=True)
        assert str(window.coherence_button.cget("state")) == "disabled"

        self._render(window, finished=True, pending_finish=12, checking_coherence=True)
        assert str(window.finish_button.cget("state")) == "disabled"


class TestDraftOnly:
    def test_a_run_is_full_unless_asked_otherwise(self, window):
        from kong.config import RunStage

        assert window.collect_settings().stage is RunStage.FULL

    def test_the_checkbox_makes_it_a_draft(self, window):
        from kong.config import RunStage

        window.draft_only_var.set(True)

        assert window.collect_settings().stage is RunStage.DRAFT
