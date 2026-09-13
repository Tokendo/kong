"""customtkinter front-end for Kong.

Same pipeline as `kong analyze`, driven from a window: pick a binary, pick a
provider, watch the run, pause it, export at any point. All widget access
happens on the main thread; the analysis itself runs in AnalysisController.
"""

from __future__ import annotations

import os
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, font as tkfont, messagebox, ttk

import customtkinter as ctk

from kong.agent.events import Event, EventType
from kong.agent.refinement import DEFAULT_REFINE_BELOW
from kong.banner import _ENV_VARS
from kong.db import get_saved_api_key, save_api_key
from kong.config import ZAI_BASE_URL, LLMProvider, RunStage
from kong.gui.controller import AnalysisController, RunSettings
from kong.llm.endpoint import suggest_limits

POLL_INTERVAL_MS = 150

# One look for everyone. Following the system theme would swing the log tag
# colours below between readable and washed out, so the window stays dark.
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# customtkinter asks for Roboto on every platform. It ships with Windows and
# macOS installs but not with a stock Ubuntu, and Tk answers a missing family
# by silently substituting one of its own — which is why the window can come
# out looking like a different program. These are the families actually
# present on each system, best first.
_FONT_CANDIDATES: dict[str, tuple[str, ...]] = {
    "linux": (
        "Ubuntu", "Cantarell", "Noto Sans", "DejaVu Sans", "Liberation Sans",
        "Roboto",
    ),
    "darwin": ("SF Pro Text", "Helvetica Neue", "Helvetica"),
    "win32": ("Segoe UI", "Roboto"),
}

#: Escape hatch for desktops Tk reads the DPI of badly — fractional scaling on
#: Wayland in particular, where the window comes out blurry or twice the size
#: it should be. KONG_UI_SCALE=1 pins it, KONG_UI_SCALE=1.5 enlarges by half.
UI_SCALE_ENV = "KONG_UI_SCALE"


def _font_family(widget: tk.Misc) -> str:
    """A font family this system actually has, for the theme and the table."""
    try:
        available = {name.lower() for name in tkfont.families(widget)}
    except tk.TclError:
        return ctk.ThemeManager.theme["CTkFont"]["family"]

    for candidate in _FONT_CANDIDATES.get(sys.platform, ()):
        if candidate.lower() in available:
            return candidate
    # Whatever Tk itself is using beats a family that resolves to a fallback.
    try:
        return tkfont.nametofont("TkDefaultFont").actual("family")
    except tk.TclError:
        return ctk.ThemeManager.theme["CTkFont"]["family"]


def apply_ui_font(widget: tk.Misc) -> str:
    """Point the theme at an installed font. Call before building widgets."""
    family = _font_family(widget)
    ctk.ThemeManager.theme["CTkFont"]["family"] = family
    return family


def ui_scale_override() -> float | None:
    """The scaling factor asked for in the environment, if it is usable."""
    raw = os.environ.get(UI_SCALE_ENV, "").strip()
    if not raw:
        return None
    try:
        scale = float(raw)
    except ValueError:
        return None
    # A factor outside this range is a typo, and applying it would leave a
    # window nobody can resize back.
    return scale if 0.5 <= scale <= 4.0 else None


def apply_ui_scale() -> float | None:
    """Honour KONG_UI_SCALE, taking Tk's own DPI guess out of the loop."""
    scale = ui_scale_override()
    if scale is None:
        return None
    ctk.deactivate_automatic_dpi_awareness()
    ctk.set_widget_scaling(scale)
    ctk.set_window_scaling(scale)
    return scale


def fit_to_screen(
    screen_width: int,
    screen_height: int,
    width: int = 1080,
    height: int = 760,
    margin: float = 0.92,
) -> tuple[int, int]:
    """Shrink the opening size to something the screen can show.

    A 1080x760 window is unremarkable on a desktop and taller than the work
    area of a scaled 1366x768 laptop, where the controls at the bottom end up
    under the panel with no way to reach them.
    """
    return (
        max(320, min(width, int(screen_width * margin))),
        max(240, min(height, int(screen_height * margin))),
    )

# A local server is usually started somewhere around 32k, so the limit fields
# open with values that fit that rather than empty. Detect replaces them with
# whatever the endpoint actually reports.
DEFAULT_CONTEXT_TOKENS = 32768

LOCAL_BASE_URL = "http://127.0.0.1:8080/v1"

# What a provider analyzes with when the model field is left empty. Mirrors
# _DEFAULT_MODELS in the CLI; shown as a hint rather than filled in, so
# switching provider never leaves another provider's model in the box.
DEFAULT_MODEL_HINTS = {
    LLMProvider.ANTHROPIC.value: "claude-opus-5",
    LLMProvider.OPENAI.value: "gpt-4o",
    LLMProvider.ZAI.value: "glm-5.3",
}

#: Files the export phase actually writes. Ghidra is not one of them: names
#: and types are written into the program database as each function is
#: analyzed, not at the end and not on request.
OUTPUT_FORMATS: list[tuple[str, str]] = [
    ("source", "C source"),
    ("json", "JSON"),
    ("python", "Python"),
    ("csharp", "C#"),
]

DEFAULT_FORMATS = ("source", "json")

# Both reconstruction formats run a second pass over the whole binary.
TRANSPILE_FORMATS = ("python", "csharp")

LOG_TAGS: dict[EventType, str] = {
    EventType.PHASE_START: "phase",
    EventType.PHASE_COMPLETE: "success",
    EventType.FUNCTION_COMPLETE: "success",
    EventType.FUNCTION_SKIPPED: "muted",
    EventType.FUNCTION_ERROR: "error",
    EventType.RUN_ERROR: "error",
    EventType.RUN_COMPLETE: "success",
    EventType.DEOBFUSCATION_DETECTED: "accent",
    EventType.DEOBFUSCATION_COMPLETE: "accent",
    EventType.EXPORT_FILE: "accent",
    # A contradiction is a finding, not a failure: the run did not break, the
    # results disagree with each other.
    EventType.COHERENCE_CHECKED: "phase",
    EventType.COHERENCE_CONFLICT: "accent",
    EventType.COHERENCE_RESOLVED: "success",
}

# Chosen to read on the dark background above, not on white.
TAG_COLORS = {
    "phase": "#58a6ff",
    "success": "#3fb950",
    "error": "#ff7b72",
    "accent": "#d2a8ff",
    "muted": "#8b949e",
}


def _dark(color: str | list[str]) -> str:
    """The dark half of a customtkinter (light, dark) colour pair."""
    return color[1] if isinstance(color, (list, tuple)) else color


def _style_results_table(family: str | None = None) -> None:
    """Dress ttk's Treeview to match its customtkinter neighbours.

    customtkinter has no table widget, so the functions tab keeps the real
    Treeview. Only the "clam" theme honours these colours; the native Windows
    and macOS themes draw their own and ignore them.

    ttk knows nothing about the theme font or about customtkinter's scaling, so
    both are handed to it here: left alone it draws the table in Tk's default
    font at 24 pixels a row, which on a scaled desktop clips every line.
    """
    theme = ctk.ThemeManager.theme
    family = family or theme["CTkFont"]["family"]
    size = theme["CTkFont"]["size"]
    row_font = tkfont.Font(family=family, size=size)
    # Row height from the font rather than a constant: the text decides how
    # much room it needs, whatever the font and the scaling turn out to be.
    row_height = int(row_font.metrics("linespace") * 1.6)

    style = ttk.Style()
    style.theme_use("clam")
    style.configure(
        "Kong.Treeview",
        background=_dark(theme["CTkFrame"]["fg_color"]),
        fieldbackground=_dark(theme["CTkFrame"]["fg_color"]),
        foreground=_dark(theme["CTkLabel"]["text_color"]),
        borderwidth=0,
        relief="flat",
        font=(family, size),
        rowheight=row_height,
    )
    style.configure(
        "Kong.Treeview.Heading",
        background=_dark(theme["CTkFrame"]["top_fg_color"]),
        foreground=_dark(theme["CTkLabel"]["text_color"]),
        borderwidth=0,
        relief="flat",
        font=(family, size, "bold"),
    )
    style.map(
        "Kong.Treeview",
        background=[("selected", _dark(theme["CTkButton"]["fg_color"]))],
        foreground=[("selected", "#ffffff")],
    )
    style.map(
        "Kong.Treeview.Heading",
        background=[("active", _dark(theme["CTkButton"]["hover_color"]))],
    )


class KongWindow(ctk.CTkFrame):
    """The main window: configuration on top, live run below."""

    def __init__(self, master: tk.Misc, initial_binary: str = "") -> None:
        super().__init__(master, fg_color="transparent")
        self.master_window = master
        self.controller: AnalysisController | None = None
        # Before any widget is built: CTkFont reads the family off the theme
        # when it is constructed, not when it is drawn.
        self.font_family = apply_ui_font(master)
        self.section_font = ctk.CTkFont(size=13, weight="bold")

        self.binary_var = tk.StringVar(value=initial_binary)
        self.output_var = tk.StringVar(value=self._default_output_dir(initial_binary))
        self.format_vars = {
            key: tk.BooleanVar(value=key in DEFAULT_FORMATS)
            for key, _ in OUTPUT_FORMATS
        }
        self.resume_var = tk.BooleanVar(value=True)
        self.provider_var = tk.StringVar(value=LLMProvider.ANTHROPIC.value)
        self.model_var = tk.StringVar()
        self.draft_model_var = tk.StringVar()
        self.refine_below_var = tk.StringVar(value=str(DEFAULT_REFINE_BELOW))
        self.draft_only_var = tk.BooleanVar(value=False)
        self.base_url_var = tk.StringVar(value=LOCAL_BASE_URL)
        # Each provider that has a base URL keeps its own, so switching back
        # and forth does not make the user retype either one.
        self._base_urls = {
            LLMProvider.CUSTOM.value: LOCAL_BASE_URL,
            LLMProvider.ZAI.value: ZAI_BASE_URL,
        }
        self._base_url_owner = LLMProvider.CUSTOM.value
        self.api_key_var = tk.StringVar()
        self.key_status_var = tk.StringVar(value="")
        # Typed keys are kept per provider for the session, so switching to
        # look at another one does not throw away what was being entered.
        self._api_keys: dict[str, str] = {}
        self._api_key_owner = LLMProvider.ANTHROPIC.value
        defaults = suggest_limits(DEFAULT_CONTEXT_TOKENS)
        self.prompt_chars_var = tk.StringVar(value=str(defaults.max_prompt_chars))
        self.chunk_functions_var = tk.StringVar(
            value=str(defaults.max_chunk_functions)
        )
        self.output_tokens_var = tk.StringVar(value=str(defaults.max_output_tokens))
        # Batch size is the one limit worth setting on a hosted API too, and it
        # means a different number there: a local server is sized by Detect,
        # while a hosted one runs on the model's own figure until a rate limit
        # or a model that drops functions from long batches says otherwise. So
        # each provider keeps its own, blank meaning "whatever the model says".
        self._chunk_sizes = {
            LLMProvider.CUSTOM.value: str(defaults.max_chunk_functions),
        }
        self._chunk_owner = LLMProvider.CUSTOM.value
        # The output budget is per provider for the same reason: a local
        # server needs a figure that fits the window it was started with,
        # while a hosted model has one of its own. Blank means "the model's".
        self._output_tokens = {
            LLMProvider.CUSTOM.value: str(defaults.max_output_tokens),
        }
        self._output_tokens_owner = LLMProvider.CUSTOM.value
        self.status_var = tk.StringVar(value="Idle.")

        self.grid(sticky="nsew", padx=10, pady=10)
        master.columnconfigure(0, weight=1)
        master.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(3, weight=1)

        self.api_key_var.set(self._saved_key(self.provider_var.get()))
        self.api_key_var.trace_add("write", lambda *_: self._refresh_key_status())

        _style_results_table(self.font_family)
        self._build_paths_section()
        self._build_provider_section()
        self._build_controls_section()
        self._build_output_section()
        self._sync_provider_fields()
        self._sync_format_hint()

        master.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(POLL_INTERVAL_MS, self._poll)

    # -------------------------------------------------------------------- build

    @staticmethod
    def _default_output_dir(binary_path: str) -> str:
        if not binary_path:
            return str(Path.cwd() / "kong_output")
        return str(Path.cwd() / f"kong_output_{Path(binary_path).stem}")

    def _section(self, title: str, row: int) -> ctk.CTkFrame:
        """A titled block: customtkinter has no LabelFrame, so build one."""
        outer = ctk.CTkFrame(self)
        outer.grid(row=row, column=0, sticky="ew", pady=(0, 8))
        outer.columnconfigure(0, weight=1)
        ctk.CTkLabel(outer, text=title, font=self.section_font, anchor="w").grid(
            row=0, column=0, sticky="ew", padx=12, pady=(8, 0)
        )
        body = ctk.CTkFrame(outer, fg_color="transparent")
        body.grid(row=1, column=0, sticky="ew", padx=12, pady=(2, 12))
        return body

    def _build_paths_section(self) -> None:
        frame = self._section("Target", row=0)
        frame.columnconfigure(1, weight=1)

        ctk.CTkLabel(frame, text="Binary").grid(
            row=0, column=0, sticky="w", padx=(0, 8)
        )
        ctk.CTkEntry(frame, textvariable=self.binary_var).grid(
            row=0, column=1, sticky="ew"
        )
        ctk.CTkButton(
            frame, text="Browse...", width=90, command=self.choose_binary
        ).grid(row=0, column=2, padx=(8, 0))

        ctk.CTkLabel(frame, text="Output").grid(
            row=1, column=0, sticky="w", padx=(0, 8), pady=(6, 0)
        )
        ctk.CTkEntry(frame, textvariable=self.output_var).grid(
            row=1, column=1, sticky="ew", pady=(6, 0)
        )
        ctk.CTkButton(
            frame, text="Browse...", width=90, command=self.choose_output_dir
        ).grid(row=1, column=2, padx=(8, 0), pady=(6, 0))

        ctk.CTkLabel(frame, text="Formats").grid(
            row=2, column=0, sticky="nw", padx=(0, 8), pady=(6, 0)
        )
        formats = ctk.CTkFrame(frame, fg_color="transparent")
        formats.grid(row=2, column=1, columnspan=2, sticky="w", pady=(6, 0))
        for column, (key, label) in enumerate(OUTPUT_FORMATS):
            ctk.CTkCheckBox(
                formats,
                text=label,
                variable=self.format_vars[key],
                onvalue=True,
                offvalue=False,
                command=self._sync_format_hint,
            ).grid(row=0, column=column, padx=(0, 12))

        self.format_hint = ctk.CTkLabel(
            formats, text="", anchor="w", text_color=TAG_COLORS["muted"]
        )
        self.format_hint.grid(row=1, column=0, columnspan=len(OUTPUT_FORMATS),
                              sticky="w", pady=(4, 0))

        self.resume_check = ctk.CTkCheckBox(
            frame,
            text="Resume where the last run over this binary stopped",
            variable=self.resume_var,
            onvalue=True,
            offvalue=False,
        )
        self.resume_check.grid(row=3, column=1, columnspan=2, sticky="w", pady=(8, 0))

    def _build_provider_section(self) -> None:
        frame = self._section("Model", row=1)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(3, weight=1)

        providers = ctk.CTkFrame(frame, fg_color="transparent")
        providers.grid(row=0, column=0, columnspan=5, sticky="w", pady=(0, 6))
        for column, provider in enumerate(
            (
                LLMProvider.ANTHROPIC,
                LLMProvider.OPENAI,
                LLMProvider.ZAI,
                LLMProvider.CUSTOM,
            )
        ):
            ctk.CTkRadioButton(
                providers,
                text=provider.display_name,
                value=provider.value,
                variable=self.provider_var,
                command=self._sync_provider_fields,
            ).grid(row=0, column=column, padx=(0, 16))

        ctk.CTkLabel(frame, text="Model").grid(row=1, column=0, sticky="w", padx=(0, 8))
        model_box = ctk.CTkFrame(frame, fg_color="transparent")
        model_box.grid(row=1, column=1, sticky="ew")
        model_box.columnconfigure(0, weight=1)
        ctk.CTkEntry(model_box, textvariable=self.model_var).grid(
            row=0, column=0, sticky="ew"
        )
        self.model_hint = ctk.CTkLabel(
            model_box, text="", anchor="w", text_color=TAG_COLORS["muted"],
        )
        self.model_hint.grid(row=0, column=1, padx=(8, 0))

        ctk.CTkLabel(frame, text="Base URL").grid(
            row=1, column=2, sticky="w", padx=(12, 8)
        )
        self.base_url_entry = ctk.CTkEntry(frame, textvariable=self.base_url_var)
        self.base_url_entry.grid(row=1, column=3, sticky="ew")

        self.detect_button = ctk.CTkButton(
            frame, text="Detect", width=80, command=self.on_detect_endpoint
        )
        self.detect_button.grid(row=1, column=4, padx=(8, 0))

        ctk.CTkLabel(frame, text="Draft model").grid(
            row=2, column=0, sticky="w", padx=(0, 8), pady=(6, 0)
        )
        ctk.CTkEntry(frame, textvariable=self.draft_model_var).grid(
            row=2, column=1, sticky="ew", pady=(6, 0)
        )

        ctk.CTkLabel(frame, text="Refine below %").grid(
            row=2, column=2, sticky="w", padx=(12, 8), pady=(6, 0)
        )
        ctk.CTkEntry(
            frame, textvariable=self.refine_below_var, width=96
        ).grid(row=2, column=3, sticky="w", pady=(6, 0))

        self.draft_only_check = ctk.CTkCheckBox(
            frame,
            text="Draft only",
            variable=self.draft_only_var,
            onvalue=True,
            offvalue=False,
        )
        self.draft_only_check.grid(row=2, column=4, sticky="w", padx=(8, 0), pady=(6, 0))

        ctk.CTkLabel(frame, text="API key").grid(
            row=3, column=0, sticky="w", padx=(0, 8), pady=(6, 0)
        )
        key_box = ctk.CTkFrame(frame, fg_color="transparent")
        key_box.grid(row=3, column=1, columnspan=4, sticky="ew", pady=(6, 0))
        key_box.columnconfigure(0, weight=1)
        self.api_key_entry = ctk.CTkEntry(
            key_box, textvariable=self.api_key_var, show="*",
        )
        self.api_key_entry.grid(row=0, column=0, sticky="ew")
        self.save_key_button = ctk.CTkButton(
            key_box, text="Save", width=80, command=self.on_save_api_key,
        )
        self.save_key_button.grid(row=0, column=1, padx=(8, 0))
        self.key_status = ctk.CTkLabel(
            key_box,
            textvariable=self.key_status_var,
            anchor="w",
            text_color=TAG_COLORS["muted"],
        )
        self.key_status.grid(row=0, column=2, padx=(8, 0))

        self.limits_frame = ctk.CTkFrame(frame, fg_color="transparent")
        self.limits_frame.grid(row=4, column=0, columnspan=5, sticky="ew", pady=(6, 0))
        # Prompt chars describes the endpoint's own window, so it is the one
        # field that only means something on a local server. The two budget
        # entries beside it apply to every provider.
        ctk.CTkLabel(self.limits_frame, text="Max prompt chars").grid(
            row=0, column=0, sticky="w", padx=(0, 6)
        )
        self.prompt_chars_entry = ctk.CTkEntry(
            self.limits_frame, textvariable=self.prompt_chars_var, width=96
        )
        self.prompt_chars_entry.grid(row=0, column=1, sticky="w")
        self.limit_entries: list[ctk.CTkEntry] = [self.prompt_chars_entry]

        ctk.CTkLabel(self.limits_frame, text="Tokens per request").grid(
            row=0, column=2, sticky="w", padx=(12, 6)
        )
        self.output_tokens_entry = ctk.CTkEntry(
            self.limits_frame, textvariable=self.output_tokens_var, width=96
        )
        self.output_tokens_entry.grid(row=0, column=3, sticky="w")

        ctk.CTkLabel(self.limits_frame, text="Functions per batch").grid(
            row=0, column=4, sticky="w", padx=(12, 6)
        )
        self.chunk_entry = ctk.CTkEntry(
            self.limits_frame, textvariable=self.chunk_functions_var, width=96
        )
        self.chunk_entry.grid(row=0, column=5, sticky="w")

        self.limits_hint = ctk.CTkLabel(
            self.limits_frame,
            text=(
                f"(prompt chars: local endpoints only, sized for a "
                f"{DEFAULT_CONTEXT_TOKENS // 1024}k context, Detect reads the "
                f"real one. Tokens per request caps what one call may spend on "
                f"its answer, and functions per batch how much it is asked for "
                f"— both apply to every provider; blank means the model's own "
                f"figure.)"
            ),
            anchor="w",
            text_color=TAG_COLORS["muted"],
        )
        self.limits_hint.grid(row=1, column=0, columnspan=6, sticky="w", pady=(4, 0))

    def _build_controls_section(self) -> None:
        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        frame.columnconfigure(5, weight=1)

        self.start_button = ctk.CTkButton(
            frame, text="Analyze", width=100, command=self.on_start
        )
        self.start_button.grid(row=0, column=0)

        self.pause_button = ctk.CTkButton(
            frame,
            text="Pause",
            width=100,
            command=self.on_toggle_pause,
            state="disabled",
        )
        self.pause_button.grid(row=0, column=1, padx=(8, 0))

        self.export_button = ctk.CTkButton(
            frame,
            text="Export now",
            width=100,
            command=self.on_export,
            state="disabled",
        )
        self.export_button.grid(row=0, column=2, padx=(8, 0))

        # The other half of a "Draft only" run, and the expensive one: it is a
        # button rather than a phase so the draft can be read before it is paid
        # for.
        self.finish_button = ctk.CTkButton(
            frame,
            text="Finish pass",
            width=120,
            command=self.on_finish_pass,
            state="disabled",
        )
        self.finish_button.grid(row=0, column=3, padx=(8, 0))

        # Manual on purpose: the pass costs LLM calls and rewrites names in
        # Ghidra, so it happens when the person reading the results asks for
        # it, not on every run.
        self.coherence_button = ctk.CTkButton(
            frame,
            text="Check coherence",
            width=140,
            command=self.on_check_coherence,
            state="disabled",
        )
        self.coherence_button.grid(row=0, column=4, padx=(8, 0))

        self.progress = ctk.CTkProgressBar(frame, mode="determinate")
        self.progress.set(0)
        self.progress.grid(row=0, column=5, sticky="ew", padx=(16, 0))

        ctk.CTkLabel(frame, textvariable=self.status_var, anchor="w").grid(
            row=1, column=0, columnspan=6, sticky="ew", pady=(6, 0)
        )

    def _build_output_section(self) -> None:
        tabs = ctk.CTkTabview(self)
        tabs.grid(row=3, column=0, sticky="nsew")

        log_tab = tabs.add("Log")
        log_tab.columnconfigure(0, weight=1)
        log_tab.rowconfigure(0, weight=1)
        # CTkTextbox brings its own scrollbars, so there is none to wire up.
        self.log = ctk.CTkTextbox(log_tab, height=320, wrap="none", state="disabled")
        self.log.grid(row=0, column=0, sticky="nsew")
        for tag, color in TAG_COLORS.items():
            self.log.tag_config(tag, foreground=color)

        table_tab = tabs.add("Functions")
        table_tab.columnconfigure(0, weight=1)
        table_tab.rowconfigure(0, weight=1)
        columns = ("address", "original", "name", "confidence", "classification")
        self.results = ttk.Treeview(
            table_tab,
            columns=columns,
            show="headings",
            height=16,
            style="Kong.Treeview",
        )
        for column, heading, width in (
            ("address", "Address", 110),
            ("original", "Original", 160),
            ("name", "Recovered", 260),
            ("confidence", "Conf.", 60),
            ("classification", "Class", 140),
        ):
            self.results.heading(column, text=heading)
            self.results.column(column, width=width, anchor="w")
        self.results.grid(row=0, column=0, sticky="nsew")
        results_scroll = ctk.CTkScrollbar(table_tab, command=self.results.yview)
        results_scroll.grid(row=0, column=1, sticky="ns")
        self.results.configure(yscrollcommand=results_scroll.set)

        coherence_tab = tabs.add("Coherence")
        coherence_tab.columnconfigure(0, weight=1)
        coherence_tab.rowconfigure(0, weight=1)
        self.conflicts = ttk.Treeview(
            coherence_tab,
            columns=("kind", "functions", "summary", "resolution"),
            show="headings",
            height=16,
            style="Kong.Treeview",
        )
        for column, heading, width in (
            ("kind", "Contradiction", 170),
            ("functions", "Functions", 180),
            ("summary", "What disagrees", 420),
            ("resolution", "Resolution", 320),
        ):
            self.conflicts.heading(column, text=heading)
            self.conflicts.column(column, width=width, anchor="w")
        self.conflicts.grid(row=0, column=0, sticky="nsew")
        conflicts_scroll = ctk.CTkScrollbar(
            coherence_tab, command=self.conflicts.yview
        )
        conflicts_scroll.grid(row=0, column=1, sticky="ns")
        self.conflicts.configure(yscrollcommand=conflicts_scroll.set)

    # ------------------------------------------------------------------ actions

    def choose_binary(self) -> None:
        path = filedialog.askopenfilename(title="Select a binary")
        if path:
            self.binary_var.set(path)
            self.output_var.set(self._default_output_dir(path))

    def choose_output_dir(self) -> None:
        path = filedialog.askdirectory(title="Select an output directory")
        if path:
            self.output_var.set(path)

    def selected_formats(self) -> list[str]:
        return [key for key, _ in OUTPUT_FORMATS if self.format_vars[key].get()]

    def _sync_format_hint(self) -> None:
        chosen = [f for f in TRANSPILE_FORMATS if self.format_vars[f].get()]
        # The writeback is not in the list because it is not optional: saying
        # so here is what keeps a shorter list from reading like a lost
        # feature.
        hint = "Names and types go back into Ghidra as the run goes, either way."
        if chosen:
            hint += (
                "\nPython/C# is a readable reconstruction, not a runnable "
                "port, and costs a second LLM pass over the binary."
            )
        self.format_hint.configure(text=hint)

    def on_detect_endpoint(self) -> None:
        """Ask the endpoint what it serves, and size the limits from it."""
        from kong.llm.endpoint import discover, suggest_limits

        base_url = self.base_url_var.get().strip()
        if not base_url:
            messagebox.showerror("Detect", "Enter a base URL first.")
            return

        try:
            info = discover(base_url)
        except Exception as exc:
            messagebox.showerror("Detect", f"Could not reach {base_url}:\n{exc}")
            return

        if info.models and not self.model_var.get().strip():
            self.model_var.set(info.models[0].id)

        context = info.effective_context
        if context is None:
            messagebox.showinfo(
                "Detect",
                f"{len(info.models)} model(s) found, but the endpoint does not "
                "report a context window. Set the limits by hand.",
            )
            return

        limits = suggest_limits(context)
        self.prompt_chars_var.set(str(limits.max_prompt_chars))
        self.chunk_functions_var.set(str(limits.max_chunk_functions))
        self.output_tokens_var.set(str(limits.max_output_tokens))
        self.status_var.set(
            f"{len(info.models)} model(s), {context:,} token context — "
            f"limits sized to fit."
        )

    def _sync_provider_fields(self) -> None:
        provider = self.provider_var.get()
        is_custom = provider == LLMProvider.CUSTOM.value
        has_base_url = provider in self._base_urls

        if has_base_url and provider != self._base_url_owner:
            self._base_urls[self._base_url_owner] = self.base_url_var.get().strip()
            self.base_url_var.set(self._base_urls[provider])
            self._base_url_owner = provider

        if provider != self._chunk_owner:
            self._chunk_sizes[self._chunk_owner] = self.chunk_functions_var.get().strip()
            self.chunk_functions_var.set(self._chunk_sizes.get(provider, ""))
            self._chunk_owner = provider

        if provider != self._output_tokens_owner:
            self._output_tokens[self._output_tokens_owner] = (
                self.output_tokens_var.get().strip()
            )
            self.output_tokens_var.set(self._output_tokens.get(provider, ""))
            self._output_tokens_owner = provider

        self.base_url_entry.configure(state="normal" if has_base_url else "disabled")
        # Detect asks the endpoint what it serves; Z.ai does not answer that.
        self.detect_button.configure(state="normal" if is_custom else "disabled")
        for entry in self.limit_entries:
            entry.configure(state="normal" if is_custom else "disabled")

        default_model = DEFAULT_MODEL_HINTS.get(provider, "")
        self.model_hint.configure(
            text=f"(empty = {default_model})" if default_model else ""
        )

        if provider != self._api_key_owner:
            self._api_keys[self._api_key_owner] = self.api_key_var.get()
            self._api_key_owner = provider
            self.api_key_var.set(self._api_keys.get(provider, self._saved_key(provider)))
        self._refresh_key_status()

    @staticmethod
    def _saved_key(provider_value: str) -> str:
        """The key already stored for this provider, if any."""
        return get_saved_api_key(LLMProvider(provider_value)) or ""

    def _refresh_key_status(self) -> None:
        """Say where the key will come from when the field is left empty."""
        provider = LLMProvider(self.provider_var.get())
        if self.api_key_var.get().strip():
            self.key_status_var.set("")
            return

        env_var = _ENV_VARS.get(provider)
        if env_var is None:
            self.key_status_var.set("(none needed for a local server)")
        elif os.environ.get(env_var):
            self.key_status_var.set(f"(using {env_var})")
        else:
            self.key_status_var.set("(no key)")

    def on_save_api_key(self) -> None:
        """Keep the key for next time, or forget it when the field is empty."""
        provider = LLMProvider(self.provider_var.get())
        key = self.api_key_var.get().strip()
        try:
            save_api_key(provider, key)
        except Exception as exc:  # sqlite failures: a read-only home, a lock
            messagebox.showerror("Could not save the key", str(exc))
            return
        self._api_keys[provider.value] = key
        self.key_status_var.set(
            f"(saved for {provider.display_name})" if key
            else f"(cleared for {provider.display_name})"
        )

    @staticmethod
    def _parse_optional_int(raw: str) -> int | None:
        raw = raw.strip()
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError as exc:
            raise ValueError(f"{raw!r} is not a whole number.") from exc

    @staticmethod
    def _parse_percentage(raw: str) -> int:
        """Read the refine-below field. Blank means the default threshold."""
        raw = raw.strip().rstrip("%").strip()
        if not raw:
            return DEFAULT_REFINE_BELOW
        try:
            return int(raw)
        except ValueError as exc:
            raise ValueError(f"{raw!r} is not a whole percentage.") from exc

    def collect_settings(self) -> RunSettings:
        provider = LLMProvider(self.provider_var.get())
        is_custom = provider is LLMProvider.CUSTOM
        return RunSettings(
            binary_path=self.binary_var.get().strip(),
            output_dir=self.output_var.get().strip(),
            formats=self.selected_formats(),
            resume=self.resume_var.get(),
            provider=provider,
            model=self.model_var.get().strip(),
            api_key=self.api_key_var.get().strip(),
            draft_model=self.draft_model_var.get().strip(),
            refine_below=self._parse_percentage(self.refine_below_var.get()),
            base_url=(
                self.base_url_var.get().strip()
                if provider.value in self._base_urls
                else ""
            ),
            max_prompt_chars=self._parse_optional_int(self.prompt_chars_var.get())
            if is_custom
            else None,
            # Not custom-only: how many functions fit in one call is a question
            # a hosted endpoint answers too, and on a bad day answers with a
            # rate limit.
            max_chunk_functions=self._parse_optional_int(
                self.chunk_functions_var.get()
            ),
            # How many tokens one request may spend on its answer is a
            # question a hosted endpoint answers too, so this is not
            # custom-only: blank leaves the model's own figure in place.
            max_output_tokens=self._parse_optional_int(self.output_tokens_var.get()),
            stage=RunStage.DRAFT if self.draft_only_var.get() else RunStage.FULL,
        )

    def on_start(self) -> None:
        try:
            settings = self.collect_settings()
            controller = AnalysisController(settings)
            controller.start()
        except (ValueError, RuntimeError) as exc:
            messagebox.showerror("Cannot start", str(exc))
            return

        self.controller = controller
        self._clear_output()
        self.start_button.configure(state="disabled")
        self.pause_button.configure(state="normal", text="Pause")
        self.export_button.configure(state="normal")
        self.finish_button.configure(state="normal")
        self.coherence_button.configure(state="normal")
        self.status_var.set("Opening the binary in Ghidra (30-60s on a cold run)...")

    def on_toggle_pause(self) -> None:
        if self.controller is None:
            return
        paused = self.controller.toggle_pause()
        self.pause_button.configure(text="Resume" if paused else "Pause")

    def on_export(self) -> None:
        if self.controller is None:
            return
        if not self.controller.request_export():
            messagebox.showinfo("Export", "Nothing to export yet.")

    def on_finish_pass(self) -> None:
        if self.controller is None:
            messagebox.showinfo(
                "Finishing pass",
                "Analyze a binary first: there is no draft to finish yet.",
            )
            return

        pending = self.controller.state.pending_finish
        if pending and not messagebox.askokcancel(
            "Finishing pass",
            f"Re-read {pending} function(s) one at a time with the main model, "
            f"then redo cleanup, synthesis and export?",
        ):
            return

        refusal = self.controller.request_finishing_pass()
        if refusal:
            messagebox.showinfo("Finishing pass", refusal)
            return
        self.finish_button.configure(state="disabled")
        self.status_var.set(f"Finishing pass on {pending} functions...")

    def on_check_coherence(self) -> None:
        if self.controller is None:
            messagebox.showinfo(
                "Coherence check",
                "Analyze a binary first: there is nothing to cross-check yet.",
            )
            return
        refusal = self.controller.request_coherence_review()
        if refusal:
            messagebox.showinfo("Coherence check", refusal)
            return
        self.coherence_button.configure(state="disabled")
        self.status_var.set("Cross-checking the analysis...")

    def on_close(self) -> None:
        if self.controller is not None and self.controller.state.running:
            if not messagebox.askokcancel(
                "Quit", "An analysis is running. Stop it and close?"
            ):
                return
        if self.controller is not None:
            self.controller.shutdown()
        self.master_window.destroy()

    # ------------------------------------------------------------------ polling

    def _clear_output(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self.results.delete(*self.results.get_children())
        self._clear_conflicts()

    def _clear_conflicts(self) -> None:
        self.conflicts.delete(*self.conflicts.get_children())

    def _append_log(self, event: Event) -> None:
        tag = LOG_TAGS.get(event.type, "")
        self.log.configure(state="normal")
        self.log.insert("end", f"{event.message}\n", tag)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _append_result(self, event: Event) -> None:
        data = event.data
        self.results.insert(
            "",
            "end",
            values=(
                f"0x{int(data.get('address', 0)):08x}",
                data.get("original_name", ""),
                data.get("name", ""),
                f"{data.get('confidence', 0)}%",
                data.get("classification", ""),
            ),
        )

    def _append_conflict(self, event: Event) -> None:
        """One detected contradiction, keyed so its resolution can find it."""
        data = event.data
        addresses = data.get("addresses", [])
        # An id Tk already holds would raise, and an exception in here stops
        # the poll loop from rearming itself — the window would freeze.
        item = str(data.get("id") or "")
        if item and self.conflicts.exists(item):
            return
        self.conflicts.insert(
            "",
            "end",
            iid=item or None,
            values=(
                str(data.get("kind", "")).replace("_", " "),
                ", ".join(f"0x{int(a):08x}" for a in addresses),
                data.get("summary", ""),
                "pending",
            ),
        )

    def _resolve_conflict(self, event: Event) -> None:
        data = event.data
        item = str(data.get("id", ""))
        if not item or not self.conflicts.exists(item):
            return
        applied = data.get("applied") or []
        self.conflicts.set(
            item,
            "resolution",
            "; ".join(applied) if applied
            else f"kept — {data.get('explanation', 'no change')}",
        )

    def _poll(self) -> None:
        controller = self.controller
        if controller is not None:
            for event in controller.poll():
                self._append_log(event)
                if event.type is EventType.FUNCTION_COMPLETE:
                    self._append_result(event)
                elif event.type is EventType.COHERENCE_CHECKED:
                    # Emitted once, before the conflicts of this pass: the
                    # table shows the current review, not every review.
                    self._clear_conflicts()
                elif event.type is EventType.COHERENCE_CONFLICT:
                    self._append_conflict(event)
                elif event.type is EventType.COHERENCE_RESOLVED:
                    self._resolve_conflict(event)
            self._render_state()
        self.after(POLL_INTERVAL_MS, self._poll)

    def _render_state(self) -> None:
        assert self.controller is not None
        state = self.controller.state
        self.progress.set(state.progress_fraction)

        if state.finished:
            self.start_button.configure(state="normal")
            self.pause_button.configure(state="disabled", text="Pause")

        self.coherence_button.configure(
            state="disabled" if state.checking_coherence or state.finishing else "normal"
        )
        self.finish_button.configure(
            state="disabled"
            if state.finishing or state.checking_coherence or not state.pending_finish
            else "normal"
        )

        if state.error:
            self.status_var.set(f"Failed: {state.error}")
            return

        pieces = [
            f"Phase: {state.phase}",
            f"{state.completed}/{state.total} functions",
            f"conf {state.high_confidence}/{state.medium_confidence}/{state.low_confidence}",
            f"{state.llm_calls} LLM calls",
            f"${state.cost_usd:.4f}",
            f"{state.elapsed_seconds:.0f}s",
        ]
        if state.llm_waiting:
            # Minutes can pass inside one request with nothing else moving,
            # which is the stretch that otherwise looks like a hung window.
            waiting = f"waiting on {state.llm_wait_label or 'the model'}"
            pieces.append(f"{waiting} ({state.llm_wait_seconds:.0f}s)")
        if state.paused:
            pieces.append("PAUSED")
        if state.pending_finish:
            pieces.append(f"{state.pending_finish} to finish")
        if state.finishing:
            pieces.append("FINISHING")
        if state.checking_coherence:
            pieces.append("CHECKING COHERENCE")
        self.status_var.set("   ".join(pieces))


def launch(initial_binary: str = "") -> None:
    """Open the Kong window and run until it is closed."""
    apply_ui_scale()
    root = ctk.CTk()
    root.title("Kong — agentic reverse engineer")
    width, height = fit_to_screen(
        root.winfo_screenwidth(), root.winfo_screenheight()
    )
    root.geometry(f"{width}x{height}")
    root.minsize(min(880, width), min(600, height))
    KongWindow(root, initial_binary=initial_binary)
    root.mainloop()
