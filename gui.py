"""
Click-to-run GUI for generating the synthetic dataset (no terminal needed).

Launch by double-clicking "Generate Images.bat", or run:
    python gui.py
"""

import json
import math
import os
import re
import sys
import time
import queue
import threading
import subprocess
import traceback
import webbrowser
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import ttk, messagebox, simpledialog

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image, ImageDraw, ImageFont, ImageTk

import random

import config
from src import fields
from src.augment import degrade, EDGE_CLIPPING_RANGES
from src.generation_profiles import (AUGMENTATION_PROFILES,
                                     StrokeDamageProfile,
                                     augmentation_profile_from_spec,
                                     augmentation_spec_from_legacy_values,
                                     get_augmentation_profile)
from src.generate_synthetic import (GenerationCancelled, field_weights_for_mode,
                                    generate, guarded_remove_dataset,
                                    zip_dataset)
from src.generation_io import verify_sha256_sidecar
from src.generation_resources import preflight_generation_resources
from src.provenance import atomic_write_json
from src.build_splits import build
from src.render import (CURSIVE_STYLE_GROUPS, fonts_for_style,
                        font_display_name, font_path_for)

# ---------------------------------------------------------------------------
# Colour palette — modern light / white theme
# ---------------------------------------------------------------------------
BG        = "#f4f5f9"   # app background (soft off-white)
SURFACE   = "#ffffff"   # card / section background
BORDER    = "#e3e6ef"   # subtle border / separator
ACCENT    = "#4f46e5"   # primary indigo accent
ACCENT_H  = "#4338ca"   # accent hover
SUCCESS   = "#16a34a"   # green for done state
WARNING   = "#d97706"   # amber for running state
ERROR_C   = "#dc2626"   # red for error / delete
TEXT      = "#1f2430"   # primary text
SUBTEXT   = "#7a8194"   # muted / hint text
BTN_FG    = "#ffffff"   # button foreground
PREVIEW_BG = "#fafbff"  # preview canvas background

PREVIEW_WIDTH = 460
SAMPLE_TEXT = "Juan Dela Cruz"
LOG_DIR = config.ROOT / "logs"
LOG_FILE = LOG_DIR / "gui.log"
PRESET_DIR = config.RESOURCES_DIR / "presets"
MAX_LOG_BYTES = 2 * 1024 * 1024
LOG_BACKUPS = 3


@dataclass(frozen=True)
class SliderSpec:
    """One Developer Studio slider: range, step, unit, and default."""

    name: str
    label: str
    low: float
    high: float
    default: float
    step: float
    unit: str = ""


DEGRADATION_SLIDER_SPECS = (
    SliderSpec("fade", "Ink contrast", 0.10, 1.00, 0.65, 0.01),
    SliderSpec("tint", "Paper tint", 0.00, 0.40, 0.15, 0.01),
    SliderSpec("stain", "Stain chance", 0.00, 1.00, 0.35, 0.01),
    SliderSpec("rotate", "Tilt angle", 0.0, 12.0, 4.5, 0.1, "°"),
    SliderSpec("blur", "Blur radius", 0.0, 3.0, 0.0, 0.1, "px"),
    SliderSpec("noise", "Grain noise", 0.0, 40.0, 12.0, 1.0),
    SliderSpec("texture", "Paper fiber", 0.0, 12.0, 3.5, 0.1),
    SliderSpec("scanline", "Scanlines", 0.0, 30.0, 10.0, 1.0),
    SliderSpec("jpeg", "JPEG quality", 20.0, 100.0, 75.0, 1.0),
)
CROP_SLIDER_SPECS = (
    SliderSpec("crop_top", "Top cut", 0.0, 15.0, 0.0, 1.0, "%"),
    SliderSpec("crop_bottom", "Bottom cut", 0.0, 15.0, 0.0, 1.0, "%"),
    SliderSpec("crop_left", "Left cut", 0.0, 15.0, 0.0, 1.0, "%"),
    SliderSpec("crop_right", "Right cut", 0.0, 15.0, 0.0, 1.0, "%"),
)
STROKE_SLIDER_SPECS = (
    SliderSpec("contrast", "Contrast-drop chance", 0.00, 1.00, 0.80, 0.01),
    SliderSpec("gap_prob", "Gap chance", 0.00, 1.00, 0.90, 0.01),
    SliderSpec("gap_count", "Max gaps", 0.0, 30.0, 16.0, 1.0),
    SliderSpec("scratch", "Scratch chance", 0.00, 1.00, 0.70, 0.01),
    SliderSpec("erode", "Stroke-thin chance", 0.00, 1.00, 0.55, 0.01),
)
SLIDER_SPECS = {
    spec.name: spec
    for spec in (*DEGRADATION_SLIDER_SPECS, *CROP_SLIDER_SPECS, *STROKE_SLIDER_SPECS)
}
STROKE_DAMAGE_SLIDERS = tuple(spec.name for spec in STROKE_SLIDER_SPECS)
SLIDER_LABEL_WIDTH = 20


def _step_decimals(step: float) -> int:
    """Number of decimal places a step needs (0.01 -> 2, 0.5 -> 1, 1 -> 0)."""
    text = f"{step:.10f}".rstrip("0").rstrip(".")
    return len(text.partition(".")[2])


def snap_value(value, step: float, low: float, high: float) -> float:
    """Clamp ``value`` to [low, high] and round it to the nearest ``step``."""
    value = float(value)
    step = float(step)
    if not math.isfinite(value):
        raise ValueError(f"value must be finite; got {value!r}")
    if not math.isfinite(step) or step <= 0:
        raise ValueError(f"step must be a positive finite number; got {step!r}")
    if low > high:
        raise ValueError(f"low ({low}) must not exceed high ({high})")
    clamped = min(high, max(low, value))
    snapped = round(round(clamped / step) * step, _step_decimals(step))
    return float(min(high, max(low, snapped)))


def format_value(value, step: float, unit: str = "") -> str:
    """Format ``value`` with exactly the precision its step implies."""
    text = f"{float(value):.{_step_decimals(step)}f}"
    if text.startswith("-") and float(text) == 0:
        text = text[1:]
    return f"{text}{unit}"


def _rotate_log(log_path: Path | None = None) -> None:
    """Keep a few bounded log generations without relying on logging threads."""
    log_path = LOG_FILE if log_path is None else Path(log_path)
    try:
        if not log_path.is_file() or log_path.stat().st_size < MAX_LOG_BYTES:
            return
        oldest = log_path.with_name(f"{log_path.name}.{LOG_BACKUPS}")
        oldest.unlink(missing_ok=True)
        for index in range(LOG_BACKUPS - 1, 0, -1):
            source = log_path.with_name(f"{log_path.name}.{index}")
            if source.exists():
                source.replace(log_path.with_name(f"{log_path.name}.{index + 1}"))
        log_path.replace(log_path.with_name(f"{log_path.name}.1"))
    except OSError:
        pass


def _log_error(context: str, detail: str) -> Path:
    """Persist a full error/traceback for GUI and pythonw launches."""
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        _rotate_log()
        timestamp = datetime.now().astimezone().isoformat()
        with LOG_FILE.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(f"[{timestamp}] {context}\n{detail.rstrip()}\n\n")
    except OSError:
        pass
    return LOG_FILE


LOG_DIR.mkdir(parents=True, exist_ok=True)
if sys.stdout is None:
    sys.stdout = open(LOG_FILE.with_name("gui-stdout.log"), "a", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(LOG_FILE, "a", encoding="utf-8")


PRESET_METADATA_KEYS = frozenset({"name", "description"})


def parse_preset_values(data) -> tuple[dict[str, float], list[str]]:
    """Validate a flat studio preset before anything is applied.

    Returns the slider values (clamped and snapped to each slider) and the
    sorted unknown keys.  Any non-numeric or non-finite slider value raises
    ``ValueError`` so a malformed preset is never half-applied.
    """
    if not isinstance(data, dict):
        raise ValueError("Preset file must contain a JSON object.")
    values: dict[str, float] = {}
    problems = []
    for key, raw in data.items():
        spec = SLIDER_SPECS.get(key)
        if spec is None:
            continue
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            problems.append(f"{key} must be a number (got {raw!r})")
            continue
        if not math.isfinite(float(raw)):
            problems.append(f"{key} must be finite (got {raw!r})")
            continue
        values[key] = snap_value(raw, spec.step, spec.low, spec.high)
    if problems:
        raise ValueError("Invalid preset values: " + "; ".join(problems))
    ignored = sorted(
        key for key in data if key not in SLIDER_SPECS and key not in PRESET_METADATA_KEYS
    )
    return values, ignored


_PRESET_PATH_CHARACTERS = frozenset('/\\:*?"<>|')
MAX_PRESET_NAME_LENGTH = 64


def preset_slug(name: str) -> str:
    """Turn a display name into a safe preset filename stem.

    Letters are lowercased and runs of other characters become ``_``.  Names
    with path characters, ``..``, control characters, Windows reserved device
    names, or an empty/over-long result are rejected.
    """
    text = (name or "").strip()
    if not text:
        raise ValueError("Preset name cannot be blank.")
    if (".." in text or any(ch in _PRESET_PATH_CHARACTERS for ch in text)
            or any(ord(ch) < 32 for ch in text)):
        raise ValueError(
            "Preset names cannot contain path characters such as / \\ : or '..'.")
    slug = re.sub(r"[^a-z0-9_-]+", "_", text.lower())
    slug = re.sub(r"_+", "_", slug).strip("_-")
    if not slug:
        raise ValueError("Preset name must contain at least one letter or digit.")
    if len(slug) > MAX_PRESET_NAME_LENGTH:
        raise ValueError(
            f"Preset name is too long (maximum {MAX_PRESET_NAME_LENGTH} characters).")
    if slug.upper() in config._WINDOWS_RESERVED_NAMES:
        raise ValueError(f"Preset name uses a reserved Windows device name: {slug!r}")
    return slug


def preset_path_for(slug: str, preset_dir: Path | None = None) -> Path:
    """Return the preset file for ``slug``, refusing anything outside the folder."""
    preset_dir = PRESET_DIR if preset_dir is None else Path(preset_dir)
    path = preset_dir / f"{slug}.json"
    if path.resolve(strict=False).parent != preset_dir.resolve(strict=False):
        raise ValueError(f"Preset path escapes the preset folder: {path}")
    return path


def preset_payload(display_name: str, slider_values: dict[str, float]) -> dict:
    """Preset JSON with every slider value snapped to its step."""
    data: dict[str, object] = {"name": display_name}
    for key, raw in slider_values.items():
        spec = SLIDER_SPECS[key]
        value = snap_value(raw, spec.step, spec.low, spec.high)
        data[key] = int(value) if spec.step >= 1 and float(spec.step).is_integer() else value
    return data


def _split_percent(fraction: float) -> str:
    """Display a split fraction as the percent text the split entries use."""
    return f"{round(fraction * 100.0, 2):g}"


def mode_field_weights(sample_mode: str) -> dict[str, int]:
    """Weights for every supported field type, 0 for types the mode skips."""
    weights = field_weights_for_mode(sample_mode)
    return {
        ftype: int(weights.get(ftype, 0))
        for ftype in sorted(config.SUPPORTED_FIELD_TYPES)
    }


class RunOptionsError(ValueError):
    """Invalid GUI input, carrying the title of the dialog that reports it."""

    def __init__(self, title: str, message: str):
        super().__init__(message)
        self.title = title


@dataclass(frozen=True)
class RunOptions:
    """Plain-value snapshot of every Tk-backed option a generation run uses."""

    count: int
    dataset: str | None
    seed: int
    names_version: str | None
    sample_mode: str
    font_style: str
    cursive_group: str
    specific_font: str
    augmentation_profile: object
    edge_clipping: object
    stroke_damage_profile: StrokeDamageProfile | None
    merge_real: bool
    package_zip: bool
    custom_field_weights: dict | None
    custom_split_fractions: tuple | None
    workers: int | None


def _open_path_portably(path: Path, *, select_file: bool = False) -> None:
    """Open a directory or select a file using the host platform."""
    path = Path(path).resolve(strict=True)
    if select_file and path.is_file():
        if sys.platform.startswith("win"):
            subprocess.Popen(["explorer", "/select,", str(path)])
            return
        path = path.parent
    if sys.platform.startswith("win"):
        os.startfile(str(path))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    elif sys.platform.startswith("linux"):
        subprocess.Popen(["xdg-open", str(path)])
    elif not webbrowser.open(path.as_uri()):
        raise OSError(f"No supported folder opener for {sys.platform}")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Synthetic Dataset Generator")
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        # Leave enough room for the host's taskbar/window chrome, including on
        # small remote-desktop and accessibility-scaled displays.  The tabs use
        # scrollable content, so a compact initial window remains usable.
        width = min(1240, max(640, screen_width - 60))
        height = min(820, max(480, screen_height - 60))
        self.geometry(f"{width}x{height}")
        self.minsize(640, 440)
        self.resizable(True, True)
        self.configure(bg=BG)

        self.q = queue.Queue()
        self.worker = None
        self.delete_worker = None
        self.cancel_event = threading.Event()
        self._closing = False
        self._pending_terminal = None
        self._pending_delete = None
        self._run_controls = []
        self._run_control_states = []
        self.start_time = None
        self._last_output = None       # path of the last finished dataset
        self._preview_img = None       # keep a ref so Tk doesn't GC it
        self._dataset_paths = {}       # tree iid -> Path
        self._refresh_generation = 0
        self._refresh_worker = None
        self._refresh_pending = False
        self._refresh_finalize_scheduled = False
        self._merge_result = None

        self.preview_seed = 42
        self.preview_text_var = tk.StringVar(value="Juan Dela Cruz")
        self.preview_field_var = tk.StringVar(value="Auto (Sampled)")
        self.preview_degraded_var = tk.BooleanVar(value=True)

        self.split_train_var = tk.StringVar(value=_split_percent(config.SYNTH_TRAIN_FRAC))
        self.split_val_var = tk.StringVar(value=_split_percent(config.SYNTH_VAL_FRAC))
        self.split_test_var = tk.StringVar(value=_split_percent(config.SYNTH_TEST_FRAC))
        self.field_weight_vars = {}
        self._field_weights_mode = config.DEFAULT_SAMPLE_MODE
        for ftype, weight in mode_field_weights(config.DEFAULT_SAMPLE_MODE).items():
            self.field_weight_vars[ftype] = tk.StringVar(value=str(weight))
        self.preset_var = tk.StringVar(value="")

        self._apply_theme()
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._update_preflight_estimates()
        self.after(150, self._update_preview)
        self.after(150, self._refresh_datasets)
        self.after(100, self._poll)

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------
    def _apply_theme(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure(".",
                        background=BG,
                        foreground=TEXT,
                        fieldbackground=SURFACE,
                        bordercolor=BORDER,
                        darkcolor=SURFACE,
                        lightcolor=SURFACE,
                        troughcolor="#e9ecf5",
                        selectbackground=ACCENT,
                        selectforeground=BTN_FG,
                        font=("Segoe UI", 10))

        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=SURFACE,
                        relief="flat", borderwidth=1)

        style.configure("TLabel", background=BG, foreground=TEXT,
                        font=("Segoe UI", 10))
        style.configure("Section.TLabel", background=BG, foreground=ACCENT,
                        font=("Segoe UI", 9, "bold"))
        style.configure("Title.TLabel", background=BG, foreground=TEXT,
                        font=("Segoe UI", 14, "bold"))
        style.configure("Sub.TLabel", background=BG, foreground=SUBTEXT,
                        font=("Segoe UI", 9))
        style.configure("Status.TLabel", background=BG, foreground=TEXT,
                        font=("Segoe UI", 9))
        style.configure("Timer.TLabel", background=BG, foreground=SUBTEXT,
                        font=("Segoe UI", 9))
        style.configure("CardLabel.TLabel", background=SURFACE, foreground=TEXT,
                        font=("Segoe UI", 10))
        style.configure("CardHint.TLabel", background=SURFACE, foreground=SUBTEXT,
                        font=("Segoe UI", 9))
        style.configure("Preview.TLabel", background=PREVIEW_BG, foreground=SUBTEXT,
                        font=("Segoe UI", 9))

        style.configure("TEntry",
                        fieldbackground="#ffffff",
                        foreground=TEXT,
                        insertcolor=TEXT,
                        bordercolor=BORDER,
                        relief="flat",
                        padding=4)
        style.map("TEntry", bordercolor=[("focus", ACCENT)])

        style.configure("TCombobox",
                        fieldbackground="#ffffff",
                        background="#ffffff",
                        foreground=TEXT,
                        arrowcolor=ACCENT,
                        bordercolor=BORDER,
                        relief="flat",
                        padding=4,
                        selectbackground="#ffffff",
                        selectforeground=TEXT)
        style.map("TCombobox",
                  fieldbackground=[("readonly", "#ffffff")],
                  foreground=[("readonly", TEXT)],
                  bordercolor=[("focus", ACCENT)])

        style.configure("TCheckbutton",
                        background=SURFACE,
                        foreground=TEXT,
                        indicatorcolor="#ffffff",
                        indicatorbackground="#ffffff",
                        focuscolor=SURFACE)
        style.map("TCheckbutton",
                  indicatorcolor=[("selected", ACCENT)],
                  background=[("active", SURFACE)],
                  foreground=[("active", TEXT)])

        style.configure("TProgressbar",
                        troughcolor="#e9ecf5",
                        background=ACCENT,
                        bordercolor="#e9ecf5",
                        lightcolor=ACCENT,
                        darkcolor=ACCENT,
                        thickness=8)

        # High-contrast sliders
        style.configure("Horizontal.TScale",
                        troughcolor="#dbe1f2",
                        background=ACCENT,
                        bordercolor=ACCENT,
                        lightcolor=ACCENT,
                        darkcolor=ACCENT,
                        thickness=14)
        style.map("Horizontal.TScale",
                  background=[("active", ACCENT_H), ("pressed", "#3730a3")],
                  troughcolor=[("active", "#cbd5e1")])

        # Notebook (tabs)
        style.configure("TNotebook", background=BG, borderwidth=0, tabmargins=(6, 6, 6, 0))
        style.configure("TNotebook.Tab",
                        background=BG,
                        foreground=SUBTEXT,
                        padding=(18, 8),
                        font=("Segoe UI", 10, "bold"))
        style.map("TNotebook.Tab",
                  background=[("selected", SURFACE)],
                  foreground=[("selected", ACCENT)])

        # Treeview (datasets list)
        style.configure("Treeview",
                        background=SURFACE,
                        fieldbackground=SURFACE,
                        foreground=TEXT,
                        rowheight=30,
                        borderwidth=0,
                        font=("Segoe UI", 10))
        style.configure("Treeview.Heading",
                        background=BG,
                        foreground=SUBTEXT,
                        relief="flat",
                        font=("Segoe UI", 9, "bold"))
        style.map("Treeview",
                  background=[("selected", ACCENT)],
                  foreground=[("selected", BTN_FG)])
        style.map("Treeview.Heading", background=[("active", BG)])

        # Buttons
        style.configure("Quick.TButton",
                        background="#eef0f7",
                        foreground=TEXT,
                        bordercolor=BORDER,
                        relief="flat",
                        padding=(6, 4),
                        font=("Segoe UI", 9))
        style.map("Quick.TButton",
                  background=[("active", "#e0e3ef"), ("disabled", "#f0f1f6")],
                  foreground=[("disabled", SUBTEXT)])

        style.configure("Generate.TButton",
                        background=ACCENT,
                        foreground=BTN_FG,
                        bordercolor=ACCENT,
                        relief="flat",
                        padding=(20, 10),
                        font=("Segoe UI", 11, "bold"))
        style.map("Generate.TButton",
                  background=[("active", ACCENT_H), ("disabled", "#c7c9d6")],
                  foreground=[("disabled", "#ffffff")])

        style.configure("Danger.TButton",
                        background=ERROR_C,
                        foreground=BTN_FG,
                        bordercolor=ERROR_C,
                        relief="flat",
                        padding=(12, 6),
                        font=("Segoe UI", 10, "bold"))
        style.map("Danger.TButton",
                  background=[("active", "#b91c1c"), ("disabled", "#e7a9a9")],
                  foreground=[("disabled", "#ffffff")])

        style.configure("Secondary.TButton",
                        background="#eef0f7",
                        foreground=TEXT,
                        bordercolor=BORDER,
                        relief="flat",
                        padding=(12, 6),
                        font=("Segoe UI", 10))
        style.map("Secondary.TButton",
                  background=[("active", "#e0e3ef"), ("disabled", "#f0f1f6")],
                  foreground=[("disabled", SUBTEXT)])

        # Combobox dropdown list colours
        self.option_add("*TCombobox*Listbox.background", SURFACE)
        self.option_add("*TCombobox*Listbox.foreground", TEXT)
        self.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
        self.option_add("*TCombobox*Listbox.selectForeground", BTN_FG)
        self.option_add("*TCombobox*Listbox.font", ("Segoe UI", 10))

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------
    def _card(self, parent, pady=(0, 0)):
        outer = ttk.Frame(parent)
        outer.pack(fill="x", padx=16, pady=pady)
        card = ttk.Frame(outer, style="Card.TFrame", padding=12)
        card.pack(fill="x")
        return card

    def _section_label(self, parent, text):
        ttk.Label(parent, text=text.upper(), style="Section.TLabel").pack(
            anchor="w", padx=16, pady=(12, 2))

    # ------------------------------------------------------------------
    # UI layout
    # ------------------------------------------------------------------
    def _build_ui(self):
        # ---- Header -------------------------------------------------------
        hdr = ttk.Frame(self)
        hdr.pack(fill="x", padx=16, pady=(16, 4))
        ttk.Label(hdr, text="Synthetic Generator", style="Title.TLabel").pack(anchor="w")
        ttk.Label(hdr, text="Synthetic Dataset Generator", style="Sub.TLabel").pack(anchor="w")

        # ---- Tabs ---------------------------------------------------------
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=8, pady=(8, 8))

        self.tab_generate = ttk.Frame(self.nb)
        self.tab_datasets = ttk.Frame(self.nb)
        self.nb.add(self.tab_generate, text="  Generate  ")
        self.nb.add(self.tab_datasets, text="  Datasets  ")
        self.nb.bind("<<NotebookTabChanged>>", self._on_tab_change)

        self._build_generate_tab(self.tab_generate)
        self._build_datasets_tab(self.tab_datasets)

    # ------------------------------------------------------------------
    def _build_generate_tab(self, parent):
        # Two-column layout: Left pane scrolls controls & Developer Studio,
        # Right pane stays fixed with Live Preview & Generation Progress.
        cols = ttk.Panedwindow(parent, orient="horizontal")
        cols.pack(fill="both", expand=True, padx=8, pady=(4, 4))

        left_container = ttk.Frame(cols)
        right_container = ttk.Frame(cols)
        cols.add(left_container, weight=5)
        cols.add(right_container, weight=5)

        # Left pane viewport (independent scrollable canvas for controls)
        viewport = tk.Canvas(
            left_container, bg=BG, highlightthickness=0, borderwidth=0
        )
        scrollbar = ttk.Scrollbar(
            left_container, orient="vertical", command=viewport.yview
        )
        viewport.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        viewport.pack(side="left", fill="both", expand=True)

        left = ttk.Frame(viewport)
        content_window = viewport.create_window((0, 0), window=left, anchor="nw")
        left.bind(
            "<Configure>",
            lambda _event: viewport.configure(scrollregion=viewport.bbox("all")),
        )
        viewport.bind(
            "<Configure>",
            lambda event: viewport.itemconfigure(content_window, width=event.width),
        )

        # Right pane viewport (independent scrollable canvas for preview & progress)
        right_viewport = tk.Canvas(
            right_container, bg=BG, highlightthickness=0, borderwidth=0
        )
        right_scrollbar = ttk.Scrollbar(
            right_container, orient="vertical", command=right_viewport.yview
        )
        right_viewport.configure(yscrollcommand=right_scrollbar.set)
        right_scrollbar.pack(side="right", fill="y")
        right_viewport.pack(side="left", fill="both", expand=True)

        right = ttk.Frame(right_viewport)
        right_window = right_viewport.create_window((0, 0), window=right, anchor="nw")
        right.bind(
            "<Configure>",
            lambda _event: right_viewport.configure(scrollregion=right_viewport.bbox("all")),
        )
        right_viewport.bind(
            "<Configure>",
            lambda event: right_viewport.itemconfigure(right_window, width=event.width),
        )

        self._target_scroll_pos = 0.0
        self._smooth_scroll_timer = None

        def _on_mousewheel(event):
            try:
                if self.nb.index(self.nb.select()) != 0:
                    return
            except Exception:
                pass

            target_vp = viewport
            try:
                w = self.winfo_containing(event.x_root, event.y_root)
                if w and (str(right_container) in str(w) or w == right_container or w == right):
                    target_vp = right_viewport
            except Exception:
                pass

            bbox = target_vp.bbox("all")
            if not bbox:
                return
            content_height = bbox[3] - bbox[1]
            visible_height = target_vp.winfo_height()
            if content_height <= visible_height or visible_height <= 0:
                return

            scrollable_height = content_height - visible_height
            current_target = target_vp.yview()[0]

            step_pixels = 22.0
            if event.num == 4:
                delta_px = -step_pixels
            elif event.num == 5:
                delta_px = step_pixels
            elif event.delta:
                raw_notches = -(event.delta / 120.0)
                clamped_notches = max(-2.5, min(2.5, raw_notches))
                delta_px = clamped_notches * step_pixels
            else:
                return

            delta_fraction = delta_px / scrollable_height
            new_target = max(0.0, min(1.0, current_target + delta_fraction))
            target_vp.yview_moveto(new_target)

        def _bind_mousewheel(_event=None):
            self.bind_all("<MouseWheel>", _on_mousewheel)
            self.bind_all("<Button-4>", _on_mousewheel)
            self.bind_all("<Button-5>", _on_mousewheel)

        def _unbind_mousewheel(_event=None):
            self.unbind_all("<MouseWheel>")
            self.unbind_all("<Button-4>")
            self.unbind_all("<Button-5>")

        self._bind_gen_mousewheel = _bind_mousewheel
        self._unbind_gen_mousewheel = _unbind_mousewheel
        self._gen_viewport = viewport

        _bind_mousewheel()

        def _bind_tree_mousewheel(widget):
            widget.bind("<MouseWheel>", _on_mousewheel, add="+")
            widget.bind("<Button-4>", _on_mousewheel, add="+")
            widget.bind("<Button-5>", _on_mousewheel, add="+")
            for child in widget.winfo_children():
                _bind_tree_mousewheel(child)

        self.after(200, lambda: _bind_tree_mousewheel(left_container))
        self.after(200, lambda: _bind_tree_mousewheel(right_container))

        # ================= LEFT: settings =================================
        # ---- Generation settings -----------------------------------------
        self._section_label(left, "Generation")
        card1 = self._card(left, pady=(0, 4))

        crow = ttk.Frame(card1, style="Card.TFrame")
        crow.pack(fill="x", pady=(0, 8))
        ttk.Label(crow, text="Samples", style="CardLabel.TLabel", width=13).pack(side="left")
        self.count_var = tk.StringVar(value=str(config.DEFAULT_COUNT))
        self.count_entry = ttk.Entry(crow, textvariable=self.count_var, width=10)
        self.count_entry.pack(side="left")
        self._run_controls.append(self.count_entry)

        srow = ttk.Frame(card1, style="Card.TFrame")
        srow.pack(fill="x", pady=(0, 8))
        ttk.Label(srow, text="Seed", style="CardLabel.TLabel", width=13).pack(side="left")
        self.seed_var = tk.StringVar(value=str(config.RANDOM_SEED))
        self.seed_entry = ttk.Entry(srow, textvariable=self.seed_var, width=10)
        self.seed_entry.pack(side="left", padx=(0, 8))
        self._run_controls.append(self.seed_entry)
        ttk.Label(srow, text="blank = random seed", style="CardHint.TLabel").pack(side="left")

        qp = ttk.Frame(card1, style="Card.TFrame")
        qp.pack(fill="x", pady=(0, 8))
        ttk.Label(qp, text="", width=13).pack(side="left")
        for n in (1_000, 5_000, 20_000, 40_000):
            quick_btn = ttk.Button(
                qp, text=f"{n:,}", width=7, style="Quick.TButton",
                command=lambda v=n: self.count_var.set(str(v)))
            quick_btn.pack(side="left", padx=2)
            self._run_controls.append(quick_btn)

        drow = ttk.Frame(card1, style="Card.TFrame")
        drow.pack(fill="x", pady=(0, 8))
        ttk.Label(drow, text="Dataset folder", style="CardLabel.TLabel", width=13).pack(side="left")
        self.dataset_var = tk.StringVar(value="(next)")
        self.dataset_entry = ttk.Entry(drow, textvariable=self.dataset_var, width=14)
        self.dataset_entry.pack(side="left", padx=(0, 8))
        self._run_controls.append(self.dataset_entry)
        ttk.Label(drow, text="blank = next", style="CardHint.TLabel").pack(side="left")

        wrow = ttk.Frame(card1, style="Card.TFrame")
        wrow.pack(fill="x", pady=(0, 8))
        ttk.Label(wrow, text="CPU Threads", style="CardLabel.TLabel", width=13).pack(side="left")
        cpu_max = os.cpu_count() or 16
        thread_opts = ["Auto (Max)"] + [str(i) for i in range(1, min(cpu_max + 1, 33))]
        self.workers_var = tk.StringVar(value="Auto (Max)")
        self.workers_cb = ttk.Combobox(wrow, textvariable=self.workers_var, values=thread_opts, state="readonly", width=12)
        self.workers_cb.pack(side="left", padx=(0, 8))
        self.workers_cb.bind("<<ComboboxSelected>>", lambda e: self._update_preflight_estimates())
        self._run_controls.append(self.workers_cb)
        ttk.Label(wrow, text=f"Max: {cpu_max} threads", style="CardHint.TLabel").pack(side="left")

        nrow = ttk.Frame(card1, style="Card.TFrame")
        nrow.pack(fill="x")
        ttk.Label(nrow, text="Names pool", style="CardLabel.TLabel", width=13).pack(side="left")
        versions = config.name_versions() or [config.NAMES_VERSION]
        default_v = config.NAMES_VERSION if config.NAMES_VERSION in versions else versions[0]
        self.names_var = tk.StringVar(value=default_v)
        self.names_cb = ttk.Combobox(
            nrow, textvariable=self.names_var, values=versions,
            state="readonly", width=12)
        self.names_cb.pack(side="left")
        self._run_controls.append(self.names_cb)

        # ---- Style settings ----------------------------------------------
        self._section_label(left, "Style")
        card2 = self._card(left, pady=(0, 4))

        mrow = ttk.Frame(card2, style="Card.TFrame")
        mrow.pack(fill="x", pady=(0, 8))
        ttk.Label(mrow, text="Sample mode", style="CardLabel.TLabel", width=13).pack(side="left")
        self.mode_by_label = {label: key for key, label in config.SAMPLE_MODES.items()}
        mode_labels = list(self.mode_by_label)
        self.sample_mode_var = tk.StringVar(value=config.SAMPLE_MODES[config.DEFAULT_SAMPLE_MODE])
        self.sample_mode_cb = ttk.Combobox(
            mrow, textvariable=self.sample_mode_var, values=mode_labels,
            state="readonly", width=26)
        self.sample_mode_cb.pack(side="left")
        self.sample_mode_cb.bind("<<ComboboxSelected>>", self._on_sample_mode_change)
        self._run_controls.append(self.sample_mode_cb)

        # Degradation Profile Dropdown
        deg_row = ttk.Frame(card2, style="Card.TFrame")
        deg_row.pack(fill="x", pady=(0, 8))
        ttk.Label(deg_row, text="Degradation", style="CardLabel.TLabel", width=13).pack(side="left")
        self.degradation_by_label = {label: key for key, label in config.DEGRADATION_PROFILES.items()}
        degradation_labels = list(self.degradation_by_label)
        self.degradation_var = tk.StringVar(value=config.DEGRADATION_PROFILES[config.DEFAULT_DEGRADATION_PROFILE])
        self.degradation_cb = ttk.Combobox(
            deg_row, textvariable=self.degradation_var, values=degradation_labels,
            state="readonly", width=26)
        self.degradation_cb.pack(side="left")
        self.degradation_cb.bind("<<ComboboxSelected>>", self._on_degradation_preset_change)
        self._run_controls.append(self.degradation_cb)

        # Character Cutoff / Edge Clipping Dropdown
        clip_row = ttk.Frame(card2, style="Card.TFrame")
        clip_row.pack(fill="x", pady=(0, 8))
        ttk.Label(clip_row, text="Char cutoff", style="CardLabel.TLabel", width=13).pack(side="left")
        self.clipping_by_label = {label: key for key, label in config.EDGE_CLIPPING_OPTIONS.items()}
        clipping_labels = list(self.clipping_by_label)
        self.edge_clipping_var = tk.StringVar(value=config.EDGE_CLIPPING_OPTIONS[config.DEFAULT_EDGE_CLIPPING])
        self.edge_clipping_cb = ttk.Combobox(
            clip_row, textvariable=self.edge_clipping_var, values=clipping_labels,
            state="readonly", width=26)
        self.edge_clipping_cb.pack(side="left")
        self.edge_clipping_cb.bind("<<ComboboxSelected>>", self._on_clipping_preset_change)
        self._run_controls.append(self.edge_clipping_cb)

        # ---- Advanced Distribution (Splits & Mix) -----------------------
        self._section_label(left, "Advanced Distribution (Splits & Mix)")
        card_dist = self._card(left, pady=(0, 4))

        split_row = ttk.Frame(card_dist, style="Card.TFrame")
        split_row.pack(fill="x", pady=(0, 6))
        ttk.Label(split_row, text="Split % (Tr/Val/Te)", style="CardLabel.TLabel", width=14).pack(side="left")

        self.split_train_entry = ttk.Entry(split_row, textvariable=self.split_train_var, width=4)
        self.split_train_entry.pack(side="left", padx=(0, 4))
        ttk.Label(split_row, text="/", style="CardHint.TLabel").pack(side="left", padx=(0, 4))

        self.split_val_entry = ttk.Entry(split_row, textvariable=self.split_val_var, width=4)
        self.split_val_entry.pack(side="left", padx=(0, 4))
        ttk.Label(split_row, text="/", style="CardHint.TLabel").pack(side="left", padx=(0, 4))

        self.split_test_entry = ttk.Entry(split_row, textvariable=self.split_test_var, width=4)
        self.split_test_entry.pack(side="left", padx=(0, 8))

        self.split_sum_lbl = ttk.Label(split_row, text="Sum: 100%", style="CardHint.TLabel")
        self.split_sum_lbl.pack(side="left")

        for entry in (self.split_train_entry, self.split_val_entry, self.split_test_entry):
            self._run_controls.append(entry)
            entry.bind("<KeyRelease>", self._update_split_sum_label)

        fw_hdr = ttk.Frame(card_dist, style="Card.TFrame")
        fw_hdr.pack(fill="x", pady=(6, 4))
        ttk.Label(fw_hdr, text="Field Type Weights", style="Sub.TLabel").pack(side="left")
        reset_fw_btn = ttk.Button(fw_hdr, text="🔄 Reset Weights", style="Secondary.TButton", command=self._reset_field_weights)
        reset_fw_btn.pack(side="right")
        self._run_controls.append(reset_fw_btn)

        fw_grid = ttk.Frame(card_dist, style="Card.TFrame")
        fw_grid.pack(fill="x")
        col_idx = 0
        row_idx = 0
        for ftype in sorted(config.SUPPORTED_FIELD_TYPES):
            cell = ttk.Frame(fw_grid, style="Card.TFrame")
            cell.grid(row=row_idx, column=col_idx, sticky="w", padx=3, pady=2)
            ttk.Label(cell, text=f"{ftype}:", style="CardHint.TLabel", width=12).pack(side="left")
            var = self.field_weight_vars[ftype]
            e = ttk.Entry(cell, textvariable=var, width=4)
            e.pack(side="left")
            self._run_controls.append(e)

            col_idx += 1
            if col_idx >= 2:
                col_idx = 0
                row_idx += 1

        # ---- Developer Studio (Custom Parameters) -----------------------
        dev_hdr_row = ttk.Frame(left)
        dev_hdr_row.pack(fill="x", pady=(8, 2))
        ttk.Label(dev_hdr_row, text="DEVELOPER STUDIO (CUSTOM PARAMETERS)", style="Section.TLabel").pack(side="left")
        
        # Studio Presets Header Bar
        preset_frame = ttk.Frame(dev_hdr_row)
        preset_frame.pack(side="right")

        self.preset_cb = ttk.Combobox(preset_frame, textvariable=self.preset_var, state="readonly", width=16)
        self.preset_cb.pack(side="left", padx=(0, 4))
        self._refresh_preset_list()

        load_preset_btn = ttk.Button(
            preset_frame, text="📂 Load", style="Secondary.TButton",
            command=self._load_selected_preset)
        load_preset_btn.pack(side="left", padx=(0, 4))
        self._run_controls.append(load_preset_btn)

        save_preset_btn = ttk.Button(
            preset_frame, text="💾 Save", style="Secondary.TButton",
            command=self._save_current_preset)
        save_preset_btn.pack(side="left", padx=(0, 4))
        self._run_controls.append(save_preset_btn)

        self.reset_btn = ttk.Button(
            preset_frame, text="🔄 Reset", style="Secondary.TButton",
            command=self._reset_studio_sliders)
        self.reset_btn.pack(side="left")
        self._run_controls.append(self.reset_btn)

        card_dev = self._card(left, pady=(0, 4))
        self.dev_sliders = {}
        self.dev_labels = {}
        self.dev_scales = {}
        self._slider_updating = False

        def _make_slider(parent, spec):
            row = ttk.Frame(parent, style="Card.TFrame")
            row.pack(fill="x", pady=(0, 4))
            ttk.Label(row, text=spec.label, style="CardLabel.TLabel",
                      width=SLIDER_LABEL_WIDTH).pack(side="left")
            val_var = tk.DoubleVar(value=spec.default)
            lbl_var = tk.StringVar(value=self._slider_text(spec.name, spec.default))

            def _on_move(val):
                if getattr(self, "_slider_updating", False):
                    return
                snapped = snap_value(val, spec.step, spec.low, spec.high)
                if snapped != val_var.get():
                    val_var.set(snapped)
                lbl_var.set(self._slider_text(spec.name, snapped))
                self._on_developer_slider_change(spec.name)

            scale = ttk.Scale(row, from_=spec.low, to=spec.high, variable=val_var,
                              command=_on_move)
            scale.pack(side="left", fill="x", expand=True, padx=(0, 6))
            ttk.Label(row, textvariable=lbl_var, style="CardHint.TLabel", width=7).pack(side="left")
            self.dev_sliders[spec.name] = val_var
            self.dev_labels[spec.name] = lbl_var
            self.dev_scales[spec.name] = scale
            self._run_controls.append(scale)

        ttk.Label(card_dev, text="Ink, paper and scanner (Ink contrast: lower = more faded)",
                  style="Sub.TLabel").pack(anchor="w", pady=(0, 4))
        for spec in DEGRADATION_SLIDER_SPECS:
            _make_slider(card_dev, spec)

        ttk.Separator(card_dev, orient="horizontal").pack(fill="x", pady=6)
        ttk.Label(card_dev, text="Character Cutoff / Edge Crops", style="Sub.TLabel").pack(anchor="w", pady=(0, 4))
        for spec in CROP_SLIDER_SPECS:
            _make_slider(card_dev, spec)

        ttk.Separator(card_dev, orient="horizontal").pack(fill="x", pady=6)
        ttk.Label(card_dev, text="Stroke Damage (semi-broken modes only)", style="Sub.TLabel").pack(anchor="w", pady=(0, 4))
        for spec in STROKE_SLIDER_SPECS:
            _make_slider(card_dev, spec)
        self._sync_stroke_damage_state()

        ttk.Separator(card_dev, orient="horizontal").pack(fill="x", pady=6)
        reset_bottom_btn = ttk.Button(
            card_dev, text="🔄 Reset All Controls to Default", style="Secondary.TButton",
            command=self._reset_all_controls)
        reset_bottom_btn.pack(fill="x", pady=(2, 2))
        self._run_controls.append(reset_bottom_btn)

        frow = ttk.Frame(card2, style="Card.TFrame")
        frow.pack(fill="x", pady=(0, 8))
        ttk.Label(frow, text="Font style", style="CardLabel.TLabel", width=13).pack(side="left")
        self.font_style_options = {"All fonts": "all", "Cursive only": "cursive"}
        self.font_style_var = tk.StringVar(value="All fonts")
        font_cb = ttk.Combobox(frow, textvariable=self.font_style_var,
                               values=list(self.font_style_options),
                               state="readonly", width=26)
        font_cb.pack(side="left")
        font_cb.bind("<<ComboboxSelected>>", self._on_font_style_change)
        self._run_controls.append(font_cb)

        # Cursive sub-style row (hidden until "Cursive only")
        self.cursive_row = ttk.Frame(card2, style="Card.TFrame")
        ttk.Label(self.cursive_row, text="Cursive style",
                  style="CardLabel.TLabel", width=13).pack(side="left")
        self.cursive_groups = ["All cursive"] + list(CURSIVE_STYLE_GROUPS.keys())
        self.cursive_group_var = tk.StringVar(value="All cursive")
        self.cursive_cb = ttk.Combobox(self.cursive_row,
                                       textvariable=self.cursive_group_var,
                                       values=self.cursive_groups,
                                       state="readonly", width=26)
        self.cursive_cb.pack(side="left")
        self.cursive_cb.bind("<<ComboboxSelected>>", self._on_cursive_group_change)
        self._run_controls.append(self.cursive_cb)

        # Specific font row (hidden until "Cursive only")
        self.specific_row = ttk.Frame(card2, style="Card.TFrame")
        ttk.Label(self.specific_row, text="Specific font",
                  style="CardLabel.TLabel", width=13).pack(side="left")
        self.specific_font_var = tk.StringVar(value="All in group")
        self.specific_cb = ttk.Combobox(self.specific_row,
                                        textvariable=self.specific_font_var,
                                        values=["All in group"],
                                        state="readonly", width=26)
        self.specific_cb.pack(side="left")
        self.specific_cb.bind("<<ComboboxSelected>>", lambda e: self._update_preview())
        self.specific_font_map = {"All in group": ""}   # display -> stem
        self._run_controls.append(self.specific_cb)

        # ---- Options -----------------------------------------------------
        self._section_label(left, "Options")
        card3 = self._card(left, pady=(0, 4))
        self.real_var = tk.BooleanVar(value=False)
        self.real_check = ttk.Checkbutton(
            card3, text="Merge real (mock) data afterwards",
            variable=self.real_var)
        self.real_check.pack(anchor="w", pady=(0, 4))
        self._run_controls.append(self.real_check)
        self.zip_var = tk.BooleanVar(value=False)
        self.zip_check = ttk.Checkbutton(
            card3, text="Package dataset as .zip when done",
            variable=self.zip_var)
        self.zip_check.pack(anchor="w")
        self._run_controls.append(self.zip_check)

        # Trace variables for preflight estimate updates
        self.count_var.trace_add("write", lambda *args: self._update_preflight_estimates())
        self.dataset_var.trace_add("write", lambda *args: self._update_preflight_estimates())
        self.zip_var.trace_add("write", lambda *args: self._update_preflight_estimates())

        # ================= RIGHT: preview + preflight + progress ===========
        self._section_label(right, "Interactive Live Preview")
        pcard = self._card(right, pady=(0, 4))

        # Controls row above preview canvas
        pctrl = ttk.Frame(pcard, style="Card.TFrame")
        pctrl.pack(fill="x", pady=(0, 6))

        ttk.Label(pctrl, text="Field:", style="CardHint.TLabel").pack(side="left", padx=(0, 2))
        field_options = ["Auto (Sampled)"] + sorted(list(config.SUPPORTED_FIELD_TYPES))
        self.preview_field_cb = ttk.Combobox(pctrl, textvariable=self.preview_field_var, values=field_options, state="readonly", width=12)
        self.preview_field_cb.pack(side="left", padx=(0, 6))
        self.preview_field_cb.bind("<<ComboboxSelected>>", lambda e: self._update_preview())
        self._run_controls.append(self.preview_field_cb)

        ttk.Label(pctrl, text="Text:", style="CardHint.TLabel").pack(side="left", padx=(0, 2))
        self.preview_text_entry = ttk.Entry(pctrl, textvariable=self.preview_text_var, width=12)
        self.preview_text_entry.pack(side="left", padx=(0, 6))
        self.preview_text_entry.bind("<KeyRelease>", lambda e: self._update_preview())
        self._run_controls.append(self.preview_text_entry)

        reroll_btn = ttk.Button(pctrl, text="🎲", width=3, style="Quick.TButton", command=self._reroll_preview)
        reroll_btn.pack(side="left", padx=(0, 6))
        self._run_controls.append(reroll_btn)

        self.preview_deg_check = ttk.Checkbutton(pctrl, text="Degraded", variable=self.preview_degraded_var, command=self._update_preview)
        self.preview_deg_check.pack(side="left")
        self._run_controls.append(self.preview_deg_check)

        self.preview_holder = tk.Frame(pcard, bg=PREVIEW_BG,
                                       highlightbackground=BORDER,
                                       highlightthickness=1, bd=0,
                                       height=260)
        self.preview_holder.pack(fill="both", expand=True)
        self.preview_holder.pack_propagate(False)
        self.preview_label = tk.Label(self.preview_holder, bg=PREVIEW_BG,
                                      text="Loading preview…", fg=SUBTEXT,
                                      font=("Segoe UI", 9))
        self.preview_label.pack(expand=True)

        # Preflight Estimates Card
        self._section_label(right, "Preflight Estimates")
        card_est = self._card(right, pady=(0, 4))
        est_grid = ttk.Frame(card_est, style="Card.TFrame")
        est_grid.pack(fill="x")

        r1 = ttk.Frame(est_grid, style="Card.TFrame")
        r1.pack(fill="x", pady=1)
        ttk.Label(r1, text="💾 Folder Space:", style="CardLabel.TLabel", width=14).pack(side="left")
        self.est_folder_lbl = ttk.Label(r1, text="Calculating...", style="CardHint.TLabel")
        self.est_folder_lbl.pack(side="left")

        r2 = ttk.Frame(est_grid, style="Card.TFrame")
        r2.pack(fill="x", pady=1)
        ttk.Label(r2, text="📦 Zip Archive:", style="CardLabel.TLabel", width=14).pack(side="left")
        self.est_zip_lbl = ttk.Label(r2, text="Calculating...", style="CardHint.TLabel")
        self.est_zip_lbl.pack(side="left")

        r3 = ttk.Frame(est_grid, style="Card.TFrame")
        r3.pack(fill="x", pady=1)
        ttk.Label(r3, text="⏱️ Est. Duration:", style="CardLabel.TLabel", width=14).pack(side="left")
        self.est_time_lbl = ttk.Label(r3, text="Calculating...", style="CardHint.TLabel")
        self.est_time_lbl.pack(side="left")

        r4 = ttk.Frame(est_grid, style="Card.TFrame")
        r4.pack(fill="x", pady=1)
        ttk.Label(r4, text="🔒 Output Target:", style="CardLabel.TLabel", width=14).pack(side="left")
        self.est_target_lbl = ttk.Label(r4, text="Calculating...", style="CardHint.TLabel")
        self.est_target_lbl.pack(side="left")

        self._section_label(right, "Progress")
        card4 = self._card(right, pady=(0, 8))
        self.progress = ttk.Progressbar(card4, mode="determinate")
        self.progress.pack(fill="x", pady=(0, 6))
        info_row = ttk.Frame(card4, style="Card.TFrame")
        info_row.pack(fill="x")
        self.status = ttk.Label(info_row, text="Ready.", style="Status.TLabel")
        self.status.pack(side="left")
        self.timer = ttk.Label(info_row, text="", style="Timer.TLabel")
        self.timer.pack(side="right")

        btn_row = ttk.Frame(right)
        btn_row.pack(fill="x", padx=16, pady=(6, 12))
        self.btn = ttk.Button(btn_row, text="⚡  Generate Dataset",
                              style="Generate.TButton", command=self.start)
        self.btn.pack(side="left", padx=(0, 8))
        self.open_btn = ttk.Button(btn_row, text="📂  Open Folder",
                                   style="Quick.TButton", command=self._open_output,
                                   state="disabled")
        self.open_btn.pack(side="left")

    # ------------------------------------------------------------------
    def _build_datasets_tab(self, parent):
        self._section_label(parent, "Generated datasets")

        card = self._card(parent, pady=(0, 4))
        tree_wrap = ttk.Frame(card, style="Card.TFrame")
        tree_wrap.pack(fill="both", expand=True)

        self.tree = ttk.Treeview(tree_wrap,
                                 columns=("folder_size", "zip_size", "checksum"),
                                 show="tree headings", height=12, selectmode="browse")
        self.tree.heading("#0",          text="Dataset")
        self.tree.heading("folder_size", text="Folder size")
        self.tree.heading("zip_size",    text="Zip size")
        self.tree.heading("checksum",    text="SHA256")
        self.tree.column("#0",          width=280, anchor="w")
        self.tree.column("folder_size", width=110, anchor="center")
        self.tree.column("zip_size",    width=110, anchor="center")
        self.tree.column("checksum",    width=80, anchor="center")
        self.tree.pack(side="left", fill="both", expand=True)
        self.tree.bind("<Double-1>", lambda e: self._open_selected_dataset())

        vsb = ttk.Scrollbar(tree_wrap, orient="vertical", command=self.tree.yview)
        vsb.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=vsb.set)

        hint = ttk.Label(parent,
                         text="📦 = .zip archive exists. Deleting a dataset also "
                              "removes its .zip and .zip.sha256 sidecar.",
                         style="Sub.TLabel")
        hint.pack(anchor="w", padx=16, pady=(2, 8))

        self.dataset_status = ttk.Label(parent, text="", style="Status.TLabel")
        self.dataset_status.pack(anchor="w", padx=16, pady=(0, 6))

        btn_row = ttk.Frame(parent)
        btn_row.pack(fill="x", padx=16, pady=(0, 12))
        self.refresh_btn = ttk.Button(
            btn_row, text="🔄  Refresh", style="Secondary.TButton",
            command=self._refresh_datasets)
        self.refresh_btn.pack(side="left")
        self.dataset_open_btn = ttk.Button(
            btn_row, text="📂  Open", style="Secondary.TButton",
            command=self._open_selected_dataset)
        self.dataset_open_btn.pack(side="left", padx=(8, 0))
        self.view_montage_btn = ttk.Button(
            btn_row, text="🖼️  Montage", style="Secondary.TButton",
            command=self._view_selected_montage)
        self.view_montage_btn.pack(side="left", padx=(8, 0))
        self.view_report_btn = ttk.Button(
            btn_row, text="📋  Report", style="Secondary.TButton",
            command=self._view_selected_report)
        self.view_report_btn.pack(side="left", padx=(8, 0))
        self.verify_sha_btn = ttk.Button(
            btn_row, text="🔍  Verify SHA", style="Secondary.TButton",
            command=self._verify_selected_sha256)
        self.verify_sha_btn.pack(side="left", padx=(8, 0))
        self.delete_btn = ttk.Button(
            btn_row, text="🗑  Delete", style="Danger.TButton",
            command=self._delete_selected_dataset)
        self.delete_btn.pack(side="right")

    # ------------------------------------------------------------------
    # Tab change
    # ------------------------------------------------------------------
    def _on_tab_change(self, _event=None):
        if self.nb.index(self.nb.select()) == 1:   # datasets tab
            if hasattr(self, "_unbind_gen_mousewheel"):
                self._unbind_gen_mousewheel()
            self._refresh_datasets()
        else:
            if hasattr(self, "_bind_gen_mousewheel"):
                self._bind_gen_mousewheel()

    # ------------------------------------------------------------------
    # Font-style preview
    # ------------------------------------------------------------------
    def _current_font_selection(self):
        font_style = self.font_style_options.get(self.font_style_var.get(), "all")
        cursive_group = ""
        specific_font = ""
        if font_style == "cursive":
            sel = self.cursive_group_var.get()
            cursive_group = sel if sel != "All cursive" else ""
            specific_font = self.specific_font_map.get(self.specific_font_var.get(), "")
        return font_style, cursive_group, specific_font

    def _populate_specific_fonts(self):
        """Fill the specific-font dropdown with the fonts in the chosen group."""
        sel = self.cursive_group_var.get()
        group = sel if sel != "All cursive" else ""
        try:
            pool = fonts_for_style("cursive", group)
        except Exception:
            pool = ()
        mapping = {"All in group": ""}
        for p in pool:
            mapping[font_display_name(p)] = Path(p).stem
        self.specific_font_map = mapping
        self.specific_cb.config(values=list(mapping.keys()))
        if self.specific_font_var.get() not in mapping:
            self.specific_font_var.set("All in group")

    def _reroll_preview(self):
        self.preview_seed = random.randint(1, 2_147_483_647)
        self._update_preview()

    def _make_preview_image(self, pool, lines=3, size=42, custom_text=None):
        """Render text in a few representative fonts, stacked vertically."""
        text_to_draw = custom_text if custom_text else SAMPLE_TEXT
        fonts = list(pool)
        if not fonts:
            return None
        if len(fonts) > lines:
            step = len(fonts) / lines
            fonts = [fonts[int(i * step)] for i in range(lines)]

        rendered = []
        for fp in fonts:
            try:
                font = ImageFont.truetype(fp, size)
            except Exception:
                continue
            measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
            box = measure.textbbox((0, 0), text_to_draw, font=font)
            w, h = box[2] - box[0], box[3] - box[1]
            im = Image.new("RGB", (w + 24, h + 20), PREVIEW_BG)
            d = ImageDraw.Draw(im)
            d.text((12 - box[0], 10 - box[1]), text_to_draw, font=font, fill=(35, 40, 55))
            if im.width > PREVIEW_WIDTH - 24:
                r = (PREVIEW_WIDTH - 24) / im.width
                im = im.resize((PREVIEW_WIDTH - 24, max(1, int(im.height * r))), Image.Resampling.LANCZOS)
            rendered.append(im)

        if not rendered:
            return None

        gap = 8
        total_h = sum(im.height for im in rendered) + gap * (len(rendered) + 1)
        canvas = Image.new("RGB", (PREVIEW_WIDTH, total_h), PREVIEW_BG)
        y = gap
        for im in rendered:
            x = (PREVIEW_WIDTH - im.width) // 2
            canvas.paste(im, (x, y))
            y += im.height + gap
        return canvas

    def _update_preview(self, _event=None):
        font_style, cursive_group, specific_font = self._current_font_selection()

        field_sel = getattr(self, "preview_field_var", None)
        text_sel = getattr(self, "preview_text_var", None)
        seed = getattr(self, "preview_seed", 42)

        field_type = field_sel.get() if field_sel else "Auto (Sampled)"
        custom_txt = text_sel.get().strip() if text_sel else ""

        if field_type != "Auto (Sampled)":
            try:
                preview_text = fields.make_value(field_type, rng=random.Random(seed))
            except Exception:
                preview_text = custom_txt or SAMPLE_TEXT
        elif custom_txt:
            preview_text = custom_txt
        else:
            preview_text = SAMPLE_TEXT

        try:
            if specific_font:
                fp = font_path_for(specific_font)
                pool = (fp,) if fp else fonts_for_style(font_style, cursive_group)
                img = self._make_preview_image(pool, lines=1, size=52, custom_text=preview_text)
            else:
                pool = fonts_for_style(font_style, cursive_group)
                img = self._make_preview_image(pool, lines=3, size=36, custom_text=preview_text)
        except Exception:
            img = None

        if img is None:
            self._preview_img = None
            self.preview_label.config(image="", text="(no preview available)")
            return

        if hasattr(self, "preview_degraded_var") and not self.preview_degraded_var.get():
            self._preview_img = ImageTk.PhotoImage(img)
            self.preview_label.config(image=self._preview_img, text="")
            return

        profile = self._build_custom_developer_profile()
        crop_tuple = self._get_custom_crop_tuple()
        stroke_damage = self._build_stroke_damage_profile()
        sample_mode_key = self._current_sample_mode()
        damage_prof = "semi_broken" if sample_mode_key.startswith("semi_broken") else "regular"
        try:
            img = degrade(
                img,
                damage_profile=damage_prof,
                augmentation_profile=profile,
                edge_clipping=crop_tuple,
                semi_broken_params=stroke_damage,
                rng=random.Random(seed),
            )
        except Exception:
            pass

        self._preview_img = ImageTk.PhotoImage(img)
        self.preview_label.config(image=self._preview_img, text="")

    def _current_sample_mode(self) -> str:
        return self.mode_by_label.get(
            self.sample_mode_var.get(), config.DEFAULT_SAMPLE_MODE)

    def _sync_stroke_damage_state(self):
        """Enable the Stroke Damage sliders only in semi-broken modes."""
        if self.__dict__.get("_run_control_states"):
            return  # a job is running; _set_job_active restores and resyncs
        state = (
            "normal" if self._current_sample_mode().startswith("semi_broken")
            else "disabled"
        )
        scales = self.__dict__.get("dev_scales", {})
        for name in STROKE_DAMAGE_SLIDERS:
            scale = scales.get(name)
            if scale is not None:
                scale.config(state=state)

    def _on_sample_mode_change(self, _event=None):
        self._sync_stroke_damage_state()
        self._apply_mode_field_weights()
        self._update_preview()

    def _grid_field_weights(self) -> dict[str, str]:
        return {ftype: var.get().strip() for ftype, var in self.field_weight_vars.items()}

    def _apply_mode_field_weights(self):
        """Fill the weight grid with the current mode's mix.

        Edits made for the previous mode are replaced, and the status line
        says so.
        """
        mode = self._current_sample_mode()
        previous = self.__dict__.get("_field_weights_mode", mode)
        expected = {k: str(v) for k, v in mode_field_weights(previous).items()}
        edited = self._grid_field_weights() != expected
        self._reset_field_weights()
        if edited and mode != previous and "status" in self.__dict__:
            self.status.config(
                text=(f"Field weights reset to the {config.SAMPLE_MODES[mode]} mix; "
                      "your edits were replaced."),
                foreground=WARNING)

    @staticmethod
    def _slider_text(name: str, value: float) -> str:
        spec = SLIDER_SPECS[name]
        return format_value(value, spec.step, spec.unit)

    def _build_stroke_damage_profile(self):
        """Stroke damage profile, or None outside semi-broken modes."""
        if not hasattr(self, "dev_sliders"):
            return None
        if not self._current_sample_mode().startswith("semi_broken"):
            return None
        return StrokeDamageProfile.from_legacy_params({
            "contrast_prob": self.dev_sliders["contrast"].get(),
            "gap_prob": self.dev_sliders["gap_prob"].get(),
            "gap_count": int(round(self.dev_sliders["gap_count"].get())),
            "scratch_prob": self.dev_sliders["scratch"].get(),
            "erode_prob": self.dev_sliders["erode"].get(),
        })

    def _build_custom_developer_profile(self):
        if not hasattr(self, "dev_sliders"):
            degradation_key = self.degradation_by_label.get(
                self.degradation_var.get(), config.DEFAULT_DEGRADATION_PROFILE
            )
            return get_augmentation_profile(degradation_key)

        preset_key = self.degradation_by_label.get(self.degradation_var.get(), "")
        if preset_key and preset_key != "custom_dev_v1" and preset_key in AUGMENTATION_PROFILES:
            return get_augmentation_profile(preset_key)

        # Interim mapping of the single-value sliders: chance 1.0 and
        # min = max, with a content-addressed profile ID.
        return augmentation_profile_from_spec(augmentation_spec_from_legacy_values(
            fade_contrast=self.dev_sliders["fade"].get(),
            paper_tint_alpha=self.dev_sliders["tint"].get(),
            stain_prob=self.dev_sliders["stain"].get(),
            rotate_deg=self.dev_sliders["rotate"].get(),
            blur_radius=self.dev_sliders["blur"].get(),
            noise_std=self.dev_sliders["noise"].get(),
            paper_texture_std=self.dev_sliders["texture"].get(),
            scanline_alpha=int(round(self.dev_sliders["scanline"].get())),
            jpeg_quality=int(round(self.dev_sliders["jpeg"].get())),
        ))

    def _get_custom_crop_tuple(self):
        if not hasattr(self, "dev_sliders"):
            preset_key = self.clipping_by_label.get(self.edge_clipping_var.get(), "none")
            return preset_key

        preset_key = self.clipping_by_label.get(self.edge_clipping_var.get(), "")
        if preset_key and preset_key != "custom":
            return preset_key

        top_p = self.dev_sliders["crop_top"].get() / 100.0
        bottom_p = self.dev_sliders["crop_bottom"].get() / 100.0
        left_p = self.dev_sliders["crop_left"].get() / 100.0
        right_p = self.dev_sliders["crop_right"].get() / 100.0
        if top_p <= 0 and bottom_p <= 0 and left_p <= 0 and right_p <= 0:
            return "none"
        return (top_p, bottom_p, left_p, right_p)

    @staticmethod
    def _profile_slider_values(profile_key: str) -> dict[str, float]:
        """Degradation slider values for a registered profile (range midpoints).

        Unregistered keys, such as the custom studio profile, give the
        slider defaults.
        """
        prof = AUGMENTATION_PROFILES.get(profile_key)
        if prof is None:
            return {spec.name: spec.default for spec in DEGRADATION_SLIDER_SPECS}

        def mid(values):
            return (values[0] + values[1]) / 2.0

        return {
            "fade": mid(prof.fade_range),
            "tint": mid(prof.paper_tint_alpha_range),
            "stain": prof.stain_probability,
            "rotate": prof.rotate_degrees,
            "blur": mid(prof.blur_radius_range),
            "noise": mid(prof.noise_std_range),
            "texture": mid(prof.paper_texture_std_range),
            "scanline": mid(prof.scanline_alpha_range),
            "jpeg": mid(prof.jpeg_quality_range),
        }

    def _set_slider_values(self, values: dict[str, float]):
        """Set sliders (snapped) and labels without triggering slider callbacks."""
        self._slider_updating = True
        try:
            for name, raw in values.items():
                spec = SLIDER_SPECS[name]
                value = snap_value(raw, spec.step, spec.low, spec.high)
                self.dev_sliders[name].set(value)
                self.dev_labels[name].set(self._slider_text(name, value))
        finally:
            self._slider_updating = False

    def _selected_degradation_key(self) -> str:
        return self.degradation_by_label.get(
            self.degradation_var.get(), config.DEFAULT_DEGRADATION_PROFILE)

    def _on_degradation_preset_change(self, _event=None):
        preset_key = self._selected_degradation_key()
        if preset_key in AUGMENTATION_PROFILES and hasattr(self, "dev_sliders"):
            self._set_slider_values(self._profile_slider_values(preset_key))
        self._update_preview()

    def _on_clipping_preset_change(self, _event=None):
        preset_label = self.edge_clipping_var.get()
        preset_key = self.clipping_by_label.get(preset_label, "none")
        if preset_key in EDGE_CLIPPING_RANGES and hasattr(self, "dev_sliders"):
            r = EDGE_CLIPPING_RANGES[preset_key]
            avg_pct = round(((r[0] + r[1]) / 2.0) * 100.0)
            self._slider_updating = True
            for side in ("crop_top", "crop_bottom", "crop_left", "crop_right"):
                self.dev_sliders[side].set(avg_pct)
                self.dev_labels[side].set(self._slider_text(side, avg_pct))
            self._slider_updating = False
        self._update_preview()

    def _on_developer_slider_change(self, slider_name=""):
        if getattr(self, "_slider_updating", False):
            return
        if slider_name.startswith("crop_"):
            if "custom" in self.clipping_by_label.values():
                self.edge_clipping_var.set(config.EDGE_CLIPPING_OPTIONS["custom"])
        else:
            if "custom_dev_v1" in self.degradation_by_label.values():
                self.degradation_var.set(config.DEGRADATION_PROFILES["custom_dev_v1"])
        self._update_preview()

    def _reset_studio_sliders(self):
        """Reset only the studio sliders.

        Degradation sliders take the selected profile's values; crop and
        stroke-damage sliders return to their defaults.
        """
        values = self._profile_slider_values(self._selected_degradation_key())
        for spec in (*CROP_SLIDER_SPECS, *STROKE_SLIDER_SPECS):
            values[spec.name] = spec.default
        self._set_slider_values(values)
        self._update_preview()

    def _reset_all_controls(self):
        """Reset every Generate-tab control, including weights and splits."""
        self.sample_mode_var.set(config.SAMPLE_MODES[config.DEFAULT_SAMPLE_MODE])
        self._sync_stroke_damage_state()
        self._reset_field_weights()
        self.split_train_var.set(_split_percent(config.SYNTH_TRAIN_FRAC))
        self.split_val_var.set(_split_percent(config.SYNTH_VAL_FRAC))
        self.split_test_var.set(_split_percent(config.SYNTH_TEST_FRAC))
        self._update_split_sum_label()
        self.degradation_var.set(config.DEGRADATION_PROFILES[config.DEFAULT_DEGRADATION_PROFILE])
        self.edge_clipping_var.set(config.EDGE_CLIPPING_OPTIONS[config.DEFAULT_EDGE_CLIPPING])
        self.count_var.set(str(config.DEFAULT_COUNT))
        self.seed_var.set(str(config.RANDOM_SEED))
        if hasattr(self, "workers_var"):
            self.workers_var.set("Auto (Max)")
        if hasattr(self, "names_var") and hasattr(config, "NAMES_VERSION"):
            self.names_var.set(config.NAMES_VERSION)
        self.font_style_var.set("All fonts")
        self._on_font_style_change()
        self.real_var.set(False)
        self.zip_var.set(False)
        self._reset_studio_sliders()

    def _update_preflight_estimates(self, _event=None):
        if not hasattr(self, "est_folder_lbl"):
            return
        try:
            cnt_str = self.count_var.get().replace(",", "").strip()
            count = int(cnt_str) if cnt_str.isdigit() else config.DEFAULT_COUNT
        except Exception:
            count = config.DEFAULT_COUNT

        create_zip = bool(self.zip_var.get())
        try:
            estimate = preflight_generation_resources(
                config.DATASETS_DIR,
                count,
                create_archive=create_zip,
            )
            folder_mb = estimate.estimated_bytes / (1024 * 1024)
            zip_mb = estimate.estimated_archive_bytes / (1024 * 1024)

            # Dynamically estimate throughput based on active CPU threads count
            active_threads = os.cpu_count() or 16
            if hasattr(self, "workers_var"):
                w_str = self.workers_var.get().strip()
                if w_str and not w_str.startswith("Auto"):
                    try:
                        active_threads = int(w_str)
                    except ValueError:
                        pass
            est_imgs_per_sec = max(30, min(600, active_threads * 22))
            time_sec = max(1, round(count / est_imgs_per_sec))

            self.est_folder_lbl.config(text=f"~{folder_mb:.1f} MB ({estimate.estimated_image_count:,} imgs)")
            self.est_zip_lbl.config(text=f"~{zip_mb:.1f} MB" if create_zip else "Disabled")
            self.est_time_lbl.config(text=f"~{time_sec}s (~{time_sec/60:.1f}m)" if time_sec >= 60 else f"~{time_sec}s")
        except Exception:
            self.est_folder_lbl.config(text="Unknown")
            self.est_zip_lbl.config(text="Unknown")
            self.est_time_lbl.config(text="Unknown")

        ds_name = self.dataset_var.get().strip()
        if not ds_name or ds_name == "(next)":
            try:
                target_folder = config.next_dataset_dir().name
            except Exception:
                target_folder = "dataset_NNN"
        else:
            target_folder = ds_name
        self.est_target_lbl.config(text=f"dataset/datasets/{target_folder}")

    def _reset_field_weights(self):
        """Reset the weight grid to the current sample mode's mix."""
        mode = self._current_sample_mode()
        for ftype, weight in mode_field_weights(mode).items():
            self.field_weight_vars[ftype].set(str(weight))
        self._field_weights_mode = mode

    def _update_split_sum_label(self, _event=None):
        if not hasattr(self, "split_sum_lbl"):
            return
        try:
            tr = float(self.split_train_var.get())
            val = float(self.split_val_var.get())
            te = float(self.split_test_var.get())
            total = tr + val + te
            if abs(total - 100.0) < 0.01:
                self.split_sum_lbl.config(text=f"Sum: {int(total)}%", foreground=SUBTEXT)
            else:
                self.split_sum_lbl.config(text=f"Sum: {total:.0f}% (Must be 100%)", foreground=ERROR_C)
        except ValueError:
            self.split_sum_lbl.config(text="Invalid numbers", foreground=ERROR_C)

    def _refresh_preset_list(self):
        PRESET_DIR.mkdir(parents=True, exist_ok=True)
        files = sorted(PRESET_DIR.glob("*.json"))
        names = [f.stem for f in files]
        if hasattr(self, "preset_cb"):
            self.preset_cb.config(values=names)
            if names and not self.preset_var.get():
                self.preset_var.set(names[0])

    def _load_selected_preset(self):
        name = self.preset_var.get()
        if not name:
            return
        preset_file = PRESET_DIR / f"{name}.json"
        if not preset_file.is_file():
            messagebox.showerror("Error", f"Preset file not found: {preset_file}")
            return
        try:
            with open(preset_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            values, ignored = parse_preset_values(data)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Error loading preset", str(exc))
            return
        self._apply_preset_values(name, values, ignored)

    def _apply_preset_values(self, name, values, ignored):
        """Apply validated preset values all at once."""
        self._slider_updating = True
        try:
            for key, value in values.items():
                self.dev_sliders[key].set(value)
                self.dev_labels[key].set(self._slider_text(key, value))
            cropped = any(
                self.dev_sliders[spec.name].get() > 0 for spec in CROP_SLIDER_SPECS
            )
            self.edge_clipping_var.set(
                config.EDGE_CLIPPING_OPTIONS["custom" if cropped else "none"])
            self.degradation_var.set(config.DEGRADATION_PROFILES["custom_dev_v1"])
        finally:
            self._slider_updating = False
        self._update_preview()
        if len(values) == len(SLIDER_SPECS):
            applied = f"all {len(values)} studio values"
        else:
            applied = ", ".join(sorted(values)) or "no studio values"
        message = f"Loaded preset {name}: applied {applied}"
        if ignored:
            message += f"; ignored {', '.join(ignored)}"
        self.status.config(text=message + ".", foreground=SUCCESS)

    def _save_current_preset(self):
        name = simpledialog.askstring("Save Preset", "Enter a name for this Developer Studio preset:")
        if name is None:
            return
        try:
            safe_name = preset_slug(name)
            preset_file = preset_path_for(safe_name)
        except ValueError as exc:
            messagebox.showerror("Invalid preset name", str(exc))
            return
        if preset_file.exists() and not messagebox.askyesno(
                "Overwrite preset",
                f"A preset named '{safe_name}' already exists. Replace it?",
                icon="warning"):
            return

        data = preset_payload(
            name.strip(), {k: var.get() for k, var in self.dev_sliders.items()})
        try:
            PRESET_DIR.mkdir(parents=True, exist_ok=True)
            atomic_write_json(preset_file, data)
        except OSError as exc:
            messagebox.showerror("Error saving preset", str(exc))
            return
        self._refresh_preset_list()
        self.preset_var.set(safe_name)
        self.status.config(text=f"Saved preset: {safe_name}", foreground=SUCCESS)

    def _get_selected_dataset_folder(self):
        selected = self.tree.selection()
        if not selected:
            return None
        return self._dataset_paths.get(selected[0])

    def _view_selected_montage(self):
        folder = self._get_selected_dataset_folder()
        if not folder or not folder.is_dir():
            messagebox.showwarning("No selection", "Please select a generated dataset from the list.")
            return
        montage_path = folder / "review-montage.jpg"
        if not montage_path.is_file():
            messagebox.showerror("Missing Montage", f"Review montage does not exist for {folder.name}:\n{montage_path}")
            return

        win = tk.Toplevel(self)
        win.title(f"Review Montage — {folder.name}")
        win.geometry("820x640")
        win.configure(bg=BG)

        try:
            pil_img = Image.open(montage_path)
            pil_img.thumbnail((800, 580), Image.Resampling.LANCZOS)
            tk_img = ImageTk.PhotoImage(pil_img)
            lbl = tk.Label(win, image=tk_img, bg=BG)
            lbl.image = tk_img
            lbl.pack(expand=True, fill="both", padx=10, pady=10)
        except Exception as exc:
            messagebox.showerror("Error displaying montage", str(exc))
            win.destroy()

    def _view_selected_report(self):
        folder = self._get_selected_dataset_folder()
        if not folder or not folder.is_dir():
            messagebox.showwarning("No selection", "Please select a generated dataset from the list.")
            return
        report_path = folder / "dataset-validation.json"
        if not report_path.is_file():
            messagebox.showerror("Missing Report", f"Validation report does not exist for {folder.name}:\n{report_path}")
            return

        try:
            with open(report_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            messagebox.showerror("Error reading report", str(exc))
            return

        win = tk.Toplevel(self)
        win.title(f"Validation Report — {folder.name}")
        win.geometry("640x520")
        win.configure(bg=BG)

        hdr_frame = tk.Frame(win, bg=SURFACE, bd=1, relief="solid")
        hdr_frame.pack(fill="x", padx=12, pady=12)

        valid = data.get("valid", False)
        status_text = "✅ VALID DATASET" if valid else "❌ INVALID DATASET"
        status_color = SUCCESS if valid else ERROR_C

        tk.Label(hdr_frame, text=status_text, font=("Segoe UI", 12, "bold"), fg=status_color, bg=SURFACE).pack(anchor="w", padx=10, pady=(8, 4))

        stats = data.get("statistics", {})
        info_txt = (
            f"Images: {stats.get('images', 0):,}  |  "
            f"Manifest Rows: {stats.get('manifest_rows', 0):,}\n"
            f"Validated At: {data.get('validated_at', 'Unknown')}\n"
            f"Manifest SHA-256: {data.get('manifest_sha256', '—')[:16]}…"
        )
        tk.Label(hdr_frame, text=info_txt, font=("Segoe UI", 9), fg=TEXT, bg=SURFACE, justify="left").pack(anchor="w", padx=10, pady=(0, 8))

        text_wrap = tk.Frame(win, bg=BG)
        text_wrap.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        stext = tk.Text(text_wrap, wrap="word", font=("Consolas", 9), bg=SURFACE, fg=TEXT, bd=1, relief="solid")
        vsb = ttk.Scrollbar(text_wrap, orient="vertical", command=stext.yview)
        stext.configure(yscrollcommand=vsb.set)
        stext.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        stext.insert("end", "=== VALIDATION DIAGNOSTICS & METRICS ===\n\n")
        stext.insert("end", f"Errors ({len(data.get('errors', []))}):\n")
        for err in data.get("errors", []):
            stext.insert("end", f"  • {err}\n")
        if not data.get("errors"):
            stext.insert("end", "  (None)\n")

        stext.insert("end", f"\nWarnings ({len(data.get('warnings', []))}):\n")
        for warn in data.get("warnings", []):
            stext.insert("end", f"  • {warn}\n")
        if not data.get("warnings"):
            stext.insert("end", "  (None)\n")

        stext.insert("end", "\nFull Report JSON:\n")
        stext.insert("end", json.dumps(data, indent=2))
        stext.config(state="disabled")

    def _verify_selected_sha256(self):
        folder = self._get_selected_dataset_folder()
        if not folder or not folder.is_dir():
            messagebox.showwarning("No selection", "Please select a generated dataset from the list.")
            return
        zip_path = self._zip_for(folder)
        checksum_path = self._checksum_for(folder)
        if not zip_path.is_file():
            messagebox.showinfo("No Archive", f"No .zip archive exists for {folder.name}.")
            return
        if not checksum_path.is_file():
            messagebox.showwarning("Missing Checksum", f"Checksum sidecar .sha256 is missing for {folder.name}.")
            return

        try:
            valid = verify_sha256_sidecar(zip_path, checksum_path)
            if valid:
                messagebox.showinfo("Checksum Verification", f"✅ SUCCESS!\n\nThe archive SHA-256 hash matches the .sha256 sidecar file.\n\nArchive: {zip_path.name}")
            else:
                messagebox.showerror("Checksum Verification", f"❌ FAILED!\n\nThe archive SHA-256 hash does NOT match the sidecar!\n\nArchive: {zip_path.name}")
        except Exception as exc:
            messagebox.showerror("Verification Error", str(exc))

    def _on_font_style_change(self, _event=None):
        if self.font_style_var.get() == "Cursive only":
            self.cursive_row.pack(fill="x", pady=(8, 0))
            self.specific_row.pack(fill="x", pady=(8, 0))
            self._populate_specific_fonts()
        else:
            self.specific_row.pack_forget()
            self.cursive_row.pack_forget()
            self.cursive_group_var.set("All cursive")
            self.specific_font_var.set("All in group")
        self._update_preview()

    def _on_cursive_group_change(self, _event=None):
        # New group -> refresh the specific-font choices and reset to "all".
        self.specific_font_var.set("All in group")
        self._populate_specific_fonts()
        self._update_preview()

    # ------------------------------------------------------------------
    # Datasets manager
    # ------------------------------------------------------------------
    @staticmethod
    def _fmt_size(bytes_: int) -> str:
        """Format a byte count as MB or GB with one decimal place."""
        if bytes_ < 0:
            return "—"
        mb = bytes_ / (1024 ** 2)
        if mb >= 1024:
            return f"{mb / 1024:.1f} GB"
        return f"{mb:.1f} MB"

    @staticmethod
    def _folder_size_fast(path) -> int:
        """
        Fast folder size using os.scandir recursion.
        Avoids Path.rglob() overhead on large trees.
        """
        total = 0
        try:
            stack = [str(path)]
            while stack:
                with os.scandir(stack.pop()) as it:
                    for entry in it:
                        if entry.is_file(follow_symlinks=False):
                            total += entry.stat().st_size
                        elif entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
        except Exception:
            return -1
        return total

    @staticmethod
    def _zip_for(folder):
        """Sibling .zip path for a dataset folder."""
        return folder.parent / (folder.name + ".zip")

    @staticmethod
    def _checksum_for(folder):
        """Sibling SHA-256 sidecar produced for a dataset ZIP."""
        return folder.parent / (folder.name + ".zip.sha256")

    # ------------------------------------------------------------------
    # Datasets list — two-phase: instant skeleton, background sizes
    # ------------------------------------------------------------------
    def _refresh_datasets(self):
        if "tree" not in self.__dict__:
            return
        self._refresh_generation = self.__dict__.get("_refresh_generation", 0) + 1
        generation = self._refresh_generation
        refresh_worker = self.__dict__.get("_refresh_worker")
        if refresh_worker is not None and refresh_worker.is_alive():
            self._refresh_pending = True
            return
        self.tree.delete(*self.tree.get_children())
        self._dataset_paths.clear()

        base = config.DATASETS_DIR
        if not base.exists():
            return
        try:
            folder_times = []
            for path in base.iterdir():
                try:
                    # Generator staging/transaction directories and reservation
                    # markers are private implementation details, never complete
                    # datasets.  Hiding all dot-prefixed entries also prevents a
                    # partial run from becoming selectable or deletable.
                    if (not path.name.startswith(".") and path.is_dir()
                            and not path.is_symlink()):
                        folder_times.append((path.stat().st_mtime, path))
                except OSError:
                    continue
            folders = [path for _mtime, path in sorted(folder_times, reverse=True)]
        except OSError as exc:
            _log_error("dataset refresh", traceback.format_exc())
            self.status.config(text=f"Could not refresh datasets: {exc}", foreground=ERROR_C)
            return

        # Phase 1: insert rows immediately with placeholder sizes
        iid_folder = []
        for folder in folders:
            zip_path = self._zip_for(folder)
            checksum_path = self._checksum_for(folder)
            has_zip = zip_path.is_file()
            has_checksum = checksum_path.is_file()
            label = f"  {'📦' if has_zip else '📁'}  {folder.name}"
            iid = self.tree.insert("", "end", text=label,
                                   values=("…", "…" if has_zip else "—",
                                           "yes" if has_checksum else "—"))
            self._dataset_paths[iid] = folder
            iid_folder.append(
                (iid, folder, zip_path, checksum_path, has_zip, has_checksum)
            )

        # Phase 2: compute sizes off the UI thread. It communicates exclusively
        # through the queue; only _poll touches Tk.  Do not start a second scan
        # while the first is traversing a large tree; coalesce refresh requests
        # and run the latest generation once the active scan exits.
        self._refresh_pending = False

        def _compute():
            try:
                for iid, folder, zip_path, checksum_path, has_zip, has_checksum in iid_folder:
                    if generation != self._refresh_generation:
                        return
                    folder_sz = self._fmt_size(self._folder_size_fast(folder))
                    try:
                        zip_sz = (
                            self._fmt_size(zip_path.stat().st_size) if has_zip else "—"
                        )
                    except OSError:
                        zip_sz = "—"
                        has_zip = False
                    checksum_status = (
                        "yes" if has_checksum and checksum_path.is_file() else "—"
                    )
                    self.q.put((
                        "dataset_size", generation, iid, folder_sz, zip_sz,
                        checksum_status,
                    ))
            finally:
                self.q.put(("refresh_complete", generation))

        self._refresh_worker = threading.Thread(target=_compute, daemon=True)
        self._refresh_worker.start()

    def _patch_row(self, generation, iid, folder_sz, zip_sz, checksum_status):
        """Update a single treeview row with computed sizes (runs on UI thread)."""
        if generation != self._refresh_generation or iid not in self._dataset_paths:
            return
        try:
            if self.tree.exists(iid):
                self.tree.item(
                    iid, values=(folder_sz, zip_sz, checksum_status)
                )
        except tk.TclError:
            pass  # row was deleted before size came back

    def _finish_refresh_worker(self):
        """Join the sole refresh worker, then run one coalesced refresh."""
        worker = self._refresh_worker
        if worker is not None and worker.is_alive():
            self.after(25, self._finish_refresh_worker)
            return
        if worker is not None:
            worker.join(timeout=0)
        self._refresh_worker = None
        self._refresh_finalize_scheduled = False
        if self._refresh_pending:
            self._refresh_pending = False
            self.after(0, self._refresh_datasets)

    def _selected_dataset(self):
        sel = self.tree.selection()
        if not sel:
            return None
        return self._dataset_paths.get(sel[0])

    def _open_selected_dataset(self):
        folder = self._selected_dataset()
        if not folder:
            messagebox.showinfo("No selection", "Select a dataset from the list first.")
            return
        try:
            _open_path_portably(folder)
        except Exception as exc:
            log_path = _log_error("open dataset folder", traceback.format_exc())
            messagebox.showerror(
                "Open failed", f"{exc}\n\nFull details were saved to:\n{log_path}"
            )

    def _delete_selected_dataset(self):
        if self._any_worker_is_alive():
            messagebox.showwarning(
                "Background work in progress",
                "Datasets cannot be deleted while another operation is running.")
            return

        folder = self._selected_dataset()
        if not folder:
            messagebox.showinfo("No selection", "Select a dataset to delete first.")
            return

        zip_path = self._zip_for(folder)
        checksum_path = self._checksum_for(folder)
        msg = f"Delete dataset '{folder.name}'?\n\nThis permanently removes the folder"
        if zip_path.exists():
            msg += f" and its archive ({zip_path.name})"
        if checksum_path.exists():
            msg += f" and checksum ({checksum_path.name})"
        msg += ".\nThis cannot be undone."

        if not messagebox.askyesno("Confirm delete", msg, icon="warning"):
            return

        self._set_dataset_operation_active(True)
        self.dataset_status.config(
            text=f"Deleting {folder.name}…", foreground=WARNING
        )
        self.delete_worker = threading.Thread(
            target=self._delete_dataset_worker,
            args=(folder, zip_path, checksum_path),
            daemon=False,
        )
        self.delete_worker.start()

    @staticmethod
    def _validated_dataset_sidecar(folder: Path, suffix: str) -> Path:
        """Return one exact direct-child archive sidecar without following links."""
        safe_folder = config.assert_safe_dataset_dir(folder)
        candidate = safe_folder.parent / f"{safe_folder.name}{suffix}"
        if candidate.parent.resolve(strict=True) != config.DATASETS_DIR.resolve(strict=True):
            raise ValueError(f"Unsafe dataset sidecar path: {candidate}")
        if candidate.is_symlink():
            raise ValueError(f"Dataset sidecar must not be a symlink: {candidate}")
        if candidate.exists() and not candidate.is_file():
            raise ValueError(f"Dataset sidecar is not a regular file: {candidate}")
        return candidate

    def _delete_dataset_worker(
        self, folder: Path, zip_path: Path, checksum_path: Path
    ) -> None:
        """Delete validated dataset artifacts without invoking any Tk method."""
        try:
            folder = config.assert_safe_dataset_dir(folder)
            expected_zip = self._validated_dataset_sidecar(folder, ".zip")
            expected_checksum = self._validated_dataset_sidecar(folder, ".zip.sha256")
            if zip_path != expected_zip or checksum_path != expected_checksum:
                raise ValueError("Dataset deletion targets changed before deletion")
            self.q.put(("delete_status", f"Deleting {folder.name} folder…"))
            removed_folder = guarded_remove_dataset(folder)
            self.q.put(("delete_status", f"Deleting {folder.name} archive…"))
            removed_zip = False
            removed_checksum = False
            if expected_zip.exists():
                expected_zip.unlink()
                removed_zip = True
            if expected_checksum.exists():
                expected_checksum.unlink()
                removed_checksum = True
            self.q.put((
                "delete_done", folder.name, removed_folder, removed_zip,
                removed_checksum,
            ))
        except Exception as exc:
            detail = traceback.format_exc()
            log_path = _log_error("dataset deletion", detail)
            self.q.put(("delete_error", str(exc) or exc.__class__.__name__, str(log_path)))

    # ------------------------------------------------------------------
    # Open last output in Explorer
    # ------------------------------------------------------------------
    def _open_output(self):
        if not self._last_output:
            return
        path = Path(self._last_output)
        try:
            _open_path_portably(path, select_file=path.suffix.lower() == ".zip")
        except Exception as exc:
            log_path = _log_error("open generated output", traceback.format_exc())
            messagebox.showerror(
                "Open failed", f"{exc}\n\nFull details were saved to:\n{log_path}"
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _worker_is_alive(self):
        """Return whether the generation worker is still executing."""
        worker = self.__dict__.get("worker")
        return worker is not None and worker.is_alive()

    def _delete_worker_is_alive(self):
        worker = self.__dict__.get("delete_worker")
        return worker is not None and worker.is_alive()

    def _any_worker_is_alive(self):
        return self._worker_is_alive() or self._delete_worker_is_alive()

    def _set_dataset_operation_active(self, active):
        """Lock dataset controls while background deletion is in flight."""
        state = "disabled" if active else "normal"
        for name in ("refresh_btn", "dataset_open_btn", "delete_btn"):
            widget = self.__dict__.get(name)
            if widget is not None:
                widget.config(state=state)
        if "btn" in self.__dict__:
            self.btn.config(state=state)

    def _set_job_active(self, active):
        """Lock or restore controls whose values affect the active run."""
        if active:
            self._run_control_states = []
            for widget in self._run_controls:
                try:
                    state = str(widget.cget("state"))
                    self._run_control_states.append((widget, state))
                    widget.config(state="disabled")
                except tk.TclError:
                    continue
            self.btn.config(text="Cancel Generation", command=self.cancel,
                            state="normal")
            self.delete_btn.config(state="disabled")
            return

        for widget, state in self._run_control_states:
            try:
                widget.config(state=state)
            except tk.TclError:
                continue
        self._run_control_states = []
        self._sync_stroke_damage_state()
        self.btn.config(text="⚡  Generate Dataset", command=self.start,
                        state="normal")
        if not self._delete_worker_is_alive():
            self.delete_btn.config(state="normal")

    def _signal_cancel(self):
        """Ask the worker to stop at its next safe cancellation boundary."""
        self.cancel_event.set()
        self.btn.config(text="Cancelling…", state="disabled")
        self.delete_btn.config(state="disabled")
        self.status.config(
            text="Cancelling… waiting for the current operation to finish safely.",
            foreground=WARNING)

    def cancel(self):
        """Confirm and request cooperative cancellation of the active job."""
        if not self._worker_is_alive() or self.cancel_event.is_set():
            return
        if messagebox.askyesno(
                "Cancel generation",
                "Cancel the active generation job?\n\n"
                "The current sample or packaging operation will finish safely first.",
                icon="warning"):
            self._signal_cancel()

    def _on_close(self):
        """Keep the window alive until an active worker exits safely."""
        if not self._any_worker_is_alive():
            self.destroy()
            return
        if self._closing:
            return
        generation_active = self._worker_is_alive()
        title = "Generation in progress" if generation_active else "Deletion in progress"
        prompt = (
            "Generation is still running. Cancel it and close after the worker "
            "exits safely?" if generation_active else
            "Dataset deletion is still running. Close after it finishes safely?"
        )
        if not messagebox.askyesno(title, prompt, icon="warning"):
            return
        self._closing = True
        if generation_active:
            self._signal_cancel()

    @staticmethod
    def _fmt(seconds: float) -> str:
        seconds = int(round(seconds))
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

    # ------------------------------------------------------------------
    # Start generation
    # ------------------------------------------------------------------
    def _snapshot_run_options(self) -> RunOptions:
        """Collect and validate every run option on the UI thread.

        The worker receives these plain Python values and never calls Tk while
        it is running.  Invalid input raises :class:`RunOptionsError`.
        """
        try:
            count = int(self.count_var.get().replace(",", "").strip())
            config.validate_config(count)
        except (TypeError, ValueError) as exc:
            raise RunOptionsError("Invalid input", str(exc)) from exc
        raw_seed = self.seed_var.get().strip()
        if not raw_seed:
            seed = random.randint(1, 2_147_483_647)
        else:
            try:
                seed = int(raw_seed)
            except ValueError:
                raise RunOptionsError(
                    "Invalid input", "Seed must be a whole number.") from None

        dataset = self.dataset_var.get().strip()
        if dataset in ("", "(next)"):
            dataset = None
        names_version = self.names_var.get().strip() or None
        sample_mode = self._current_sample_mode()
        font_style, cursive_group, specific_font = self._current_font_selection()

        custom_split_fractions = None
        if hasattr(self, "split_train_var"):
            try:
                tr_pct = float(self.split_train_var.get())
                val_pct = float(self.split_val_var.get())
                te_pct = float(self.split_test_var.get())
            except ValueError:
                raise RunOptionsError(
                    "Invalid split ratios", "Split ratios must be valid numbers."
                ) from None
            total_pct = tr_pct + val_pct + te_pct
            if abs(total_pct - 100.0) > 0.01 or tr_pct < 0 or val_pct < 0 or te_pct < 0:
                raise RunOptionsError(
                    "Invalid split ratios",
                    "Train, Val, and Test split percentages must be non-negative "
                    "and sum to 100%.")
            custom_split_fractions = (tr_pct / 100.0, val_pct / 100.0, te_pct / 100.0)

        custom_field_weights = None
        if hasattr(self, "field_weight_vars") and self.field_weight_vars:
            try:
                custom_field_weights = {}
                for ftype, var in self.field_weight_vars.items():
                    w = int(var.get())
                    if w < 0:
                        raise ValueError(f"Weight for {ftype} cannot be negative.")
                    custom_field_weights[ftype] = w
            except ValueError as exc:
                raise RunOptionsError(
                    "Invalid field weights",
                    f"Field weights must be non-negative integers: {exc}") from None
            if not any(v > 0 for v in custom_field_weights.values()):
                raise RunOptionsError(
                    "Invalid field weights",
                    "At least one field weight must be greater than zero.")

        workers_count = None
        if hasattr(self, "workers_var"):
            w_str = self.workers_var.get().strip()
            if w_str and not w_str.startswith("Auto"):
                try:
                    workers_count = int(w_str)
                except ValueError:
                    workers_count = None

        return RunOptions(
            count=count,
            dataset=dataset,
            seed=seed,
            names_version=names_version,
            sample_mode=sample_mode,
            font_style=font_style,
            cursive_group=cursive_group,
            specific_font=specific_font,
            augmentation_profile=self._build_custom_developer_profile(),
            edge_clipping=self._get_custom_crop_tuple(),
            stroke_damage_profile=self._build_stroke_damage_profile(),
            merge_real=bool(self.real_var.get()),
            package_zip=bool(self.zip_var.get()),
            custom_field_weights=custom_field_weights,
            custom_split_fractions=custom_split_fractions,
            workers=workers_count,
        )

    def start(self):
        if self._any_worker_is_alive():
            return
        try:
            options = self._snapshot_run_options()
        except RunOptionsError as exc:
            messagebox.showerror(exc.title, str(exc))
            return
        warning_count = getattr(config, "LARGE_GENERATION_WARNING_COUNT", None)
        if warning_count is not None and options.count >= warning_count:
            if not messagebox.askyesno(
                "Large generation job",
                f"Generate {options.count:,} samples? This may require substantial disk "
                "space and time.",
                icon="warning",
            ):
                return

        self.cancel_event.clear()
        self._pending_terminal = None
        self._set_job_active(True)
        self.open_btn.config(state="disabled")
        self._last_output = None
        self.progress.config(maximum=options.count, value=0)
        self.status.config(text="Starting…", foreground=WARNING)
        self.timer.config(text="")
        self.start_time = time.time()

        self.worker = threading.Thread(
            target=self._run,
            args=(options.count, options.dataset, options.seed,
                  options.names_version, options.sample_mode,
                  options.font_style, options.cursive_group, options.specific_font,
                  options.merge_real, options.package_zip, self.cancel_event,
                  options.augmentation_profile, options.edge_clipping,
                  options.stroke_damage_profile, options.custom_field_weights,
                  options.custom_split_fractions, options.workers),
            daemon=False)
        self.worker.start()

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------
    def _run(self, count, dataset, seed, names_version, sample_mode,
             font_style, cursive_group, specific_font,
             merge_real, package_zip, cancel_event,
             augmentation_profile=config.DEFAULT_DEGRADATION_PROFILE,
             edge_clipping=config.DEFAULT_EDGE_CLIPPING,
             stroke_damage_profile=None,
             custom_field_weights=None,
             custom_split_fractions=None,
             workers=None):
        try:
            def cb(done, total, field_type):
                self.q.put(("progress", done, total, field_type))

            out_dir = generate(count, dataset=dataset, seed=seed,
                               names_version=names_version,
                               sample_mode=sample_mode, font_style=font_style,
                               cursive_group=cursive_group, specific_font=specific_font,
                               progress_callback=cb, show_bar=False,
                               cancel_event=cancel_event,
                               archive_planned=package_zip,
                               augmentation_profile=augmentation_profile,
                               edge_clipping=edge_clipping,
                               stroke_damage_profile=stroke_damage_profile,
                               custom_field_weights=custom_field_weights,
                               custom_split_fractions=custom_split_fractions,
                               workers=workers)
            if cancel_event.is_set():
                self.q.put(("cancelled", "Generation cancelled safely."))
                return
            merge_payload = None
            if merge_real:
                self.q.put(("status", "Merging real data…"))
                merge_result = build(out_dir.name)
                merge_payload = {
                    name: getattr(merge_result, name)
                    for name in (
                        "copied", "unchanged", "removed", "skipped", "failed"
                    )
                }
                self.q.put(("merge_result", merge_payload))
            if cancel_event.is_set():
                self.q.put(("cancelled", "Generation cancelled safely."))
                return
            out = str(out_dir)
            if package_zip:
                self.q.put(("status", "Packaging .zip…"))
                out = str(zip_dataset(out_dir, cancel_event=cancel_event))
            if cancel_event.is_set():
                self.q.put(("cancelled", "Generation cancelled safely."))
                return
            self.q.put(("done", count, out, seed, merge_payload))
        except GenerationCancelled as exc:
            detail = str(exc).strip()
            self.q.put(("cancelled", detail or "Generation cancelled safely."))
        except Exception as e:
            detail = str(e).strip()
            log_path = _log_error("generation worker", traceback.format_exc())
            self.q.put((
                "error", detail or e.__class__.__name__, str(log_path)
            ))

    # ------------------------------------------------------------------
    # Poll queue (UI thread)
    # ------------------------------------------------------------------
    def _finish_job(self, msg):
        """Apply one terminal worker result after the worker has exited."""
        kind = msg[0]
        total_time = time.time() - self.start_time
        self._set_job_active(False)

        if kind == "done":
            self.progress.config(value=self.progress["maximum"])
            self.status.config(
                text=f"✓  Done — {msg[1]:,} samples generated.",
                foreground=SUCCESS)
            self.timer.config(text=f"Total time: {self._fmt(total_time)}")
            self._last_output = msg[2]
            self.open_btn.config(state="normal")
            self._refresh_datasets()
            merge_result = msg[4] if len(msg) > 4 else None
            merge_summary = ""
            no_op = False
            if merge_result is not None:
                merge_summary = (
                    "\n\nReal merge:\n"
                    f"  copied: {merge_result['copied']}\n"
                    f"  unchanged: {merge_result['unchanged']}\n"
                    f"  removed: {merge_result['removed']}\n"
                    f"  skipped: {merge_result['skipped']}\n"
                    f"  failed: {merge_result['failed']}"
                )
                no_op = (
                    merge_result["copied"] == 0
                    and merge_result["removed"] == 0
                    and merge_result["failed"] == 0
                )
                if no_op:
                    merge_summary += "\n\nWarning: real merge made no dataset changes."
            messagebox.showinfo(
                "Generation complete",
                f"Generated {msg[1]:,} samples in {self._fmt(total_time)}.\n"
                f"Effective seed: {msg[3]}\n\n"
                f"Output:\n{msg[2]}{merge_summary}")
            return

        self.open_btn.config(state="disabled")
        self.timer.config(text=f"Elapsed: {self._fmt(total_time)}")
        self._refresh_datasets()
        if kind == "cancelled":
            self.status.config(text="Generation cancelled safely.",
                               foreground=WARNING)
            return

        self.status.config(text="✗  Error — see details.", foreground=ERROR_C)
        log_note = f"\n\nFull details were saved to:\n{msg[2]}" if len(msg) > 2 else ""
        messagebox.showerror("Error", f"{msg[1]}{log_note}")

    def _poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                kind = msg[0]
                if kind == "progress":
                    if getattr(self, "cancel_event", threading.Event()).is_set():
                        continue
                    _, done, total, field_type = msg
                    self.progress.config(value=done)
                    pct = done / total * 100 if total else 0
                    self.status.config(
                        text=f"{done:,} / {total:,}  ({pct:.1f}%)  ·  {field_type}",
                        foreground=WARNING)
                    elapsed = time.time() - self.start_time
                    if done > 0:
                        rate = done / elapsed
                        remaining = (total - done) / rate if rate > 0 else 0
                        self.timer.config(
                            text=f"{self._fmt(elapsed)} elapsed  ·  "
                                 f"~{self._fmt(remaining)} left  ·  "
                                 f"{rate:.0f} img/s")
                elif kind == "status":
                    if not self.cancel_event.is_set():
                        self.status.config(text=msg[1], foreground=WARNING)
                elif kind == "merge_result":
                    self._merge_result = msg[1]
                elif kind == "dataset_size":
                    self._patch_row(*msg[1:])
                elif kind == "refresh_complete":
                    if not self.__dict__.get("_refresh_finalize_scheduled", False):
                        self._refresh_finalize_scheduled = True
                        self.after(0, self._finish_refresh_worker)
                elif kind == "delete_status":
                    if "dataset_status" in self.__dict__:
                        self.dataset_status.config(text=msg[1], foreground=WARNING)
                elif kind in ("delete_done", "delete_error"):
                    self._pending_delete = msg
                elif kind in ("done", "cancelled", "error"):
                    self._pending_terminal = msg
        except queue.Empty:
            pass

        if self._pending_terminal is not None:
            # A terminal result is queued immediately before _run returns. Do
            # not re-enable destructive controls or close Tk until it is gone.
            if self._worker_is_alive():
                self.after(50, self._poll)
                return
            msg = self._pending_terminal
            self._pending_terminal = None
            if self.worker is not None:
                self.worker.join(timeout=0)
            self.worker = None
            if self._closing and not self._delete_worker_is_alive():
                self.destroy()
                return
            self._finish_job(msg)

        pending_delete = self.__dict__.get("_pending_delete")
        if pending_delete is not None:
            if self._delete_worker_is_alive():
                self.after(50, self._poll)
                return
            self._pending_delete = None
            if self.delete_worker is not None:
                self.delete_worker.join(timeout=0)
            self.delete_worker = None
            if self._closing and not self._worker_is_alive():
                self.destroy()
                return
            self._set_dataset_operation_active(False)
            if pending_delete[0] == "delete_done":
                _kind, name, removed_folder, removed_zip, removed_checksum = pending_delete
                removed = sum((removed_folder, removed_zip, removed_checksum))
                self.dataset_status.config(
                    text=f"Deleted {name} ({removed} artifact(s)).",
                    foreground=SUCCESS,
                )
                self._refresh_datasets()
            else:
                self.dataset_status.config(text="Deletion failed.", foreground=ERROR_C)
                messagebox.showerror(
                    "Delete failed",
                    f"{pending_delete[1]}\n\nFull details were saved to:\n"
                    f"{pending_delete[2]}",
                )

        if self.worker is not None and not self._worker_is_alive():
            # A normal/cancelled/error path always queues a terminal result.
            # Recheck once after observing thread exit to cover the narrow
            # queue hand-off race before reporting an unexpected stop.
            try:
                msg = self.q.get_nowait()
            except queue.Empty:
                msg = ("error", "Generation worker stopped without a result.")
            if msg[0] in ("done", "cancelled", "error"):
                self._pending_terminal = msg
            else:
                self._pending_terminal = (
                    "error", "Generation worker stopped without a result.")
            self.after(0, self._poll)
            return

        if self.delete_worker is not None and not self._delete_worker_is_alive():
            # Mirror the generation hand-off guard for unexpected delete-worker
            # exits that did not enqueue a result.
            self._pending_delete = (
                "delete_error",
                "Dataset deletion worker stopped without a result.",
                str(LOG_FILE),
            )
            self.after(0, self._poll)
            return
        self.after(100, self._poll)


if __name__ == "__main__":
    App().mainloop()
