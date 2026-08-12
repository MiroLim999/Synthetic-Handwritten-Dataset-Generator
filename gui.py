"""
Click-to-run GUI for generating the synthetic dataset (no terminal needed).

Launch by double-clicking "Generate Images.bat", or run:
    python gui.py
"""

import os
import sys
import time
import queue
import threading
import subprocess
import traceback
import webbrowser
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import ttk, messagebox

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image, ImageDraw, ImageFont, ImageTk

import random

import config
from src.augment import degrade, EDGE_CLIPPING_RANGES
from src.generation_profiles import (AUGMENTATION_PROFILES,
                                     create_custom_augmentation_profile,
                                     get_augmentation_profile)
from src.generate_synthetic import (GenerationCancelled, generate,
                                    guarded_remove_dataset, zip_dataset)
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
MAX_LOG_BYTES = 2 * 1024 * 1024
LOG_BACKUPS = 3


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
        width = min(1040, max(480, screen_width - 80))
        height = min(740, max(360, screen_height - 100))
        self.geometry(f"{width}x{height}")
        self.minsize(min(480, width), min(360, height))
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

        self._apply_theme()
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
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
        right = ttk.Frame(cols)
        cols.add(left_container, weight=5)
        cols.add(right, weight=5)

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

        self._target_scroll_pos = 0.0
        self._smooth_scroll_timer = None

        def _update_smooth_scroll():
            current = viewport.yview()[0]
            target = getattr(self, "_target_scroll_pos", current)
            diff = target - current
            if abs(diff) < 0.0001:
                viewport.yview_moveto(target)
                self._smooth_scroll_timer = None
            else:
                new_pos = current + diff * 0.14
                viewport.yview_moveto(new_pos)
                self._smooth_scroll_timer = self.after(16, _update_smooth_scroll)

        def _on_mousewheel(event):
            try:
                if self.nb.index(self.nb.select()) != 0:
                    return
            except Exception:
                pass

            bbox = viewport.bbox("all")
            if not bbox:
                return
            content_height = bbox[3] - bbox[1]
            visible_height = viewport.winfo_height()
            if content_height <= visible_height or visible_height <= 0:
                return

            scrollable_height = content_height - visible_height
            current_target = getattr(self, "_target_scroll_pos", viewport.yview()[0])

            step_pixels = 18.0
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
            self._target_scroll_pos = new_target

            if getattr(self, "_smooth_scroll_timer", None) is None:
                _update_smooth_scroll()

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
        self.sample_mode_cb.bind("<<ComboboxSelected>>", lambda e: self._update_preview())
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

        # ---- Developer Studio (Custom Parameters) -----------------------
        dev_hdr_row = ttk.Frame(left)
        dev_hdr_row.pack(fill="x", pady=(8, 2))
        ttk.Label(dev_hdr_row, text="DEVELOPER STUDIO (CUSTOM PARAMETERS)", style="Section.TLabel").pack(side="left")
        self.reset_btn = ttk.Button(
            dev_hdr_row, text="🔄 Reset Defaults", style="Secondary.TButton",
            command=self._reset_developer_defaults)
        self.reset_btn.pack(side="right")
        self._run_controls.append(self.reset_btn)

        card_dev = self._card(left, pady=(0, 4))
        self.dev_sliders = {}
        self.dev_labels = {}
        self._slider_updating = False

        def _make_slider(parent, name, label_text, from_, to_, default, resolution=0.01, unit=""):
            row = ttk.Frame(parent, style="Card.TFrame")
            row.pack(fill="x", pady=(0, 4))
            ttk.Label(row, text=label_text, style="CardLabel.TLabel", width=13).pack(side="left")
            val_var = tk.DoubleVar(value=default)
            lbl_var = tk.StringVar(value=f"{default:.2f}{unit}" if resolution < 1.0 else f"{int(default)}{unit}")

            def _on_move(val):
                if getattr(self, "_slider_updating", False):
                    return
                v = float(val)
                lbl_var.set(f"{v:.2f}{unit}" if resolution < 1.0 else f"{int(v)}{unit}")
                self._on_developer_slider_change(name)

            scale = ttk.Scale(row, from_=from_, to=to_, variable=val_var, command=_on_move)
            scale.pack(side="left", fill="x", expand=True, padx=(0, 6))
            ttk.Label(row, textvariable=lbl_var, style="CardHint.TLabel", width=7).pack(side="left")
            self.dev_sliders[name] = val_var
            self.dev_labels[name] = lbl_var
            self._run_controls.append(scale)

        _make_slider(card_dev, "fade", "Ink Fading", 0.10, 1.00, 0.65)
        _make_slider(card_dev, "tint", "Paper Tint", 0.00, 0.40, 0.15)
        _make_slider(card_dev, "stain", "Stain Prob", 0.00, 1.00, 0.35)
        _make_slider(card_dev, "rotate", "Tilt Angle", 0.0, 12.0, 4.5, unit="°")
        _make_slider(card_dev, "blur", "Blur Radius", 0.0, 3.0, 0.8, unit="px")
        _make_slider(card_dev, "noise", "Grain Noise", 0.0, 40.0, 12.0, resolution=1.0)
        _make_slider(card_dev, "texture", "Paper Fiber", 0.0, 12.0, 3.5)
        _make_slider(card_dev, "scanline", "Scanlines", 0.0, 30.0, 10.0, resolution=1.0)
        _make_slider(card_dev, "jpeg", "JPEG Quality", 20.0, 100.0, 75.0, resolution=1.0)

        ttk.Separator(card_dev, orient="horizontal").pack(fill="x", pady=6)
        ttk.Label(card_dev, text="Character Cutoff / Edge Crops", style="Sub.TLabel").pack(anchor="w", pady=(0, 4))

        _make_slider(card_dev, "crop_top", "Top Cut %", 0.0, 15.0, 0.0, resolution=1.0, unit="%")
        _make_slider(card_dev, "crop_bottom", "Bottom Cut %", 0.0, 15.0, 0.0, resolution=1.0, unit="%")
        _make_slider(card_dev, "crop_left", "Left Cut %", 0.0, 15.0, 0.0, resolution=1.0, unit="%")
        _make_slider(card_dev, "crop_right", "Right Cut %", 0.0, 15.0, 0.0, resolution=1.0, unit="%")

        ttk.Separator(card_dev, orient="horizontal").pack(fill="x", pady=6)
        ttk.Label(card_dev, text="Stroke Damage (Semi-Broken)", style="Sub.TLabel").pack(anchor="w", pady=(0, 4))

        _make_slider(card_dev, "gap_prob", "Gap Prob", 0.00, 1.00, 0.90)
        _make_slider(card_dev, "gap_count", "Gap Count", 0.0, 30.0, 16.0, resolution=1.0)
        _make_slider(card_dev, "scratch", "Scratch Prob", 0.00, 1.00, 0.70)
        _make_slider(card_dev, "erode", "Stroke Thin", 0.00, 1.00, 0.55)

        ttk.Separator(card_dev, orient="horizontal").pack(fill="x", pady=6)
        reset_bottom_btn = ttk.Button(
            card_dev, text="🔄 Reset All Controls to Default", style="Secondary.TButton",
            command=self._reset_developer_defaults)
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

        # ================= RIGHT: preview + progress ======================
        self._section_label(right, "Preview")
        pcard = self._card(right, pady=(0, 4))
        self.preview_holder = tk.Frame(pcard, bg=PREVIEW_BG,
                                       highlightbackground=BORDER,
                                       highlightthickness=1, bd=0,
                                       height=280)
        self.preview_holder.pack(fill="both", expand=True)
        self.preview_holder.pack_propagate(False)
        self.preview_label = tk.Label(self.preview_holder, bg=PREVIEW_BG,
                                      text="Loading preview…", fg=SUBTEXT,
                                      font=("Segoe UI", 9))
        self.preview_label.pack(expand=True)

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

    def _make_preview_image(self, pool, lines=3, size=42):
        """Render SAMPLE_TEXT in a few representative fonts, stacked vertically."""
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
            box = measure.textbbox((0, 0), SAMPLE_TEXT, font=font)
            w, h = box[2] - box[0], box[3] - box[1]
            im = Image.new("RGB", (w + 24, h + 20), PREVIEW_BG)
            d = ImageDraw.Draw(im)
            d.text((12 - box[0], 10 - box[1]), SAMPLE_TEXT, font=font, fill=(35, 40, 55))
            if im.width > PREVIEW_WIDTH:
                r = PREVIEW_WIDTH / im.width
                im = im.resize((PREVIEW_WIDTH, max(1, int(im.height * r))))
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
        try:
            if specific_font:
                fp = font_path_for(specific_font)
                pool = (fp,) if fp else fonts_for_style(font_style, cursive_group)
                img = self._make_preview_image(pool, lines=1, size=64)
            else:
                pool = fonts_for_style(font_style, cursive_group)
                img = self._make_preview_image(pool, lines=3, size=42)
        except Exception:
            img = None

        if img is None:
            self._preview_img = None
            self.preview_label.config(image="", text="(no preview available)")
            return

        profile = self._build_custom_developer_profile()
        crop_tuple = self._get_custom_crop_tuple()
        semi_params = self._build_semi_broken_params()
        sample_mode_key = self.mode_by_label.get(self.sample_mode_var.get(), "regular")
        damage_prof = "semi_broken" if sample_mode_key.startswith("semi_broken") else "regular"
        try:
            img = degrade(
                img,
                damage_profile=damage_prof,
                augmentation_profile=profile,
                edge_clipping=crop_tuple,
                semi_broken_params=semi_params,
                rng=random.Random(42),
            )
        except Exception:
            pass
        self._preview_img = ImageTk.PhotoImage(img)
        self.preview_label.config(image=self._preview_img, text="")

    def _build_semi_broken_params(self):
        if not hasattr(self, "dev_sliders"):
            return None
        return {
            "gap_prob": self.dev_sliders["gap_prob"].get(),
            "gap_count": int(round(self.dev_sliders["gap_count"].get())),
            "scratch_prob": self.dev_sliders["scratch"].get(),
            "erode_prob": self.dev_sliders["erode"].get(),
        }

    def _build_custom_developer_profile(self):
        if not hasattr(self, "dev_sliders"):
            degradation_key = self.degradation_by_label.get(
                self.degradation_var.get(), config.DEFAULT_DEGRADATION_PROFILE
            )
            return get_augmentation_profile(degradation_key)

        preset_key = self.degradation_by_label.get(self.degradation_var.get(), "")
        if preset_key and preset_key != "custom_dev_v1" and preset_key in AUGMENTATION_PROFILES:
            return get_augmentation_profile(preset_key)

        return create_custom_augmentation_profile(
            fade_contrast=self.dev_sliders["fade"].get(),
            paper_tint_alpha=self.dev_sliders["tint"].get(),
            stain_prob=self.dev_sliders["stain"].get(),
            rotate_deg=self.dev_sliders["rotate"].get(),
            blur_radius=self.dev_sliders["blur"].get(),
            noise_std=self.dev_sliders["noise"].get(),
            paper_texture_std=self.dev_sliders["texture"].get(),
            scanline_alpha=int(round(self.dev_sliders["scanline"].get())),
            jpeg_quality=int(round(self.dev_sliders["jpeg"].get())),
        )

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

    def _on_degradation_preset_change(self, _event=None):
        preset_label = self.degradation_var.get()
        preset_key = self.degradation_by_label.get(preset_label, "historical_scan_v1")
        if preset_key in AUGMENTATION_PROFILES and hasattr(self, "dev_sliders"):
            prof = AUGMENTATION_PROFILES[preset_key]
            self._slider_updating = True
            self.dev_sliders["fade"].set((prof.fade_range[0] + prof.fade_range[1]) / 2.0)
            self.dev_sliders["tint"].set((prof.paper_tint_alpha_range[0] + prof.paper_tint_alpha_range[1]) / 2.0)
            self.dev_sliders["stain"].set(prof.stain_probability)
            self.dev_sliders["rotate"].set(prof.rotate_degrees)
            self.dev_sliders["blur"].set((prof.blur_radius_range[0] + prof.blur_radius_range[1]) / 2.0)
            self.dev_sliders["noise"].set((prof.noise_std_range[0] + prof.noise_std_range[1]) / 2.0)
            self.dev_sliders["texture"].set((prof.paper_texture_std_range[0] + prof.paper_texture_std_range[1]) / 2.0)
            self.dev_sliders["scanline"].set((prof.scanline_alpha_range[0] + prof.scanline_alpha_range[1]) / 2.0)
            self.dev_sliders["jpeg"].set((prof.jpeg_quality_range[0] + prof.jpeg_quality_range[1]) / 2.0)
            for k, var in self.dev_sliders.items():
                if k.startswith("crop_"):
                    continue
                v = var.get()
                unit = "°" if k == "rotate" else ("px" if k == "blur" else "")
                res = 1.0 if k in ("noise", "scanline", "jpeg") else 0.01
                self.dev_labels[k].set(f"{v:.2f}{unit}" if res < 1.0 else f"{int(v)}{unit}")
            self._slider_updating = False
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
                self.dev_labels[side].set(f"{int(avg_pct)}%")
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

    def _reset_developer_defaults(self):
        """Reset all controls and sliders back to standard defaults."""
        self._slider_updating = True

        self.sample_mode_var.set(config.SAMPLE_MODES[config.DEFAULT_SAMPLE_MODE])
        self.degradation_var.set(config.DEGRADATION_PROFILES[config.DEFAULT_DEGRADATION_PROFILE])
        self.edge_clipping_var.set(config.EDGE_CLIPPING_OPTIONS[config.DEFAULT_EDGE_CLIPPING])
        self.count_var.set(str(config.DEFAULT_COUNT))
        self.seed_var.set(str(config.RANDOM_SEED))
        if hasattr(self, "names_var") and hasattr(config, "NAMES_VERSION"):
            self.names_var.set(config.NAMES_VERSION)
        self.font_style_var.set("All fonts")
        self._on_font_style_change()
        self.real_var.set(False)
        self.zip_var.set(False)

        defaults = {
            "fade": 0.65,
            "tint": 0.15,
            "stain": 0.35,
            "rotate": 4.5,
            "blur": 0.8,
            "noise": 12.0,
            "texture": 3.5,
            "scanline": 10.0,
            "jpeg": 75.0,
            "crop_top": 0.0,
            "crop_bottom": 0.0,
            "crop_left": 0.0,
            "crop_right": 0.0,
            "gap_prob": 0.90,
            "gap_count": 16.0,
            "scratch": 0.70,
            "erode": 0.55,
        }

        for name, def_val in defaults.items():
            if hasattr(self, "dev_sliders") and name in self.dev_sliders:
                self.dev_sliders[name].set(def_val)
                unit = "°" if name == "rotate" else ("px" if name == "blur" else ("%" if "crop" in name else ""))
                res = 1.0 if name in ("noise", "scanline", "jpeg", "gap_count") or "crop" in name else 0.01
                self.dev_labels[name].set(f"{def_val:.2f}{unit}" if res < 1.0 else f"{int(def_val)}{unit}")

        self._slider_updating = False
        self._update_preview()

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
    def start(self):
        if self._any_worker_is_alive():
            return
        try:
            count = int(self.count_var.get().replace(",", "").strip())
            config.validate_config(count)
        except (TypeError, ValueError) as exc:
            messagebox.showerror("Invalid input", str(exc))
            return
        warning_count = getattr(config, "LARGE_GENERATION_WARNING_COUNT", None)
        if warning_count is not None and count >= warning_count:
            if not messagebox.askyesno(
                "Large generation job",
                f"Generate {count:,} samples? This may require substantial disk "
                "space and time.",
                icon="warning",
            ):
                return
        raw_seed = self.seed_var.get().strip()
        if not raw_seed:
            seed = random.randint(1, 2_147_483_647)
        else:
            try:
                seed = int(raw_seed)
            except ValueError:
                messagebox.showerror("Invalid input", "Seed must be a whole number.")
                return

        # Snapshot every Tk-backed option on the UI thread. The worker receives
        # plain Python values and never calls Tk while it is running.
        dataset = self.dataset_var.get().strip()
        if dataset in ("", "(next)"):
            dataset = None
        names_version = self.names_var.get().strip() or None
        sample_mode = self.mode_by_label.get(
            self.sample_mode_var.get(), config.DEFAULT_SAMPLE_MODE)
        font_style, cursive_group, specific_font = self._current_font_selection()
        augmentation_profile = self._build_custom_developer_profile()
        edge_clipping = self._get_custom_crop_tuple()
        semi_broken_params = self._build_semi_broken_params()
        merge_real = bool(self.real_var.get())
        package_zip = bool(self.zip_var.get())

        self.cancel_event.clear()
        self._pending_terminal = None
        self._set_job_active(True)
        self.open_btn.config(state="disabled")
        self._last_output = None
        self.progress.config(maximum=count, value=0)
        self.status.config(text="Starting…", foreground=WARNING)
        self.timer.config(text="")
        self.start_time = time.time()

        self.worker = threading.Thread(
            target=self._run,
            args=(count, dataset, seed, names_version, sample_mode,
                  font_style, cursive_group, specific_font,
                  merge_real, package_zip, self.cancel_event,
                  augmentation_profile, edge_clipping, semi_broken_params),
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
             semi_broken_params=None):
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
                               semi_broken_params=semi_broken_params)
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
