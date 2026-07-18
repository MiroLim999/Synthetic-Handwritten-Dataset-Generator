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
import tkinter as tk
from tkinter import ttk, messagebox

if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.generate_synthetic import generate, zip_dataset
from src.build_splits import build
from src.render import CURSIVE_STYLE_GROUPS

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
BG        = "#1e1e2e"   # main background
SURFACE   = "#2a2a3d"   # card / section background
BORDER    = "#3a3a55"   # subtle border / separator
ACCENT    = "#7c6af7"   # primary purple accent
ACCENT_H  = "#9d8fff"   # accent hover
SUCCESS   = "#50fa7b"   # green for done state
WARNING   = "#ffb86c"   # orange for running state
ERROR_C   = "#ff5555"   # red for error
TEXT      = "#cdd6f4"   # primary text
SUBTEXT   = "#6e6e9a"   # muted / hint text
BTN_FG    = "#ffffff"   # button foreground


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Civil Registry Dataset Generator")
        self.geometry("560x680")
        self.resizable(False, False)
        self.configure(bg=BG)

        self.q = queue.Queue()
        self.worker = None
        self.start_time = None

        self._apply_theme()
        self._build_ui()

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
                        troughcolor=SURFACE,
                        selectbackground=ACCENT,
                        selectforeground=BTN_FG,
                        font=("Segoe UI", 10))

        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=SURFACE,
                        relief="flat", borderwidth=1)

        style.configure("TLabel", background=BG, foreground=TEXT,
                        font=("Segoe UI", 10))
        style.configure("Hint.TLabel", background=SURFACE, foreground=SUBTEXT,
                        font=("Segoe UI", 9))
        style.configure("Section.TLabel", background=BG, foreground=ACCENT,
                        font=("Segoe UI", 9, "bold"))
        style.configure("Title.TLabel", background=BG, foreground=TEXT,
                        font=("Segoe UI", 13, "bold"))
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

        style.configure("TEntry",
                        fieldbackground=SURFACE,
                        foreground=TEXT,
                        insertcolor=TEXT,
                        bordercolor=BORDER,
                        relief="flat")
        style.map("TEntry",
                  bordercolor=[("focus", ACCENT)])

        style.configure("TCombobox",
                        fieldbackground=SURFACE,
                        background=SURFACE,
                        foreground=TEXT,
                        arrowcolor=ACCENT,
                        bordercolor=BORDER,
                        relief="flat",
                        selectbackground=SURFACE,
                        selectforeground=TEXT)
        style.map("TCombobox",
                  fieldbackground=[("readonly", SURFACE)],
                  foreground=[("readonly", TEXT)],
                  bordercolor=[("focus", ACCENT)])

        style.configure("TCheckbutton",
                        background=BG,
                        foreground=TEXT,
                        indicatorcolor=SURFACE,
                        indicatorbackground=SURFACE)
        style.map("TCheckbutton",
                  indicatorcolor=[("selected", ACCENT)],
                  foreground=[("active", TEXT)])

        style.configure("TProgressbar",
                        troughcolor=SURFACE,
                        background=ACCENT,
                        bordercolor=BORDER,
                        lightcolor=ACCENT,
                        darkcolor=ACCENT,
                        thickness=8)

        # Quick-pick buttons
        style.configure("Quick.TButton",
                        background=SURFACE,
                        foreground=TEXT,
                        bordercolor=BORDER,
                        relief="flat",
                        padding=(6, 4),
                        font=("Segoe UI", 9))
        style.map("Quick.TButton",
                  background=[("active", BORDER)],
                  foreground=[("active", TEXT)])

        # Primary generate button
        style.configure("Generate.TButton",
                        background=ACCENT,
                        foreground=BTN_FG,
                        bordercolor=ACCENT,
                        relief="flat",
                        padding=(20, 10),
                        font=("Segoe UI", 11, "bold"))
        style.map("Generate.TButton",
                  background=[("active", ACCENT_H), ("disabled", BORDER)],
                  foreground=[("disabled", SUBTEXT)])

        # Apply combobox dropdown list colours via option_add
        self.option_add("*TCombobox*Listbox.background", SURFACE)
        self.option_add("*TCombobox*Listbox.foreground", TEXT)
        self.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
        self.option_add("*TCombobox*Listbox.selectForeground", BTN_FG)
        self.option_add("*TCombobox*Listbox.font", ("Segoe UI", 10))

    # ------------------------------------------------------------------
    # UI layout
    # ------------------------------------------------------------------
    def _card(self, parent, pady=(0, 0)):
        """Create a rounded-looking card frame."""
        outer = ttk.Frame(parent)
        outer.pack(fill="x", padx=16, pady=pady)
        card = ttk.Frame(outer, style="Card.TFrame", padding=12)
        card.pack(fill="x")
        return card

    def _section_label(self, parent, text):
        ttk.Label(parent, text=text.upper(), style="Section.TLabel").pack(
            anchor="w", padx=16, pady=(14, 2))

    def _build_ui(self):
        # ---- Header -------------------------------------------------------
        hdr = ttk.Frame(self)
        hdr.pack(fill="x", padx=16, pady=(16, 4))
        ttk.Label(hdr, text="Civil Registry", style="Title.TLabel").pack(anchor="w")
        ttk.Label(hdr, text="Synthetic Dataset Generator", style="Sub.TLabel").pack(anchor="w")

        sep = tk.Frame(self, height=1, bg=BORDER)
        sep.pack(fill="x", padx=16, pady=(8, 0))

        # ---- Generation settings -----------------------------------------
        self._section_label(self, "Generation")
        card1 = self._card(self, pady=(0, 4))

        # Count row
        crow = ttk.Frame(card1, style="Card.TFrame")
        crow.pack(fill="x", pady=(0, 8))
        ttk.Label(crow, text="Samples", style="CardLabel.TLabel", width=14).pack(side="left")
        self.count_var = tk.StringVar(value="20000")
        cnt_entry = ttk.Entry(crow, textvariable=self.count_var, width=10)
        cnt_entry.pack(side="left", padx=(0, 10))

        qp = ttk.Frame(crow, style="Card.TFrame")
        qp.pack(side="left")
        for n in (1_000, 5_000, 20_000, 40_000):
            ttk.Button(qp, text=f"{n:,}", width=7, style="Quick.TButton",
                       command=lambda v=n: self.count_var.set(str(v))
                       ).pack(side="left", padx=2)

        # Dataset folder row
        drow = ttk.Frame(card1, style="Card.TFrame")
        drow.pack(fill="x", pady=(0, 8))
        ttk.Label(drow, text="Dataset folder", style="CardLabel.TLabel", width=14).pack(side="left")
        self.dataset_var = tk.StringVar(value="(next)")
        ttk.Entry(drow, textvariable=self.dataset_var, width=14).pack(side="left", padx=(0, 8))
        ttk.Label(drow, text="name, number, or blank = next",
                  style="CardHint.TLabel").pack(side="left")

        # Names version row
        nrow = ttk.Frame(card1, style="Card.TFrame")
        nrow.pack(fill="x")
        ttk.Label(nrow, text="Names pool", style="CardLabel.TLabel", width=14).pack(side="left")
        versions = config.name_versions() or [config.NAMES_VERSION]
        default_v = config.NAMES_VERSION if config.NAMES_VERSION in versions else versions[0]
        self.names_var = tk.StringVar(value=default_v)
        ttk.Combobox(nrow, textvariable=self.names_var, values=versions,
                     state="readonly", width=12).pack(side="left")

        # ---- Style settings ----------------------------------------------
        self._section_label(self, "Style")
        card2 = self._card(self, pady=(0, 4))

        # Sample mode row
        mrow = ttk.Frame(card2, style="Card.TFrame")
        mrow.pack(fill="x", pady=(0, 8))
        ttk.Label(mrow, text="Sample mode", style="CardLabel.TLabel", width=14).pack(side="left")
        self.mode_by_label = {label: key for key, label in config.SAMPLE_MODES.items()}
        mode_labels = list(self.mode_by_label)
        self.sample_mode_var = tk.StringVar(value=config.SAMPLE_MODES[config.DEFAULT_SAMPLE_MODE])
        ttk.Combobox(mrow, textvariable=self.sample_mode_var, values=mode_labels,
                     state="readonly", width=28).pack(side="left")

        # Font style row
        frow = ttk.Frame(card2, style="Card.TFrame")
        frow.pack(fill="x", pady=(0, 0))
        ttk.Label(frow, text="Font style", style="CardLabel.TLabel", width=14).pack(side="left")
        self.font_style_options = {
            "All fonts":    "all",
            "Cursive only": "cursive",
        }
        self.font_style_var = tk.StringVar(value="All fonts")
        font_cb = ttk.Combobox(frow, textvariable=self.font_style_var,
                               values=list(self.font_style_options),
                               state="readonly", width=16)
        font_cb.pack(side="left")
        font_cb.bind("<<ComboboxSelected>>", self._on_font_style_change)

        # Cursive sub-style row (hidden initially)
        self.cursive_row = ttk.Frame(card2, style="Card.TFrame")
        ttk.Label(self.cursive_row, text="Cursive style",
                  style="CardLabel.TLabel", width=14).pack(side="left")
        self.cursive_groups = ["All cursive"] + list(CURSIVE_STYLE_GROUPS.keys())
        self.cursive_group_var = tk.StringVar(value="All cursive")
        self.cursive_cb = ttk.Combobox(self.cursive_row,
                                       textvariable=self.cursive_group_var,
                                       values=self.cursive_groups,
                                       state="readonly", width=28)
        self.cursive_cb.pack(side="left")

        # ---- Options -----------------------------------------------------
        self._section_label(self, "Options")
        card3 = self._card(self, pady=(0, 4))

        self.real_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(card3, text="Merge real (mock) data afterwards",
                        variable=self.real_var).pack(anchor="w", pady=(0, 4))

        self.zip_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(card3, text="Package dataset as .zip when done",
                        variable=self.zip_var).pack(anchor="w")

        # ---- Progress ----------------------------------------------------
        self._section_label(self, "Progress")
        card4 = self._card(self, pady=(0, 8))

        self.progress = ttk.Progressbar(card4, length=490, mode="determinate")
        self.progress.pack(fill="x", pady=(0, 6))

        info_row = ttk.Frame(card4, style="Card.TFrame")
        info_row.pack(fill="x")
        self.status = ttk.Label(info_row, text="Ready.", style="Status.TLabel")
        self.status.pack(side="left")
        self.timer = ttk.Label(info_row, text="", style="Timer.TLabel")
        self.timer.pack(side="right")

        # ---- Buttons row -------------------------------------------------
        btn_row = ttk.Frame(self)
        btn_row.pack(pady=(4, 16))

        self.btn = ttk.Button(btn_row, text="⚡  Generate Dataset",
                              style="Generate.TButton", command=self.start)
        self.btn.pack(side="left", padx=(0, 8))

        self.open_btn = ttk.Button(btn_row, text="📂  Open Folder",
                                   style="Quick.TButton", command=self._open_output,
                                   state="disabled")
        self.open_btn.pack(side="left")

        self._last_output = None   # path of the last finished dataset

    # ------------------------------------------------------------------
    # Event: font style changed
    # ------------------------------------------------------------------
    def _on_font_style_change(self, _event=None):
        if self.font_style_var.get() == "Cursive only":
            self.cursive_row.pack(fill="x", pady=(6, 0))
        else:
            self.cursive_row.pack_forget()
            self.cursive_group_var.set("All cursive")

    # ------------------------------------------------------------------
    # Open output folder / zip in Explorer
    # ------------------------------------------------------------------
    def _open_output(self):
        if not self._last_output:
            return
        import subprocess
        path = self._last_output
        # If it's a .zip, select it in the parent folder; otherwise open the folder.
        if path.endswith(".zip"):
            subprocess.Popen(["explorer", "/select,", path])
        else:
            subprocess.Popen(["explorer", path])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _fmt(seconds: float) -> str:
        seconds = int(round(seconds))
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

    # ------------------------------------------------------------------
    # Start
    # ------------------------------------------------------------------
    def start(self):
        if self.worker and self.worker.is_alive():
            return
        try:
            count = int(self.count_var.get().replace(",", "").strip())
            if count <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid input", "Please enter a positive whole number.")
            return

        self.btn.config(state="disabled")
        self.open_btn.config(state="disabled")
        self._last_output = None
        self.progress.config(maximum=count, value=0)
        self.status.config(text="Starting…", foreground=WARNING)
        self.timer.config(text="")
        self.start_time = time.time()

        dataset = self.dataset_var.get().strip()
        if dataset in ("", "(next)"):
            dataset = None
        names_version = self.names_var.get().strip() or None
        sample_mode = self.mode_by_label.get(
            self.sample_mode_var.get(), config.DEFAULT_SAMPLE_MODE)
        font_style = self.font_style_options.get(
            self.font_style_var.get(), "all")
        cursive_group = ""
        if font_style == "cursive":
            sel = self.cursive_group_var.get()
            cursive_group = sel if sel != "All cursive" else ""

        self.worker = threading.Thread(
            target=self._run,
            args=(count, dataset, names_version, sample_mode, font_style, cursive_group),
            daemon=True)
        self.worker.start()
        self.after(100, self._poll)

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------
    def _run(self, count, dataset, names_version, sample_mode, font_style, cursive_group):
        try:
            def cb(done, total, field_type):
                self.q.put(("progress", done, total, field_type))

            out_dir = generate(count, dataset=dataset, names_version=names_version,
                               sample_mode=sample_mode, font_style=font_style,
                               cursive_group=cursive_group,
                               progress_callback=cb, show_bar=False)
            if self.real_var.get():
                self.q.put(("status", "Merging real data…"))
                build(out_dir.name)
            out = str(out_dir)
            if self.zip_var.get():
                self.q.put(("status", "Packaging .zip…"))
                out = str(zip_dataset(out_dir))
            self.q.put(("done", count, out))
        except Exception as e:
            self.q.put(("error", str(e)))

    # ------------------------------------------------------------------
    # Poll queue (UI thread)
    # ------------------------------------------------------------------
    def _poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                kind = msg[0]
                if kind == "progress":
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
                    self.status.config(text=msg[1], foreground=WARNING)
                elif kind == "done":
                    total_time = time.time() - self.start_time
                    self.progress.config(value=self.progress["maximum"])
                    self.status.config(
                        text=f"✓  Done — {msg[1]:,} samples generated.",
                        foreground=SUCCESS)
                    self.timer.config(
                        text=f"Total time: {self._fmt(total_time)}")
                    self.btn.config(state="normal")
                    self._last_output = msg[2]
                    self.open_btn.config(state="normal")
                    messagebox.showinfo(
                        "Generation complete",
                        f"Generated {msg[1]:,} samples in {self._fmt(total_time)}.\n\nOutput:\n{msg[2]}")
                    return
                elif kind == "error":
                    self.status.config(text="✗  Error — see details.",
                                       foreground=ERROR_C)
                    self.btn.config(state="normal")
                    messagebox.showerror("Error", msg[1])
                    return
        except queue.Empty:
            pass
        self.after(100, self._poll)


if __name__ == "__main__":
    App().mainloop()
