"""
Click-to-run GUI for generating the synthetic dataset (no terminal needed).

Launch by double-clicking "Generate Images.bat", or run:
    python gui.py
"""

import os
import sys
import time
import queue
import shutil
import threading
import subprocess
import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox

if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image, ImageDraw, ImageFont, ImageTk

import config
from src.generate_synthetic import generate, zip_dataset
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


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Civil Registry Dataset Generator")
        self.geometry("1040x740")
        self.minsize(1040, 740)
        self.resizable(False, False)
        self.configure(bg=BG)

        self.q = queue.Queue()
        self.worker = None
        self.start_time = None
        self._last_output = None       # path of the last finished dataset
        self._preview_img = None       # keep a ref so Tk doesn't GC it
        self._dataset_paths = {}       # tree iid -> Path

        self._apply_theme()
        self._build_ui()
        self.after(150, self._update_preview)
        self.after(150, self._refresh_datasets)

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
        ttk.Label(hdr, text="Civil Registry", style="Title.TLabel").pack(anchor="w")
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
        # Two-column landscape layout.
        cols = ttk.Frame(parent)
        cols.pack(fill="both", expand=True, padx=8, pady=(4, 4))
        left = ttk.Frame(cols, width=480)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        right = ttk.Frame(cols)
        right.pack(side="left", fill="both", expand=True)

        # ================= LEFT: settings =================================
        # ---- Generation settings -----------------------------------------
        self._section_label(left, "Generation")
        card1 = self._card(left, pady=(0, 4))

        crow = ttk.Frame(card1, style="Card.TFrame")
        crow.pack(fill="x", pady=(0, 8))
        ttk.Label(crow, text="Samples", style="CardLabel.TLabel", width=13).pack(side="left")
        self.count_var = tk.StringVar(value="20000")
        ttk.Entry(crow, textvariable=self.count_var, width=10).pack(side="left")

        qp = ttk.Frame(card1, style="Card.TFrame")
        qp.pack(fill="x", pady=(0, 8))
        ttk.Label(qp, text="", width=13).pack(side="left")
        for n in (1_000, 5_000, 20_000, 40_000):
            ttk.Button(qp, text=f"{n:,}", width=7, style="Quick.TButton",
                       command=lambda v=n: self.count_var.set(str(v))
                       ).pack(side="left", padx=2)

        drow = ttk.Frame(card1, style="Card.TFrame")
        drow.pack(fill="x", pady=(0, 8))
        ttk.Label(drow, text="Dataset folder", style="CardLabel.TLabel", width=13).pack(side="left")
        self.dataset_var = tk.StringVar(value="(next)")
        ttk.Entry(drow, textvariable=self.dataset_var, width=14).pack(side="left", padx=(0, 8))
        ttk.Label(drow, text="blank = next", style="CardHint.TLabel").pack(side="left")

        nrow = ttk.Frame(card1, style="Card.TFrame")
        nrow.pack(fill="x")
        ttk.Label(nrow, text="Names pool", style="CardLabel.TLabel", width=13).pack(side="left")
        versions = config.name_versions() or [config.NAMES_VERSION]
        default_v = config.NAMES_VERSION if config.NAMES_VERSION in versions else versions[0]
        self.names_var = tk.StringVar(value=default_v)
        ttk.Combobox(nrow, textvariable=self.names_var, values=versions,
                     state="readonly", width=12).pack(side="left")

        # ---- Style settings ----------------------------------------------
        self._section_label(left, "Style")
        card2 = self._card(left, pady=(0, 4))

        mrow = ttk.Frame(card2, style="Card.TFrame")
        mrow.pack(fill="x", pady=(0, 8))
        ttk.Label(mrow, text="Sample mode", style="CardLabel.TLabel", width=13).pack(side="left")
        self.mode_by_label = {label: key for key, label in config.SAMPLE_MODES.items()}
        mode_labels = list(self.mode_by_label)
        self.sample_mode_var = tk.StringVar(value=config.SAMPLE_MODES[config.DEFAULT_SAMPLE_MODE])
        ttk.Combobox(mrow, textvariable=self.sample_mode_var, values=mode_labels,
                     state="readonly", width=26).pack(side="left")

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

        # ---- Options -----------------------------------------------------
        self._section_label(left, "Options")
        card3 = self._card(left, pady=(0, 4))
        self.real_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(card3, text="Merge real (mock) data afterwards",
                        variable=self.real_var).pack(anchor="w", pady=(0, 4))
        self.zip_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(card3, text="Package dataset as .zip when done",
                        variable=self.zip_var).pack(anchor="w")

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

        self.tree = ttk.Treeview(tree_wrap, columns=("archive",),
                                 show="tree headings", height=12, selectmode="browse")
        self.tree.heading("#0", text="Dataset")
        self.tree.heading("archive", text="Archive")
        self.tree.column("#0", width=340, anchor="w")
        self.tree.column("archive", width=140, anchor="center")
        self.tree.pack(side="left", fill="both", expand=True)
        self.tree.bind("<Double-1>", lambda e: self._open_selected_dataset())

        vsb = ttk.Scrollbar(tree_wrap, orient="vertical", command=self.tree.yview)
        vsb.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=vsb.set)

        hint = ttk.Label(parent,
                         text="A 📦 tag means a .zip archive exists next to the folder. "
                              "Deleting a dataset also deletes its .zip.",
                         style="Sub.TLabel")
        hint.pack(anchor="w", padx=16, pady=(2, 8))

        btn_row = ttk.Frame(parent)
        btn_row.pack(fill="x", padx=16, pady=(0, 12))
        ttk.Button(btn_row, text="🔄  Refresh", style="Secondary.TButton",
                   command=self._refresh_datasets).pack(side="left")
        ttk.Button(btn_row, text="📂  Open", style="Secondary.TButton",
                   command=self._open_selected_dataset).pack(side="left", padx=(8, 0))
        ttk.Button(btn_row, text="🗑  Delete", style="Danger.TButton",
                   command=self._delete_selected_dataset).pack(side="right")

    # ------------------------------------------------------------------
    # Tab change
    # ------------------------------------------------------------------
    def _on_tab_change(self, _event=None):
        if self.nb.index(self.nb.select()) == 1:   # datasets tab
            self._refresh_datasets()

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
        self._preview_img = ImageTk.PhotoImage(img)
        self.preview_label.config(image=self._preview_img, text="")

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
    def _zip_for(folder):
        """Sibling .zip path for a dataset folder."""
        return folder.parent / (folder.name + ".zip")

    def _refresh_datasets(self):
        if not hasattr(self, "tree"):
            return
        self.tree.delete(*self.tree.get_children())
        self._dataset_paths.clear()

        base = config.DATASETS_DIR
        if not base.exists():
            return
        folders = [p for p in base.iterdir() if p.is_dir()]
        folders.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        for folder in folders:
            has_zip = self._zip_for(folder).exists()
            tag = "📦 .zip" if has_zip else "—"
            iid = self.tree.insert("", "end", text=f"  📁  {folder.name}",
                                   values=(tag,))
            self._dataset_paths[iid] = folder

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
        if folder.exists():
            subprocess.Popen(["explorer", str(folder)])

    def _delete_selected_dataset(self):
        folder = self._selected_dataset()
        if not folder:
            messagebox.showinfo("No selection", "Select a dataset to delete first.")
            return

        zip_path = self._zip_for(folder)
        msg = f"Delete dataset '{folder.name}'?\n\nThis permanently removes the folder"
        if zip_path.exists():
            msg += f" and its archive ({zip_path.name})"
        msg += ".\nThis cannot be undone."

        if not messagebox.askyesno("Confirm delete", msg, icon="warning"):
            return

        try:
            if folder.exists():
                shutil.rmtree(folder)
            if zip_path.exists():
                zip_path.unlink()
        except Exception as e:
            messagebox.showerror("Delete failed", str(e))
            return

        self._refresh_datasets()

    # ------------------------------------------------------------------
    # Open last output in Explorer
    # ------------------------------------------------------------------
    def _open_output(self):
        if not self._last_output:
            return
        path = self._last_output
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
    # Start generation
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
        font_style, cursive_group, specific_font = self._current_font_selection()

        self.worker = threading.Thread(
            target=self._run,
            args=(count, dataset, names_version, sample_mode,
                  font_style, cursive_group, specific_font),
            daemon=True)
        self.worker.start()
        self.after(100, self._poll)

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------
    def _run(self, count, dataset, names_version, sample_mode,
             font_style, cursive_group, specific_font):
        try:
            def cb(done, total, field_type):
                self.q.put(("progress", done, total, field_type))

            out_dir = generate(count, dataset=dataset, names_version=names_version,
                               sample_mode=sample_mode, font_style=font_style,
                               cursive_group=cursive_group, specific_font=specific_font,
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
                    self.timer.config(text=f"Total time: {self._fmt(total_time)}")
                    self.btn.config(state="normal")
                    self._last_output = msg[2]
                    self.open_btn.config(state="normal")
                    self._refresh_datasets()
                    messagebox.showinfo(
                        "Generation complete",
                        f"Generated {msg[1]:,} samples in {self._fmt(total_time)}.\n\nOutput:\n{msg[2]}")
                    return
                elif kind == "error":
                    self.status.config(text="✗  Error — see details.", foreground=ERROR_C)
                    self.btn.config(state="normal")
                    messagebox.showerror("Error", msg[1])
                    return
        except queue.Empty:
            pass
        self.after(100, self._poll)


if __name__ == "__main__":
    App().mainloop()
