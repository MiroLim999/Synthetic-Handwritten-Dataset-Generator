"""
Click-to-run GUI for generating the synthetic dataset (no terminal needed).

Launch by double-clicking "Generate Images.bat", or run:
    python gui.py

Lets you choose how many samples to generate, watch a live progress bar,
and optionally build the train/val/test splits afterwards.
"""

import os
import sys
import time
import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox

# Under pythonw.exe there is no console, so sys.stdout/stderr are None and any
# print() would crash. Redirect them to a throwaway sink before importing code
# that prints.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

# make sure the project root is importable when launched by double-click
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.generate_synthetic import generate
from src.build_splits import build


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Civil Registry Dataset Generator")
        self.geometry("460x330")
        self.resizable(False, False)

        self.q = queue.Queue()
        self.worker = None
        self.start_time = None

        pad = {"padx": 12, "pady": 6}

        # --- count input -------------------------------------------------
        frm = ttk.Frame(self)
        frm.pack(fill="x", **pad)
        ttk.Label(frm, text="Number of samples:").pack(side="left")
        self.count_var = tk.StringVar(value="20000")
        ttk.Entry(frm, textvariable=self.count_var, width=12).pack(side="left", padx=8)

        # quick-pick buttons
        qp = ttk.Frame(self)
        qp.pack(fill="x", padx=12)
        for n in (1000, 5000, 20000, 40000):
            ttk.Button(qp, text=f"{n:,}", width=8,
                       command=lambda v=n: self.count_var.set(str(v))).pack(side="left", padx=3)

        # --- options -----------------------------------------------------
        self.splits_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self, text="Build train/val/test splits afterwards",
                        variable=self.splits_var).pack(anchor="w", padx=12, pady=(10, 0))

        # --- progress ----------------------------------------------------
        self.progress = ttk.Progressbar(self, length=420, mode="determinate")
        self.progress.pack(padx=12, pady=(16, 4))
        self.status = ttk.Label(self, text="Ready.")
        self.status.pack(anchor="w", padx=12)
        self.timer = ttk.Label(self, text="", foreground="#555")
        self.timer.pack(anchor="w", padx=12)

        # --- generate button --------------------------------------------
        self.btn = ttk.Button(self, text="Generate", command=self.start)
        self.btn.pack(pady=14)

    # ---------------------------------------------------------------------
    @staticmethod
    def _fmt(seconds: float) -> str:
        """Format seconds as M:SS or H:MM:SS."""
        seconds = int(round(seconds))
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"

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
        self.progress.config(maximum=count, value=0)
        self.status.config(text="Starting...")
        self.timer.config(text="")
        self.start_time = time.time()

        self.worker = threading.Thread(target=self._run, args=(count,), daemon=True)
        self.worker.start()
        self.after(100, self._poll)

    def _run(self, count):
        """Runs in a background thread."""
        try:
            def cb(done, total, field_type):
                self.q.put(("progress", done, total, field_type))
            generate(count, progress_callback=cb, show_bar=False)
            if self.splits_var.get():
                self.q.put(("status", "Building splits..."))
                build()
            self.q.put(("done", count))
        except Exception as e:  # surface any error to the UI
            self.q.put(("error", str(e)))

    def _poll(self):
        """Runs in the UI thread; drains messages from the worker."""
        try:
            while True:
                msg = self.q.get_nowait()
                kind = msg[0]
                if kind == "progress":
                    _, done, total, field_type = msg
                    self.progress.config(value=done)
                    self.status.config(text=f"{done:,} / {total:,}   •   {field_type}")
                    # elapsed + estimated time remaining
                    elapsed = time.time() - self.start_time
                    if done > 0:
                        rate = done / elapsed                 # images per second
                        remaining = (total - done) / rate if rate > 0 else 0
                        self.timer.config(
                            text=f"Elapsed {self._fmt(elapsed)}   |   "
                                 f"Remaining ~{self._fmt(remaining)}   |   "
                                 f"{rate:.0f} img/s"
                        )
                elif kind == "status":
                    self.status.config(text=msg[1])
                elif kind == "done":
                    total_time = time.time() - self.start_time
                    self.progress.config(value=self.progress["maximum"])
                    self.status.config(text=f"Done. Generated {msg[1]:,} samples.")
                    self.timer.config(text=f"Total time: {self._fmt(total_time)}")
                    self.btn.config(state="normal")
                    out = config.SPLITS_DIR if self.splits_var.get() else config.SYNTHETIC_DIR
                    messagebox.showinfo(
                        "Finished",
                        f"Generated {msg[1]:,} samples in {self._fmt(total_time)}.\n\nOutput:\n{out}"
                    )
                    return
                elif kind == "error":
                    self.status.config(text="Error.")
                    self.btn.config(state="normal")
                    messagebox.showerror("Error", msg[1])
                    return
        except queue.Empty:
            pass
        self.after(100, self._poll)


if __name__ == "__main__":
    App().mainloop()
