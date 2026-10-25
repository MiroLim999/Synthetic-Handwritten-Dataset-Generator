"""Slider widgets for the Developer Studio.

:class:`RangeModel` is pure (no Tk) and holds the rules: clamp, snap to the
step, keep min <= max, drag either handle, nudge from the keyboard.
:class:`RangeSlider` is a thin ``tk.Canvas`` view of a model with one handle
(a value or chance) or two handles (a min/max range).
"""

from __future__ import annotations

import math
import re
import tkinter as tk

ACCENT = "#4f46e5"
TRACK = "#dbe1f2"
DISABLED = "#c7c9d6"
SURFACE = "#ffffff"
TEXT = "#1f2430"
SUBTEXT = "#7a8194"
LARGE_STEP = 10


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


class RangeModel:
    """State of a one-handle (``single``) or two-handle slider."""

    def __init__(self, low: float, high: float, step: float, value, *,
                 single: bool = False, integer: bool = False,
                 scale: float = 1.0, unit: str = ""):
        if low >= high:
            raise ValueError("low must be below high")
        self.low, self.high, self.step = float(low), float(high), float(step)
        self.single = single
        self.integer = integer
        self.scale = scale
        self.unit = unit
        self._values = [self.low] if single else [self.low, self.high]
        self.set_value(value)

    @property
    def handles(self) -> tuple[str, ...]:
        return ("value",) if self.single else ("low", "high")

    @property
    def value(self):
        values = [int(v) if self.integer else v for v in self._values]
        return values[0] if self.single else (values[0], values[1])

    def handle_value(self, handle: str) -> float:
        return self._values[self.handles.index(handle)]

    def _snap(self, value) -> float:
        return snap_value(value, self.step, self.low, self.high)

    def set_value(self, value) -> bool:
        """Set every handle (snapped, clamped, ordered); return whether it changed."""
        before = list(self._values)
        if self.single:
            self._values = [self._snap(value)]
        else:
            low, high = sorted((self._snap(value[0]), self._snap(value[1])))
            self._values = [low, high]
        return self._values != before

    def set_handle(self, handle: str, value) -> bool:
        """Move one handle; a range handle stops at the other handle."""
        index = self.handles.index(handle)
        snapped = self._snap(value)
        if not self.single:
            if handle == "low":
                snapped = min(snapped, self._values[1])
            else:
                snapped = max(snapped, self._values[0])
        changed = snapped != self._values[index]
        self._values[index] = snapped
        return changed

    def nudge(self, handle: str, steps: int) -> bool:
        return self.set_handle(handle, self.handle_value(handle) + steps * self.step)

    def fraction(self, value: float) -> float:
        return (value - self.low) / (self.high - self.low)

    def value_at(self, fraction: float) -> float:
        return self._snap(self.low + max(0.0, min(1.0, fraction)) * (self.high - self.low))

    def nearest_handle(self, value: float) -> str:
        if self.single:
            return "value"
        low, high = self._values
        if low == high:
            return "high" if value >= high else "low"
        return "low" if abs(value - low) <= abs(value - high) else "high"

    def display(self, value: float) -> str:
        return format_value(value * self.scale, self.step * self.scale, self.unit)

    def text(self) -> str:
        if self.single:
            return self.display(self._values[0])
        return f"{self.display(self._values[0])} - {self.display(self._values[1])}"

    def parse(self, typed: str):
        """Parse typed display units ("4.5", "10 - 25", "10, 25%") into a value."""
        numbers = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", typed)]
        expected = 1 if self.single else 2
        if len(numbers) == 1 and not self.single:
            numbers = numbers * 2
        if len(numbers) != expected:
            raise ValueError(f"Enter {'a number' if self.single else 'min and max'}")
        values = [number / self.scale for number in numbers]
        return values[0] if self.single else (values[0], values[1])


class RangeSlider(tk.Canvas):
    """Canvas slider with keyboard focus per handle and type-in editing.

    Keys: Left/Right (or Down/Up) move the focused handle one step, with
    Shift ten steps; Home/End jump to the ends; Tab moves from the min
    handle to the max handle before leaving the widget.  Double-click to
    type a value.  ``command(value)`` runs after every user change.
    """

    PAD = 9
    TEXT_WIDTH = 96

    def __init__(self, master, model: RangeModel, *, command=None,
                 width: int = 260, height: int = 26, background: str = SURFACE):
        super().__init__(master, width=width, height=height, highlightthickness=1,
                         highlightbackground=background, highlightcolor=ACCENT,
                         background=background, takefocus=1, borderwidth=0)
        self.model = model
        self.command = command
        self.active = 0
        self.enabled = True
        self._dragging = None
        self._editor = None
        self.bind("<Configure>", lambda _event: self.redraw())
        self.bind("<Button-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", lambda _event: setattr(self, "_dragging", None))
        self.bind("<Double-Button-1>", self._open_editor)
        self.bind("<FocusIn>", lambda _event: self.redraw())
        self.bind("<FocusOut>", lambda _event: self.redraw())
        for key, steps in (("Left", -1), ("Down", -1), ("Right", 1), ("Up", 1)):
            self.bind(f"<{key}>", lambda _event, s=steps: self._key_nudge(s))
            self.bind(f"<Shift-{key}>", lambda _event, s=steps: self._key_nudge(s * LARGE_STEP))
        self.bind("<Home>", lambda _event: self._key_jump(self.model.low))
        self.bind("<End>", lambda _event: self._key_jump(self.model.high))
        self.bind("<Return>", self._open_editor)
        self.bind("<Tab>", self._on_tab)
        self.bind("<Shift-Tab>", self._on_shift_tab)
        self.redraw()

    # -- geometry ---------------------------------------------------------
    def _track(self) -> tuple[float, float, float]:
        width = max(int(self.winfo_width()), int(self["width"]))
        height = max(int(self.winfo_height()), int(self["height"]))
        return self.PAD, max(self.PAD + 20, width - self.TEXT_WIDTH - self.PAD), height / 2

    def _x_for(self, value: float) -> float:
        left, right, _middle = self._track()
        return left + self.model.fraction(value) * (right - left)

    def _value_for(self, x: float) -> float:
        left, right, _middle = self._track()
        return self.model.value_at((x - left) / max(1.0, right - left))

    # -- drawing ----------------------------------------------------------
    def redraw(self) -> None:
        self.delete("all")
        left, right, middle = self._track()
        color = ACCENT if self.enabled else DISABLED
        self.create_line(left, middle, right, middle, fill=TRACK, width=4, capstyle="round")
        values = [self.model.handle_value(handle) for handle in self.model.handles]
        start = self._x_for(self.model.low if self.model.single else values[0])
        self.create_line(start, middle, self._x_for(values[-1]), middle,
                         fill=color, width=4, capstyle="round")
        focused = self.enabled and self.focus_get() is self
        for index, value in enumerate(values):
            x = self._x_for(value)
            ring = 3 if focused and index == self.active else 1
            self.create_oval(x - 7, middle - 7, x + 7, middle + 7, fill=SURFACE,
                             outline=color, width=ring)
        self.create_text(right + self.PAD, middle, anchor="w", text=self.model.text(),
                         fill=TEXT if self.enabled else SUBTEXT, font=("Segoe UI", 9))

    # -- public API -------------------------------------------------------
    def set_value(self, value) -> None:
        """Programmatic update; does not call ``command``."""
        self.model.set_value(value)
        self.redraw()

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self.configure(takefocus=1 if self.enabled else 0)
        self.redraw()

    # Lets App._set_job_active lock and restore sliders like ttk widgets.
    def cget(self, key):  # type: ignore[override]
        if key == "state":
            return "normal" if self.enabled else "disabled"
        return super().cget(key)

    def config(self, cnf=None, **kwargs):  # type: ignore[override]
        if "state" in kwargs:
            self.set_enabled(kwargs.pop("state") != "disabled")
        if cnf or kwargs:
            return super().config(cnf, **kwargs)
        return None

    configure = config  # type: ignore[assignment]

    # -- interaction ------------------------------------------------------
    def _changed(self) -> None:
        self.redraw()
        if self.command is not None:
            self.command(self.model.value)

    def _handle_name(self) -> str:
        return self.model.handles[self.active]

    def _on_press(self, event):
        if not self.enabled:
            return "break"
        self.focus_set()
        value = self._value_for(event.x)
        handle = self.model.nearest_handle(value)
        self.active = self.model.handles.index(handle)
        self._dragging = handle
        if self.model.set_handle(handle, value):
            self._changed()
        else:
            self.redraw()
        return "break"

    def _on_drag(self, event):
        if self.enabled and self._dragging and self.model.set_handle(
                self._dragging, self._value_for(event.x)):
            self._changed()
        return "break"

    def _key_nudge(self, steps: int):
        if self.enabled and self.model.nudge(self._handle_name(), steps):
            self._changed()
        return "break"

    def _key_jump(self, value: float):
        if self.enabled and self.model.set_handle(self._handle_name(), value):
            self._changed()
        return "break"

    def _on_tab(self, _event):
        if self.enabled and self.active < len(self.model.handles) - 1:
            self.active += 1
            self.redraw()
            return "break"
        self.active = 0
        return None

    def _on_shift_tab(self, _event):
        if self.enabled and self.active > 0:
            self.active -= 1
            self.redraw()
            return "break"
        return None

    def _open_editor(self, _event=None):
        if not self.enabled or self._editor is not None:
            return "break"
        left, right, middle = self._track()
        entry = tk.Entry(self, width=14, font=("Segoe UI", 9))
        entry.insert(0, self.model.text().replace(self.model.unit, ""))
        entry.select_range(0, "end")
        self._editor = self.create_window(right + self.PAD, middle, anchor="w", window=entry)
        entry.focus_set()

        def close(commit: bool):
            text = entry.get()
            self.delete(self._editor)
            self._editor = None
            entry.destroy()
            self.focus_set()
            if commit:
                try:
                    changed = self.model.set_value(self.model.parse(text))
                except ValueError:
                    self.bell()
                    changed = False
                if changed:
                    self._changed()
            self.redraw()

        entry.bind("<Return>", lambda _e: (close(True), "break")[1])
        entry.bind("<KP_Enter>", lambda _e: (close(True), "break")[1])
        entry.bind("<Escape>", lambda _e: (close(False), "break")[1])
        entry.bind("<FocusOut>", lambda _e: close(True) if self._editor else None)
        return "break"


class Tooltip:
    """Hover (or keyboard-focus) help text for a widget."""

    DELAY_MS = 500

    def __init__(self, widget, text: str):
        self.widget = widget
        self.text = text
        self._after = None
        self._window = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self.hide, add="+")
        widget.bind("<ButtonPress>", self.hide, add="+")
        widget.bind("<FocusIn>", self._schedule, add="+")
        widget.bind("<FocusOut>", self.hide, add="+")
        widget.bind("<Destroy>", self.hide, add="+")

    def _schedule(self, _event=None) -> None:
        self._cancel()
        self._after = self.widget.after(self.DELAY_MS, self.show)

    def _cancel(self) -> None:
        if self._after is not None:
            try:
                self.widget.after_cancel(self._after)
            except tk.TclError:
                pass
            self._after = None

    def show(self) -> None:
        self._after = None
        if self._window is not None or not self.text:
            return
        try:
            x = self.widget.winfo_rootx() + 12
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        except tk.TclError:
            return
        self._window = window = tk.Toplevel(self.widget)
        window.wm_overrideredirect(True)
        window.wm_geometry(f"+{x}+{y}")
        tk.Label(window, text=self.text, justify="left", wraplength=320,
                 background="#1f2430", foreground="#ffffff", padx=8, pady=4,
                 font=("Segoe UI", 9)).pack()

    def hide(self, _event=None) -> None:
        self._cancel()
        if self._window is not None:
            try:
                self._window.destroy()
            except tk.TclError:
                pass
            self._window = None


def model_for_spec(spec, value) -> RangeModel:
    """A model matching a :class:`src.studio_spec.ParamSpec`."""
    scale = 100.0 if spec.unit == "%" else 1.0
    return RangeModel(spec.low, spec.high, spec.step, value,
                      single=spec.kind != "range", integer=spec.integer,
                      scale=scale, unit=spec.unit)


def chance_slider(master, value: float, *, command=None, **kwargs) -> RangeSlider:
    """A one-handle 0-100% slider for an effect's chance."""
    model = RangeModel(0.0, 1.0, 0.01, value, single=True, scale=100.0, unit="%")
    return RangeSlider(master, model, command=command, **kwargs)
