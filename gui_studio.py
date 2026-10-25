"""Developer Studio tab: every effect as a chance plus a min/max range.

The panel is built from :mod:`src.studio_spec` and edits a pure
:class:`src.studio_recipe.StudioState`.  Any edit fires ``<<StudioChanged>>``;
the owner reads :meth:`StudioPanel.get_state`.  Tk code stays thin: the rules
live in the state, the spec, and :mod:`gui_widgets`.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import config
from gui_widgets import RangeSlider, model_for_spec
from src import studio_spec
from src.generation_profiles import (AUGMENTATION_PROFILES, CUTOFF_PROFILES,
                                     cutoff_spec_from_profile)
from src.studio_recipe import StudioState

STUDIO_CHANGED = "<<StudioChanged>>"
SECTION_TITLES = {
    "augmentation": "augmentation",
    "stroke_damage": "stroke damage",
    "cutoff": "cutoff",
}
STROKE_HINT = (
    "Stroke damage applies only in the semi-broken sample modes. "
    "Choose one on the Generate tab to edit these values."
)
CUTOFF_NOTE = (
    "Cuts are measured from the ink, so letters are really clipped. Labels of "
    "3 or fewer characters get no side cuts and capped vertical cuts; side "
    "cuts on longer labels are capped at about a third of one character."
)


def effect_title(effect: str) -> str:
    return effect.replace("_", " ").capitalize()


def specs_by_group() -> dict[str, list[studio_spec.ParamSpec]]:
    groups: dict[str, list[studio_spec.ParamSpec]] = {group: [] for group in studio_spec.GROUPS}
    for spec in studio_spec.PARAMS:
        groups[spec.group].append(spec)
    return groups


class StudioPanel(ttk.Frame):
    """All studio controls for one :class:`StudioState`."""

    def __init__(self, master, *, state: StudioState | None = None,
                 card_style: str = "Card.TFrame", label_style: str = "CardLabel.TLabel",
                 hint_style: str = "CardHint.TLabel", section_style: str = "Section.TLabel",
                 background: str = "#ffffff", preset_dir=None,
                 on_recipe_loaded=None, run_section=None, cli_options=None):
        super().__init__(master)
        self._state = state or StudioState.default()
        self._preset_dir = preset_dir
        self._on_recipe_loaded = on_recipe_loaded
        self._run_section = run_section
        self._cli_options = cli_options
        self._styles = {"card": card_style, "label": label_style, "hint": hint_style,
                        "section": section_style}
        self._background = background
        self.sliders: dict[str, RangeSlider] = {}
        self._controls: list[tk.Widget] = []
        self._stroke_enabled = True
        self._build()
        self.set_state(self._state)

    # -- layout -------------------------------------------------------------
    def _card(self, title: str, reset_command=None) -> ttk.Frame:
        header = ttk.Frame(self)
        header.pack(fill="x", padx=16, pady=(12, 2))
        ttk.Label(header, text=title.upper(), style=self._styles["section"]).pack(side="left")
        if reset_command is not None:
            button = ttk.Button(header, text="Reset", style="Quick.TButton",
                                command=reset_command)
            button.pack(side="right")
            self._controls.append(button)
        outer = ttk.Frame(self)
        outer.pack(fill="x", padx=16)
        card = ttk.Frame(outer, style=self._styles["card"], padding=12)
        card.pack(fill="x")
        return card

    def _build(self) -> None:
        source = self._card("Studio recipe")
        row = ttk.Frame(source, style=self._styles["card"])
        row.pack(fill="x", pady=(0, 6))
        ttk.Label(row, text="Current", style=self._styles["label"], width=14).pack(side="left")
        self.state_label = ttk.Label(row, text="", style=self._styles["label"])
        self.state_label.pack(side="left")

        row = ttk.Frame(source, style=self._styles["card"])
        row.pack(fill="x")
        ttk.Label(row, text="Base profile", style=self._styles["label"], width=14).pack(side="left")
        self.base_profile_var = tk.StringVar(value="")
        self.base_profile_cb = ttk.Combobox(
            row, textvariable=self.base_profile_var, state="readonly", width=28,
            values=[config.DEGRADATION_PROFILES.get(key, key) for key in AUGMENTATION_PROFILES])
        self.base_profile_cb.pack(side="left", padx=(0, 8))
        self.base_profile_cb.bind("<<ComboboxSelected>>", self._on_base_profile)
        self._controls.append(self.base_profile_cb)
        reset_all = ttk.Button(row, text="Reset studio", style="Secondary.TButton",
                               command=self.reset_studio)
        reset_all.pack(side="right")
        self._controls.append(reset_all)
        ttk.Label(source, text="Loads that profile's exact values; any edit marks "
                               "the recipe as custom.",
                  style=self._styles["hint"]).pack(anchor="w", pady=(4, 0))

        for group, specs in specs_by_group().items():
            card = self._card(group, lambda specs=specs: self.reset_group(specs))
            if group == studio_spec.STROKE_DAMAGE:
                self.stroke_hint = ttk.Label(card, text=STROKE_HINT, wraplength=420,
                                             style=self._styles["hint"])
            if group == studio_spec.CHARACTER_CUTOFF:
                self._build_cutoff_picks(card)
            effect = None
            for spec in specs:
                if spec.effect != effect and spec.kind != "value":
                    effect = spec.effect
                    ttk.Label(card, text=effect_title(effect),
                              style=self._styles["label"]).pack(anchor="w", pady=(6, 0))
                self._build_row(card, spec)
            if group == studio_spec.CHARACTER_CUTOFF:
                ttk.Label(card, text=CUTOFF_NOTE, wraplength=420,
                          style=self._styles["hint"]).pack(anchor="w", pady=(6, 0))

    def _build_cutoff_picks(self, card) -> None:
        row = ttk.Frame(card, style=self._styles["card"])
        row.pack(fill="x", pady=(0, 4))
        ttk.Label(row, text="Quick pick", style=self._styles["label"], width=14).pack(side="left")
        for key in CUTOFF_PROFILES:
            button = ttk.Button(row, text=key.capitalize(), style="Quick.TButton",
                                command=lambda key=key: self.pick_cutoff(key))
            button.pack(side="left", padx=2)
            self._controls.append(button)

    def _build_row(self, card, spec: studio_spec.ParamSpec) -> None:
        row = ttk.Frame(card, style=self._styles["card"])
        row.pack(fill="x", pady=1)
        label = spec.label
        if spec.kind == "chance":
            label = "Chance"
        ttk.Label(row, text=f"  {label}", style=self._styles["hint"], width=30).pack(side="left")
        slider = RangeSlider(row, model_for_spec(spec, self._state.value(spec)),
                             command=lambda value, spec=spec: self._on_slider(spec, value),
                             background=self._background)
        slider.pack(side="left", fill="x", expand=True)
        self.sliders[spec.path] = slider
        self._controls.append(slider)

    # -- state --------------------------------------------------------------
    def get_state(self) -> StudioState:
        return self._state

    def set_state(self, state: StudioState) -> None:
        """Show ``state`` without firing ``<<StudioChanged>>``."""
        self._state = state
        for path, slider in self.sliders.items():
            slider.set_value(state.value(studio_spec.PARAMS_BY_PATH[path]))
        self._refresh_labels()

    def _refresh_labels(self) -> None:
        self.state_label.config(text=self._state.label)
        name = self._state.source_name
        self.base_profile_var.set(
            config.DEGRADATION_PROFILES.get(name, name) if name in AUGMENTATION_PROFILES else "")

    def _changed(self, state: StudioState) -> None:
        self.set_state(state)
        self.event_generate(STUDIO_CHANGED, when="tail")

    def _on_slider(self, spec: studio_spec.ParamSpec, value) -> None:
        plain = list(value) if isinstance(value, tuple) else value
        self._state = self._state.with_value(spec, plain)
        self._refresh_labels()
        self.event_generate(STUDIO_CHANGED, when="tail")

    def _on_base_profile(self, _event=None) -> None:
        label = self.base_profile_var.get()
        for key in AUGMENTATION_PROFILES:
            if config.DEGRADATION_PROFILES.get(key, key) == label:
                self._changed(self._state.with_base_profile(key))
                return

    def pick_cutoff(self, key: str) -> None:
        self._changed(self._state.with_section("cutoff", cutoff_spec_from_profile(key)))

    def reset_group(self, specs) -> None:
        state = self._state
        for section in dict.fromkeys(spec.section for spec in specs):
            state = state.reset_section(
                section, [spec for spec in specs if spec.section == section])
        self._changed(state)

    def reset_studio(self) -> None:
        self._changed(StudioState.default())

    # -- enable / lock -------------------------------------------------------
    def set_stroke_enabled(self, enabled: bool) -> None:
        """Enable the stroke-damage sliders only in semi-broken modes."""
        self._stroke_enabled = bool(enabled)
        for path, slider in self.sliders.items():
            if path.startswith("stroke_damage."):
                slider.set_enabled(self._stroke_enabled)
        if self._stroke_enabled:
            self.stroke_hint.pack_forget()
        else:
            self.stroke_hint.pack(anchor="w", pady=(0, 4), before=self._first_stroke_widget())

    def _first_stroke_widget(self):
        children = [child for child in self.stroke_hint.master.winfo_children()
                    if child is not self.stroke_hint]
        return children[0] if children else None

    def controls(self) -> list[tk.Widget]:
        """Every interactive widget, for locking while a run is active."""
        return list(self._controls)
