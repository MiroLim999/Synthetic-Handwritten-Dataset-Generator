"""Developer Studio tab: every effect as a chance plus a min/max range.

The panel is built from :mod:`src.studio_spec` and edits a pure
:class:`src.studio_recipe.StudioState`.  Any edit fires ``<<StudioChanged>>``;
the owner reads :meth:`StudioPanel.get_state`.  Tk code stays thin: the rules
live in the state, the spec, and :mod:`gui_widgets`.
"""

from __future__ import annotations

import os
from dataclasses import replace
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

import config
from gui_widgets import RangeSlider, model_for_spec
from src import studio_spec
from src.generation_profiles import (AUGMENTATION_PROFILES, CUTOFF_PROFILES,
                                     cutoff_spec_from_profile)
from src.studio_recipe import (EFFECT_SECTIONS, PresetEntry, PresetExistsError,
                               ReadOnlyPresetError, RecipeError, StudioState,
                               delete_user_preset, duplicate_preset,
                               export_preset, import_preset, list_presets,
                               load_recipe, preset_slug, rename_user_preset,
                               save_recipe, save_user_preset)

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


SECTION_CHIPS = {"augmentation": "aug", "stroke_damage": "stroke",
                 "cutoff": "cutoff", "run": "run"}


def section_chips(sections) -> str:
    """Short labels for the sections a preset contains."""
    return "  ".join(f"[{SECTION_CHIPS[name]}]" for name in sections)


def cli_command(recipe_path: Path, sections, options: dict, root: Path) -> str:
    """The command line that runs a saved preset with the current count/seed."""
    try:
        shown = Path(os.path.relpath(recipe_path, root)).as_posix()
    except ValueError:  # another drive on Windows
        shown = Path(recipe_path).as_posix()
    parts = ["python -m src.generate_synthetic", f'--recipe "{shown}"']
    if options.get("count"):
        parts.append(f"--count {options['count']}")
    if options.get("seed"):
        parts.append(f"--seed {options['seed']}")
    if "run" not in sections and options.get("mode"):
        parts.append(f"--mode {options['mode']}")
    return " ".join(parts)


def ask_save_as(parent, *, name: str, description: str, sections) -> dict | None:
    """Modal 'Save as' dialog: name, description, and which sections to save."""
    dialog = tk.Toplevel(parent)
    dialog.title("Save preset as")
    dialog.transient(parent.winfo_toplevel())
    dialog.resizable(False, False)
    result: dict = {}
    frame = ttk.Frame(dialog, padding=12)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="Name").grid(row=0, column=0, sticky="w")
    name_var = tk.StringVar(value=name)
    name_entry = ttk.Entry(frame, textvariable=name_var, width=36)
    name_entry.grid(row=0, column=1, sticky="we", pady=2)
    ttk.Label(frame, text="Description").grid(row=1, column=0, sticky="w")
    description_var = tk.StringVar(value=description)
    ttk.Entry(frame, textvariable=description_var, width=36).grid(
        row=1, column=1, sticky="we", pady=2)
    ttk.Label(frame, text="Sections").grid(row=2, column=0, sticky="nw", pady=(6, 0))
    checks = {}
    box = ttk.Frame(frame)
    box.grid(row=2, column=1, sticky="w", pady=(6, 0))
    labels = {"augmentation": "Augmentation", "stroke_damage": "Stroke damage",
              "cutoff": "Character cutoff", "run": "Run settings (Generate tab)"}
    for section, label in labels.items():
        var = tk.BooleanVar(value=section in sections)
        ttk.Checkbutton(box, text=label, variable=var).pack(anchor="w")
        checks[section] = var

    def accept(_event=None):
        chosen = tuple(section for section, var in checks.items() if var.get())
        if not name_var.get().strip() or not chosen:
            dialog.bell()
            return
        result.update(name=name_var.get().strip(),
                      description=description_var.get().strip(), sections=chosen)
        dialog.destroy()

    buttons = ttk.Frame(frame)
    buttons.grid(row=3, column=0, columnspan=2, sticky="e", pady=(10, 0))
    ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side="right")
    ttk.Button(buttons, text="Save", command=accept).pack(side="right", padx=(0, 6))
    dialog.bind("<Return>", accept)
    dialog.bind("<Escape>", lambda _event: dialog.destroy())
    name_entry.focus_set()
    dialog.grab_set()
    parent.wait_window(dialog)
    return result or None


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

    def _build_preset_bar(self) -> None:
        card = self._card("Presets")
        columns = ("sections",)
        self.preset_tree = ttk.Treeview(card, columns=columns, show="tree headings",
                                        height=6, selectmode="browse")
        self.preset_tree.heading("#0", text="Preset (🔒 = built-in, read-only)")
        self.preset_tree.heading("sections", text="Sections")
        self.preset_tree.column("#0", width=260, anchor="w")
        self.preset_tree.column("sections", width=220, anchor="w")
        self.preset_tree.pack(fill="x")
        self.preset_tree.bind("<Double-1>", lambda _event: self.load_selected())
        self.preset_tree.bind("<<TreeviewSelect>>", lambda _event: self._update_buttons())
        self._controls.append(self.preset_tree)
        rows = [ttk.Frame(card, style=self._styles["card"]) for _ in range(2)]
        for row in rows:
            row.pack(fill="x", pady=(6, 0))
        self.preset_buttons = {}
        for row, specs in zip(rows, (
                (("load", "Load", self.load_selected), ("save", "Save", self.save),
                 ("save_as", "Save as...", self.save_as),
                 ("duplicate", "Duplicate", self.duplicate),
                 ("rename", "Rename", self.rename), ("delete", "Delete", self.delete)),
                (("import", "Import...", self.import_preset),
                 ("export", "Export...", self.export),
                 ("copy_cli", "Copy CLI command", self.copy_cli)))):
            for key, text, command in specs:
                button = ttk.Button(row, text=text, style="Quick.TButton", command=command)
                button.pack(side="left", padx=(0, 4))
                self.preset_buttons[key] = button
                self._controls.append(button)
        self.preset_message = ttk.Label(card, text="", style=self._styles["hint"])
        self.preset_message.pack(anchor="w", pady=(4, 0))
        self.refresh_presets()

    def _build(self) -> None:
        if self._preset_dir is not None:
            self._build_preset_bar()
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
        state = self._state
        text = f"{state.source_name}*  (unsaved changes)" if state.dirty else state.source_name
        self.state_label.config(text=text)
        name = self._state.source_name
        self.base_profile_var.set(
            config.DEGRADATION_PROFILES.get(name, name) if name in AUGMENTATION_PROFILES else "")

    def _changed(self, state: StudioState) -> None:
        self.set_state(state)
        self.event_generate(STUDIO_CHANGED, when="tail")
        if self._preset_dir is not None:
            self._update_buttons()

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

    # -- presets --------------------------------------------------------------
    def _dirs(self) -> dict:
        return {"builtin_dir": self._preset_dir, "user_dir": Path(self._preset_dir) / "user"}

    def refresh_presets(self, select: str | None = None) -> None:
        tree = self.preset_tree
        tree.delete(*tree.get_children())
        self._entries: dict[str, PresetEntry] = {}
        for entry in list_presets(**self._dirs()):
            self._entries[entry.key] = entry
            text = f"{'🔒 ' if entry.builtin else ''}{entry.name}"
            chips = "(invalid file)" if entry.error else section_chips(entry.sections)
            tree.insert("", "end", iid=entry.key, text=text, values=(chips,))
        if select in self._entries:
            tree.selection_set(select)
            tree.see(select)
        self._update_buttons()

    def selected_entry(self) -> PresetEntry | None:
        selection = self.preset_tree.selection()
        return self._entries.get(selection[0]) if selection else None

    def _loaded_entry(self) -> PresetEntry | None:
        return self._entries.get(self._state.source_key or "")

    def _update_buttons(self) -> None:
        entry = self.selected_entry()
        loaded = self._loaded_entry()
        user_selected = entry is not None and not entry.builtin
        states = {
            "load": entry is not None and entry.error is None,
            "save": loaded is not None and not loaded.builtin,
            "duplicate": entry is not None and entry.error is None,
            "rename": user_selected, "delete": user_selected,
            "export": entry is not None and entry.error is None,
        }
        for key, enabled in states.items():
            self.preset_buttons[key].config(state="normal" if enabled else "disabled")

    def _message(self, text: str) -> None:
        self.preset_message.config(text=text)

    def _mark_saved(self, recipe, key: str) -> None:
        """Make the saved recipe the new clean baseline."""
        self._changed(self._state.loaded(recipe, source_name=recipe.name or key,
                                         source_key=key))
        self.refresh_presets(select=key)

    def load_selected(self) -> None:
        entry = self.selected_entry()
        if entry is None:
            return
        if entry.error:
            messagebox.showerror("Invalid preset", entry.error, parent=self)
            return
        if self._state.dirty and not messagebox.askyesno(
                "Unsaved changes",
                f"Discard your changes to '{self._state.source_name}' and load "
                f"'{entry.name}'?", icon="warning", parent=self):
            return
        try:
            recipe = load_recipe(entry.path)
        except RecipeError as exc:
            messagebox.showerror("Invalid preset", str(exc), parent=self)
            return
        if self._on_recipe_loaded is not None:
            self._on_recipe_loaded(entry.key, recipe)
        else:
            self._changed(self._state.loaded(recipe, source_name=recipe.name or entry.key,
                                             source_key=entry.key))
        self._update_buttons()
        self._message(f"Loaded {entry.name}.")

    def _recipe_for(self, name: str, description: str, sections):
        effect = tuple(section for section in EFFECT_SECTIONS if section in sections)
        run = self._run_section() if "run" in sections and self._run_section else None
        return self._state.to_recipe(name, description, sections=effect, run=run)

    def save(self) -> None:
        entry = self._loaded_entry()
        if entry is None or entry.builtin:
            messagebox.showinfo("Read-only preset",
                                "Built-in presets are read-only. Use 'Save as...' to "
                                "save your changes as a user preset.", parent=self)
            return
        try:
            current = load_recipe(entry.path)
            recipe = self._recipe_for(current.name, current.description, current.sections)
            save_recipe(entry.path, recipe)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Save failed", str(exc), parent=self)
            return
        self._mark_saved(recipe, entry.key)
        self._message(f"Saved {entry.name}.")

    def save_as(self) -> None:
        loaded = self._loaded_entry()
        default_sections = loaded.sections if loaded is not None else EFFECT_SECTIONS
        answer = ask_save_as(self, name=self._state.source_name, description="",
                             sections=default_sections or EFFECT_SECTIONS)
        if answer is None:
            return
        try:
            recipe = self._recipe_for(answer["name"], answer["description"],
                                      answer["sections"])
            try:
                save_user_preset(recipe, answer["name"], **self._dirs())
            except PresetExistsError:
                if not messagebox.askyesno(
                        "Overwrite preset",
                        f"A user preset named '{preset_slug(answer['name'])}' already "
                        "exists. Replace it?", icon="warning", parent=self):
                    return
                save_user_preset(recipe, answer["name"], overwrite=True, **self._dirs())
        except ReadOnlyPresetError as exc:
            messagebox.showerror("Name in use", str(exc), parent=self)
            return
        except (OSError, ValueError) as exc:
            messagebox.showerror("Save failed", str(exc), parent=self)
            return
        self._mark_saved(recipe, f"user/{preset_slug(answer['name'])}")
        self._message(f"Saved {answer['name']} as a user preset.")

    def _ask_name(self, title: str, initial: str) -> str | None:
        return simpledialog.askstring(title, "Preset name:", initialvalue=initial,
                                      parent=self)

    def _user_preset_action(self, title: str, action) -> None:
        try:
            action()
        except PresetExistsError as exc:
            messagebox.showerror(title, f"{exc} Choose another name.", parent=self)
        except (ReadOnlyPresetError, RecipeError, OSError, ValueError) as exc:
            messagebox.showerror(title, str(exc), parent=self)

    def duplicate(self) -> None:
        entry = self.selected_entry()
        name = entry and self._ask_name("Duplicate preset", f"{entry.name} copy")
        if entry and name:
            def action():
                duplicate_preset(entry, name, **self._dirs())
                self.refresh_presets(select=f"user/{preset_slug(name)}")
                self._message(f"Duplicated {entry.name} as {name}.")
            self._user_preset_action("Duplicate failed", action)

    def rename(self) -> None:
        entry = self.selected_entry()
        if entry is None or entry.builtin:
            return
        name = self._ask_name("Rename preset", entry.name)
        if name:
            def action():
                rename_user_preset(entry, name, **self._dirs())
                new_key = f"user/{preset_slug(name)}"
                if self._state.source_key == entry.key:
                    self._changed(replace(self._state, source_key=new_key,
                                          source_name=name))
                self.refresh_presets(select=new_key)
                self._message(f"Renamed to {name}.")
            self._user_preset_action("Rename failed", action)

    def delete(self) -> None:
        entry = self.selected_entry()
        if entry is None or entry.builtin:
            return
        if not messagebox.askyesno("Delete preset",
                                   f"Delete the user preset '{entry.name}'? "
                                   "This cannot be undone.", icon="warning", parent=self):
            return

        def action():
            delete_user_preset(entry)
            self.refresh_presets()
            self._message(f"Deleted {entry.name}.")
        self._user_preset_action("Delete failed", action)

    def import_preset(self) -> None:
        path = filedialog.askopenfilename(
            parent=self, title="Import preset",
            filetypes=[("Studio recipe", "*.json"), ("All files", "*.*")])
        if not path:
            return

        def action():
            saved = import_preset(path, **self._dirs())
            self.refresh_presets(select=f"user/{saved.stem}")
            self._message(f"Imported {Path(path).name} as user/{saved.stem}.")
        self._user_preset_action("Import failed", action)

    def export(self) -> None:
        entry = self.selected_entry()
        if entry is None:
            return
        path = filedialog.asksaveasfilename(
            parent=self, title="Export preset", defaultextension=".json",
            initialfile=f"{entry.slug}.json", filetypes=[("Studio recipe", "*.json")])
        if path:
            def action():
                export_preset(entry, path)
                self._message(f"Exported {entry.name} to {path}.")
            self._user_preset_action("Export failed", action)

    def copy_cli(self) -> None:
        entry = self._loaded_entry()
        if self._state.dirty or entry is None:
            messagebox.showinfo(
                "Save first",
                "The CLI command runs a saved preset. Save your changes first "
                "('Save' or 'Save as...'), then copy the command.", parent=self)
            return
        options = self._cli_options() if self._cli_options else {}
        command = cli_command(entry.path, entry.sections, options, config.ROOT)
        self.clipboard_clear()
        self.clipboard_append(command)
        self._message(f"Copied: {command}")

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
