"""Studio Recipe v2: one preset format with optional sections.

A recipe has a ``name``, a ``description``, and any of four sections:
``augmentation``, ``stroke_damage``, ``cutoff`` (validated against
:mod:`src.studio_spec`) and ``run`` (sample mode, field mix, splits, fonts,
names, writer profile).  Loading a recipe applies only the sections it
contains.  Old flat presets (no ``schema_version``) are migrated in memory.

Precedence, lowest to highest: built-in defaults < recipe < explicit CLI
flags or later GUI edits.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any
from numbers import Real
from pathlib import Path

import config
from src import studio_spec
from src.generation_profiles import (DEFAULT_AUGMENTATION_PROFILE_ID,
                                     DEFAULT_WRITER_PROFILE_ID,
                                     WRITER_STYLE_PROFILES,
                                     augmentation_profile_from_spec,
                                     augmentation_spec_from_legacy_values,
                                     augmentation_spec_from_profile,
                                     canonical_json, cutoff_profile_from_spec,
                                     cutoff_spec_from_profile,
                                     stroke_damage_from_config,
                                     stroke_damage_from_spec,
                                     stroke_damage_spec_from_profile)
from src.provenance import atomic_write_json
from src.render import CURSIVE_STYLE_GROUPS, FONT_STYLE_ALL, FONT_STYLES

RECIPE_FORMAT = "synthetic-generator.studio-recipe"
RECIPE_SCHEMA_VERSION = 2
RECIPE_FILENAME = "studio-recipe.json"
EFFECT_SECTIONS = studio_spec.SECTIONS
SECTIONS = (*EFFECT_SECTIONS, "run")
RUN_KEYS = (
    "sample_mode", "field_weights", "split_fractions", "font_style",
    "cursive_group", "specific_font", "names_version", "writer_profile",
    "samples_per_writer",
)
_TOP_LEVEL_KEYS = frozenset({"format", "schema_version", "name", "description", *SECTIONS})


class RecipeError(ValueError):
    """An invalid recipe; ``errors`` lists every problem with its JSON path."""

    def __init__(self, errors: list[str], origin: str = "recipe"):
        self.errors = list(errors)
        super().__init__(f"Invalid {origin}: " + "; ".join(self.errors))


@dataclass(frozen=True)
class StudioRecipe:
    """A validated recipe.  Absent sections are ``None``."""

    name: str = ""
    description: str = ""
    augmentation: dict | None = None
    stroke_damage: dict | None = None
    cutoff: dict | None = None
    run: dict | None = None
    # How the recipe was obtained (not part of its content or hash).
    migrated_from_v1: bool = field(default=False, compare=False)

    @property
    def sections(self) -> tuple[str, ...]:
        return tuple(name for name in SECTIONS if getattr(self, name) is not None)

    def section(self, name: str) -> dict | None:
        value = getattr(self, name)
        return None if value is None else copy.deepcopy(value)

    def to_dict(self) -> dict:
        data: dict[str, object] = {
            "format": RECIPE_FORMAT,
            "schema_version": RECIPE_SCHEMA_VERSION,
            "name": self.name,
            "description": self.description,
        }
        for name in self.sections:
            data[name] = copy.deepcopy(getattr(self, name))
        return data

    def canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def sha256(self) -> str:
        return recipe_sha256(self)


def recipe_sha256(recipe: StudioRecipe) -> str:
    """SHA-256 of the recipe's canonical JSON (sorted keys, compact)."""
    return hashlib.sha256(recipe.canonical_json().encode("utf-8")).hexdigest()


def _is_number(value) -> bool:
    return (not isinstance(value, bool) and isinstance(value, Real)
            and math.isfinite(float(value)))


def validate_run(run, prefix: str = "run") -> tuple[dict, list[str]]:
    """Validate a ``run`` section; keys may be partial."""
    errors: list[str] = []
    if not isinstance(run, Mapping):
        return {}, [f"{prefix} must be a JSON object"]
    normalized: dict[str, object] = {}
    for key in sorted(set(run) - set(RUN_KEYS) - {"based_on"}):
        errors.append(f"{prefix}.{key}: unknown key")

    def check(key, ok, message, value=None):
        if key not in run:
            return
        if ok(run[key]):
            normalized[key] = run[key] if value is None else value(run[key])
        else:
            errors.append(f"{prefix}.{key}: {message}")

    check("sample_mode", lambda v: v in config.SAMPLE_MODES,
          f"must be one of {', '.join(config.SAMPLE_MODES)}")
    check("font_style", lambda v: v in FONT_STYLES,
          f"must be one of {', '.join(FONT_STYLES)}")
    check("cursive_group", lambda v: v == "" or v in CURSIVE_STYLE_GROUPS,
          "must be empty or a known cursive group")
    check("specific_font", lambda v: isinstance(v, str), "must be a string")
    check("names_version", lambda v: v in config.name_versions(),
          f"must be one of {', '.join(config.name_versions())}")
    check("writer_profile", lambda v: v in WRITER_STYLE_PROFILES,
          f"must be one of {', '.join(WRITER_STYLE_PROFILES)}")
    check("samples_per_writer",
          lambda v: not isinstance(v, bool) and isinstance(v, int) and v > 0,
          "must be a positive whole number")
    check("based_on", lambda v: isinstance(v, str), "must be a string")

    if "field_weights" in run:
        weights = run["field_weights"]
        if weights is None:
            normalized["field_weights"] = None
        elif not isinstance(weights, Mapping):
            errors.append(f"{prefix}.field_weights: must be a JSON object")
        else:
            clean = {}
            for key in sorted(weights):
                value = weights[key]
                if key not in config.SUPPORTED_FIELD_TYPES:
                    errors.append(f"{prefix}.field_weights.{key}: unknown field type")
                elif not _is_number(value) or value < 0:
                    errors.append(f"{prefix}.field_weights.{key}: "
                                  "must be a nonnegative number")
                else:
                    clean[key] = int(value) if float(value).is_integer() else float(value)
            if clean and not any(value > 0 for value in clean.values()):
                errors.append(f"{prefix}.field_weights: at least one weight must be positive")
            elif not clean and not errors:
                errors.append(f"{prefix}.field_weights: must not be empty")
            normalized["field_weights"] = clean

    if "split_fractions" in run:
        fractions = run["split_fractions"]
        if (not isinstance(fractions, (list, tuple)) or len(fractions) != 3
                or not all(_is_number(v) for v in fractions)):
            errors.append(f"{prefix}.split_fractions: must be [train, val, test] numbers")
        else:
            try:
                normalized["split_fractions"] = list(
                    config._validate_fraction_set("split_fractions", fractions))
            except ValueError as exc:
                errors.append(f"{prefix}.split_fractions: {exc}")
    return normalized, errors


def parse_recipe(data, *, origin: str = "recipe") -> StudioRecipe:
    """Validate a recipe object strictly (v2), or migrate an old flat preset."""
    if not isinstance(data, Mapping):
        raise RecipeError(["recipe must be a JSON object"], origin)
    if "schema_version" not in data and "format" not in data:
        return migrate_v1(data, origin=origin)
    errors: list[str] = []
    version = data.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise RecipeError(["schema_version: must be a whole number"], origin)
    if version > RECIPE_SCHEMA_VERSION:
        raise RecipeError([
            f"schema_version: {version} was written by a newer generator; this "
            f"version reads schema_version {RECIPE_SCHEMA_VERSION} or older"], origin)
    if version != RECIPE_SCHEMA_VERSION:
        raise RecipeError([f"schema_version: unsupported value {version}"], origin)
    if data.get("format") != RECIPE_FORMAT:
        errors.append(f"format: must be {RECIPE_FORMAT!r}")
    for key in sorted(set(data) - _TOP_LEVEL_KEYS):
        errors.append(f"{key}: unknown key")
    for key in ("name", "description"):
        if not isinstance(data.get(key, ""), str):
            errors.append(f"{key}: must be a string")
    sections: dict[str, dict | None] = {}
    for name in EFFECT_SECTIONS:
        if name in data:
            sections[name], problems = studio_spec.validate_section(name, data[name])
            errors.extend(problems)
    if "run" in data:
        sections["run"], problems = validate_run(data["run"])
        errors.extend(problems)
    if errors:
        raise RecipeError(errors, origin)
    return StudioRecipe(
        name=str(data.get("name", "")),
        description=str(data.get("description", "")),
        augmentation=sections.get("augmentation"),
        stroke_damage=sections.get("stroke_damage"),
        cutoff=sections.get("cutoff"),
        run=sections.get("run"),
    )


def _legacy_number(data: Mapping, key: str, default: float, errors: list[str]) -> float:
    value = data.get(key, default)
    if not _is_number(value):
        errors.append(f"{key}: must be a finite number")
        return float(default)
    return float(value)


_V1_KEYS = frozenset({
    "name", "description", "fade", "tint", "stain", "rotate", "blur", "noise",
    "texture", "scanline", "jpeg", "contrast", "gap_prob", "gap_count", "scratch",
    "erode", "crop_top", "crop_bottom", "crop_left", "crop_right",
})


def migrate_v1(data: Mapping, *, origin: str = "recipe") -> StudioRecipe:
    """Convert an old flat studio preset (single-value sliders) in memory.

    Augmentation uses the clamps of ``create_custom_augmentation_profile``;
    stroke damage keeps the config ranges; ``crop_*`` percentages become
    ink-relative cutoff fractions.
    """
    errors: list[str] = [
        f"{key}: unknown key" for key in sorted(set(data) - _V1_KEYS)
    ]
    num = lambda key, default: _legacy_number(data, key, default, errors)  # noqa: E731
    augmentation = augmentation_spec_from_legacy_values(
        fade_contrast=num("fade", 0.65),
        paper_tint_alpha=num("tint", 0.15),
        stain_prob=num("stain", 0.35),
        rotate_deg=num("rotate", 4.5),
        blur_radius=num("blur", 0.8),
        noise_std=num("noise", 12.0),
        paper_texture_std=num("texture", 3.5),
        scanline_alpha=int(round(num("scanline", 10.0))),
        jpeg_quality=int(round(num("jpeg", 75.0))),
    )
    base = stroke_damage_from_config()

    def chance(key: str, default: float) -> float:
        return max(0.0, min(1.0, num(key, default)))

    stroke = stroke_damage_spec_from_profile(base)
    stroke["contrast"]["probability"] = chance("contrast", base.contrast_probability)
    stroke["erode"]["probability"] = chance("erode", base.erode_probability)
    stroke["scratches"]["probability"] = chance("scratch", base.scratch_probability)
    stroke["gaps"]["probability"] = chance("gap_prob", base.gap_probability)
    if "gap_count" in data:
        top = min(60, max(1, int(num("gap_count", 1))))
        stroke["gaps"]["count_range"] = [1, top]

    cutoff = studio_spec.default_section("cutoff")
    for side in studio_spec.CUTOFF_SIDES:
        percent = num(f"crop_{side}", 0.0)
        high = studio_spec.PARAMS_BY_PATH[f"cutoff.{side}.range"].high
        fraction = max(0.0, min(high, percent / 100.0))
        if fraction > 0:
            cutoff[side] = {"probability": 1.0, "range": [fraction, fraction]}
    name = data.get("name", "")
    if not isinstance(name, str):
        errors.append("name: must be a string")
    if errors:
        raise RecipeError(errors, origin)
    recipe = parse_recipe({
        "format": RECIPE_FORMAT,
        "schema_version": RECIPE_SCHEMA_VERSION,
        "name": name if isinstance(name, str) else "",
        "description": "",
        "augmentation": augmentation,
        "stroke_damage": stroke,
        "cutoff": cutoff,
    }, origin=origin)
    return replace(recipe, migrated_from_v1=True)


def load_recipe(path: str | Path) -> StudioRecipe:
    """Read and validate a recipe file (v2, or a migrated flat v1 preset)."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RecipeError([f"could not read {path.name}: {exc}"], path.name) from exc
    return parse_recipe(data, origin=path.name)


def save_recipe(path: str | Path, recipe: StudioRecipe) -> None:
    """Atomically write a recipe; saving always writes schema version 2."""
    atomic_write_json(Path(path), recipe.to_dict())


# ---------------------------------------------------------------------------
# Settings and apply rules
# ---------------------------------------------------------------------------
def default_run() -> dict:
    """Run defaults.  ``field_weights: None`` follows the sample mode."""
    return {
        "sample_mode": config.DEFAULT_SAMPLE_MODE,
        "field_weights": None,
        "split_fractions": list(config.synthetic_split_fractions()),
        "font_style": FONT_STYLE_ALL,
        "cursive_group": "",
        "specific_font": "",
        "names_version": config.NAMES_VERSION,
        "writer_profile": DEFAULT_WRITER_PROFILE_ID,
        "samples_per_writer": 32,
    }


def default_settings() -> dict:
    """Fully resolved built-in defaults for every section."""
    return {
        "augmentation": augmentation_spec_from_profile(DEFAULT_AUGMENTATION_PROFILE_ID),
        "stroke_damage": stroke_damage_spec_from_profile(stroke_damage_from_config()),
        "cutoff": cutoff_spec_from_profile("none"),
        "run": default_run(),
    }


def apply_recipe(settings: Mapping, recipe: StudioRecipe) -> dict:
    """Return new settings with only the recipe's sections applied.

    Effect sections are replaced as a whole; ``run`` keys are merged, so a
    recipe may set only the sample mode.  Informational ``based_on`` values
    are dropped.
    """
    result = copy.deepcopy(dict(settings))
    for name in EFFECT_SECTIONS:
        section = recipe.section(name)
        if section is not None:
            section.pop("based_on", None)
            result[name] = section
    if recipe.run is not None:
        run = dict(result.get("run") or default_run())
        for key, value in recipe.run.items():
            if key != "based_on":
                run[key] = copy.deepcopy(value)
        if "sample_mode" in recipe.run and "field_weights" not in recipe.run:
            run["field_weights"] = None  # follow the recipe's sample mode
        result["run"] = run
    return result


def compose_settings(recipe: StudioRecipe | None = None, *,
                     overrides: Mapping[str, Mapping] | None = None) -> dict:
    """Resolve settings in precedence order: defaults < recipe < overrides.

    ``overrides`` maps a section name to explicit values (CLI flags or GUI
    edits).  An effect-section override replaces that whole section; a
    ``run`` override merges its keys.  Overrides are validated like recipe
    sections and raise :class:`RecipeError` with JSON paths.
    """
    settings = default_settings()
    if recipe is not None:
        settings = apply_recipe(settings, recipe)
    errors: list[str] = []
    for section, values in (overrides or {}).items():
        if section in EFFECT_SECTIONS:
            normalized, problems = studio_spec.validate_section(section, values)
            normalized.pop("based_on", None)
        elif section == "run":
            normalized, problems = validate_run(values)
            normalized = {**settings["run"], **normalized}
            if "sample_mode" in values and "field_weights" not in values:
                normalized["field_weights"] = None
        else:
            problems = [f"{section}: unknown section"]
        errors.extend(problems)
        if not problems:
            settings[section] = normalized
    if errors:
        raise RecipeError(errors, "settings")
    return settings


def resolved_recipe(settings: Mapping, *, name: str = "",
                    description: str = "") -> StudioRecipe:
    """A recipe holding every section of ``settings``."""
    data = {"format": RECIPE_FORMAT, "schema_version": RECIPE_SCHEMA_VERSION,
            "name": name, "description": description}
    for section in SECTIONS:
        value = copy.deepcopy(settings[section])
        if section == "run":
            value = {key: v for key, v in value.items() if v is not None}
        data[section] = value
    return parse_recipe(data)


# ---------------------------------------------------------------------------
# Preset storage
# ---------------------------------------------------------------------------
BUILTIN_PRESET_DIR = config.RESOURCES_DIR / "presets"
USER_PRESET_SUBDIR = "user"
_PRESET_PATH_CHARACTERS = frozenset('/\\:*?"<>|')
MAX_PRESET_NAME_LENGTH = 64


class PresetExistsError(FileExistsError):
    """A user preset with this name exists; retry with ``overwrite=True``."""


class ReadOnlyPresetError(PermissionError):
    """Built-in presets cannot be overwritten, renamed, or deleted."""


@dataclass(frozen=True)
class PresetEntry:
    """One listed preset.  ``error`` is set when the file does not validate."""

    slug: str
    name: str
    path: Path
    builtin: bool
    sections: tuple[str, ...] = ()
    description: str = ""
    error: str | None = None

    @property
    def key(self) -> str:
        """Unique handle: ``slug`` for built-ins, ``user/slug`` for user presets."""
        return self.slug if self.builtin else f"{USER_PRESET_SUBDIR}/{self.slug}"


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


def preset_path_for(slug: str, preset_dir: str | Path) -> Path:
    """Return the preset file for ``slug``, refusing anything outside the folder."""
    preset_dir = Path(preset_dir)
    path = preset_dir / f"{slug}.json"
    if path.resolve(strict=False).parent != preset_dir.resolve(strict=False):
        raise ValueError(f"Preset path escapes the preset folder: {path}")
    return path


def _dirs(builtin_dir, user_dir) -> tuple[Path, Path]:
    builtin = Path(BUILTIN_PRESET_DIR if builtin_dir is None else builtin_dir)
    user = builtin / USER_PRESET_SUBDIR if user_dir is None else Path(user_dir)
    return builtin, user


def _entry(path: Path, builtin: bool) -> PresetEntry:
    try:
        recipe = load_recipe(path)
    except RecipeError as exc:
        return PresetEntry(path.stem, path.stem, path, builtin, error=str(exc))
    return PresetEntry(path.stem, recipe.name or path.stem, path, builtin,
                       recipe.sections, recipe.description)


def list_presets(builtin_dir=None, user_dir=None) -> list[PresetEntry]:
    """Built-in presets, then user presets, each sorted by slug."""
    builtin, user = _dirs(builtin_dir, user_dir)
    entries: list[PresetEntry] = []
    for folder, is_builtin in ((builtin, True), (user, False)):
        if folder.is_dir():
            entries.extend(_entry(path, is_builtin)
                           for path in sorted(folder.glob("*.json")))
    return entries


def find_preset(key: str, builtin_dir=None, user_dir=None) -> PresetEntry:
    """Look up a preset by ``slug`` (built-in) or ``user/slug``."""
    for entry in list_presets(builtin_dir, user_dir):
        if entry.key == key:
            return entry
    raise FileNotFoundError(f"No preset named {key!r}")


def save_user_preset(recipe: StudioRecipe, name: str, *, overwrite: bool = False,
                     builtin_dir=None, user_dir=None) -> Path:
    """Save ``recipe`` as a user preset named ``name`` (always schema v2).

    A name that clashes with a built-in is refused; replacing an existing
    user preset requires ``overwrite=True``.
    """
    builtin, user = _dirs(builtin_dir, user_dir)
    slug = preset_slug(name)
    if (builtin / f"{slug}.json").is_file():
        raise ReadOnlyPresetError(
            f"'{slug}' is a built-in preset name; choose another name.")
    path = preset_path_for(slug, user)
    if path.exists() and not overwrite:
        raise PresetExistsError(f"A user preset named '{slug}' already exists.")
    user.mkdir(parents=True, exist_ok=True)
    save_recipe(path, replace(recipe, name=name.strip()))
    return path


def _require_user(entry: PresetEntry, action: str) -> None:
    if entry.builtin:
        raise ReadOnlyPresetError(f"Built-in presets are read-only; cannot {action} it.")


def duplicate_preset(entry: PresetEntry, new_name: str, *, overwrite: bool = False,
                     builtin_dir=None, user_dir=None) -> Path:
    """Copy any preset (built-in or user) into a new user preset."""
    return save_user_preset(load_recipe(entry.path), new_name, overwrite=overwrite,
                            builtin_dir=builtin_dir, user_dir=user_dir)


def rename_user_preset(entry: PresetEntry, new_name: str, *, overwrite: bool = False,
                       builtin_dir=None, user_dir=None) -> Path:
    _require_user(entry, "rename")
    path = save_user_preset(load_recipe(entry.path), new_name, overwrite=overwrite,
                            builtin_dir=builtin_dir, user_dir=user_dir)
    if path.resolve() != entry.path.resolve():
        entry.path.unlink()
    return path


def delete_user_preset(entry: PresetEntry) -> None:
    _require_user(entry, "delete")
    entry.path.unlink()


def import_preset(source: str | Path, *, name: str | None = None,
                  overwrite: bool = False, builtin_dir=None, user_dir=None) -> Path:
    """Validate a recipe file, then copy it into the user presets as v2."""
    recipe = load_recipe(source)
    return save_user_preset(recipe, name or recipe.name or Path(source).stem,
                            overwrite=overwrite, builtin_dir=builtin_dir,
                            user_dir=user_dir)


def export_preset(entry: PresetEntry, destination: str | Path) -> Path:
    """Write a preset (validated, schema v2) to any path."""
    destination = Path(destination)
    save_recipe(destination, load_recipe(entry.path))
    return destination


# ---------------------------------------------------------------------------
# Studio state (what the Studio tab edits)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StudioState:
    """The three effect sections being edited, plus where they came from.

    ``baseline`` holds the sections as last loaded (a preset or a base
    profile); any difference from it makes the state dirty, shown as
    "custom (based on X)".  The run section lives on the Generate tab.
    """

    augmentation: dict
    stroke_damage: dict
    cutoff: dict
    source_name: str = DEFAULT_AUGMENTATION_PROFILE_ID
    source_key: str | None = None
    baseline: dict = field(default_factory=dict, compare=False)

    @classmethod
    def default(cls) -> "StudioState":
        settings = default_settings()
        sections = {name: settings[name] for name in EFFECT_SECTIONS}
        return cls(**copy.deepcopy(sections), baseline=copy.deepcopy(sections))

    def sections(self) -> dict[str, dict]:
        return {name: copy.deepcopy(getattr(self, name)) for name in EFFECT_SECTIONS}

    def with_section(self, section: str, data: Mapping) -> "StudioState":
        """A new state with one section replaced (validated, defaults filled)."""
        normalized = studio_spec.require_valid_section(section, data)
        normalized.pop("based_on", None)
        return self._replace_section(section, normalized)

    def _replace_section(self, section: str, data: dict) -> "StudioState":
        if section not in EFFECT_SECTIONS:
            raise ValueError(f"Unknown studio section {section!r}")
        changes: dict[str, Any] = {section: data}
        return replace(self, **changes)

    def with_value(self, spec: studio_spec.ParamSpec, value) -> "StudioState":
        section = copy.deepcopy(getattr(self, spec.section))
        studio_spec.set_value(section, spec, value)
        return self.with_section(spec.section, section)

    def value(self, spec: studio_spec.ParamSpec):
        return studio_spec.get_value(getattr(self, spec.section), spec)

    def loaded(self, recipe: StudioRecipe, *, source_name: str,
               source_key: str | None = None) -> "StudioState":
        """Apply a recipe's effect sections; the result is the new baseline."""
        state = self
        for name in EFFECT_SECTIONS:
            section = recipe.section(name)
            if section is not None:
                state = state.with_section(name, section)
        return replace(state, source_name=source_name, source_key=source_key,
                       baseline=state.sections())

    def with_base_profile(self, profile_id: str) -> "StudioState":
        """Load a built-in augmentation profile's exact values."""
        state = self.with_section("augmentation", augmentation_spec_from_profile(profile_id))
        baseline = {**self.baseline, "augmentation": state.sections()["augmentation"]}
        return replace(state, source_name=profile_id, source_key=None, baseline=baseline)

    def reset_section(self, section: str, specs=None) -> "StudioState":
        """Return ``section`` (or only ``specs`` of it) to the baseline values."""
        baseline = self.baseline.get(section) or default_settings()[section]
        if specs is None:
            return self.with_section(section, baseline)
        state = self
        for spec in specs:
            state = state.with_value(spec, studio_spec.get_value(baseline, spec))
        return state

    @property
    def dirty_sections(self) -> tuple[str, ...]:
        return tuple(name for name in EFFECT_SECTIONS
                     if getattr(self, name) != self.baseline.get(name))

    @property
    def dirty(self) -> bool:
        return bool(self.dirty_sections)

    @property
    def label(self) -> str:
        return f"custom (based on {self.source_name})" if self.dirty else self.source_name

    def to_recipe(self, name: str = "", description: str = "", *,
                  sections=EFFECT_SECTIONS, run: Mapping | None = None) -> StudioRecipe:
        """A recipe with the chosen sections (and optionally a run section)."""
        data: dict[str, object] = {
            "format": RECIPE_FORMAT, "schema_version": RECIPE_SCHEMA_VERSION,
            "name": name, "description": description,
        }
        for section in sections:
            data[section] = copy.deepcopy(getattr(self, section))
        if run is not None:
            data["run"] = {key: value for key, value in run.items() if value is not None}
        return parse_recipe(data)

    def with_bypassed(self, effects) -> "StudioState":
        """A copy with the given ``(section, effect)`` chances set to 0.

        For A/B previews only; ``("cutoff", "*")`` turns every cut side off.
        """
        state = self
        for section, effect in effects:
            data = copy.deepcopy(getattr(state, section))
            names = (studio_spec.CUTOFF_SIDES if section == "cutoff" and effect == "*"
                     else (effect,))
            for name in names:
                data[name]["probability"] = 0.0
            state = state._replace_section(section, data)
        return state

    def augmentation_profile(self):
        return augmentation_profile_from_spec(self.augmentation)

    def stroke_damage_profile(self):
        return stroke_damage_from_spec(self.stroke_damage)

    def cutoff_profile(self):
        return cutoff_profile_from_spec(self.cutoff)

    def baseline_recipe(self) -> StudioRecipe:
        """The loaded source as a recipe (used to record whether a run changed it)."""
        return parse_recipe({
            "format": RECIPE_FORMAT, "schema_version": RECIPE_SCHEMA_VERSION,
            "name": self.source_name, "description": "",
            **{name: copy.deepcopy(self.baseline.get(name) or getattr(self, name))
               for name in EFFECT_SECTIONS},
        })
