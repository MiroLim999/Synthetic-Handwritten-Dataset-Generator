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
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from numbers import Real
from pathlib import Path

import config
from src import studio_spec
from src.generation_profiles import (DEFAULT_AUGMENTATION_PROFILE_ID,
                                     DEFAULT_WRITER_PROFILE_ID,
                                     WRITER_STYLE_PROFILES,
                                     augmentation_spec_from_legacy_values,
                                     augmentation_spec_from_profile,
                                     canonical_json, cutoff_spec_from_profile,
                                     stroke_damage_from_config,
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
        **{name: sections.get(name) for name in SECTIONS},
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
