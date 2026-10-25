"""One parameter table for the Developer Studio.

Every studio control, recipe value, and profile field is described by a
:class:`ParamSpec`.  The table drives the GUI widgets, recipe validation, and
the conversion between recipe sections and profile dataclasses, so the three
can never disagree about bounds, steps, units, or defaults.

Recipe sections are nested dictionaries.  A spec's ``path`` is
``<section>.<effect>.<key>`` (for example ``augmentation.blur.range``) or
``<section>.<key>`` for a plain value (``cutoff.max_ink_loss``).  ``chance``
params are probabilities, ``range`` params are ``[min, max]`` pairs, and
``value`` params are single numbers.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real

SECTIONS = ("augmentation", "stroke_damage", "cutoff")
KINDS = ("chance", "range", "value")
CUTOFF_SIDES = ("top", "bottom", "left", "right")


@dataclass(frozen=True)
class ParamSpec:
    """One studio parameter.

    ``fields`` names the target profile field(s).  A ``range`` maps to one
    tuple field, or to two scalar fields (min, max) when two names are given.
    """

    path: str
    group: str
    label: str
    kind: str
    low: float
    high: float
    step: float
    default: float | tuple[float, float]
    tooltip: str
    fields: tuple[str, ...]
    unit: str = ""

    @property
    def section(self) -> str:
        return self.path.split(".", 1)[0]

    @property
    def keys(self) -> tuple[str, ...]:
        """Path below the section: ``("blur", "range")`` or ``("max_ink_loss",)``."""
        return tuple(self.path.split(".")[1:])

    @property
    def effect(self) -> str:
        return self.keys[0]

    @property
    def integer(self) -> bool:
        return float(self.step).is_integer()


def _chance(path, group, label, default, tooltip, field):
    return ParamSpec(path, group, label, "chance", 0.0, 1.0, 0.01, default,
                     tooltip, (field,), "%")


def _range(path, group, label, low, high, step, default, tooltip, fields, unit=""):
    return ParamSpec(path, group, label, "range", low, high, step,
                     tuple(default), tooltip, tuple(fields), unit)


INK_AND_PAPER = "Ink and paper"
GEOMETRY = "Geometry"
SCANNER = "Scanner"
STROKE_DAMAGE = "Stroke damage"
CHARACTER_CUTOFF = "Character cutoff"
GROUPS = (INK_AND_PAPER, GEOMETRY, SCANNER, STROKE_DAMAGE, CHARACTER_CUTOFF)

# Augmentation defaults are the historical_scan_v1 profile.
_AUGMENTATION = (
    _chance("augmentation.fade.probability", INK_AND_PAPER, "Fade chance", 0.60,
            "Chance that ink contrast is reduced.", "fade_probability"),
    _range("augmentation.fade.range", INK_AND_PAPER, "Ink contrast", 0.10, 1.00,
           0.01, (0.50, 0.85),
           "Contrast factor when fading applies; lower = more faded.",
           ("fade_range",)),
    _chance("augmentation.brightness.probability", INK_AND_PAPER,
            "Brightness chance", 0.65, "Chance that brightness changes.",
            "brightness_probability"),
    _range("augmentation.brightness.range", INK_AND_PAPER, "Brightness", 0.30,
           1.60, 0.01, (0.65, 1.15),
           "Brightness factor; 1.0 leaves the image unchanged.",
           ("brightness_range",)),
    _chance("augmentation.paper_tint.probability", INK_AND_PAPER,
            "Paper tint chance", 0.75, "Chance of an aged-paper tint.",
            "paper_tint_probability"),
    _range("augmentation.paper_tint.range", INK_AND_PAPER, "Paper tint", 0.0,
           0.50, 0.01, (0.08, 0.20), "Strength of the yellowish paper overlay.",
           ("paper_tint_alpha_range",)),
    _chance("augmentation.stains.probability", INK_AND_PAPER, "Stain chance",
            0.35, "Chance of brownish foxing spots.", "stain_probability"),
    _range("augmentation.stains.count_range", INK_AND_PAPER, "Stain count", 0,
           20, 1, (2, 6), "How many spots a stained image gets.",
           ("stain_count_range",)),
    _range("augmentation.stains.opacity_range", INK_AND_PAPER, "Stain opacity",
           0, 120, 1, (15, 45), "Opacity of each spot (0-255 alpha).",
           ("stain_opacity_range",)),
    _chance("augmentation.rotation.probability", GEOMETRY, "Tilt chance", 0.85,
            "Chance the crop is rotated.", "rotate_probability"),
    _range("augmentation.rotation.range", GEOMETRY, "Tilt", 0.0, 15.0, 0.1,
           (0.0, 4.5), "Minimum and maximum absolute tilt; the sign is random.",
           ("rotate_min_degrees", "rotate_degrees"), "°"),
    _chance("augmentation.blur.probability", SCANNER, "Blur chance", 0.65,
            "Chance of a Gaussian blur.", "blur_probability"),
    _range("augmentation.blur.range", SCANNER, "Blur radius", 0.0, 5.0, 0.1,
           (0.4, 1.8), "Gaussian blur radius.", ("blur_radius_range",), "px"),
    _chance("augmentation.noise.probability", SCANNER, "Grain chance", 0.65,
            "Chance of scanner grain.", "noise_probability"),
    _range("augmentation.noise.range", SCANNER, "Grain noise", 0.0, 50.0, 0.5,
           (4.0, 22.0), "Standard deviation of the pixel noise.",
           ("noise_std_range",)),
    _chance("augmentation.paper_texture.probability", INK_AND_PAPER,
            "Paper fiber chance", 0.50, "Chance of low-frequency paper texture.",
            "paper_texture_probability"),
    _range("augmentation.paper_texture.range", INK_AND_PAPER, "Paper fiber", 0.0,
           20.0, 0.1, (1.5, 6.0), "Strength of the paper texture.",
           ("paper_texture_std_range",)),
    _chance("augmentation.scanlines.probability", SCANNER, "Scanline chance",
            0.20, "Chance of faint horizontal scanner lines.",
            "scanline_probability"),
    _range("augmentation.scanlines.range", SCANNER, "Scanline alpha", 0, 60, 1,
           (4, 16), "Opacity of each scanline (0-255 alpha).",
           ("scanline_alpha_range",)),
    _chance("augmentation.jpeg.probability", SCANNER, "JPEG chance", 0.30,
            "Chance of a lossy JPEG round trip.", "compression_probability"),
    _range("augmentation.jpeg.range", SCANNER, "JPEG quality", 10, 100, 1,
           (58, 88), "JPEG quality; lower = more artifacts.",
           ("jpeg_quality_range",)),
)

# Stroke-damage defaults are the config BROKEN_* values.
_STROKE_DAMAGE = (
    _chance("stroke_damage.contrast.probability", STROKE_DAMAGE,
            "Contrast-drop chance", 0.70, "Chance of an extra contrast drop.",
            "contrast_probability"),
    _range("stroke_damage.contrast.range", STROKE_DAMAGE, "Contrast drop", 0.05,
           1.00, 0.01, (0.35, 0.75), "Contrast factor of the drop.",
           ("contrast_range",)),
    _chance("stroke_damage.erode.probability", STROKE_DAMAGE,
            "Stroke-thin chance", 0.55, "Chance strokes are thinned.",
            "erode_probability"),
    _range("stroke_damage.erode.range", STROKE_DAMAGE, "Stroke thinning", 0.0,
           1.0, 0.01, (0.15, 0.45), "Blend amount of the thinning filter.",
           ("erode_blend_range",)),
    _chance("stroke_damage.gaps.probability", STROKE_DAMAGE, "Gap chance", 0.90,
            "Chance of small gaps in strokes.", "gap_probability"),
    _range("stroke_damage.gaps.count_range", STROKE_DAMAGE, "Gap count", 1, 60,
           1, (8, 24), "How many gaps a damaged image gets.",
           ("gap_count_range",)),
    _range("stroke_damage.gaps.size_range", STROKE_DAMAGE, "Gap size", 1, 40, 1,
           (3, 16), "Width of each gap.", ("gap_size_range",), "px"),
    _chance("stroke_damage.scratches.probability", STROKE_DAMAGE,
            "Scratch chance", 0.70, "Chance of paper-colored scratches.",
            "scratch_probability"),
    _range("stroke_damage.scratches.count_range", STROKE_DAMAGE, "Scratch count",
           1, 20, 1, (1, 5), "How many scratches a damaged image gets.",
           ("scratch_count_range",)),
)

_SIDE_EXTENT = {"top": "letter height", "bottom": "letter height",
                "left": "word width", "right": "word width"}
_CUTOFF = tuple(
    spec
    for side in CUTOFF_SIDES
    for spec in (
        _chance(f"cutoff.{side}.probability", CHARACTER_CUTOFF,
                f"{side.title()} cut chance", 0.0,
                f"Chance the {side} of the ink is cut off.",
                f"{side}.probability"),
        _range(f"cutoff.{side}.range", CHARACTER_CUTOFF,
               f"{side.title()} cut - % of {_SIDE_EXTENT[side]}", 0.0, 0.40,
               0.01, (0.0, 0.0),
               f"Fraction of the ink box's {_SIDE_EXTENT[side]} removed "
               f"from the {side}.", (f"{side}.fraction_range",), "%"),
    )
) + (
    ParamSpec("cutoff.max_ink_loss", CHARACTER_CUTOFF, "Max ink loss", "value",
              0.0, 0.50, 0.01, 0.15,
              "Largest share of ink pixels a cut may remove; bigger cuts "
              "are shrunk.", ("max_ink_loss",), "%"),
)

PARAMS: tuple[ParamSpec, ...] = _AUGMENTATION + _STROKE_DAMAGE + _CUTOFF
PARAMS_BY_PATH: dict[str, ParamSpec] = {spec.path: spec for spec in PARAMS}


def params_for_section(section: str) -> tuple[ParamSpec, ...]:
    if section not in SECTIONS:
        raise ValueError(f"Unknown studio section {section!r}")
    return tuple(spec for spec in PARAMS if spec.section == section)


def effects_for_section(section: str) -> tuple[str, ...]:
    """Effect names in table order (``fade``, ``brightness`` ...)."""
    seen: dict[str, None] = {}
    for spec in params_for_section(section):
        seen.setdefault(spec.effect, None)
    return tuple(seen)


def get_value(section_data: Mapping, spec: ParamSpec):
    node = section_data
    for key in spec.keys:
        node = node[key]
    return node


def set_value(section_data: dict, spec: ParamSpec, value) -> None:
    node = section_data
    for key in spec.keys[:-1]:
        node = node.setdefault(key, {})
    node[spec.keys[-1]] = value


def _plain(spec: ParamSpec, value):
    """JSON-friendly value: ints for integer params, lists for ranges."""
    if spec.kind == "range":
        low, high = value
        if spec.integer:
            return [int(low), int(high)]
        return [float(low), float(high)]
    return int(value) if spec.integer and spec.kind == "value" else float(value)


def default_section(section: str) -> dict:
    """A complete section filled with every spec default."""
    data: dict = {}
    for spec in params_for_section(section):
        set_value(data, spec, _plain(spec, spec.default))
    return data


def _is_number(value) -> bool:
    return (not isinstance(value, bool) and isinstance(value, Real)
            and math.isfinite(float(value)))


def check_value(spec: ParamSpec, value) -> str | None:
    """Return a problem description, or None when ``value`` is valid."""
    if spec.kind == "range":
        if (not isinstance(value, (list, tuple)) or len(value) != 2
                or not all(_is_number(item) for item in value)):
            return "must be a [min, max] pair of finite numbers"
        low, high = float(value[0]), float(value[1])
        if spec.integer and not all(float(item).is_integer() for item in value):
            return "must contain whole numbers"
        if low > high:
            return "min must not exceed max"
        if low < spec.low or high > spec.high:
            return f"must stay within [{spec.low:g}, {spec.high:g}]"
        return None
    if not _is_number(value):
        return "must be a finite number"
    if spec.integer and not float(value).is_integer():
        return "must be a whole number"
    if not spec.low <= float(value) <= spec.high:
        return f"must be between {spec.low:g} and {spec.high:g}"
    return None


def validate_section(section: str, data, *, prefix: str | None = None,
                     allowed_extra: frozenset[str] = frozenset({"based_on"}),
                     ) -> tuple[dict, list[str]]:
    """Strictly validate one recipe section.

    Missing params take their spec defaults.  Returns the normalized,
    complete section (JSON-friendly values) and a list of problems, each
    naming its JSON path.  Unknown keys are problems too.
    """
    prefix = section if prefix is None else prefix
    errors: list[str] = []
    if not isinstance(data, Mapping):
        return default_section(section), [f"{prefix} must be a JSON object"]
    specs = params_for_section(section)
    normalized = default_section(section)
    for key in sorted(set(data) - set(effects_for_section(section)) - allowed_extra):
        errors.append(f"{prefix}.{key}: unknown key")
    for effect in effects_for_section(section):
        if effect not in data:
            continue
        effect_specs = [spec for spec in specs if spec.effect == effect]
        node = data[effect]
        if len(effect_specs) == 1 and len(effect_specs[0].keys) == 1:
            spec = effect_specs[0]
            problem = check_value(spec, node)
            if problem:
                errors.append(f"{prefix}.{effect}: {problem}")
            else:
                set_value(normalized, spec, _plain(spec, node))
            continue
        if not isinstance(node, Mapping):
            errors.append(f"{prefix}.{effect}: must be a JSON object")
            continue
        known = {spec.keys[1] for spec in effect_specs}
        for key in sorted(set(node) - known):
            errors.append(f"{prefix}.{effect}.{key}: unknown key")
        for spec in effect_specs:
            key = spec.keys[1]
            if key not in node:
                continue
            problem = check_value(spec, node[key])
            if problem:
                errors.append(f"{prefix}.{effect}.{key}: {problem}")
            else:
                set_value(normalized, spec, _plain(spec, node[key]))
    if "based_on" in data and "based_on" in allowed_extra:
        if not isinstance(data["based_on"], str):
            errors.append(f"{prefix}.based_on: must be a string")
        else:
            normalized["based_on"] = data["based_on"]
    return normalized, errors


def require_valid_section(section: str, data) -> dict:
    """Validate a section and raise ``ValueError`` listing every problem."""
    normalized, errors = validate_section(section, data)
    if errors:
        raise ValueError("Invalid studio values: " + "; ".join(errors))
    return normalized


def section_fields(section: str, data: Mapping) -> dict[str, object]:
    """Map a validated section onto profile keyword arguments."""
    kwargs: dict[str, object] = {}
    for spec in params_for_section(section):
        value = get_value(data, spec)
        if spec.kind == "range" and len(spec.fields) == 2:
            kwargs[spec.fields[0]] = float(value[0])
            kwargs[spec.fields[1]] = float(value[1])
        elif spec.kind == "range":
            cast = int if spec.integer else float
            kwargs[spec.fields[0]] = (cast(value[0]), cast(value[1]))
        else:
            kwargs[spec.fields[0]] = float(value)
    return kwargs


def section_from_fields(section: str, values: Mapping[str, object]) -> dict:
    """Build a complete section from profile field values (no validation)."""
    data: dict = {}
    for spec in params_for_section(section):
        if spec.kind == "range" and len(spec.fields) == 2:
            value = (values[spec.fields[0]], values[spec.fields[1]])
        else:
            value = values[spec.fields[0]]
        set_value(data, spec, _plain(spec, value))
    return data


def copy_section(data: Mapping) -> dict:
    return copy.deepcopy(dict(data))
