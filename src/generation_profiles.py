"""Versioned realism profiles used by the synthetic image generator.

The profiles in this module are intentionally explicit and serializable.  A
run can record the selected profile ID, and recreating a writer style requires
only that ID, a writer identifier, and the run seed.  The numeric defaults are
engineering ranges rather than claims that they are calibrated to a real
archive; :mod:`src.image_statistics` provides the comparison hook needed for
future calibration against representative scans.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import warnings
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from numbers import Integral, Real

from src import studio_spec


PROFILE_SCHEMA_VERSION = 1
# Shape version of AugmentationProfile metadata.  Kept separate from
# PROFILE_SCHEMA_VERSION, which seeds writer styles and must never change.
AUGMENTATION_SCHEMA_VERSION = 2
STROKE_DAMAGE_SCHEMA_VERSION = 1
# Length-aware safeguard for short labels (age, sex, single characters).
SHORT_LABEL_CHARS = 3
SHORT_LABEL_MAX_ERODE_PROBABILITY = 0.35
# Attempts after this number progressively soften stochastic damage.
DAMAGE_BACKOFF_AFTER_ATTEMPT = 8
DEFAULT_WRITER_PROFILE_ID = "writer_style_v1"
DEFAULT_AUGMENTATION_PROFILE_ID = "historical_scan_v1"


def _finite(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number; got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite; got {value!r}")
    return result


def _range(name: str, value, *, minimum: float | None = None,
           maximum: float | None = None) -> tuple[float, float]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"{name} must be a two-value tuple")
    low = _finite(f"{name}[0]", value[0])
    high = _finite(f"{name}[1]", value[1])
    if low > high:
        raise ValueError(f"{name} must be ordered low-to-high")
    if minimum is not None and low < minimum:
        raise ValueError(f"{name} values must be at least {minimum}")
    if maximum is not None and high > maximum:
        raise ValueError(f"{name} values must be at most {maximum}")
    return low, high


def _probability(name: str, value: Real) -> float:
    result = _finite(name, value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return result


@dataclass(frozen=True)
class WriterStyleProfile:
    """Ranges from which a stable writer identity is sampled."""

    profile_id: str
    baseline_amplitude_range: tuple[float, float]
    baseline_period_range: tuple[float, float]
    spacing_range: tuple[float, float]
    connection_overlap_range: tuple[float, float]
    slant_range: tuple[float, float]
    stroke_width_range: tuple[int, int]
    pressure_bias_range: tuple[int, int]
    pressure_variation_range: tuple[int, int]
    ink_texture_density_range: tuple[float, float]
    glyph_width_scale_range: tuple[float, float]
    font_size_scale_range: tuple[float, float]

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise ValueError("profile_id cannot be empty")
        _range("baseline_amplitude_range", self.baseline_amplitude_range, minimum=0)
        _range("baseline_period_range", self.baseline_period_range, minimum=1)
        _range("spacing_range", self.spacing_range)
        _range("connection_overlap_range", self.connection_overlap_range, minimum=0)
        _range("slant_range", self.slant_range, minimum=-0.5, maximum=0.5)
        stroke = _range("stroke_width_range", self.stroke_width_range, minimum=0,
                        maximum=4)
        if any(not isinstance(value, Integral) or isinstance(value, bool)
               for value in self.stroke_width_range):
            raise ValueError("stroke_width_range values must be integers")
        if stroke[1] > 4:
            raise ValueError("stroke_width_range values must be at most 4")
        for name, value in (
                ("pressure_bias_range", self.pressure_bias_range),
                ("pressure_variation_range", self.pressure_variation_range)):
            _range(name, value, minimum=-80 if name == "pressure_bias_range" else 0,
                   maximum=80)
            if any(not isinstance(item, Integral) or isinstance(item, bool)
                   for item in value):
                raise ValueError(f"{name} values must be integers")
        _range("ink_texture_density_range", self.ink_texture_density_range,
               minimum=0, maximum=0.10)
        _range("glyph_width_scale_range", self.glyph_width_scale_range,
               minimum=0.70, maximum=1.30)
        _range("font_size_scale_range", self.font_size_scale_range,
               minimum=0.75, maximum=1.25)

    def to_metadata(self) -> dict[str, object]:
        return {"profile_schema_version": PROFILE_SCHEMA_VERSION, **asdict(self)}


@dataclass(frozen=True)
class WriterStyle:
    """Stable visual traits for one synthetic writer.

    ``glyph_width_scale`` and ``font_preference`` are practical proxies for
    glyph-form consistency when OpenType fonts do not expose alternate glyphs.
    ``connection_overlap`` tightens character spacing for fonts whose glyphs
    visually connect.  The renderer records the profile and style IDs so these
    approximations remain auditable.
    """

    profile_id: str
    style_id: str
    writer_id: str
    baseline_amplitude: float
    baseline_period: float
    spacing: float
    connection_overlap: float
    slant: float
    stroke_width: int
    pressure_bias: int
    pressure_variation: int
    ink_texture_density: float
    glyph_width_scale: float
    font_size_scale: float
    font_preference: float
    baseline_phase: float

    def __post_init__(self) -> None:
        for name, value in (
                ("profile_id", self.profile_id),
                ("style_id", self.style_id),
                ("writer_id", self.writer_id)):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a nonempty string")
        if _finite("baseline_amplitude", self.baseline_amplitude) < 0:
            raise ValueError("baseline_amplitude must be nonnegative")
        if _finite("baseline_period", self.baseline_period) <= 0:
            raise ValueError("baseline_period must be positive")
        _finite("spacing", self.spacing)
        if _finite("connection_overlap", self.connection_overlap) < 0:
            raise ValueError("connection_overlap must be nonnegative")
        if not -0.5 <= _finite("slant", self.slant) <= 0.5:
            raise ValueError("slant must be between -0.5 and 0.5")
        if (isinstance(self.stroke_width, bool)
                or not isinstance(self.stroke_width, Integral)
                or not 0 <= self.stroke_width <= 4):
            raise ValueError("stroke_width must be an integer from 0 to 4")
        for name, value in (
                ("pressure_bias", self.pressure_bias),
                ("pressure_variation", self.pressure_variation)):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer")
        if self.pressure_variation < 0:
            raise ValueError("pressure_variation must be nonnegative")
        if not 0 <= _finite("ink_texture_density", self.ink_texture_density) <= 0.10:
            raise ValueError("ink_texture_density must be between 0 and 0.10")
        if not 0.70 <= _finite("glyph_width_scale", self.glyph_width_scale) <= 1.30:
            raise ValueError("glyph_width_scale must be between 0.70 and 1.30")
        if not 0.75 <= _finite("font_size_scale", self.font_size_scale) <= 1.25:
            raise ValueError("font_size_scale must be between 0.75 and 1.25")
        if not 0.0 <= _finite("font_preference", self.font_preference) <= 1.0:
            raise ValueError("font_preference must be between 0 and 1")
        _finite("baseline_phase", self.baseline_phase)

    def to_metadata(self) -> dict[str, object]:
        return {"profile_schema_version": PROFILE_SCHEMA_VERSION, **asdict(self)}


@dataclass(frozen=True)
class AugmentationProfile:
    """A fully versioned scan/degradation recipe.

    These values are fixed rather than read from mutable global configuration,
    which makes a recorded profile ID meaningful for reproduction.
    """

    profile_id: str
    fade_probability: float
    fade_range: tuple[float, float]
    brightness_probability: float
    brightness_range: tuple[float, float]
    paper_tint_probability: float
    paper_tint_alpha_range: tuple[float, float]
    stain_probability: float
    rotate_probability: float
    rotate_degrees: float
    blur_probability: float
    blur_radius_range: tuple[float, float]
    noise_probability: float
    noise_std_range: tuple[float, float]
    paper_texture_probability: float
    paper_texture_std_range: tuple[float, float]
    scanline_probability: float
    scanline_alpha_range: tuple[int, int]
    compression_probability: float
    jpeg_quality_range: tuple[int, int]
    # Fields added in augmentation schema 2.  Their defaults reproduce the
    # schema-1 behavior and RNG draws exactly.
    rotate_min_degrees: float = 0.0
    stain_count_range: tuple[int, int] = (2, 6)
    stain_opacity_range: tuple[int, int] = (15, 45)

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise ValueError("profile_id cannot be empty")
        for name in (
                "fade_probability", "brightness_probability",
                "paper_tint_probability", "stain_probability",
                "rotate_probability", "blur_probability", "noise_probability",
                "paper_texture_probability", "scanline_probability",
                "compression_probability"):
            _probability(name, getattr(self, name))
        _range("fade_range", self.fade_range, minimum=0)
        _range("brightness_range", self.brightness_range, minimum=0)
        _range("paper_tint_alpha_range", self.paper_tint_alpha_range,
               minimum=0, maximum=1)
        rotate = _finite("rotate_degrees", self.rotate_degrees)
        if rotate < 0:
            raise ValueError("rotate_degrees must be nonnegative")
        rotate_min = _finite("rotate_min_degrees", self.rotate_min_degrees)
        if not 0 <= rotate_min <= rotate:
            raise ValueError(
                "rotate_min_degrees must be between 0 and rotate_degrees")
        _range("blur_radius_range", self.blur_radius_range, minimum=0)
        _range("noise_std_range", self.noise_std_range, minimum=0)
        _range("paper_texture_std_range", self.paper_texture_std_range, minimum=0)
        for name, value, low, high in (
                ("scanline_alpha_range", self.scanline_alpha_range, 0, 255),
                ("jpeg_quality_range", self.jpeg_quality_range, 1, 100),
                ("stain_count_range", self.stain_count_range, 0, 1000),
                ("stain_opacity_range", self.stain_opacity_range, 0, 255)):
            _range(name, value, minimum=low, maximum=high)
            if any(not isinstance(item, Integral) or isinstance(item, bool)
                   for item in value):
                raise ValueError(f"{name} values must be integers")

    def to_metadata(self) -> dict[str, object]:
        return {
            "profile_schema_version": PROFILE_SCHEMA_VERSION,
            "augmentation_schema_version": AUGMENTATION_SCHEMA_VERSION,
            **asdict(self),
        }


def _whole_range(name: str, value, *, minimum: int) -> tuple[int, int]:
    _range(name, value, minimum=minimum)
    if any(not isinstance(item, Integral) or isinstance(item, bool) for item in value):
        raise ValueError(f"{name} values must be integers")
    return int(value[0]), int(value[1])


@dataclass(frozen=True)
class StrokeDamageProfile:
    """Localized stroke damage for semi-broken samples.

    The defaults (:func:`stroke_damage_from_config`) reproduce the config
    ``BROKEN_*`` behavior and RNG draws exactly.
    """

    contrast_probability: float
    contrast_range: tuple[float, float]
    erode_probability: float
    erode_blend_range: tuple[float, float]
    gap_probability: float
    gap_count_range: tuple[int, int]
    gap_size_range: tuple[int, int]
    scratch_probability: float
    scratch_count_range: tuple[int, int]

    def __post_init__(self) -> None:
        for name in ("contrast_probability", "erode_probability",
                     "gap_probability", "scratch_probability"):
            _probability(name, getattr(self, name))
        low, _high = _range("contrast_range", self.contrast_range,
                            minimum=0, maximum=1)
        if low <= 0:
            raise ValueError("contrast_range values must be greater than 0")
        _range("erode_blend_range", self.erode_blend_range, minimum=0, maximum=1)
        _whole_range("gap_count_range", self.gap_count_range, minimum=1)
        _whole_range("gap_size_range", self.gap_size_range, minimum=1)
        _whole_range("scratch_count_range", self.scratch_count_range, minimum=1)

    @property
    def profile_id(self) -> str:
        return content_id("stroke", _stroke_values(asdict(self)))

    def to_metadata(self) -> dict[str, object]:
        return {
            "stroke_damage_schema_version": STROKE_DAMAGE_SCHEMA_VERSION,
            "profile_id": self.profile_id,
            **asdict(self),
        }

    @classmethod
    def from_legacy_params(cls, params: Mapping | None) -> "StrokeDamageProfile":
        """Convert the old GUI dict (contrast_prob, gap_count, and so on).

        ``gap_count: N`` becomes ``gap_count_range = (1, max(1, int(N)))``;
        missing keys fall back to the config defaults, as the old
        ``_semi_broken`` did.
        """
        base = stroke_damage_from_config()
        params = dict(params or {})
        unknown = sorted(set(params) - _LEGACY_STROKE_KEYS)
        if unknown:
            raise ValueError(f"Unknown stroke damage params: {', '.join(unknown)}")
        changes: dict[str, object] = {}
        for legacy, field in (("contrast_prob", "contrast_probability"),
                              ("erode_prob", "erode_probability"),
                              ("gap_prob", "gap_probability"),
                              ("scratch_prob", "scratch_probability")):
            if legacy in params:
                changes[field] = float(params[legacy])
        if params.get("gap_count") is not None:
            changes["gap_count_range"] = (1, max(1, int(params["gap_count"])))
        return replace(base, **changes)

    def constrained_for_label(self, label: str, attempt: int) -> "StrokeDamageProfile":
        """Apply the short-label safeguard and the late-attempt backoff.

        Labels of 3 or fewer characters cap both ends of the gap-count range
        at 3 x length and the erode chance at 0.35.  Attempts above 8 scale
        the gap chance, erode chance, and gap-count range by
        ``max(0.1, 1 - 0.1 * (attempt - 8))``, keeping at least one gap.
        """
        profile = self
        length = len(label.strip())
        if length <= SHORT_LABEL_CHARS:
            cap = max(1, length * 3)
            low, high = profile.gap_count_range
            profile = replace(
                profile,
                gap_count_range=(min(low, cap), min(high, cap)),
                erode_probability=min(profile.erode_probability,
                                      SHORT_LABEL_MAX_ERODE_PROBABILITY),
            )
        if attempt > DAMAGE_BACKOFF_AFTER_ATTEMPT:
            scale = max(0.1, 1.0 - ((attempt - DAMAGE_BACKOFF_AFTER_ATTEMPT) * 0.1))
            low, high = profile.gap_count_range
            profile = replace(
                profile,
                gap_probability=profile.gap_probability * scale,
                erode_probability=profile.erode_probability * scale,
                gap_count_range=(max(1, int(low * scale)), max(1, int(high * scale))),
            )
        return profile


_LEGACY_STROKE_KEYS = frozenset({
    "contrast_prob", "erode_prob", "gap_prob", "gap_count", "scratch_prob",
})
_STROKE_INTEGER_FIELDS = frozenset({
    "gap_count_range", "gap_size_range", "scratch_count_range",
})


def _stroke_values(fields: Mapping[str, object]) -> dict[str, object]:
    return {
        name: _normalized_number(value, name in _STROKE_INTEGER_FIELDS)
        for name, value in fields.items()
    }


def stroke_damage_from_config() -> StrokeDamageProfile:
    """The config ``BROKEN_*`` values, read at call time."""
    import config

    return StrokeDamageProfile(
        contrast_probability=config.BROKEN_CONTRAST_PROB,
        contrast_range=tuple(config.BROKEN_CONTRAST_RANGE),
        erode_probability=config.BROKEN_ERODE_PROB,
        erode_blend_range=tuple(config.BROKEN_ERODE_BLEND),
        gap_probability=config.BROKEN_INK_GAP_PROB,
        gap_count_range=tuple(config.BROKEN_INK_GAP_COUNT),
        gap_size_range=tuple(config.BROKEN_INK_GAP_SIZE),
        scratch_probability=config.BROKEN_SCRATCH_PROB,
        scratch_count_range=tuple(config.BROKEN_SCRATCH_COUNT),
    )


def as_stroke_damage_profile(value) -> StrokeDamageProfile:
    """Accept a profile, a legacy params dict, or None (config defaults)."""
    if isinstance(value, StrokeDamageProfile):
        return value
    if value is None:
        return stroke_damage_from_config()
    if isinstance(value, Mapping):
        return StrokeDamageProfile.from_legacy_params(value)
    raise TypeError(f"Unsupported stroke damage settings: {value!r}")


def stroke_damage_from_spec(spec: Mapping) -> StrokeDamageProfile:
    """Build a profile from a studio ``stroke_damage`` section."""
    data = studio_spec.require_valid_section("stroke_damage", spec)
    return StrokeDamageProfile(**studio_spec.section_fields("stroke_damage", data))


def stroke_damage_spec_from_profile(profile: StrokeDamageProfile) -> dict:
    return studio_spec.section_from_fields("stroke_damage", asdict(profile))


WRITER_STYLE_PROFILES: dict[str, WriterStyleProfile] = {
    DEFAULT_WRITER_PROFILE_ID: WriterStyleProfile(
        profile_id=DEFAULT_WRITER_PROFILE_ID,
        baseline_amplitude_range=(0.0, 2.2),
        baseline_period_range=(3.5, 9.0),
        spacing_range=(-0.6, 1.8),
        connection_overlap_range=(0.0, 1.6),
        slant_range=(-0.12, 0.22),
        stroke_width_range=(0, 2),
        pressure_bias_range=(-14, 20),
        pressure_variation_range=(1, 15),
        ink_texture_density_range=(0.001, 0.012),
        glyph_width_scale_range=(0.88, 1.12),
        font_size_scale_range=(0.90, 1.10),
    ),
}


AUGMENTATION_PROFILES: dict[str, AugmentationProfile] = {
    DEFAULT_AUGMENTATION_PROFILE_ID: AugmentationProfile(
        profile_id=DEFAULT_AUGMENTATION_PROFILE_ID,
        fade_probability=0.60,
        fade_range=(0.50, 0.85),
        brightness_probability=0.65,
        brightness_range=(0.65, 1.15),
        paper_tint_probability=0.75,
        paper_tint_alpha_range=(0.08, 0.20),
        stain_probability=0.35,
        rotate_probability=0.85,
        rotate_degrees=4.5,
        blur_probability=0.65,
        blur_radius_range=(0.4, 1.8),
        noise_probability=0.65,
        noise_std_range=(4.0, 22.0),
        paper_texture_probability=0.50,
        paper_texture_std_range=(1.5, 6.0),
        scanline_probability=0.20,
        scanline_alpha_range=(4, 16),
        compression_probability=0.30,
        jpeg_quality_range=(58, 88),
    ),
    "historical_1914_ledger": AugmentationProfile(
        profile_id="historical_1914_ledger",
        fade_probability=0.85,
        fade_range=(0.40, 0.70),
        brightness_probability=0.75,
        brightness_range=(0.70, 1.15),
        paper_tint_probability=0.90,
        paper_tint_alpha_range=(0.12, 0.25),
        stain_probability=0.45,
        rotate_probability=0.85,
        rotate_degrees=3.5,
        blur_probability=0.55,
        blur_radius_range=(0.4, 1.4),
        noise_probability=0.60,
        noise_std_range=(4.0, 18.0),
        paper_texture_probability=0.80,
        paper_texture_std_range=(2.5, 7.0),
        scanline_probability=0.20,
        scanline_alpha_range=(4, 14),
        compression_probability=0.35,
        jpeg_quality_range=(60, 85),
    ),
    "faded_ink": AugmentationProfile(
        profile_id="faded_ink",
        fade_probability=0.95,
        fade_range=(0.30, 0.60),
        brightness_probability=0.80,
        brightness_range=(0.80, 1.25),
        paper_tint_probability=0.90,
        paper_tint_alpha_range=(0.14, 0.28),
        stain_probability=0.25,
        rotate_probability=0.75,
        rotate_degrees=3.5,
        blur_probability=0.45,
        blur_radius_range=(0.3, 1.2),
        noise_probability=0.50,
        noise_std_range=(3.0, 15.0),
        paper_texture_probability=0.60,
        paper_texture_std_range=(2.0, 7.0),
        scanline_probability=0.15,
        scanline_alpha_range=(4, 12),
        compression_probability=0.25,
        jpeg_quality_range=(65, 90),
    ),
    "bleed_through": AugmentationProfile(
        profile_id="bleed_through",
        fade_probability=0.70,
        fade_range=(0.45, 0.80),
        brightness_probability=0.70,
        brightness_range=(0.60, 1.10),
        paper_tint_probability=0.85,
        paper_tint_alpha_range=(0.12, 0.26),
        stain_probability=0.60,
        rotate_probability=0.85,
        rotate_degrees=4.0,
        blur_probability=0.70,
        blur_radius_range=(0.5, 2.0),
        noise_probability=0.60,
        noise_std_range=(5.0, 20.0),
        paper_texture_probability=0.95,
        paper_texture_std_range=(4.0, 10.0),
        scanline_probability=0.30,
        scanline_alpha_range=(6, 18),
        compression_probability=0.40,
        jpeg_quality_range=(50, 80),
    ),
    "scan_noise": AugmentationProfile(
        profile_id="scan_noise",
        fade_probability=0.65,
        fade_range=(0.45, 0.80),
        brightness_probability=0.75,
        brightness_range=(0.55, 1.15),
        paper_tint_probability=0.75,
        paper_tint_alpha_range=(0.08, 0.22),
        stain_probability=0.40,
        rotate_probability=0.90,
        rotate_degrees=5.0,
        blur_probability=0.80,
        blur_radius_range=(0.8, 2.2),
        noise_probability=0.95,
        noise_std_range=(12.0, 30.0),
        paper_texture_probability=0.75,
        paper_texture_std_range=(3.0, 8.0),
        scanline_probability=0.85,
        scanline_alpha_range=(10, 28),
        compression_probability=0.85,
        jpeg_quality_range=(35, 65),
    ),
    "heavy_smudge": AugmentationProfile(
        profile_id="heavy_smudge",
        fade_probability=0.85,
        fade_range=(0.35, 0.70),
        brightness_probability=0.80,
        brightness_range=(0.50, 1.10),
        paper_tint_probability=0.90,
        paper_tint_alpha_range=(0.14, 0.28),
        stain_probability=0.85,
        rotate_probability=0.90,
        rotate_degrees=5.5,
        blur_probability=0.85,
        blur_radius_range=(0.8, 2.4),
        noise_probability=0.85,
        noise_std_range=(10.0, 26.0),
        paper_texture_probability=0.85,
        paper_texture_std_range=(3.0, 9.0),
        scanline_probability=0.50,
        scanline_alpha_range=(8, 22),
        compression_probability=0.75,
        jpeg_quality_range=(40, 70),
    ),
    "clean": AugmentationProfile(
        profile_id="clean",
        fade_probability=0.0,
        fade_range=(1.0, 1.0),
        brightness_probability=0.0,
        brightness_range=(1.0, 1.0),
        paper_tint_probability=0.0,
        paper_tint_alpha_range=(0.0, 0.0),
        stain_probability=0.0,
        rotate_probability=0.0,
        rotate_degrees=0.0,
        blur_probability=0.0,
        blur_radius_range=(0.0, 0.0),
        noise_probability=0.0,
        noise_std_range=(0.0, 0.0),
        paper_texture_probability=0.0,
        paper_texture_std_range=(0.0, 0.0),
        scanline_probability=0.0,
        scanline_alpha_range=(0, 0),
        compression_probability=0.0,
        jpeg_quality_range=(100, 100),
    ),
    "held_out_scan_v1": AugmentationProfile(
        profile_id="held_out_scan_v1",
        fade_probability=0.75,
        fade_range=(0.40, 0.75),
        brightness_probability=0.80,
        brightness_range=(0.55, 1.10),
        paper_tint_probability=0.80,
        paper_tint_alpha_range=(0.12, 0.25),
        stain_probability=0.50,
        rotate_probability=0.90,
        rotate_degrees=5.5,
        blur_probability=0.85,
        blur_radius_range=(1.0, 2.4),
        noise_probability=0.80,
        noise_std_range=(10.0, 28.0),
        paper_texture_probability=0.75,
        paper_texture_std_range=(3.0, 9.0),
        scanline_probability=0.55,
        scanline_alpha_range=(8, 24),
        compression_probability=0.75,
        jpeg_quality_range=(38, 68),
    ),
}


def get_writer_style_profile(
        profile: str | WriterStyleProfile = DEFAULT_WRITER_PROFILE_ID,
) -> WriterStyleProfile:
    if isinstance(profile, WriterStyleProfile):
        return profile
    try:
        return WRITER_STYLE_PROFILES[profile]
    except (KeyError, TypeError):
        valid = ", ".join(sorted(WRITER_STYLE_PROFILES))
        raise ValueError(f"Unknown writer-style profile {profile!r}; choose {valid}") from None


def get_augmentation_profile(
        profile: str | AugmentationProfile = DEFAULT_AUGMENTATION_PROFILE_ID,
) -> AugmentationProfile:
    if isinstance(profile, AugmentationProfile):
        return profile
    try:
        return AUGMENTATION_PROFILES[profile]
    except (KeyError, TypeError):
        valid = ", ".join(sorted(AUGMENTATION_PROFILES))
        raise ValueError(f"Unknown augmentation profile {profile!r}; choose {valid}") from None


def create_custom_augmentation_profile(
    *,
    fade_contrast: float = 0.65,
    paper_tint_alpha: float = 0.15,
    stain_prob: float = 0.35,
    rotate_deg: float = 4.5,
    blur_radius: float = 0.8,
    noise_std: float = 12.0,
    paper_texture_std: float = 3.5,
    scanline_alpha: int = 10,
    jpeg_quality: int = 75,
    profile_id: str = "custom_dev_v1",
) -> AugmentationProfile:
    """Dynamically construct a validated AugmentationProfile from developer UI controls.

    .. deprecated::
        Use :func:`augmentation_spec_from_legacy_values` with
        :func:`augmentation_profile_from_spec`, which give the same values
        with a content-addressed profile ID.
    """
    warnings.warn(
        "create_custom_augmentation_profile is deprecated; use "
        "augmentation_profile_from_spec(augmentation_spec_from_legacy_values(...))",
        DeprecationWarning,
        stacklevel=2,
    )
    fade_contrast = max(0.10, min(1.0, float(fade_contrast)))
    paper_tint_alpha = max(0.0, min(0.50, float(paper_tint_alpha)))
    stain_prob = max(0.0, min(1.0, float(stain_prob)))
    rotate_deg = max(0.0, min(15.0, float(rotate_deg)))
    blur_radius = max(0.0, min(5.0, float(blur_radius)))
    noise_std = max(0.0, min(50.0, float(noise_std)))
    paper_texture_std = max(0.0, min(20.0, float(paper_texture_std)))
    scanline_alpha = max(0, min(60, int(scanline_alpha)))
    jpeg_quality = max(10, min(100, int(jpeg_quality)))

    return AugmentationProfile(
        profile_id=profile_id,
        fade_probability=1.0 if fade_contrast < 0.98 else 0.0,
        fade_range=(fade_contrast, fade_contrast),
        brightness_probability=0.0,
        brightness_range=(1.0, 1.0),
        paper_tint_probability=1.0 if paper_tint_alpha > 0 else 0.0,
        paper_tint_alpha_range=(paper_tint_alpha, paper_tint_alpha),
        stain_probability=stain_prob,
        rotate_probability=1.0 if rotate_deg > 0 else 0.0,
        rotate_degrees=rotate_deg,
        blur_probability=1.0 if blur_radius > 0 else 0.0,
        blur_radius_range=(blur_radius, blur_radius),
        noise_probability=1.0 if noise_std > 0 else 0.0,
        noise_std_range=(noise_std, noise_std),
        paper_texture_probability=1.0 if paper_texture_std > 0 else 0.0,
        paper_texture_std_range=(paper_texture_std, paper_texture_std),
        scanline_probability=1.0 if scanline_alpha > 0 else 0.0,
        scanline_alpha_range=(scanline_alpha, scanline_alpha),
        compression_probability=1.0 if jpeg_quality < 100 else 0.0,
        jpeg_quality_range=(jpeg_quality, jpeg_quality),
    )


_AUGMENTATION_INTEGER_FIELDS = frozenset({
    "scanline_alpha_range", "jpeg_quality_range",
    "stain_count_range", "stain_opacity_range",
})


def canonical_json(values: Mapping) -> str:
    """Sorted, compact JSON used for content hashes."""
    return json.dumps(values, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def content_id(prefix: str, values: Mapping) -> str:
    """``<prefix>_<first 12 hex of sha256(canonical JSON)>``."""
    digest = hashlib.sha256(canonical_json(values).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:12]}"


def _normalized_number(value, integer: bool):
    if isinstance(value, (tuple, list)):
        return [_normalized_number(item, integer) for item in value]
    return int(value) if integer else float(value)


def _augmentation_values(fields: Mapping[str, object]) -> dict[str, object]:
    """Every profile value except the ID, with ints and floats normalized."""
    return {
        name: _normalized_number(value, name in _AUGMENTATION_INTEGER_FIELDS)
        for name, value in fields.items()
        if name != "profile_id"
    }


def _registered_augmentation_ids() -> dict[str, str]:
    return {
        canonical_json(_augmentation_values(asdict(profile))): profile_id
        for profile_id, profile in AUGMENTATION_PROFILES.items()
    }


def augmentation_profile_id(fields: Mapping[str, object]) -> str:
    """Registered ID when the values match a built-in, else a content ID."""
    values = _augmentation_values(fields)
    registered = _registered_augmentation_ids().get(canonical_json(values))
    return registered or content_id("custom", values)


def augmentation_profile_from_spec(spec: Mapping) -> AugmentationProfile:
    """Build a profile from a studio ``augmentation`` section.

    Missing params take the spec defaults; invalid or unknown values raise
    ``ValueError``.  The ID is content-addressed (or a built-in's ID when the
    values match it exactly).
    """
    data = studio_spec.require_valid_section("augmentation", spec)
    fields = studio_spec.section_fields("augmentation", data)
    return AugmentationProfile(profile_id=augmentation_profile_id(fields), **fields)


def augmentation_spec_from_profile(
        profile: str | AugmentationProfile) -> dict:
    """The complete studio ``augmentation`` section for a profile."""
    profile = get_augmentation_profile(profile)
    return studio_spec.section_from_fields("augmentation", asdict(profile))


def augmentation_spec_from_legacy_values(
    *,
    fade_contrast: float = 0.65,
    paper_tint_alpha: float = 0.15,
    stain_prob: float = 0.35,
    rotate_deg: float = 4.5,
    blur_radius: float = 0.8,
    noise_std: float = 12.0,
    paper_texture_std: float = 3.5,
    scanline_alpha: int = 10,
    jpeg_quality: int = 75,
) -> dict:
    """The ``augmentation`` section equivalent to the old single-value sliders.

    Uses the same clamps and on/off rules as
    :func:`create_custom_augmentation_profile`, so the resulting profile is
    equal to it field for field (ID aside).
    """
    fade = max(0.10, min(1.0, float(fade_contrast)))
    tint = max(0.0, min(0.50, float(paper_tint_alpha)))
    stain = max(0.0, min(1.0, float(stain_prob)))
    rotate = max(0.0, min(15.0, float(rotate_deg)))
    blur = max(0.0, min(5.0, float(blur_radius)))
    noise = max(0.0, min(50.0, float(noise_std)))
    texture = max(0.0, min(20.0, float(paper_texture_std)))
    scanline = max(0, min(60, int(scanline_alpha)))
    jpeg = max(10, min(100, int(jpeg_quality)))
    return {
        "fade": {"probability": 1.0 if fade < 0.98 else 0.0, "range": [fade, fade]},
        "brightness": {"probability": 0.0, "range": [1.0, 1.0]},
        "paper_tint": {"probability": 1.0 if tint > 0 else 0.0, "range": [tint, tint]},
        "stains": {"probability": stain, "count_range": [2, 6],
                   "opacity_range": [15, 45]},
        "rotation": {"probability": 1.0 if rotate > 0 else 0.0, "range": [0.0, rotate]},
        "blur": {"probability": 1.0 if blur > 0 else 0.0, "range": [blur, blur]},
        "noise": {"probability": 1.0 if noise > 0 else 0.0, "range": [noise, noise]},
        "paper_texture": {"probability": 1.0 if texture > 0 else 0.0,
                          "range": [texture, texture]},
        "scanlines": {"probability": 1.0 if scanline > 0 else 0.0,
                      "range": [scanline, scanline]},
        "jpeg": {"probability": 1.0 if jpeg < 100 else 0.0, "range": [jpeg, jpeg]},
    }


CUTOFF_SCHEMA_VERSION = 1
CUTOFF_METHOD = "ink_bbox_v1"
CUTOFF_SIDE_NAMES = ("top", "bottom", "left", "right")


@dataclass(frozen=True)
class CutoffSide:
    """Chance and fraction range for cutting one side of the ink box.

    Fractions are of the ink box: its height for top/bottom, its width for
    left/right.
    """

    probability: float = 0.0
    fraction_range: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        _probability("probability", self.probability)
        _, high = _range("fraction_range", self.fraction_range, minimum=0)
        if high >= 1:
            raise ValueError("fraction_range values must be below 1")

    @property
    def active(self) -> bool:
        return self.probability > 0


@dataclass(frozen=True)
class CutoffProfile:
    """Ink-relative character cutoff (method ``ink_bbox_v1``)."""

    profile_id: str
    top: CutoffSide = CutoffSide()
    bottom: CutoffSide = CutoffSide()
    left: CutoffSide = CutoffSide()
    right: CutoffSide = CutoffSide()
    max_ink_loss: float = 0.0
    short_label_chars: int = 3
    short_label_max_fraction: float = 0.10
    method: str = CUTOFF_METHOD

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise ValueError("profile_id cannot be empty")
        for name in CUTOFF_SIDE_NAMES:
            if not isinstance(getattr(self, name), CutoffSide):
                raise ValueError(f"{name} must be a CutoffSide")
        loss = _finite("max_ink_loss", self.max_ink_loss)
        if not 0 <= loss < 1:
            raise ValueError("max_ink_loss must be in [0, 1)")
        if (isinstance(self.short_label_chars, bool)
                or not isinstance(self.short_label_chars, Integral)
                or self.short_label_chars < 0):
            raise ValueError("short_label_chars must be a nonnegative integer")
        cap = _finite("short_label_max_fraction", self.short_label_max_fraction)
        if not 0 <= cap < 1:
            raise ValueError("short_label_max_fraction must be in [0, 1)")
        if self.method != CUTOFF_METHOD:
            raise ValueError(f"Unsupported cutoff method {self.method!r}")

    def sides(self) -> tuple[tuple[str, CutoffSide], ...]:
        return tuple((name, getattr(self, name)) for name in CUTOFF_SIDE_NAMES)

    @property
    def active(self) -> bool:
        """Whether any side can be cut (only then is an ink mask built)."""
        return any(side.active for _name, side in self.sides())

    def constrained_for_label(self, label: str) -> "CutoffProfile":
        """Short labels get no side cuts and capped vertical cuts."""
        if len(label.strip()) > self.short_label_chars:
            return self
        cap = self.short_label_max_fraction

        def capped(side: CutoffSide) -> CutoffSide:
            low, high = side.fraction_range
            return replace(side, fraction_range=(min(low, cap), min(high, cap)))

        return replace(self, top=capped(self.top), bottom=capped(self.bottom),
                       left=CutoffSide(), right=CutoffSide())

    def to_metadata(self) -> dict[str, object]:
        return {
            "cutoff_schema_version": CUTOFF_SCHEMA_VERSION,
            **asdict(self),
        }


def _cutoff_side(probability: float, low: float, high: float) -> CutoffSide:
    return CutoffSide(probability, (low, high))


CUTOFF_PROFILES: dict[str, CutoffProfile] = {
    "none": CutoffProfile(profile_id="none"),
    "light": CutoffProfile(
        profile_id="light",
        top=_cutoff_side(0.50, 0.03, 0.08),
        bottom=_cutoff_side(0.50, 0.03, 0.08),
        max_ink_loss=0.10,
    ),
    "moderate": CutoffProfile(
        profile_id="moderate",
        top=_cutoff_side(0.60, 0.06, 0.14),
        bottom=_cutoff_side(0.60, 0.06, 0.14),
        left=_cutoff_side(0.15, 0.02, 0.05),
        right=_cutoff_side(0.15, 0.02, 0.05),
        max_ink_loss=0.15,
    ),
    "heavy": CutoffProfile(
        profile_id="heavy",
        top=_cutoff_side(0.75, 0.10, 0.22),
        bottom=_cutoff_side(0.75, 0.10, 0.22),
        left=_cutoff_side(0.30, 0.03, 0.08),
        right=_cutoff_side(0.30, 0.03, 0.08),
        max_ink_loss=0.20,
    ),
}


def _cutoff_values(profile: CutoffProfile) -> dict[str, object]:
    """Every profile value except the ID, with numbers normalized."""
    values: dict[str, object] = {
        name: {"probability": float(side.probability),
               "fraction_range": [float(v) for v in side.fraction_range]}
        for name, side in profile.sides()
    }
    values.update({
        "max_ink_loss": float(profile.max_ink_loss),
        "short_label_chars": int(profile.short_label_chars),
        "short_label_max_fraction": float(profile.short_label_max_fraction),
        "method": profile.method,
    })
    return values


def cutoff_profile_id(profile: CutoffProfile) -> str:
    """Registered ID when the values match a built-in, else a content ID."""
    values = _cutoff_values(profile)
    for profile_id, registered in CUTOFF_PROFILES.items():
        if _cutoff_values(registered) == values:
            return profile_id
    return content_id("cutoff", values)


def _with_content_id(profile: CutoffProfile) -> CutoffProfile:
    return replace(profile, profile_id=cutoff_profile_id(profile))


def cutoff_profile_from_spec(spec: Mapping) -> CutoffProfile:
    """Build a profile from a studio ``cutoff`` section."""
    data = studio_spec.require_valid_section("cutoff", spec)
    sides = {
        name: CutoffSide(float(data[name]["probability"]),
                         tuple(float(v) for v in data[name]["range"]))
        for name in CUTOFF_SIDE_NAMES
    }
    return _with_content_id(CutoffProfile(
        profile_id="pending", max_ink_loss=float(data["max_ink_loss"]), **sides))


def cutoff_spec_from_profile(profile: str | CutoffProfile) -> dict:
    """The complete studio ``cutoff`` section for a profile."""
    profile = get_cutoff_profile(profile)
    data: dict[str, object] = {
        name: {"probability": float(side.probability),
               "range": [float(v) for v in side.fraction_range]}
        for name, side in profile.sides()
    }
    data["max_ink_loss"] = float(profile.max_ink_loss)
    return data


def get_cutoff_profile(profile: str | CutoffProfile = "none") -> CutoffProfile:
    if isinstance(profile, CutoffProfile):
        return profile
    try:
        return CUTOFF_PROFILES[profile]
    except (KeyError, TypeError):
        valid = ", ".join(CUTOFF_PROFILES)
        raise ValueError(f"Unknown cutoff profile {profile!r}; choose {valid}") from None


def as_cutoff_profile(value) -> CutoffProfile:
    """Accept a profile, a registered ID, None, or a legacy 4-tuple.

    A ``(top, bottom, left, right)`` tuple of fractions becomes a custom
    profile that always cuts each nonzero side by exactly that fraction of
    the ink box, with the spec's default ink-loss limit.
    """
    if value is None or value == "":
        return CUTOFF_PROFILES["none"]
    if isinstance(value, (tuple, list)):
        if len(value) != 4:
            raise ValueError("A cutoff tuple needs (top, bottom, left, right)")
        sides = {}
        for name, raw in zip(CUTOFF_SIDE_NAMES, value):
            fraction = max(0.0, float(raw))
            sides[name] = (CutoffSide(1.0, (fraction, fraction)) if fraction > 0
                           else CutoffSide())
        default_loss = studio_spec.PARAMS_BY_PATH["cutoff.max_ink_loss"].default
        return _with_content_id(CutoffProfile(
            profile_id="pending", max_ink_loss=float(default_loss), **sides))
    return get_cutoff_profile(value)


def _style_seed(seed: int, writer_id: str, profile_id: str) -> tuple[int, str]:
    payload = (
        f"writer-style\0{PROFILE_SCHEMA_VERSION}\0{profile_id}\0{seed}\0{writer_id}"
        .encode("utf-8")
    )
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:16], "big"), digest.hex()[:16]


def writer_style_for(
        writer_id: str,
        seed: int,
        profile: str | WriterStyleProfile = DEFAULT_WRITER_PROFILE_ID,
) -> WriterStyle:
    """Derive a stable style without consuming the run's sample RNG."""
    if not isinstance(writer_id, str) or not writer_id.strip():
        raise ValueError("writer_id must be a nonempty string")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    selected = get_writer_style_profile(profile)
    derived_seed, digest_prefix = _style_seed(int(seed), writer_id, selected.profile_id)
    rng = random.Random(derived_seed)

    def uniform(values: tuple[float, float]) -> float:
        return rng.uniform(*values)

    return WriterStyle(
        profile_id=selected.profile_id,
        style_id=f"{selected.profile_id}:{digest_prefix}",
        writer_id=writer_id,
        baseline_amplitude=uniform(selected.baseline_amplitude_range),
        baseline_period=uniform(selected.baseline_period_range),
        spacing=uniform(selected.spacing_range),
        connection_overlap=uniform(selected.connection_overlap_range),
        slant=uniform(selected.slant_range),
        stroke_width=rng.randint(*selected.stroke_width_range),
        pressure_bias=rng.randint(*selected.pressure_bias_range),
        pressure_variation=rng.randint(*selected.pressure_variation_range),
        ink_texture_density=uniform(selected.ink_texture_density_range),
        glyph_width_scale=uniform(selected.glyph_width_scale_range),
        font_size_scale=uniform(selected.font_size_scale_range),
        font_preference=rng.random(),
        baseline_phase=rng.uniform(0.0, math.tau),
    )
