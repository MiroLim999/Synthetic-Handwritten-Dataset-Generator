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
import math
import random
from dataclasses import asdict, dataclass
from numbers import Integral, Real


PROFILE_SCHEMA_VERSION = 1
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
        _range("blur_radius_range", self.blur_radius_range, minimum=0)
        _range("noise_std_range", self.noise_std_range, minimum=0)
        _range("paper_texture_std_range", self.paper_texture_std_range, minimum=0)
        for name, value, low, high in (
                ("scanline_alpha_range", self.scanline_alpha_range, 0, 255),
                ("jpeg_quality_range", self.jpeg_quality_range, 1, 100)):
            _range(name, value, minimum=low, maximum=high)
            if any(not isinstance(item, Integral) or isinstance(item, bool)
                   for item in value):
                raise ValueError(f"{name} values must be integers")

    def to_metadata(self) -> dict[str, object]:
        return {"profile_schema_version": PROFILE_SCHEMA_VERSION, **asdict(self)}


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
    """Dynamically construct a validated AugmentationProfile from developer UI controls."""
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
