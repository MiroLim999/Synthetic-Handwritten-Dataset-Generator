"""Deterministic held-out-domain policy for synthetic evaluation splits.

The policy deliberately keeps validation in distribution for model selection
and reserves test for an out-of-domain check.  When two or more fonts are
available, complete font files are held out from train/validation and used only
for test.  Test images may additionally receive a low-resolution/strong-blur
post-process that is absent from train and validation.

The version-1 manifest has a fixed schema, so per-row evaluation annotations
are returned as sidecar-ready dictionaries by :func:`evaluation_annotations`.
The full policy definition is serializable through :meth:`to_metadata` for the
run metadata and evaluation report.
"""

from __future__ import annotations

import hashlib
import math
import os
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageFilter

from src.fields import (BASE_FORMAT_PROFILE, DATE_FORMAT_FIELD_TYPES,
                        DATE_FORMAT_PATTERN_IDS, HELD_OUT_DATE_FORMAT_PROFILE)


POLICY_VERSION = "1"
BASE_DEGRADATION_PROFILE = "base"
HELD_OUT_DEGRADATION_PROFILE = "held_out_low_resolution_blur_v1"
HELD_OUT_SCALE_RANGE = (0.55, 0.70)
HELD_OUT_BLUR_RADIUS_RANGE = (2.0, 2.6)

EVALUATION_ANNOTATION_COLUMNS = (
    "filename",
    "split",
    "evaluation_condition",
    "format_profile",
    "format_id",
    "label_seen_in_train",
    "font_seen_in_train",
    "format_seen_in_train",
)


def _require_split(split: str) -> str:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unknown split {split!r}; expected train, val, or test")
    return split


def _font_key(font: str) -> str:
    """Portable identity for a font path used for ordering and disjointness."""
    return os.path.normpath(str(font)).replace("\\", "/").casefold()


def _font_display(font: str) -> str:
    """Manifest-compatible font filename for policy metadata."""
    return Path(font).name


def _derived_seed(seed: int, namespace: str) -> int:
    """Derive stable independent randomness without Python's salted hash()."""
    digest = hashlib.sha256(f"{POLICY_VERSION}\0{seed}\0{namespace}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _normalise_fonts(fonts: Sequence[str]) -> tuple[str, ...]:
    if isinstance(fonts, (str, bytes)):
        raise ValueError("fonts must be a sequence of font paths, not one string")
    by_key: dict[str, str] = {}
    by_filename: dict[str, str] = {}
    for raw_font in fonts:
        font = str(raw_font).strip()
        if not font:
            raise ValueError("Font paths cannot be empty")
        key = _font_key(font)
        filename_key = _font_display(font).casefold()
        if key in by_key:
            continue
        if filename_key in by_filename and _font_key(by_filename[filename_key]) != key:
            raise ValueError(
                "Font filenames must be unique because the manifest records only "
                f"the filename: {_font_display(font)!r}"
            )
        by_key[key] = font
        by_filename[filename_key] = font
    if not by_key:
        raise ValueError("At least one font is required")
    return tuple(by_key[key] for key in sorted(by_key))


@dataclass(frozen=True)
class SyntheticEvaluationPolicy:
    """Immutable font/degradation policy for train, validation, and test."""

    seed: int
    train_fonts: tuple[str, ...]
    val_fonts: tuple[str, ...]
    test_fonts: tuple[str, ...]
    font_holdout_enabled: bool
    degradation_holdout_enabled: bool
    format_holdout_enabled: bool
    requested_font_holdout_fraction: float

    def fonts_for_split(self, split: str) -> tuple[str, ...]:
        """Return the only font pool permitted for ``split``."""
        split = _require_split(split)
        return {
            "train": self.train_fonts,
            "val": self.val_fonts,
            "test": self.test_fonts,
        }[split]

    def choose_font(self, split: str, rng) -> str:
        """Choose from the permitted split pool using an explicit run RNG."""
        if rng is None or not hasattr(rng, "choice"):
            raise TypeError("choose_font requires an explicit random-compatible RNG")
        return rng.choice(self.fonts_for_split(split))

    def degradation_profile_for_split(self, split: str) -> str:
        """Return the named post-degradation profile for ``split``."""
        split = _require_split(split)
        if split == "test" and self.degradation_holdout_enabled:
            return HELD_OUT_DEGRADATION_PROFILE
        return BASE_DEGRADATION_PROFILE

    def format_profile_for_field(self, split: str, field_type: str) -> str:
        """Return the format domain allowed for one field in one split.

        Only explicit date fields receive the held-out format profile, and only
        in test. Train and validation always use the original base patterns.
        """
        split = _require_split(split)
        if (split == "test" and self.format_holdout_enabled
                and field_type in DATE_FORMAT_FIELD_TYPES):
            return HELD_OUT_DATE_FORMAT_PROFILE
        return BASE_FORMAT_PROFILE

    def evaluation_condition_for_split(self, split: str) -> str:
        """Return a stable metric/reporting label for the split's domain."""
        split = _require_split(split)
        if split == "train":
            return "synthetic_training"
        if split == "val":
            return "synthetic_in_distribution"
        holdouts = []
        if self.font_holdout_enabled:
            holdouts.append("unseen_font")
        if self.degradation_holdout_enabled:
            holdouts.append("held_out_degradation")
        return "synthetic_" + ("+".join(holdouts) if holdouts else "in_distribution")

    def evaluation_condition_for_sample(self, split: str, field_type: str) -> str:
        """Return the row-specific domain label, including date format holdout."""
        split = _require_split(split)
        if split != "test":
            return self.evaluation_condition_for_split(split)
        holdouts = []
        if self.font_holdout_enabled:
            holdouts.append("unseen_font")
        if self.degradation_holdout_enabled:
            holdouts.append("held_out_degradation")
        if self.format_profile_for_field(split, field_type) != BASE_FORMAT_PROFILE:
            holdouts.append("held_out_format")
        return "synthetic_" + ("+".join(holdouts) if holdouts else "in_distribution")

    def apply_degradation_holdout(
        self,
        image: Image.Image,
        split: str,
        *,
        sample_key: str,
    ) -> Image.Image:
        """Apply the test-only degradation deterministically per sample.

        This is a post-process for the generator's normal augmentation chain.
        Train and validation images are returned unchanged.  Test images are
        downsampled/upscaled and blurred beyond the normal configured blur
        radius, representing an explicitly held-out low-resolution scan domain.
        ``sample_key`` should be the stable output filename or sample ID.
        """
        split = _require_split(split)
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL Image")
        if not isinstance(sample_key, str) or not sample_key:
            raise ValueError("sample_key must be a nonempty stable string")
        if self.degradation_profile_for_split(split) == BASE_DEGRADATION_PROFILE:
            return image

        rng = random.Random(_derived_seed(self.seed, f"degradation\0{split}\0{sample_key}"))
        scale = rng.uniform(*HELD_OUT_SCALE_RANGE)
        reduced_size = (
            max(1, round(image.width * scale)),
            max(1, round(image.height * scale)),
        )
        resampling = getattr(Image, "Resampling", Image)
        degraded = image.convert("RGB").resize(reduced_size, resampling.LANCZOS)
        degraded = degraded.resize(image.size, resampling.BILINEAR)
        return degraded.filter(
            ImageFilter.GaussianBlur(rng.uniform(*HELD_OUT_BLUR_RADIUS_RANGE))
        )

    def to_metadata(self) -> dict[str, object]:
        """Return a JSON-serializable definition for run provenance/reports."""
        return {
            "policy_version": POLICY_VERSION,
            "seed": self.seed,
            "font_holdout": {
                "enabled": self.font_holdout_enabled,
                "requested_test_fraction": self.requested_font_holdout_fraction,
                "train_fonts": [_font_display(font) for font in self.train_fonts],
                "val_fonts": [_font_display(font) for font in self.val_fonts],
                "test_fonts": [_font_display(font) for font in self.test_fonts],
                "test_fonts_seen_in_train": bool(
                    {_font_key(font) for font in self.train_fonts}
                    & {_font_key(font) for font in self.test_fonts}
                ),
            },
            "degradation_holdout": {
                "enabled": self.degradation_holdout_enabled,
                "profiles": {
                    split: self.degradation_profile_for_split(split)
                    for split in ("train", "val", "test")
                },
                "test_scale_range": list(HELD_OUT_SCALE_RANGE),
                "test_blur_radius_range": list(HELD_OUT_BLUR_RADIUS_RANGE),
            },
            "format_holdout": {
                "enabled": self.format_holdout_enabled,
                "fields": sorted(DATE_FORMAT_FIELD_TYPES),
                "profiles": {
                    split: {
                        field_type: self.format_profile_for_field(split, field_type)
                        for field_type in sorted(DATE_FORMAT_FIELD_TYPES)
                    }
                    for split in ("train", "val", "test")
                },
                "pattern_ids": {
                    field_type: {
                        profile: list(pattern_ids)
                        for profile, pattern_ids in profiles.items()
                    }
                    for field_type, profiles in DATE_FORMAT_PATTERN_IDS.items()
                },
                "empirically_calibrated": False,
                "note": (
                    "Representative synthetic punctuation/order variants for "
                    "domain testing; frequencies are not corpus estimates."
                ),
            },
            "evaluation_conditions": {
                split: self.evaluation_condition_for_split(split)
                for split in ("train", "val", "test")
            },
            "test_evaluation_conditions_by_field": {
                field_type: self.evaluation_condition_for_sample("test", field_type)
                for field_type in sorted(DATE_FORMAT_FIELD_TYPES)
            },
        }


def build_synthetic_evaluation_policy(
    fonts: Sequence[str],
    seed: int,
    *,
    test_font_fraction: float = 0.20,
    enable_font_holdout: bool = True,
    enable_degradation_holdout: bool = True,
    enable_format_holdout: bool = True,
) -> SyntheticEvaluationPolicy:
    """Build a deterministic policy independent of input font ordering.

    With at least two unique fonts, ``ceil(n * test_font_fraction)`` complete
    fonts (capped so train retains at least one) are assigned exclusively to
    test. Validation deliberately shares the train pool so it remains suitable
    for model selection. With one font, font holdout is disabled explicitly;
    the test degradation holdout may still provide an out-of-domain check.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed must be an integer; got {seed!r}")
    if (isinstance(test_font_fraction, bool)
            or not isinstance(test_font_fraction, (int, float))
            or not math.isfinite(float(test_font_fraction))
            or not 0.0 < float(test_font_fraction) < 1.0):
        raise ValueError("test_font_fraction must be finite and strictly between 0 and 1")

    normalised = _normalise_fonts(fonts)
    holdout_enabled = bool(enable_font_holdout and len(normalised) >= 2)
    if holdout_enabled:
        shuffled = list(normalised)
        random.Random(_derived_seed(seed, "font_holdout")).shuffle(shuffled)
        test_count = min(len(shuffled) - 1, max(1, math.ceil(
            len(shuffled) * float(test_font_fraction)
        )))
        test_fonts = tuple(shuffled[:test_count])
        train_fonts = tuple(shuffled[test_count:])
    else:
        train_fonts = normalised
        test_fonts = normalised

    return SyntheticEvaluationPolicy(
        seed=seed,
        train_fonts=train_fonts,
        val_fonts=train_fonts,
        test_fonts=test_fonts,
        font_holdout_enabled=holdout_enabled,
        degradation_holdout_enabled=bool(enable_degradation_holdout),
        format_holdout_enabled=bool(enable_format_holdout),
        requested_font_holdout_fraction=float(test_font_fraction),
    )


def evaluation_annotations(
    rows: Iterable[Mapping[str, object]],
    policy: SyntheticEvaluationPolicy,
) -> tuple[dict[str, object], ...]:
    """Build sidecar rows identifying domain and exact seen/unseen status.

    Labels are compared exactly because exact-match OCR uses the same criterion.
    Fonts use the policy's portable path identity.  These annotations belong in
    run metadata/evaluation artifacts while manifest schema v1 remains fixed.
    """
    materialised = [dict(row) for row in rows]
    train_labels = {
        str(row.get("label", ""))
        for row in materialised
        if row.get("split") == "train"
    }
    train_fonts = {
        _font_key(str(row.get("font", "")))
        for row in materialised
        if row.get("split") == "train" and str(row.get("font", ""))
    }

    annotations = []
    for row in materialised:
        split = _require_split(str(row.get("split", "")))
        filename = str(row.get("filename", ""))
        if not filename:
            raise ValueError("Every evaluation row requires a filename")
        label = str(row.get("label", ""))
        font = str(row.get("font", ""))
        field_type = str(row.get("field_type", ""))
        expected_profile = policy.format_profile_for_field(split, field_type)
        format_profile = str(row.get("format_profile", expected_profile))
        if format_profile != expected_profile:
            raise ValueError(
                f"Row {filename!r} uses format profile {format_profile!r}; "
                f"policy requires {expected_profile!r}"
            )
        format_id = str(row.get("format_id", ""))
        allowed_patterns = DATE_FORMAT_PATTERN_IDS.get(field_type, {}).get(
            format_profile, ()
        )
        if format_id and allowed_patterns and format_id not in allowed_patterns:
            raise ValueError(
                f"Row {filename!r} uses format ID {format_id!r} outside "
                f"profile {format_profile!r}"
            )
        annotations.append({
            "filename": filename,
            "split": split,
            "evaluation_condition": policy.evaluation_condition_for_sample(
                split, field_type
            ),
            "format_profile": format_profile,
            "format_id": format_id,
            "label_seen_in_train": label in train_labels,
            "font_seen_in_train": bool(font and _font_key(font) in train_fonts),
            "format_seen_in_train": format_profile == BASE_FORMAT_PROFILE,
        })
    return tuple(annotations)
