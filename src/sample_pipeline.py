"""One sample, one attempt: shared by the generator and the studio preview.

:func:`attempt_sample` makes exactly one try (value, render, degrade,
test-split holdout, quality check).  :func:`generate_valid_sample` retries it
the way a run does.  Both make the same RNG calls in the same order as the
generator always has, so a preview tile is the image a run would write.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

import numpy as np
from PIL import Image

import config
from src import fields
from src.augment import degrade_detailed
from src.cutoff import CutoffResult
from src.dataset_validation import assess_image_quality
from src.generation_io import SyntheticWriterAssigner, iter_generation_plan
from src.generation_profiles import (AugmentationProfile, CutoffProfile,
                                     StrokeDamageProfile, WriterStyle)
from src.render import render_text

MAX_SAMPLE_ATTEMPTS = 20
# Attempts after this number drop stroke damage and cutoff so huge runs never
# fail on one stubborn sample.
SAFE_FALLBACK_AFTER_ATTEMPT = 15


@dataclass(frozen=True)
class PlannedSample:
    """One entry of a run plan: what to generate, where, and from which seed."""

    index: int
    field_type: str
    split: str
    file_name: str
    writer_id: str
    item_seed: int

    @property
    def relative_path(self) -> str:
        return f"{self.split}/{self.file_name}"


def item_seed_for(seed: int, index: int) -> int:
    """Per-sample seed; independent of worker count and scheduling."""
    return (seed + index * 1_000_003) & 0x7FFFFFFF


def iter_run_plan(count: int, seed: int, field_weights: Mapping[str, Real],
                  split_fractions, writer_assigner: SyntheticWriterAssigner,
                  ) -> Iterator[PlannedSample]:
    """The exact sample plan a run with these settings follows."""
    rng = random.Random(seed)
    split_counts = config.allocate_split_counts(count, split_fractions)
    for item in iter_generation_plan(count, field_weights, split_counts, rng,
                                     split_order=config.SPLIT_NAMES):
        yield PlannedSample(
            index=item.index,
            field_type=item.field_type,
            split=item.split,
            file_name=f"syn_{item.index:06d}.png",
            writer_id=writer_assigner.writer_id_for(item.split),
            item_seed=item_seed_for(seed, item.index),
        )


@dataclass(frozen=True)
class SampleAttempt:
    """The outcome of one attempt at one sample."""

    attempt: int
    label: str
    format_profile: str
    format_id: str
    font: str
    clean_image: Image.Image
    image: Image.Image
    cutoff: CutoffResult
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def attempt_sample(*, attempt: int, field_type: str, split: str, file_name: str,
                   names_dir: Path, font_style: str, cursive_group: str,
                   evaluation_policy, writer_style: WriterStyle,
                   augmentation_profile: AugmentationProfile,
                   rng: random.Random, np_rng: np.random.Generator,
                   damage_profile: str,
                   stroke_damage_profile: StrokeDamageProfile | None,
                   cutoff_profile: CutoffProfile) -> SampleAttempt:
    """Make one attempt at a sample and quality-check the result."""
    format_profile = evaluation_policy.format_profile_for_field(split, field_type)
    label, format_id = fields.make_value_with_format(
        field_type,
        rng=rng,
        names_dir=names_dir,
        format_profile=format_profile,
    )
    clean_image, font_used = render_text(
        label,
        font_style=font_style,
        cursive_group=cursive_group,
        specific_font="",
        rng=rng,
        writer_style=writer_style,
        font_pool=evaluation_policy.fonts_for_split(split),
    )

    if attempt > SAFE_FALLBACK_AFTER_ATTEMPT:
        current_damage_profile = "regular"
        current_stroke_damage = None
        current_cutoff: CutoffProfile | str = "none"
    else:
        current_damage_profile = damage_profile
        current_cutoff = cutoff_profile.constrained_for_label(label)
        # Short-label safeguard and progressive backoff for every
        # semi-broken run.
        current_stroke_damage = (
            stroke_damage_profile.constrained_for_label(label, attempt)
            if damage_profile == "semi_broken" and stroke_damage_profile is not None
            else None
        )

    degraded = degrade_detailed(
        clean_image,
        damage_profile=current_damage_profile,
        rng=rng,
        np_rng=np_rng,
        augmentation_profile=augmentation_profile,
        edge_clipping=current_cutoff,
        semi_broken_params=current_stroke_damage,
    )
    image = evaluation_policy.apply_degradation_holdout(
        degraded.image, split, sample_key=file_name
    )
    _statistics, errors, _warnings = assess_image_quality(
        image, cut_sides=degraded.cutoff.cut_sides)
    return SampleAttempt(
        attempt=attempt,
        label=label,
        format_profile=format_profile,
        format_id=format_id,
        font=font_used,
        clean_image=clean_image,
        image=image,
        cutoff=degraded.cutoff,
        errors=tuple(errors),
    )


def generate_valid_sample(*, field_type: str,
                          on_attempt: Callable[[SampleAttempt], None] | None = None,
                          **attempt_options) -> SampleAttempt:
    """Retry :func:`attempt_sample` until one passes the quality check.

    ``on_attempt`` sees every attempt (used for retry statistics).  Raises
    ``ValueError`` after :data:`MAX_SAMPLE_ATTEMPTS` failures.
    """
    failures: tuple[str, ...] = ()
    for attempt in range(1, MAX_SAMPLE_ATTEMPTS + 1):
        result = attempt_sample(attempt=attempt, field_type=field_type, **attempt_options)
        if on_attempt is not None:
            on_attempt(result)
        if result.ok:
            return result
        failures = result.errors
    detail = "; ".join(failures) or "unknown image-quality failure"
    raise ValueError(
        f"Could not produce a valid {field_type!r} sample after "
        f"{MAX_SAMPLE_ATTEMPTS} attempts: {detail}"
    )
