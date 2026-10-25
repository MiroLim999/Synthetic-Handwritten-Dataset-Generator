"""Studio preview: the real first samples of a planned run.

:func:`preview_samples` walks the same plan a run would (field and split
choices, writer IDs, item seeds, font pools, retries) and returns tiles that
are pixel-identical to the images the run writes.  :func:`retry_statistics`
measures how often the current settings make the quality check retry.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

import config
from src.generate_synthetic import (ResolvedRun, _preflight_generation,
                                    effective_fonts_for, resolve_run)
from src.generation_io import SyntheticWriterAssigner
from src.generation_profiles import (CUTOFF_SIDE_NAMES, StrokeDamageProfile,
                                     writer_style_for)
from src.sample_pipeline import (PlannedSample, SampleAttempt,
                                 generate_valid_sample, iter_run_plan)
from src.split_policy import (SyntheticEvaluationPolicy,
                              build_synthetic_evaluation_policy)
from src.studio_recipe import StudioRecipe


@dataclass(frozen=True)
class PreviewTile:
    """One previewed sample, exactly as the run would write it."""

    index: int
    path: str
    field_type: str
    split: str
    label: str = ""
    font: str = ""
    cut_sides: tuple[str, ...] = ()
    attempts: int = 0
    image: Image.Image | None = None
    clean_image: Image.Image | None = None
    error: str | None = None


@dataclass
class _RunContext:
    count: int
    seed: int
    resolved: ResolvedRun
    names_dir: Path
    evaluation_policy: SyntheticEvaluationPolicy
    writer_assigner: SyntheticWriterAssigner
    stroke_damage: StrokeDamageProfile | None


def _context(options: Mapping, recipe: StudioRecipe | None, count: int,
             seed: int) -> _RunContext:
    config.validate_config(count)
    resolved = resolve_run(recipe=recipe, **dict(options))
    sample_mode, names_dir = _preflight_generation(
        count, resolved.names_version, resolved.sample_mode, resolved.font_style,
        resolved.cursive_group, resolved.specific_font)
    fonts = effective_fonts_for(resolved.font_style, resolved.cursive_group,
                                resolved.specific_font)
    return _RunContext(
        count=count,
        seed=seed,
        resolved=resolved,
        names_dir=names_dir,
        evaluation_policy=build_synthetic_evaluation_policy(fonts, seed),
        writer_assigner=SyntheticWriterAssigner(
            samples_per_writer=resolved.samples_per_writer),
        stroke_damage=(resolved.stroke_damage if sample_mode.startswith("semi_broken")
                       else None),
    )


def _plan(context: _RunContext):
    resolved = context.resolved
    return iter_run_plan(context.count, context.seed, resolved.field_weights,
                         resolved.split_fractions, context.writer_assigner)


def _sample(context: _RunContext, planned: PlannedSample, on_attempt=None) -> SampleAttempt:
    resolved = context.resolved
    rng = random.Random(planned.item_seed)
    np_rng = np.random.default_rng(planned.item_seed)
    return generate_valid_sample(
        field_type=planned.field_type,
        split=planned.split,
        file_name=planned.file_name,
        names_dir=context.names_dir,
        font_style=resolved.font_style,
        cursive_group=resolved.cursive_group,
        evaluation_policy=context.evaluation_policy,
        writer_style=writer_style_for(planned.writer_id, context.seed,
                                      resolved.writer_profile),
        augmentation_profile=resolved.augmentation_profile,
        rng=rng,
        np_rng=np_rng,
        damage_profile=("semi_broken" if resolved.sample_mode.startswith("semi_broken")
                        else "regular"),
        stroke_damage_profile=context.stroke_damage,
        cutoff_profile=resolved.cutoff_profile,
        on_attempt=on_attempt,
    )


def preview_samples(options: Mapping, recipe: StudioRecipe | None = None, *,
                    count: int, seed: int, n: int, offset: int = 0,
                    split: str | None = None) -> list[PreviewTile]:
    """The ``n`` samples a run would produce, starting at ``offset``.

    ``options`` are :func:`src.generate_synthetic.generate` keyword arguments
    (sample mode, profiles, weights, splits, fonts ...).  With ``split`` set,
    only that split's samples are shown (offset counts within the split).
    A sample that cannot be produced becomes a tile with ``error`` set.
    """
    if split is not None and split not in config.SPLIT_NAMES:
        raise ValueError(f"Unknown split {split!r}")
    context = _context(options, recipe, count, seed)
    tiles: list[PreviewTile] = []
    skipped = 0
    for planned in _plan(context):
        if split is not None and planned.split != split:
            continue
        if skipped < offset:
            skipped += 1
            continue
        base = {"index": planned.index, "path": planned.relative_path,
                "field_type": planned.field_type, "split": planned.split}
        try:
            result = _sample(context, planned)
        except ValueError as exc:
            tiles.append(PreviewTile(**base, error=str(exc)))
        else:
            tiles.append(PreviewTile(
                **base, label=result.label, font=result.font,
                cut_sides=tuple(side for side in CUTOFF_SIDE_NAMES
                                if side in result.cutoff.cut_sides),
                attempts=result.attempt, image=result.image,
                clean_image=result.clean_image))
        if len(tiles) >= n:
            break
    return tiles


@dataclass(frozen=True)
class RetryStatistics:
    """How often first attempts fail the quality check, and why."""

    samples: int
    retried: int
    reasons: dict[str, int] = field(default_factory=dict)
    mean_attempts: float = 1.0

    def share(self, count: int) -> float:
        return count / self.samples if self.samples else 0.0

    def describe(self) -> str:
        reasons = ", ".join(
            f"{name} {self.share(self.reasons.get(name, 0)) * 100:.0f}%"
            for name in ("edge ink", "blank", "low contrast"))
        return (f"about {self.share(self.retried) * 100:.0f}% would be retried "
                f"({reasons}); {self.mean_attempts:.2f} attempts per sample")


def _reason(error: str) -> str:
    if "image edge" in error:
        return "edge ink"
    if "nearly blank" in error:
        return "blank"
    if "contrast" in error:
        return "low contrast"
    return "other"


def retry_statistics(options: Mapping, recipe: StudioRecipe | None = None, *,
                     count: int, seed: int, samples: int = 48) -> RetryStatistics:
    """Run the first ``samples`` planned samples and count retries."""
    context = _context(options, recipe, count, seed)
    reasons: Counter[str] = Counter()
    retried = attempts_total = measured = 0
    for planned in _plan(context):
        if measured >= samples:
            break
        attempts: list[SampleAttempt] = []
        try:
            _sample(context, planned, on_attempt=attempts.append)
        except ValueError:
            pass
        measured += 1
        attempts_total += len(attempts)
        first = attempts[0]
        if not first.ok:
            retried += 1
            for name in dict.fromkeys(_reason(error) for error in first.errors):
                reasons[name] += 1
    return RetryStatistics(
        samples=measured,
        retried=retried,
        reasons=dict(reasons),
        mean_attempts=attempts_total / measured if measured else 1.0,
    )
