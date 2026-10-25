"""Ink-relative character cutoff (method ``ink_bbox_v1``).

An ink mask is built from the clean render, rotated with the image, and the
cut lines are measured inward from the mask's bounding box, so a cut really
clips letters instead of trimming blank padding.  Sides that are not cut keep
their full extent and padding.  Ink loss is the share of mask pixels a cut
removes; a cut that loses too much ink is shrunk by halving every fraction
(no new random draws) and abandoned if it still loses too much.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from src.generation_profiles import CUTOFF_SIDE_NAMES, CutoffProfile

CUTOFF_ANNOTATIONS_FILENAME = "cutoff-annotations.csv"
CUTOFF_ANNOTATION_COLUMNS = (
    "filename", "split", "cut_top", "cut_bottom", "cut_left", "cut_right", "ink_loss",
)
INK_THRESHOLD = 200
MIN_CROP_PX = 10
MAX_SHRINK_STEPS = 3
_VERTICAL = frozenset({"top", "bottom"})


@dataclass(frozen=True)
class CutoffResult:
    """Applied cut per side (fraction of the ink box) and the ink lost."""

    top: float = 0.0
    bottom: float = 0.0
    left: float = 0.0
    right: float = 0.0
    ink_loss: float = 0.0

    @property
    def cut_sides(self) -> frozenset[str]:
        return frozenset(side for side in CUTOFF_SIDE_NAMES if getattr(self, side) > 0)

    @property
    def is_cut(self) -> bool:
        return bool(self.cut_sides)


NO_CUT = CutoffResult()


def annotation_row(filename: str, split: str, result: CutoffResult) -> dict[str, str]:
    """One ``cutoff-annotations.csv`` row (values to 4 decimal places)."""
    return {
        "filename": filename,
        "split": split,
        **{f"cut_{side}": f"{getattr(result, side):.4f}" for side in CUTOFF_SIDE_NAMES},
        "ink_loss": f"{result.ink_loss:.4f}",
    }


def ink_mask(image: Image.Image) -> Image.Image:
    """Binary ``L`` mask (255 = ink) of pixels darker than ``INK_THRESHOLD``."""
    gray = np.asarray(image.convert("L"))
    return Image.fromarray(np.where(gray < INK_THRESHOLD, 255, 0).astype(np.uint8), "L")


def rotate_mask(mask: Image.Image, angle: float) -> Image.Image:
    """Rotate a mask exactly like the image (same angle, ``expand=True``)."""
    return mask.rotate(angle, expand=True, fillcolor=0, resample=Image.NEAREST)


def draw_fractions(profile: CutoffProfile, rng) -> dict[str, float]:
    """Draw the requested fraction for each side, in top/bottom/left/right order.

    Sides with zero chance make no draws, so an inactive profile consumes no
    randomness at all.
    """
    fractions = {}
    for name, side in profile.sides():
        fraction = 0.0
        if side.probability > 0 and rng.random() < side.probability:
            fraction = rng.uniform(*side.fraction_range)
        fractions[name] = fraction
    return fractions


def plan_cut(mask: np.ndarray, fractions: dict[str, float],
             max_ink_loss: float) -> tuple[tuple[int, int, int, int] | None, CutoffResult]:
    """Return the crop box and result for a boolean ink mask, or no cut."""
    height, width = mask.shape
    total = int(mask.sum())
    if total == 0 or not any(value > 0 for value in fractions.values()):
        return None, NO_CUT
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    extent = {"top": y1 - y0, "bottom": y1 - y0,
              "left": x1 - x0, "right": x1 - x0}

    for step in range(MAX_SHRINK_STEPS + 1):
        scale = 0.5 ** step
        offsets = {
            side: int(round(fractions[side] * scale * extent[side]))
            for side in CUTOFF_SIDE_NAMES
        }
        if not any(offsets.values()):
            return None, NO_CUT
        top = y0 + offsets["top"] if offsets["top"] else 0
        bottom = y1 - offsets["bottom"] if offsets["bottom"] else height
        left = x0 + offsets["left"] if offsets["left"] else 0
        right = x1 - offsets["right"] if offsets["right"] else width
        if bottom - top < MIN_CROP_PX or right - left < MIN_CROP_PX:
            continue
        kept = int(mask[top:bottom, left:right].sum())
        loss = 1.0 - kept / total
        if loss <= max_ink_loss:
            applied = {side: offsets[side] / extent[side] for side in CUTOFF_SIDE_NAMES}
            return (left, top, right, bottom), CutoffResult(ink_loss=loss, **applied)
    return None, NO_CUT


def apply_cut(image: Image.Image, mask: Image.Image, profile: CutoffProfile,
              rng) -> tuple[Image.Image, CutoffResult]:
    """Draw the cut for ``profile`` and apply it to ``image``."""
    fractions = draw_fractions(profile, rng)
    if image.size != mask.size:
        raise ValueError(f"Image {image.size} and ink mask {mask.size} are misaligned")
    box, result = plan_cut(np.asarray(mask) > 0, fractions, profile.max_ink_loss)
    if box is None:
        return image, NO_CUT
    return image.crop(box), result
