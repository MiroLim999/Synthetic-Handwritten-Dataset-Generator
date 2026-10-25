"""StrokeDamageProfile parity with the pre-profile stroke-damage code."""

from __future__ import annotations

import pickle
import random
import unittest
from unittest import mock

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

import config
from src import augment
from src.generation_profiles import (StrokeDamageProfile,
                                     as_stroke_damage_profile,
                                     stroke_damage_from_config,
                                     stroke_damage_from_spec,
                                     stroke_damage_spec_from_profile)


# --- Frozen copy of the stroke-damage code before StrokeDamageProfile ------
def _legacy_ink_gaps(img, rng, max_gap_count=None):
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    gray = np.asarray(img.convert("L"))
    ys, xs = np.where(gray < 210)
    if len(xs) == 0 or len(ys) == 0:
        return img
    min_gap, max_gap = config.BROKEN_INK_GAP_SIZE
    gap_count_range = (config.BROKEN_INK_GAP_COUNT if max_gap_count is None
                       else (1, max(1, int(max_gap_count))))
    count = rng.randint(*gap_count_range)
    for _ in range(count):
        w = rng.randint(min_gap, max_gap)
        h = rng.randint(1, max(2, max_gap // 2))
        idx = rng.randrange(len(xs))
        x, y = int(xs[idx]), int(ys[idx])
        fill = rng.choice([(255, 255, 255), (248, 242, 220), (238, 228, 200)])
        alpha = rng.randint(175, 255)
        box = [x - w // 2, y - h // 2, x + w // 2, y + h // 2]
        if rng.random() < 0.35:
            draw.ellipse(box, fill=fill + (alpha,))
        else:
            draw.rectangle(box, fill=fill + (alpha,))
    return img


def _legacy_scratches(img, rng):
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    _x0, y0, _x1, y1 = augment._ink_bbox(img)
    for _ in range(rng.randint(*config.BROKEN_SCRATCH_COUNT)):
        length = rng.randint(max(8, w // 5), max(10, w))
        angle = rng.uniform(-0.35, 0.35)
        sx = rng.randint(0, max(0, w - 1))
        sy = rng.randint(max(0, y0 - 3), min(h - 1, max(y0, y1 + 3)))
        ex, ey = sx + int(length), sy + int(length * angle)
        fill = rng.choice([(250, 246, 226), (238, 230, 205), (225, 214, 185)])
        draw.line((sx, sy, ex, ey), fill=fill + (rng.randint(120, 230),),
                  width=rng.randint(1, 3))
    return img


def _legacy_semi_broken(img, rng, params=None):
    params = params or {}
    if rng.random() < params.get("contrast_prob", config.BROKEN_CONTRAST_PROB):
        img = ImageEnhance.Contrast(img).enhance(
            rng.uniform(*config.BROKEN_CONTRAST_RANGE))
    if rng.random() < params.get("erode_prob", config.BROKEN_ERODE_PROB):
        eroded = img.filter(ImageFilter.MaxFilter(3))
        img = Image.blend(img.convert("RGB"), eroded.convert("RGB"),
                          rng.uniform(*config.BROKEN_ERODE_BLEND))
    if rng.random() < params.get("gap_prob", config.BROKEN_INK_GAP_PROB):
        img = _legacy_ink_gaps(img, rng, params.get("gap_count"))
    if rng.random() < params.get("scratch_prob", config.BROKEN_SCRATCH_PROB):
        img = _legacy_scratches(img, rng)
    return img


def _legacy_safeguard(params, label, attempt):
    """The dict safeguard formerly inlined in _generate_valid_sample."""
    current = dict(params) if params else {}
    if len(label.strip()) <= 3:
        max_gaps = max(1, len(label.strip()) * 3)
        if "gap_count" in current:
            current["gap_count"] = min(current["gap_count"], max_gaps)
        if "erode_prob" in current:
            current["erode_prob"] = min(current.get("erode_prob", 0.5), 0.35)
    if attempt > 8 and current:
        scale = max(0.1, 1.0 - ((attempt - 8) * 0.1))
        if "gap_count" in current:
            current["gap_count"] = max(1, int(current["gap_count"] * scale))
        if "erode_prob" in current:
            current["erode_prob"] = current["erode_prob"] * scale
        if "gap_prob" in current:
            current["gap_prob"] = current["gap_prob"] * scale
    return current
# ---------------------------------------------------------------------------


def _ink_image(text="Maasin 1948"):
    image = Image.new("RGB", (240, 80), "white")
    ImageDraw.Draw(image).text((20, 25), text, fill="black")
    return image


GUI_DICTS = (
    {"contrast_prob": 0.8, "gap_prob": 0.9, "gap_count": 16, "scratch_prob": 0.7,
     "erode_prob": 0.55},
    {"contrast_prob": 1.0, "gap_prob": 1.0, "gap_count": 30, "scratch_prob": 1.0,
     "erode_prob": 1.0},
    {"contrast_prob": 0.0, "gap_prob": 0.08, "gap_count": 1, "scratch_prob": 0.09,
     "erode_prob": 0.63},
    {"contrast_prob": 0.5, "gap_prob": 0.5, "gap_count": 0, "scratch_prob": 0.5,
     "erode_prob": 0.2},
)


class StrokeDamageParityTests(unittest.TestCase):
    def test_config_defaults_are_byte_identical_to_the_previous_path(self):
        for seed in range(25):
            old = _legacy_semi_broken(_ink_image(), random.Random(seed))
            new = augment._semi_broken(_ink_image(), random.Random(seed))
            self.assertEqual(old.tobytes(), new.tobytes(), seed)
            via_profile = augment._semi_broken(
                _ink_image(), random.Random(seed), stroke_damage_from_config())
            self.assertEqual(old.tobytes(), via_profile.tobytes(), seed)

    def test_legacy_dicts_match_the_previous_path(self):
        partial = ({"gap_count": 5}, {"erode_prob": 1.0, "gap_prob": 1.0,
                                       "gap_count": 30})
        for params in GUI_DICTS + partial:
            profile = StrokeDamageProfile.from_legacy_params(params)
            for seed in range(10):
                old = _legacy_semi_broken(_ink_image(), random.Random(seed), params)
                new = augment._semi_broken(_ink_image(), random.Random(seed), profile)
                self.assertEqual(old.tobytes(), new.tobytes(), (params, seed))

    def test_degrade_accepts_dict_profile_or_none(self):
        params = GUI_DICTS[1]
        outputs = []
        for value in (params, StrokeDamageProfile.from_legacy_params(params)):
            outputs.append(augment.degrade(
                _ink_image(), "semi_broken", rng=random.Random(4),
                np_rng=np.random.default_rng(4),
                augmentation_profile="historical_scan_v1",
                semi_broken_params=value).tobytes())
        self.assertEqual(outputs[0], outputs[1])
        self.assertIsInstance(augment.degrade(
            _ink_image(), "semi_broken", rng=random.Random(4),
            semi_broken_params=None), Image.Image)

    def test_constrained_for_label_matches_the_old_safeguard(self):
        for params in GUI_DICTS:
            profile = StrokeDamageProfile.from_legacy_params(params)
            for label in ("A", "12", "abc", " ab ", "Juan", "Juan Dela Cruz"):
                for attempt in range(1, 16):
                    expected = StrokeDamageProfile.from_legacy_params(
                        _legacy_safeguard(params, label, attempt))
                    self.assertEqual(
                        profile.constrained_for_label(label, attempt), expected,
                        (params, label, attempt))

    def test_short_labels_cap_both_ends_of_the_gap_range(self):
        profile = stroke_damage_from_config()  # gap_count_range (8, 24)
        constrained = profile.constrained_for_label("7", 1)
        self.assertEqual(constrained.gap_count_range, (3, 3))
        self.assertEqual(constrained.erode_probability, 0.35)
        self.assertEqual(profile.constrained_for_label("Juan", 1), profile)
        late = profile.constrained_for_label("Juan", 18)
        self.assertEqual(late.gap_count_range, (1, 2))
        self.assertAlmostEqual(late.gap_probability, config.BROKEN_INK_GAP_PROB * 0.1)


class StrokeDamageProfileTests(unittest.TestCase):
    def test_config_values_are_read_at_call_time(self):
        with mock.patch.multiple(config, BROKEN_INK_GAP_COUNT=(2, 3),
                                 BROKEN_SCRATCH_PROB=0.25):
            profile = stroke_damage_from_config()
            self.assertEqual(profile.gap_count_range, (2, 3))
            self.assertEqual(as_stroke_damage_profile(None).scratch_probability, 0.25)
        self.assertEqual(stroke_damage_from_config().gap_count_range,
                         config.BROKEN_INK_GAP_COUNT)

    def test_from_legacy_params_conversion(self):
        profile = StrokeDamageProfile.from_legacy_params({"gap_count": 7.9})
        self.assertEqual(profile.gap_count_range, (1, 7))
        self.assertEqual(profile.contrast_probability, config.BROKEN_CONTRAST_PROB)
        self.assertEqual(
            StrokeDamageProfile.from_legacy_params({"gap_count": 0}).gap_count_range,
            (1, 1))
        with self.assertRaisesRegex(ValueError, "Unknown stroke damage params"):
            StrokeDamageProfile.from_legacy_params({"gap_counts": 3})

    def test_validation_metadata_pickling_and_spec_round_trip(self):
        profile = StrokeDamageProfile.from_legacy_params(GUI_DICTS[0])
        self.assertEqual(pickle.loads(pickle.dumps(profile)), profile)
        metadata = profile.to_metadata()
        self.assertEqual(metadata["stroke_damage_schema_version"], 1)
        self.assertRegex(metadata["profile_id"], r"^stroke_[0-9a-f]{12}$")
        self.assertEqual(stroke_damage_from_spec(
            stroke_damage_spec_from_profile(profile)), profile)
        for bad in ({"gap_probability": 1.5}, {"contrast_range": (0.0, 0.5)},
                    {"gap_count_range": (0, 3)}, {"scratch_count_range": (2.5, 3)}):
            values = {**profile.__dict__, **bad}
            with self.assertRaises(ValueError, msg=str(bad)):
                StrokeDamageProfile(**values)


if __name__ == "__main__":
    unittest.main()
