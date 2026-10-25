"""Real, ink-relative character cutoff: engine, quality check, and sidecar."""

from __future__ import annotations

import contextlib
import csv
import io
import json
import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image, ImageDraw

import config
from src import cutoff as cutoff_engine
from src.augment import degrade_detailed
from src.build_splits import build
from src.cutoff import CUTOFF_ANNOTATIONS_FILENAME, plan_cut
from src.dataset_validation import assess_image_quality, validate_dataset
from src.generate_synthetic import generate, zip_dataset
from src.generation_profiles import (AUGMENTATION_PROFILES, CUTOFF_PROFILES,
                                     CutoffProfile, CutoffSide)
from src.handoff_validation import validate_handoff_files


def _ink_image(width=240, height=100, box=(40, 30, 199, 69)):
    image = Image.new("RGB", (width, height), "white")
    ImageDraw.Draw(image).rectangle(box, fill="black")
    return image


def _edge_ink(image):
    ink = np.asarray(image.convert("L")) < 128
    return {"top": int(ink[0].sum()), "bottom": int(ink[-1].sum()),
            "left": int(ink[:, 0].sum()), "right": int(ink[:, -1].sum())}


def _always(**sides):
    return CutoffProfile(
        profile_id="test",
        **{name: CutoffSide(1.0, (value, value)) for name, value in sides.items()},
        max_ink_loss=0.5,
    )


class CutoffEngineTests(unittest.TestCase):
    def test_top_cut_removes_ink_at_the_top_only(self):
        result = degrade_detailed(_ink_image(), augmentation_profile="clean",
                                  edge_clipping=_always(top=0.25),
                                  rng=random.Random(1))
        edges = _edge_ink(result.image)
        self.assertGreater(edges["top"], 0)
        self.assertEqual((edges["bottom"], edges["left"], edges["right"]), (0, 0, 0))
        self.assertEqual(result.cutoff.cut_sides, frozenset({"top"}))
        self.assertAlmostEqual(result.cutoff.ink_loss, 0.25)

    def test_ink_loss_is_bounded_and_shrinking_adds_no_draws(self):
        mask = np.asarray(cutoff_engine.ink_mask(_ink_image())) > 0
        fractions = {"top": 0.6, "bottom": 0.0, "left": 0.0, "right": 0.0}
        box, result = plan_cut(mask, fractions, max_ink_loss=0.2)
        # 0.6 -> 0.3 -> 0.15: the second halving fits under the limit.
        self.assertIsNotNone(box)
        self.assertLessEqual(result.ink_loss, 0.2)
        self.assertAlmostEqual(result.top, 0.15)

        _box, never = plan_cut(mask, fractions, max_ink_loss=0.01)
        self.assertFalse(never.is_cut)

        profile = CutoffProfile(profile_id="t", top=CutoffSide(1.0, (0.6, 0.6)),
                                max_ink_loss=0.01)
        rng, reference = random.Random(5), random.Random(5)
        cutoff_engine.apply_cut(_ink_image(), cutoff_engine.ink_mask(_ink_image()),
                                profile, rng)
        reference.random()
        reference.uniform(0.6, 0.6)
        self.assertEqual(rng.getstate(), reference.getstate())

    def test_blank_image_and_tiny_boxes_are_not_cut(self):
        blank = np.zeros((50, 50), dtype=bool)
        self.assertFalse(plan_cut(blank, {"top": 0.3, "bottom": 0, "left": 0,
                                          "right": 0}, 0.5)[1].is_cut)
        small = np.zeros((12, 12), dtype=bool)
        small[1:11, 1:11] = True
        box, _result = plan_cut(small, {"top": 0.3, "bottom": 0.3, "left": 0,
                                        "right": 0}, 0.9)
        if box is not None:
            self.assertGreaterEqual(box[3] - box[1], cutoff_engine.MIN_CROP_PX)

    def test_short_labels_get_no_side_cuts_and_capped_vertical_cuts(self):
        heavy = CUTOFF_PROFILES["heavy"]
        short = heavy.constrained_for_label(" 42 ")
        self.assertFalse(short.left.active or short.right.active)
        self.assertLessEqual(short.top.fraction_range[1], heavy.short_label_max_fraction)
        self.assertEqual(heavy.constrained_for_label("Juan"), heavy)

    def test_rotation_and_cut_stay_aligned(self):
        rotating = replace(AUGMENTATION_PROFILES["clean"], rotate_probability=1.0,
                           rotate_degrees=10.0, rotate_min_degrees=8.0)
        for seed in range(10):
            result = degrade_detailed(_ink_image(), augmentation_profile=rotating,
                                      edge_clipping=_always(bottom=0.2),
                                      rng=random.Random(seed))
            self.assertTrue(result.cutoff.is_cut, seed)
            edges = _edge_ink(result.image)
            self.assertGreater(edges["bottom"], 0)
            self.assertEqual(edges["top"], 0)
        with self.assertRaisesRegex(ValueError, "misaligned"):
            cutoff_engine.apply_cut(_ink_image(), Image.new("L", (10, 10)),
                                    _always(top=0.1), random.Random(0))

    def test_cutoff_none_builds_no_mask_and_makes_no_extra_draws(self):
        image = _ink_image()
        with mock.patch.object(cutoff_engine, "ink_mask",
                               side_effect=AssertionError("mask built")):
            first_rng = random.Random(8)
            first = degrade_detailed(image, augmentation_profile="historical_scan_v1",
                                     edge_clipping="none", rng=first_rng,
                                     np_rng=np.random.default_rng(8))
            second_rng = random.Random(8)
            second = degrade_detailed(image, augmentation_profile="historical_scan_v1",
                                      edge_clipping=None, rng=second_rng,
                                      np_rng=np.random.default_rng(8))
        self.assertEqual(first.image.tobytes(), second.image.tobytes())
        self.assertEqual(first_rng.getstate(), second_rng.getstate())
        self.assertFalse(first.cutoff.is_cut)


class CutAwareQualityTests(unittest.TestCase):
    def test_ink_is_allowed_only_at_cut_sides(self):
        image = _ink_image(box=(40, 0, 199, 69))  # ink touches the top edge
        _stats, errors, _warnings = assess_image_quality(image)
        self.assertTrue(any("edge" in error for error in errors))
        self.assertEqual(assess_image_quality(image, cut_sides=frozenset({"top"}))[1], [])
        _stats, errors, _warnings = assess_image_quality(
            image, cut_sides=frozenset({"bottom", "left"}))
        self.assertTrue(any("edge" in error for error in errors))

    def test_empty_cut_sides_match_the_previous_result(self):
        for box in ((40, 30, 199, 69), (0, 30, 199, 69), (40, 30, 239, 99)):
            image = _ink_image(box=box)
            self.assertEqual(assess_image_quality(image),
                             assess_image_quality(image, cut_sides=frozenset()))


class _DatasetCase(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.root = Path(self._temporary.name)
        patcher = mock.patch.object(config, "DATASETS_DIR", self.root / "datasets")
        patcher.start()
        self.addCleanup(patcher.stop)

    def _generate(self, name, **kwargs):
        options = {"seed": 11, "show_bar": False, "edge_clipping": "heavy",
                   "workers": 1}
        options.update(kwargs)
        with contextlib.redirect_stdout(io.StringIO()):
            return generate(24, dataset=name, **options)


class CutoffSidecarTests(_DatasetCase):
    def test_generated_sidecar_is_valid_and_bound_in_metadata(self):
        out_dir = self._generate("heavy")
        sidecar = out_dir / CUTOFF_ANNOTATIONS_FILENAME
        with sidecar.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertTrue(rows)
        metadata = json.loads((out_dir / "run-metadata.json").read_text("utf-8"))
        self.assertEqual(metadata["cutoff_annotations"], CUTOFF_ANNOTATIONS_FILENAME)
        self.assertEqual(metadata["effective_selection"]["edge_clipping"], "heavy")
        self.assertEqual(metadata["cutoff_profile"]["cutoff_schema_version"], 1)
        report = validate_dataset(out_dir, write_artifacts=False)
        self.assertTrue(report.valid, report.errors)
        self.assertEqual(report.statistics["cutoff"]["samples"], len(rows))
        self.assertLessEqual(report.statistics["cutoff"]["max_ink_loss"], 0.20)

        none_dir = self._generate("none", edge_clipping="none")
        self.assertFalse((none_dir / CUTOFF_ANNOTATIONS_FILENAME).exists())
        none_metadata = json.loads((none_dir / "run-metadata.json").read_text("utf-8"))
        self.assertIsNone(none_metadata["cutoff_annotations"])

    def test_validator_rejects_bad_sidecar_rows(self):
        out_dir = self._generate("bad_rows")
        sidecar = out_dir / CUTOFF_ANNOTATIONS_FILENAME
        original = sidecar.read_text(encoding="utf-8")
        header, first, *_rest = original.splitlines()
        cells = first.split(",")
        cases = {
            "orphan": [header, first.replace(cells[0], "syn_999999.png", 1)],
            "duplicate": [header, first, first],
            "out of range": [header, ",".join(cells[:2] + ["1.5"] + cells[3:])],
            "no cut": [header, ",".join(cells[:2] + ["0.0000"] * 5)],
            "malformed": [header, ",".join(cells[:3])],
            "wrong split": [header, ",".join([cells[0], "nosplit"] + cells[2:])],
        }
        for name, lines in cases.items():
            sidecar.write_text("\n".join(lines) + "\n", encoding="utf-8")
            report = validate_dataset(out_dir, write_artifacts=False)
            self.assertFalse(report.valid, name)
            self.assertTrue(any(CUTOFF_ANNOTATIONS_FILENAME in error
                                for error in report.errors), name)

    def test_handoff_rejects_sidecar_hash_mismatch_and_missing_file(self):
        out_dir = self._generate("hash")
        report = validate_dataset(out_dir, write_artifacts=False)
        hashes = {"manifest_sha256": report.manifest_sha256,
                  "images_sha256": report.images_sha256}
        validate_handoff_files(out_dir, **hashes)
        sidecar = out_dir / CUTOFF_ANNOTATIONS_FILENAME
        content = sidecar.read_bytes()
        sidecar.write_bytes(content + b"\n")
        with self.assertRaisesRegex(ValueError, "cutoff annotation hash mismatch"):
            validate_handoff_files(out_dir, **hashes)
        sidecar.unlink()
        with self.assertRaisesRegex(ValueError, "Missing cutoff-annotations.csv"):
            validate_handoff_files(out_dir, **hashes)

    def test_zip_and_real_merge_keep_the_sidecar_valid(self):
        real_dir = self.root / "real"
        (real_dir / "images").mkdir(parents=True)
        with (real_dir / "labels.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["filename", "label", "writer_id"])
            for index in range(1, 7):
                name = f"real_{index:03d}.png"
                writer.writerow([name, f"Real {index}", f"W{index:03d}"])
                _ink_image().save(real_dir / "images" / name)

        out_dir = self._generate("merge_me")
        with (mock.patch.object(config, "REAL_DIR", real_dir),
              contextlib.redirect_stdout(io.StringIO())):
            result = build(out_dir.name)
        self.assertEqual(result.copied, 6)
        report = validate_dataset(out_dir)
        self.assertTrue(report.valid, report.errors)
        with contextlib.redirect_stdout(io.StringIO()):
            archive = zip_dataset(out_dir)
        self.assertTrue(archive.is_file())

    def test_thread_and_process_paths_write_identical_sidecars(self):
        threaded = self._generate("threads", workers=1)
        processes = self._generate("processes", workers=2)
        self.assertEqual(
            (threaded / CUTOFF_ANNOTATIONS_FILENAME).read_bytes(),
            (processes / CUTOFF_ANNOTATIONS_FILENAME).read_bytes())
        self.assertEqual((threaded / "manifest.csv").read_bytes(),
                         (processes / "manifest.csv").read_bytes())


if __name__ == "__main__":
    unittest.main()
