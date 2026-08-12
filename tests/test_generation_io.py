import csv
import random
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from src.generation_io import (
    AtomicCsvWriter,
    GenerationCounters,
    SyntheticWriterAssigner,
    iter_exact_assignments,
    iter_generation_plan,
    iter_weighted_choices,
    verify_sha256_sidecar,
    write_sha256_sidecar,
)


class GenerationIoTests(unittest.TestCase):
    def test_atomic_csv_streams_rows_and_publishes_on_success(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "manifest.csv"
            with AtomicCsvWriter(path, ("filename", "label")) as writer:
                writer.write({"filename": "a.png", "label": "A"})
                writer.write({"filename": "b.png", "label": "B"})
                self.assertFalse(path.exists())
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(writer.row_count, 2)
            self.assertEqual(rows[1], {"filename": "b.png", "label": "B"})

    def test_atomic_csv_removes_private_temp_on_error(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "manifest.csv"
            with self.assertRaisesRegex(RuntimeError, "injected"):
                with AtomicCsvWriter(path, ("filename",)) as writer:
                    writer.write({"filename": "a.png"})
                    raise RuntimeError("injected")
            self.assertFalse(path.exists())
            self.assertEqual(list(root.iterdir()), [])

    def test_streamed_exact_assignments_have_exact_counts_and_seed(self):
        counts = {"train": 8, "val": 1, "test": 2}
        first = list(iter_exact_assignments(
            counts, random.Random(42), order=("train", "val", "test")
        ))
        second = list(iter_exact_assignments(
            counts, random.Random(42), order=("train", "val", "test")
        ))
        self.assertEqual(first, second)
        self.assertEqual(Counter(first), Counter(counts))

    def test_weighted_iterator_is_order_independent_and_lazy(self):
        first = iter_weighted_choices({"b": 2, "a": 1}, 20, random.Random(5))
        second = iter_weighted_choices({"a": 1, "b": 2}, 20, random.Random(5))
        self.assertNotIsInstance(first, list)
        self.assertEqual(list(first), list(second))

    def test_generation_plan_streams_pairs_and_counters(self):
        plan = iter_generation_plan(
            7,
            {"name": 3, "date": 1},
            {"train": 4, "val": 2, "test": 1},
            random.Random(8),
        )
        self.assertNotIsInstance(plan, list)
        items = list(plan)
        self.assertEqual([item.index for item in items], list(range(1, 8)))
        self.assertEqual(Counter(item.split for item in items), {
            "train": 4, "val": 2, "test": 1,
        })

        counters = GenerationCounters()
        for item in items:
            counters.observe(split=item.split, field_type=item.field_type)
        self.assertEqual(counters.total, 7)
        self.assertEqual(sum(counters.field_types.values()), 7)

    def test_archive_sidecar_detects_tampering_and_refuses_collision(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / "dataset_001.zip"
            archive.write_bytes(b"archive-v1")
            sidecar = write_sha256_sidecar(archive)

            self.assertTrue(verify_sha256_sidecar(archive))
            self.assertTrue(sidecar.name.endswith(".zip.sha256"))
            with self.assertRaises(FileExistsError):
                write_sha256_sidecar(archive)
            archive.write_bytes(b"archive-v2")
            self.assertFalse(verify_sha256_sidecar(archive))

    def test_pseudo_writers_repeat_but_never_cross_splits(self):
        assigner = SyntheticWriterAssigner(samples_per_writer=2)
        train = [assigner.writer_id_for("train") for _ in range(5)]
        test = [assigner.writer_id_for("test") for _ in range(3)]

        self.assertEqual(train[0], train[1])
        self.assertNotEqual(train[1], train[2])
        self.assertTrue(set(train).isdisjoint(test))
        metadata = assigner.to_metadata()
        self.assertEqual(metadata["writers_by_split"], {"test": 2, "train": 3})
        self.assertTrue(metadata["split_exclusive"])


if __name__ == "__main__":
    unittest.main()
