import json
import tempfile
import unittest
from pathlib import Path

from src.provenance import (atomic_write_json, configuration_snapshot,
                            hash_dataset_images, hash_named_files, json_safe,
                            resource_hashes, sha256_file, source_hashes)


class ProvenanceTests(unittest.TestCase):
    def test_json_safe_snapshot_is_deterministic(self):
        class Settings:
            ROOT = Path("ignored")
            ALPHA = {"z": 2, "a": (1, Path("relative/file"))}
            VALUES = frozenset({"b", "a"})

        self.assertEqual(
            configuration_snapshot(Settings),
            {
                "ALPHA": {"a": [1, "relative/file"], "z": 2},
                "VALUES": ["a", "b"],
            },
        )
        self.assertEqual(json_safe(Path("a") / "b"), "a/b")

    def test_hashes_are_deterministic_and_content_sensitive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "a.txt"
            second = root / "b.txt"
            first.write_text("alpha", encoding="utf-8")
            second.write_text("beta", encoding="utf-8")

            one = hash_named_files([("b", second), ("a", first)])
            two = hash_named_files([("a", first), ("b", second)])
            self.assertEqual(one, two)
            second.write_text("changed", encoding="utf-8")
            self.assertNotEqual(one, hash_named_files([("a", first), ("b", second)]))
            self.assertEqual(len(sha256_file(first)), 64)

    def test_dataset_hash_includes_logical_split_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "train").mkdir()
            (root / "test").mkdir()
            (root / "train" / "same.png").write_bytes(b"image")
            first = hash_dataset_images(root, ("train", "test"))
            (root / "train" / "same.png").replace(root / "test" / "same.png")
            second = hash_dataset_images(root, ("train", "test"))
            self.assertNotEqual(first, second)

    def test_resource_hashes_and_atomic_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "nested").mkdir()
            (root / "nested" / "values.txt").write_text("x\n", encoding="utf-8")
            (root / ".gitkeep").write_text("ignored", encoding="utf-8")
            self.assertEqual(set(resource_hashes(root)), {"nested/values.txt"})

            output = root / "metadata.json"
            atomic_write_json(output, {"unicode": "Niño", "value": 1})
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["value"], 1)
            self.assertFalse(any(path.suffix == ".tmp" for path in root.iterdir()))

    def test_source_hashes_reject_escape_and_bind_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "module.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            before = source_hashes(root, ["module.py"])
            source.write_text("VALUE = 2\n", encoding="utf-8")
            self.assertNotEqual(before, source_hashes(root, ["module.py"]))
            with self.assertRaises(ValueError):
                source_hashes(root, ["../outside.py"])


if __name__ == "__main__":
    unittest.main()
