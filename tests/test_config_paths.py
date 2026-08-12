import concurrent.futures
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config


class DatasetPathTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.datasets_dir = Path(self.temp_dir.name) / "datasets"
        self.patch = mock.patch.object(config, "DATASETS_DIR", self.datasets_dir)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_rejects_unsafe_names_without_touching_filesystem(self):
        unsafe = [
            "", "   ", ".", "..", "../x", "..\\..", "foo/bar", "foo\\bar",
            r"C:\Temp\x", "C:relative", "/tmp/x", r"\\server\share\x",
            "CON", "con.txt", "NUL", "LPT9.log", "-1", "0", "000",
            ".hidden", "trailing.", " leading", "trailing ", "bad:name",
        ]
        for name in unsafe:
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    config.resolve_dataset_dir(name)
                with self.assertRaises(ValueError):
                    config.reserve_dataset_dir(name)
                self.assertFalse(self.datasets_dir.exists())

    def test_resolves_positive_ids_and_safe_custom_names(self):
        expected_root = self.datasets_dir.resolve()
        self.assertEqual(
            config.resolve_dataset_dir(2), expected_root / "dataset_002"
        )
        self.assertEqual(
            config.resolve_dataset_dir("002"), expected_root / "dataset_002"
        )
        self.assertEqual(
            config.resolve_dataset_dir("my_test_run"), expected_root / "my_test_run"
        )
        self.assertFalse(self.datasets_dir.exists())

    def test_assert_safe_dataset_dir_requires_resolved_direct_child(self):
        safe = self.datasets_dir / "dataset_001"
        self.assertEqual(config.assert_safe_dataset_dir(safe), safe.resolve())
        for path in (
            self.datasets_dir,
            self.datasets_dir / "dataset_001" / "train",
            self.datasets_dir.parent / "outside",
            self.datasets_dir / ".." / "outside",
        ):
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    config.assert_safe_dataset_dir(path)

    def test_assert_safe_dataset_dir_rejects_symlink_escape(self):
        self.datasets_dir.mkdir(parents=True)
        outside = Path(self.temp_dir.name) / "outside"
        outside.mkdir()
        link = self.datasets_dir / "linked_run"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"Symlinks unavailable: {exc}")
        with self.assertRaises(ValueError):
            config.assert_safe_dataset_dir(link)

    def test_next_number_considers_directories_archives_and_reservations(self):
        self.datasets_dir.mkdir(parents=True)
        (self.datasets_dir / "dataset_003").mkdir()
        (self.datasets_dir / "dataset_013.zip").write_bytes(b"zip placeholder")
        (self.datasets_dir / ".dataset_021.reserve").write_text("claim", encoding="ascii")
        (self.datasets_dir / "custom_run").mkdir()

        self.assertEqual(config.next_dataset_dir().name, "dataset_022")
        self.assertEqual(
            [path.name for path in config.existing_datasets()], ["dataset_003"]
        )

    def test_zip_only_number_is_not_reused(self):
        self.datasets_dir.mkdir(parents=True)
        (self.datasets_dir / "dataset_013.zip").write_bytes(b"zip placeholder")
        self.assertEqual(config.next_dataset_dir().name, "dataset_014")

    def test_orphan_checksum_number_and_name_are_not_reused(self):
        self.datasets_dir.mkdir(parents=True)
        (self.datasets_dir / "dataset_013.zip.sha256").write_text(
            "digest  dataset_013.zip\n", encoding="ascii"
        )
        self.assertEqual(config.next_dataset_dir().name, "dataset_014")
        with self.assertRaises(FileExistsError):
            config.reserve_dataset_dir(13)

    def test_explicit_reservation_refuses_existing_output_or_archive(self):
        self.datasets_dir.mkdir(parents=True)
        output = self.datasets_dir / "dataset_002"
        output.mkdir()
        with self.assertRaises(FileExistsError):
            config.reserve_dataset_dir(2)

        archive = self.datasets_dir / "named_run.zip"
        archive.write_bytes(b"existing archive")
        with self.assertRaises(FileExistsError):
            config.reserve_dataset_dir("named_run")

    def test_reservation_leaves_final_path_absent_and_releases_marker(self):
        reservation = config.reserve_dataset_dir("safe_run")
        self.assertEqual(reservation.path, self.datasets_dir.resolve() / "safe_run")
        self.assertFalse(reservation.path.exists())
        self.assertTrue((self.datasets_dir / ".safe_run.reserve").is_file())

        reservation.release()
        reservation.release()
        self.assertTrue(reservation.released)
        self.assertFalse((self.datasets_dir / ".safe_run.reserve").exists())

    def test_reservation_is_a_context_manager(self):
        with config.reserve_dataset_dir(7) as reservation:
            self.assertEqual(reservation.path.name, "dataset_007")
            self.assertTrue((self.datasets_dir / ".dataset_007.reserve").exists())
        self.assertTrue(reservation.released)
        self.assertFalse((self.datasets_dir / ".dataset_007.reserve").exists())

    def test_simultaneous_automatic_reservations_are_unique(self):
        workers = 8

        def reserve(_):
            return config.reserve_dataset_dir()

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            reservations = list(executor.map(reserve, range(workers)))
        self.addCleanup(lambda: [item.release() for item in reservations])

        names = [item.path.name for item in reservations]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(
            set(names), {f"dataset_{number:03d}" for number in range(1, workers + 1)}
        )
        self.assertTrue(all(not item.path.exists() for item in reservations))


if __name__ == "__main__":
    unittest.main()
