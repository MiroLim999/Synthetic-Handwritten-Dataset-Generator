"""
Synthetic data generator.

Generates fake handwriting field images + labels and writes them STRAIGHT
into a numbered dataset folder, already divided into train / val / test:

  1. choose a field type (weighted by config.FIELD_WEIGHTS)
  2. generate a realistic value for it
  3. render it in a random handwriting font
  4. degrade it to look like an old scan
  5. assign it to train / val / test and save it there

Each run gets its own folder so nothing is ever overwritten:

  dataset/datasets/dataset_001/train/syn_000001.png ...
  dataset/datasets/dataset_001/val/...
  dataset/datasets/dataset_001/test/...
  dataset/datasets/dataset_001/labels.csv          (filename, label, split)
  dataset/datasets/dataset_001/manifest.csv        (filename, label, split, field_type, font, sample_mode)

Usage:
  python -m src.generate_synthetic --count 20000
  python -m src.generate_synthetic --count 5000 --dataset 2
  python -m src.generate_synthetic --count 200 --dataset my_test_run
  python -m src.generate_synthetic --count 2000 --mode semi_broken_mixed
"""

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import stat
import tempfile
import uuid
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

import config
from src import fields
from src.augment import degrade
from src.dataset_validation import (REPORT_FILENAME, assess_image_quality,
                                    validate_dataset, MONTAGE_FILENAME)
from src.dataset_mutation import (assert_no_sibling_archive,
                                  dataset_mutation_lock)
from src.handoff_validation import (validate_handoff_bytes,
                                    validate_handoff_files)
from src.image_preprocessing import (PREPROCESSING_ID, TROCR_IMAGE_SIZE,
                                     TROCR_PAD_COLOR)
from src.image_statistics import (compare_image_statistics, iter_image_paths,
                                  summarize_image_statistics)
from src.generation_io import (AtomicCsvWriter, GenerationCounters,
                               SyntheticWriterAssigner, iter_generation_plan,
                               verify_sha256_sidecar, write_sha256_sidecar)
from src.generation_profiles import (DEFAULT_AUGMENTATION_PROFILE_ID,
                                     DEFAULT_WRITER_PROFILE_ID,
                                     get_augmentation_profile,
                                     get_writer_style_profile,
                                     writer_style_for)
from src.generation_resources import preflight_generation_resources
from src.manifest import (MANIFEST_COLUMNS, MANIFEST_SCHEMA_VERSION,
                          RUN_METADATA_FILENAME)
from src.path_safety import is_reparse_point, snapshot_regular_tree
from src.provenance import (atomic_write_json, configuration_snapshot,
                            git_is_dirty, git_revision, installed_versions,
                            resource_hashes, sha256_file, source_hashes,
                            utc_now_iso)
from src.render import (CURSIVE_STYLE_GROUPS, FONT_STYLES, FONT_STYLE_ALL,
                        font_path_for, fonts_for_style, render_text)
from src.split_policy import (EVALUATION_ANNOTATION_COLUMNS,
                              build_synthetic_evaluation_policy)


EVALUATION_ANNOTATIONS_FILENAME = "evaluation-annotations.csv"
IMAGE_STATISTICS_FILENAME = "image-statistics.json"
STATISTICS_SAMPLE_LIMIT = 512
MAX_SAMPLE_ATTEMPTS = 5
GENERATOR_SOURCE_FILES = (
    "config.py",
    "src/augment.py",
    "src/dataset_validation.py",
    "src/fields.py",
    "src/generation_io.py",
    "src/generation_profiles.py",
    "src/generation_resources.py",
    "src/generate_synthetic.py",
    "src/image_preprocessing.py",
    "src/image_statistics.py",
    "src/manifest.py",
    "src/provenance.py",
    "src/render.py",
    "src/split_policy.py",
)
FORMAT_SOURCE_COLUMNS = (
    "filename", "field_type", "format_profile", "format_id",
)


class GenerationCancelled(RuntimeError):
    """Raised when a caller requests cancellation at a safe sample boundary."""


def _is_cancelled(cancel_event) -> bool:
    return bool(cancel_event is not None and cancel_event.is_set())


def _raise_if_cancelled(cancel_event) -> None:
    if _is_cancelled(cancel_event):
        raise GenerationCancelled("Dataset generation was cancelled.")


def _assert_internal_dataset_child(path) -> Path:
    """Require `path` to resolve to a direct child of the datasets directory."""
    base = config.DATASETS_DIR.resolve()
    candidate = Path(path)
    resolved = candidate.resolve(strict=False)
    if resolved == base or resolved.parent != base:
        raise ValueError(f"Unsafe dataset path outside the output root: {candidate}")
    return candidate


def guarded_remove_dataset(path, *, _allow_internal: bool = False) -> bool:
    """Safely remove one direct-child dataset or an internal staging folder."""
    candidate = _assert_internal_dataset_child(path)
    if not _allow_internal:
        candidate = config.assert_safe_dataset_dir(candidate)
    if not candidate.exists():
        return False
    if not candidate.is_dir():
        raise ValueError(f"Refusing to recursively remove a non-directory: {candidate}")
    if is_reparse_point(candidate):
        raise ValueError(f"Refusing to recursively remove a link or junction: {candidate}")
    for descendant in candidate.rglob("*"):
        if is_reparse_point(descendant):
            raise ValueError(
                "Refusing to recursively remove a tree containing a link or "
                f"junction: {descendant}"
            )
    shutil.rmtree(candidate)
    return True


def _require_nonempty_resource(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Resource list not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        if not any(line.strip() for line in handle):
            raise ValueError(f"Resource list is empty: {path}")


def _preflight_generation(count: int, names_version: str | None,
                          sample_mode: str, font_style: str,
                          cursive_group: str, specific_font: str) -> tuple[str, Path]:
    """Validate a run completely before reserving or creating output paths."""
    config.validate_config(count)

    sample_mode = _normalize_sample_mode(sample_mode)
    if font_style not in FONT_STYLES:
        raise ValueError(f"Unknown font style '{font_style}'. Valid styles: {', '.join(FONT_STYLES)}")
    if cursive_group and cursive_group not in CURSIVE_STYLE_GROUPS:
        raise ValueError(f"Unknown cursive group: {cursive_group}")

    version = names_version or config.NAMES_VERSION
    if version not in config.name_versions():
        raise ValueError(f"Unknown names version '{version}'. Available: {', '.join(config.name_versions())}")
    names_dir = config.RESOURCES_DIR / version

    for filename in ("first_names.txt", "middle_names.txt", "last_names.txt"):
        _require_nonempty_resource(names_dir / filename)
    _require_nonempty_resource(config.PLACES_FILE)
    for vocab_path in sorted(config.VOCAB_DIR.glob("*.txt")):
        _require_nonempty_resource(vocab_path)

    pool = fonts_for_style(font_style, cursive_group)
    if not pool:
        raise RuntimeError("The selected font pool is empty.")
    if specific_font:
        selected = font_path_for(specific_font)
        if selected is None:
            raise ValueError(f"Requested font is not available: {specific_font}")
        selected_resolved = Path(selected).resolve()
        if all(Path(candidate).resolve() != selected_resolved for candidate in pool):
            raise ValueError(
                f"Requested font {specific_font!r} is outside the selected font style/group"
            )

    return sample_mode, names_dir


def _normalize_sample_mode(sample_mode: str | None) -> str:
    """Return a valid sample mode, falling back to the configured default."""
    sample_mode = sample_mode or config.DEFAULT_SAMPLE_MODE
    if sample_mode not in config.SAMPLE_MODES:
        valid = ", ".join(config.SAMPLE_MODES)
        raise ValueError(f"Unknown sample mode '{sample_mode}'. Valid modes: {valid}")
    return sample_mode


def _field_weights_for_mode(sample_mode: str) -> dict[str, int]:
    """Field mix for regular and semi-broken generation modes."""
    if sample_mode == "regular":
        return dict(config.FIELD_WEIGHTS)

    if sample_mode == "semi_broken_characters":
        return {"character": 1}

    if sample_mode == "semi_broken_numerics":
        return {
            "date_numeric": config.FIELD_WEIGHTS.get("date_numeric", 8),
            "age": config.FIELD_WEIGHTS.get("age", 3),
            "numeric": 10,
        }

    if sample_mode == "semi_broken_words":
        numeric_types = {"date_numeric", "age"}
        return {
            k: v for k, v in config.FIELD_WEIGHTS.items()
            if k not in numeric_types
        }

    if sample_mode == "semi_broken_mixed":
        weights = dict(config.FIELD_WEIGHTS)
        weights["character"] = 10
        weights["numeric"] = 10
        return weights

    raise AssertionError(f"Unhandled sample mode: {sample_mode}")


def _weighted_field_types(n: int, sample_mode: str, rng=None) -> list[str]:
    """Build a list of n field-type choices following the selected mode."""
    rng = random if rng is None else rng
    mode_weights = _field_weights_for_mode(sample_mode)
    types = list(mode_weights.keys())
    weights = list(mode_weights.values())
    return rng.choices(types, weights=weights, k=n)


def _split_assignments(count: int, rng=None) -> list[str]:
    """
    Return a shuffled list of length `count` where each entry is
    'train', 'val' or 'test' following the configured fractions.
    """
    rng = random if rng is None else rng
    counts = config.allocate_synthetic_split_counts(count)
    assignments = [
        split
        for split in config.SPLIT_NAMES
        for _ in range(counts[split])
    ]
    rng.shuffle(assignments)
    return assignments


def _generate_valid_sample(field_type: str, split: str, file_name: str,
                           names_dir: Path, sample_mode: str,
                           font_style: str, cursive_group: str,
                           evaluation_policy, writer_style,
                           augmentation_profile,
                           rng: random.Random,
                           np_rng: np.random.Generator,
                           edge_clipping: str = "none"):
    """Generate one quality-checked crop, retrying stochastic failures."""
    damage_profile = (
        "semi_broken" if sample_mode.startswith("semi_broken") else "regular"
    )
    failures = []
    for _attempt in range(1, MAX_SAMPLE_ATTEMPTS + 1):
        format_profile = evaluation_policy.format_profile_for_field(
            split, field_type
        )
        label, format_id = fields.make_value_with_format(
            field_type,
            rng=rng,
            names_dir=names_dir,
            format_profile=format_profile,
        )
        image, font_used = render_text(
            label,
            font_style=font_style,
            cursive_group=cursive_group,
            specific_font="",
            rng=rng,
            writer_style=writer_style,
            font_pool=evaluation_policy.fonts_for_split(split),
        )
        image = degrade(
            image,
            damage_profile=damage_profile,
            rng=rng,
            np_rng=np_rng,
            augmentation_profile=augmentation_profile,
            edge_clipping=edge_clipping,
        )
        image = evaluation_policy.apply_degradation_holdout(
            image, split, sample_key=file_name
        )
        _statistics, errors, _warnings = assess_image_quality(image)
        if not errors:
            return label, image, font_used, format_profile, format_id
        failures = errors
    detail = "; ".join(failures) or "unknown image-quality failure"
    raise ValueError(
        f"Could not produce a valid {field_type!r} sample after "
        f"{MAX_SAMPLE_ATTEMPTS} attempts: {detail}"
    )


def _run_metadata(*, seed: int, count: int, sample_mode: str,
                  names_dir: Path, font_style: str, cursive_group: str,
                  specific_font: str, effective_fonts: tuple[str, ...],
                  split_counts: dict[str, int], manifest_path: Path,
                  validation_report, evaluation_policy,
                  generation_started_at: str, resource_estimate,
                  writer_profile, augmentation_profile,
                  counters: GenerationCounters, writer_assigner,
                  edge_clipping: str = "none") -> dict:
    """Build the complete provenance record written before publication."""
    return {
        "metadata_schema_version": 1,
        "generation_started_at": generation_started_at,
        "generation_completed_at": utc_now_iso(),
        "generator_revision": git_revision(config.ROOT),
        "generator_worktree_dirty": git_is_dirty(config.ROOT),
        "source_sha256": source_hashes(config.ROOT, GENERATOR_SOURCE_FILES),
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "seed": seed,
        "requested_count": count,
        "image_count": validation_report.statistics.get("images", 0),
        "row_count": validation_report.statistics.get("manifest_rows", 0),
        "split_counts": split_counts,
        "effective_selection": {
            "names_version": names_dir.name,
            "sample_mode": sample_mode,
            "font_style": font_style,
            "cursive_group": cursive_group,
            "specific_font": specific_font,
            "edge_clipping": edge_clipping,
            "font_files": [Path(path).name for path in effective_fonts],
        },
        "preprocessing": {
            "id": PREPROCESSING_ID,
            "canvas_size": TROCR_IMAGE_SIZE,
            "pad_color_rgb": list(TROCR_PAD_COLOR),
            "aspect_preserving": True,
        },
        "evaluation_policy": evaluation_policy.to_metadata(),
        "evaluation_annotations": EVALUATION_ANNOTATIONS_FILENAME,
        "evaluation_annotations_sha256": sha256_file(
            manifest_path.parent / EVALUATION_ANNOTATIONS_FILENAME
        ),
        "image_statistics": IMAGE_STATISTICS_FILENAME,
        "writer_style_profile": writer_profile.to_metadata(),
        "augmentation_profile": augmentation_profile.to_metadata(),
        "generation_resources": resource_estimate.to_metadata(),
        "generation_counters": counters.to_metadata(),
        "synthetic_writer_assignment": writer_assigner.to_metadata(),
        "configuration": configuration_snapshot(config),
        "resource_sha256": resource_hashes(config.RESOURCES_DIR),
        "dependencies": installed_versions(("numpy", "Pillow", "tqdm")),
        "manifest_sha256": sha256_file(manifest_path),
        "images_sha256": validation_report.images_sha256,
        "validation_report": REPORT_FILENAME,
        "real_merge": None,
    }


def _write_evaluation_annotations_streaming(
        manifest_path: Path, format_source_path: Path, output_path: Path,
        evaluation_policy) -> None:
    """Build the evaluation sidecar without retaining manifest rows in RAM."""
    train_labels: set[str] = set()
    train_fonts: set[str] = set()
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == "train":
                train_labels.add(row["label"])
                if row["font"]:
                    train_fonts.add(row["font"].casefold())

    with (manifest_path.open("r", newline="", encoding="utf-8") as manifest_handle,
          format_source_path.open("r", newline="", encoding="utf-8") as format_handle,
          AtomicCsvWriter(output_path, EVALUATION_ANNOTATION_COLUMNS) as output):
        manifest_rows = csv.DictReader(manifest_handle)
        format_rows = csv.DictReader(format_handle)
        row_count = 0
        for manifest_row, format_row in zip(manifest_rows, format_rows):
            row_count += 1
            if manifest_row["filename"] != format_row["filename"]:
                raise ValueError(
                    "Manifest/evaluation-source row order mismatch at "
                    f"row {row_count + 1}"
                )
            split = manifest_row["split"]
            field_type = manifest_row["field_type"]
            expected_profile = evaluation_policy.format_profile_for_field(
                split, field_type
            )
            if format_row["format_profile"] != expected_profile:
                raise ValueError(
                    f"Unexpected format profile for {manifest_row['filename']}: "
                    f"{format_row['format_profile']!r}"
                )
            output.write({
                "filename": manifest_row["filename"],
                "split": split,
                "evaluation_condition":
                    evaluation_policy.evaluation_condition_for_sample(
                        split, field_type
                    ),
                "format_profile": format_row["format_profile"],
                "format_id": format_row["format_id"],
                "label_seen_in_train": manifest_row["label"] in train_labels,
                "font_seen_in_train": (
                    bool(manifest_row["font"])
                    and manifest_row["font"].casefold() in train_fonts
                ),
                "format_seen_in_train": (
                    format_row["format_profile"] == fields.BASE_FORMAT_PROFILE
                ),
            })
        if next(manifest_rows, None) is not None or next(format_rows, None) is not None:
            raise ValueError("Manifest/evaluation-source row counts do not match")


def _write_image_statistics_artifact(dataset_dir: Path) -> None:
    """Write bounded descriptive generated/real image statistics."""
    per_split = max(1, STATISTICS_SAMPLE_LIMIT // len(config.SPLIT_NAMES))
    generated_paths = []
    for split in config.SPLIT_NAMES:
        generated_paths.extend(
            sorted((dataset_dir / split).glob("*.png"), key=lambda path: path.name)
            [:per_split]
        )
    generated_paths = generated_paths[:STATISTICS_SAMPLE_LIMIT]
    real_root = config.REAL_DIR / "images"
    real_paths = (
        list(iter_image_paths(real_root))[:STATISTICS_SAMPLE_LIMIT]
        if real_root.is_dir() else []
    )
    if real_paths:
        value = compare_image_statistics(generated_paths, real_paths).to_metadata()
        value["comparison_available"] = True
    else:
        value = {
            "statistics_schema_version": 1,
            "comparison_available": False,
            "generated": summarize_image_statistics(generated_paths).to_metadata(),
            "real": None,
            "interpretation": (
                "No real image folder was available; generated statistics are "
                "descriptive only and are not evidence of domain realism."
            ),
        }
    value["sample_limit_per_group"] = STATISTICS_SAMPLE_LIMIT
    atomic_write_json(dataset_dir / IMAGE_STATISTICS_FILENAME, value)


def generate(count: int, dataset=None, seed: int = config.RANDOM_SEED,
             names_version: str = None,
             sample_mode: str = config.DEFAULT_SAMPLE_MODE,
             font_style: str = FONT_STYLE_ALL,
             cursive_group: str = "",
             specific_font: str = "",
             progress_callback=None, show_bar: bool = True,
             cancel_event=None, archive_planned: bool = False,
             writer_profile=DEFAULT_WRITER_PROFILE_ID,
             augmentation_profile=DEFAULT_AUGMENTATION_PROFILE_ID,
             samples_per_writer: int = 32,
             edge_clipping: str = "none"):
    """
    Generate `count` synthetic samples into a numbered dataset folder.

    dataset:
        None        -> auto-pick the next free folder (dataset_001, _002, ...)
        int/str     -> dataset/datasets/dataset_<n> or dataset/datasets/<name>
    names_version:
        None        -> use config.NAMES_VERSION (default name pool)
        'name1'/'name2'/... -> draw names from resources/<names_version>
    sample_mode:
        regular / semi_broken_mixed / semi_broken_words /
        semi_broken_characters / semi_broken_numerics
    font_style:
        'all'     -> use every available handwriting font (default)
        'cursive' -> restrict to cursive/script fonts only
    cursive_group:
        Sub-style group name (only used when font_style == 'cursive').
        E.g. 'Palmer / School cursive', 'Elegant calligraphy', etc.
        Empty string means all cursive fonts.
    specific_font:
        Exact font filename/stem to render every sample in. Empty means
        use the whole pool selected by font_style / cursive_group.

    Returns the output Path of the dataset folder that was created.
    """
    sample_mode, names_dir = _preflight_generation(
        count,
        names_version,
        sample_mode,
        font_style,
        cursive_group,
        specific_font,
    )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be a whole number")
    if not isinstance(archive_planned, bool):
        raise ValueError("archive_planned must be a boolean")
    _raise_if_cancelled(cancel_event)

    effective_fonts = (
        (font_path_for(specific_font),)
        if specific_font
        else fonts_for_style(font_style, cursive_group)
    )
    if not all(effective_fonts):
        raise ValueError("The effective font selection contains an unavailable font")
    effective_fonts = tuple(str(font) for font in effective_fonts)
    writer_profile = get_writer_style_profile(writer_profile)
    augmentation_profile = get_augmentation_profile(augmentation_profile)
    writer_assigner = SyntheticWriterAssigner(samples_per_writer=samples_per_writer)
    resource_estimate = preflight_generation_resources(
        config.DATASETS_DIR,
        count,
        create_archive=archive_planned,
        large_job_count=config.LARGE_GENERATION_WARNING_COUNT,
    )
    evaluation_policy = build_synthetic_evaluation_policy(effective_fonts, seed)
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    generation_started_at = utc_now_iso()

    reservation = None
    staging_dir = None
    progress = None

    try:
        reservation = config.reserve_dataset_dir(dataset)
        out_dir = config.assert_safe_dataset_dir(reservation.path)
        base_dir = config.DATASETS_DIR.resolve()
        staging_dir = Path(tempfile.mkdtemp(
            prefix=f".{out_dir.name}.tmp-",
            dir=str(base_dir),
        ))
        _assert_internal_dataset_child(staging_dir)

        split_dirs = {}
        for split in config.SPLIT_NAMES:
            d = staging_dir / split
            d.mkdir()
            split_dirs[split] = d

        # Intro info BEFORE the bar starts, so the user immediately sees what's running.
        if specific_font:
            font_info = f"single font: {specific_font}"
        else:
            n_fonts = len(fonts_for_style(font_style, cursive_group))
            font_info = f"{n_fonts} fonts  (style: {font_style}"
            if font_style == "cursive" and cursive_group:
                font_info += f" / {cursive_group}"
            font_info += ")"
        print(f"Generating {count:,} synthetic samples")
        print(f"  output : {out_dir}")
        print(f"  splits : {', '.join(config.SPLIT_NAMES)}")
        print(f"  names  : {names_dir.name}")
        print(f"  seed   : {seed}")
        print(f"  mode   : {config.SAMPLE_MODES[sample_mode]}")
        print(f"  fonts  : {font_info}")
        print()

        split_counts = config.allocate_synthetic_split_counts(count)
        counters = GenerationCounters()
        plan = iter_generation_plan(
            count,
            _field_weights_for_mode(sample_mode),
            split_counts,
            rng,
            split_order=config.SPLIT_NAMES,
        )
        iterator = plan

        if show_bar:
            progress = tqdm(
                iterator,
                total=count,
                desc="Generating",
                unit="img",
                ncols=80,
                dynamic_ncols=True,
                smoothing=0.1,
            )
            iterator = progress

        manifest_path = staging_dir / "manifest.csv"
        labels_path = staging_dir / "labels.csv"
        format_source_path = staging_dir / ".evaluation-source.csv"
        with (AtomicCsvWriter(manifest_path, MANIFEST_COLUMNS) as manifest_writer,
              AtomicCsvWriter(
                  labels_path, ("filename", "label", "split")) as labels_writer,
              AtomicCsvWriter(
                  format_source_path, FORMAT_SOURCE_COLUMNS) as format_writer):
            for item in iterator:
                _raise_if_cancelled(cancel_event)
                i, field_type, split = item.index, item.field_type, item.split
                file_name = f"syn_{i:06d}.png"
                synthetic_writer_id = writer_assigner.writer_id_for(split)
                style = writer_style_for(
                    synthetic_writer_id, seed, writer_profile
                )
                (label, img, font_used, format_profile,
                 format_id) = _generate_valid_sample(
                    field_type,
                    split,
                    file_name,
                    names_dir,
                    sample_mode,
                    font_style,
                    cursive_group,
                    evaluation_policy,
                    style,
                    augmentation_profile,
                    rng,
                    np_rng,
                    edge_clipping=edge_clipping,
                )
                img.save(split_dirs[split] / file_name)
                manifest_writer.write({
                    "filename": file_name,
                    "label": label,
                    "split": split,
                    "source": "synthetic",
                    "field_type": field_type,
                    "font": font_used,
                    "sample_mode": sample_mode,
                    "writer_id": "",
                    "schema_version": MANIFEST_SCHEMA_VERSION,
                })
                labels_writer.write({
                    "filename": file_name, "label": label, "split": split,
                })
                format_writer.write({
                    "filename": file_name,
                    "field_type": field_type,
                    "format_profile": format_profile,
                    "format_id": format_id,
                })
                counters.observe(split=split, field_type=field_type)

                if progress is not None and i % 50 == 0:
                    progress.set_postfix_str(f"{split}/{field_type}")
                if progress_callback is not None:
                    progress_callback(i, count, field_type)

        _raise_if_cancelled(cancel_event)
        _write_evaluation_annotations_streaming(
            manifest_path,
            format_source_path,
            staging_dir / EVALUATION_ANNOTATIONS_FILENAME,
            evaluation_policy,
        )
        format_source_path.unlink()
        validation_report = validate_dataset(staging_dir, expected_count=count)
        validation_report.raise_for_errors()
        _write_image_statistics_artifact(staging_dir)
        atomic_write_json(
            staging_dir / RUN_METADATA_FILENAME,
            _run_metadata(
                seed=seed,
                count=count,
                sample_mode=sample_mode,
                names_dir=names_dir,
                font_style=font_style,
                cursive_group=cursive_group,
                specific_font=specific_font,
                effective_fonts=effective_fonts,
                split_counts=split_counts,
                manifest_path=manifest_path,
                validation_report=validation_report,
                evaluation_policy=evaluation_policy,
                generation_started_at=generation_started_at,
                resource_estimate=resource_estimate,
                writer_profile=writer_profile,
                augmentation_profile=augmentation_profile,
                counters=counters,
                writer_assigner=writer_assigner,
                edge_clipping=edge_clipping,
            ),
        )
        _raise_if_cancelled(cancel_event)

        # The reservation marker prevents cooperating processes from claiming the
        # same final path. The final directory remains absent until this rename.
        if out_dir.exists():
            raise FileExistsError(f"Dataset output already exists: {out_dir}")
        staging_dir.rename(out_dir)
        staging_dir = None

        print(f"\nGenerated {count:,} samples -> {out_dir}")
        print(f"Manifest: {out_dir / 'manifest.csv'}")
        _print_split_distribution(split_counts)
        _print_distribution(counters.field_types)
        return out_dir
    finally:
        if progress is not None:
            progress.close()
        if staging_dir is not None and staging_dir.exists():
            guarded_remove_dataset(staging_dir, _allow_internal=True)
        if reservation is not None:
            reservation.release()


def zip_dataset(out_dir, remove_dir: bool = False, cancel_event=None) -> "Path":
    """
    Compress a dataset folder into a sibling .zip archive.

    dataset/datasets/dataset_001/  ->  dataset/datasets/dataset_001.zip

    remove_dir:
        True  -> delete the original folder after zipping (keep only the .zip)
        False -> leave the folder in place alongside the .zip
    Returns the path of the created .zip file.
    """
    _raise_if_cancelled(cancel_event)
    out_dir = config.assert_safe_dataset_dir(out_dir)
    _assert_internal_dataset_child(out_dir)
    if not out_dir.is_dir():
        raise FileNotFoundError(f"Dataset folder does not exist: {out_dir}")
    if is_reparse_point(out_dir):
        raise ValueError(f"Refusing to archive a link or junction: {out_dir}")

    temp_base = out_dir.parent / f".{out_dir.name}.{uuid.uuid4().hex}.archive"
    temp_zip = Path(f"{temp_base}.zip")
    zip_path = out_dir.parent / f"{out_dir.name}.zip"
    checksum_path = zip_path.with_name(f"{zip_path.name}.sha256")

    try:
        with dataset_mutation_lock(out_dir, purpose="package dataset"):
            zip_path, checksum_path = assert_no_sibling_archive(out_dir)
            validation_report = validate_dataset(
                out_dir,
                split_names=config.SPLIT_NAMES,
                require_nonempty_splits=True,
                write_artifacts=False,
                create_montage=False,
            )
            validation_report.raise_for_errors()
            _require_fresh_validation_artifacts(out_dir, validation_report)
            _require_fresh_handoff_artifacts(out_dir, validation_report)
            _raise_if_cancelled(cancel_event)

            # The complete snapshot both rejects links/devices and supplies the
            # byte contract verified against the temporary ZIP below.
            source_snapshot = snapshot_regular_tree(out_dir)
            created = Path(shutil.make_archive(
                base_name=str(temp_base),
                format="zip",
                root_dir=out_dir.parent,
                base_dir=out_dir.name,
            ))
            if created != temp_zip or not temp_zip.is_file():
                raise RuntimeError(
                    f"Archive creation did not produce the expected file: {temp_zip}"
                )
            _raise_if_cancelled(cancel_event)
            if snapshot_regular_tree(out_dir) != source_snapshot:
                raise RuntimeError(
                    "Dataset changed during packaging; the archive was not published"
                )
            _validate_zip_archive(temp_zip, out_dir.name, source_snapshot)
            _raise_if_cancelled(cancel_event)
            # Minimise the final non-cooperating-writer window before publish.
            if snapshot_regular_tree(out_dir) != source_snapshot:
                raise RuntimeError(
                    "Dataset changed before archive publication; retry packaging"
                )
            assert_no_sibling_archive(out_dir)
            try:
                # Hard-link publication is create-if-absent on both Windows and
                # POSIX; unlike rename it cannot overwrite a racing artifact.
                os.link(temp_zip, zip_path)
                temp_zip.unlink()
                checksum_path = write_sha256_sidecar(zip_path)
                if not verify_sha256_sidecar(zip_path, checksum_path):
                    raise ValueError(f"Archive checksum verification failed: {zip_path}")
                if _is_cancelled(cancel_event):
                    raise GenerationCancelled("Dataset packaging was cancelled.")
            except Exception:
                checksum_path.unlink(missing_ok=True)
                zip_path.unlink(missing_ok=True)
                raise

            if remove_dir:
                guarded_remove_dataset(out_dir)
    finally:
        temp_zip.unlink(missing_ok=True)

    print(f"Zipped dataset -> {zip_path}")
    print(f"SHA-256       -> {checksum_path}")
    if remove_dir:
        print(f"Removed folder -> {out_dir}")
    return zip_path


def _require_fresh_validation_artifacts(out_dir: Path, validation_report) -> None:
    """Require the published report/montage to match the exact archived bytes."""
    report_path = out_dir / REPORT_FILENAME
    montage_path = out_dir / MONTAGE_FILENAME
    if not report_path.is_file():
        raise FileNotFoundError(f"Missing validation report: {report_path}")
    if not montage_path.is_file():
        raise FileNotFoundError(f"Missing review montage: {montage_path}")
    try:
        saved = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unreadable validation report {report_path}: {exc}") from exc
    if (saved.get("valid") is not True
            or saved.get("manifest_sha256") != validation_report.manifest_sha256
            or saved.get("images_sha256") != validation_report.images_sha256):
        raise ValueError(
            "Dataset validation report is missing, failed, or stale; "
            "run dataset validation before packaging"
        )


def _require_fresh_handoff_artifacts(out_dir: Path, validation_report) -> None:
    """Require the notebook's provenance and annotation inputs to be current."""
    validate_handoff_files(
        out_dir,
        manifest_sha256=validation_report.manifest_sha256,
        images_sha256=validation_report.images_sha256,
    )


def _validate_zip_archive(
    path: Path, expected_root: str, expected_snapshot: dict[str, str]
) -> None:
    """Verify archive paths, exact bytes, CRCs, and Kaggle handoff semantics."""
    with zipfile.ZipFile(path, "r") as archive:
        members = archive.infolist()
        if not members:
            raise ValueError(f"Dataset archive is empty: {path}")
        archived_files: dict[str, zipfile.ZipInfo] = {}
        folded_names: set[str] = set()
        for member in members:
            parts = Path(member.filename.replace("\\", "/")).parts
            if (not parts or parts[0] != expected_root or
                    member.filename.startswith(("/", "\\")) or ".." in parts):
                raise ValueError(f"Unsafe or unexpected archive member: {member.filename}")
            unix_mode = member.external_attr >> 16
            if stat.S_ISLNK(unix_mode):
                raise ValueError(f"Archive contains a symbolic link: {member.filename}")
            if member.is_dir():
                continue
            logical_name = Path(*parts[1:]).as_posix()
            folded = logical_name.casefold()
            if not logical_name or folded in folded_names:
                raise ValueError(f"Duplicate archive member: {member.filename}")
            folded_names.add(folded)
            archived_files[logical_name] = member
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"Archive CRC check failed for: {bad_member}")

        if set(archived_files) != set(expected_snapshot):
            missing = sorted(set(expected_snapshot) - set(archived_files))
            extra = sorted(set(archived_files) - set(expected_snapshot))
            raise ValueError(
                "Archive file set differs from the validated dataset "
                f"(missing={missing[:3]}, extra={extra[:3]})"
            )
        for logical_name, expected_digest in expected_snapshot.items():
            digest = hashlib.sha256()
            with archive.open(archived_files[logical_name], "r") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected_digest:
                raise ValueError(
                    f"Archive member changed during packaging: {logical_name}"
                )

        image_digest = hashlib.sha256()
        image_members = [
            (logical_name, member)
            for logical_name, member in archived_files.items()
            if len(Path(logical_name).parts) == 2
            and Path(logical_name).parts[0] in config.SPLIT_NAMES
        ]
        for logical_name, member in sorted(image_members):
            encoded_name = logical_name.encode("utf-8")
            image_digest.update(len(encoded_name).to_bytes(8, "big"))
            image_digest.update(encoded_name)
            with archive.open(member, "r") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    image_digest.update(chunk)

        validate_handoff_bytes(
            manifest_content=archive.read(archived_files["manifest.csv"]),
            metadata_content=archive.read(archived_files[RUN_METADATA_FILENAME]),
            annotations_content=archive.read(
                archived_files[EVALUATION_ANNOTATIONS_FILENAME]
            ),
            validation_content=archive.read(archived_files[REPORT_FILENAME]),
            manifest_sha256=expected_snapshot["manifest.csv"],
            images_sha256=image_digest.hexdigest(),
        )


def _validate_staged_dataset(staging_dir: Path, expected_count: int) -> None:
    """Compatibility wrapper around the canonical dataset validator."""
    report = validate_dataset(staging_dir, expected_count=expected_count)
    report.raise_for_errors()


def _write_csv(path, header, rows):
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with open(temp_path, "x", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(header)
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    temp_path.replace(path)


def _print_split_distribution(split_for):
    counts = (
        dict(split_for)
        if hasattr(split_for, "items")
        else dict(Counter(split_for))
    )
    print("\nSplit distribution:")
    for split in config.SPLIT_NAMES:
        print(f"  {split:<6} {counts.get(split, 0)}")


def _print_distribution(rows):
    if hasattr(rows, "items"):
        counts = dict(rows)
    else:
        counts = {}
        for row in rows:
            field_type = row["field_type"] if isinstance(row, dict) else row[3]
            counts[field_type] = counts.get(field_type, 0) + 1
    print("\nField distribution:")
    for field_type, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {field_type:<14} {n}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic handwriting data.")
    parser.add_argument("--count", type=int, default=config.DEFAULT_COUNT,
                        help=f"number of samples (default {config.DEFAULT_COUNT})")
    parser.add_argument("--dataset", type=str, default=None,
                        help="dataset name or number (default: next free dataset_NNN)")
    parser.add_argument("--names", type=str, default=None,
                        help="names version folder under resources/ (e.g. name1, name2)")
    parser.add_argument("--mode", type=str, default=config.DEFAULT_SAMPLE_MODE,
                        choices=list(config.SAMPLE_MODES),
                        help="sample style to generate")
    parser.add_argument("--font-style", type=str, default=FONT_STYLE_ALL,
                        choices=["all", "cursive"],
                        help="font pool: 'all' (default) or 'cursive' only")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--writer-profile", default=DEFAULT_WRITER_PROFILE_ID,
                        help="versioned synthetic writer-style profile")
    parser.add_argument("--augmentation-profile",
                        default=DEFAULT_AUGMENTATION_PROFILE_ID,
                        help="versioned scan/augmentation profile")
    parser.add_argument("--samples-per-writer", type=int, default=32,
                        help="samples sharing one synthetic writer style")
    parser.add_argument("--zip", action="store_true",
                        help="also package the finished dataset as a .zip archive")
    parser.add_argument("--zip-only", action="store_true",
                        help="with --zip, delete the folder and keep only the .zip")
    parser.add_argument("--yes", action="store_true",
                        help="confirm a job at or above the large-run warning threshold")
    args = parser.parse_args()

    if args.count >= config.LARGE_GENERATION_WARNING_COUNT and not args.yes:
        parser.error(
            f"{args.count:,} samples is a large job; review disk/time estimates "
            "and rerun with --yes to confirm"
        )

    out_dir = generate(args.count, args.dataset, args.seed, names_version=args.names,
                       sample_mode=args.mode, font_style=args.font_style,
                       archive_planned=bool(args.zip or args.zip_only),
                       writer_profile=args.writer_profile,
                       augmentation_profile=args.augmentation_profile,
                       samples_per_writer=args.samples_per_writer)
    if args.zip or args.zip_only:
        zip_dataset(out_dir, remove_dir=args.zip_only)


if __name__ == "__main__":
    main()
