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
  dataset/datasets/dataset_001/manifest.csv        (filename, label, split, field_type, font)

Usage:
  python -m src.generate_synthetic --count 20000
  python -m src.generate_synthetic --count 5000 --dataset 2
  python -m src.generate_synthetic --count 200 --dataset my_test_run
"""

import argparse
import csv
import random
import shutil
from pathlib import Path

from tqdm import tqdm

import config
from src import fields
from src.augment import degrade
from src.render import render_text, available_fonts


def _weighted_field_types(n: int) -> list[str]:
    """Build a list of n field-type choices following FIELD_WEIGHTS."""
    types = list(config.FIELD_WEIGHTS.keys())
    weights = list(config.FIELD_WEIGHTS.values())
    return random.choices(types, weights=weights, k=n)


def _split_assignments(count: int) -> list[str]:
    """
    Return a shuffled list of length `count` where each entry is
    'train', 'val' or 'test' following the configured fractions.
    """
    n_train = int(count * config.SYNTH_TRAIN_FRAC)
    n_val = int(count * config.SYNTH_VAL_FRAC)
    # test gets the remainder so the counts always add up to `count`
    n_test = count - n_train - n_val

    assignments = (["train"] * n_train) + (["val"] * n_val) + (["test"] * n_test)
    random.shuffle(assignments)
    return assignments


def generate(count: int, dataset=None, seed: int = config.RANDOM_SEED,
             names_version: str = None,
             progress_callback=None, show_bar: bool = True):
    """
    Generate `count` synthetic samples into a numbered dataset folder.

    dataset:
        None        -> auto-pick the next free folder (dataset_001, _002, ...)
        int/str     -> dataset/datasets/dataset_<n> or dataset/datasets/<name>
    names_version:
        None        -> use config.NAMES_VERSION (default name pool)
        'name1'/'name2'/... -> draw names from resources/<names_version>

    Returns the output Path of the dataset folder that was created.
    """
    random.seed(seed)

    # Select which name pool to draw from for this run.
    if names_version:
        config.NAMES_VERSION = names_version
        config.NAMES_DIR = config.RESOURCES_DIR / names_version

    out_dir = config.resolve_dataset_dir(dataset)
    split_dirs = {}
    for split in config.SPLIT_NAMES:
        d = out_dir / split
        d.mkdir(parents=True, exist_ok=True)
        split_dirs[split] = d

    # Intro info BEFORE the bar starts, so the user immediately sees what's running.
    n_fonts = len(available_fonts())
    print(f"Generating {count:,} synthetic samples")
    print(f"  output : {out_dir}")
    print(f"  splits : {', '.join(config.SPLIT_NAMES)}")
    print(f"  names  : {config.NAMES_VERSION}")
    print(f"  fonts  : {n_fonts} handwriting fonts")
    print()

    manifest_rows = []
    labels_rows = []

    field_types = _weighted_field_types(count)
    split_for = _split_assignments(count)

    iterator = list(enumerate(zip(field_types, split_for), start=1))
    progress = None
    if show_bar:
        progress = tqdm(
            iterator,
            desc="Generating",
            unit="img",
            ncols=80,
            dynamic_ncols=True,
            smoothing=0.1,
        )
        iterator = progress

    for i, (field_type, split) in iterator:
        label = fields.make_value(field_type)
        img, font_used = render_text(label)
        img = degrade(img)

        file_name = f"syn_{i:06d}.png"
        img.save(split_dirs[split] / file_name)

        manifest_rows.append([file_name, label, split, field_type, font_used])
        labels_rows.append([file_name, label, split])

        if progress is not None and i % 50 == 0:
            progress.set_postfix_str(f"{split}/{field_type}")
        # let an external UI (e.g. the GUI) track progress
        if progress_callback is not None:
            progress_callback(i, count, field_type)

    if progress is not None:
        progress.close()

    _write_csv(out_dir / "manifest.csv",
               ["filename", "label", "split", "field_type", "font"], manifest_rows)
    _write_csv(out_dir / "labels.csv",
               ["filename", "label", "split"], labels_rows)

    print(f"\nGenerated {count:,} samples -> {out_dir}")
    print(f"Manifest: {out_dir / 'manifest.csv'}")
    _print_split_distribution(split_for)
    _print_distribution(manifest_rows)
    return out_dir


def zip_dataset(out_dir, remove_dir: bool = False) -> "Path":
    """
    Compress a dataset folder into a sibling .zip archive.

    dataset/datasets/dataset_001/  ->  dataset/datasets/dataset_001.zip

    remove_dir:
        True  -> delete the original folder after zipping (keep only the .zip)
        False -> leave the folder in place alongside the .zip
    Returns the path of the created .zip file.
    """
    out_dir = Path(out_dir)
    # make_archive adds the .zip extension itself, so base_name has none.
    zip_path = shutil.make_archive(
        base_name=str(out_dir),
        format="zip",
        root_dir=out_dir.parent,
        base_dir=out_dir.name,
    )
    print(f"Zipped dataset -> {zip_path}")
    if remove_dir:
        shutil.rmtree(out_dir)
        print(f"Removed folder -> {out_dir}")
    return Path(zip_path)


def _write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(header)
        writer.writerows(rows)


def _print_split_distribution(split_for):
    counts = {}
    for split in split_for:
        counts[split] = counts.get(split, 0) + 1
    print("\nSplit distribution:")
    for split in config.SPLIT_NAMES:
        print(f"  {split:<6} {counts.get(split, 0)}")


def _print_distribution(rows):
    counts = {}
    for _, _, _, field_type, _ in rows:
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
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--zip", action="store_true",
                        help="also package the finished dataset as a .zip archive")
    parser.add_argument("--zip-only", action="store_true",
                        help="with --zip, delete the folder and keep only the .zip")
    args = parser.parse_args()

    out_dir = generate(args.count, args.dataset, args.seed, names_version=args.names)
    if args.zip or args.zip_only:
        zip_dataset(out_dir, remove_dir=args.zip_only)


if __name__ == "__main__":
    main()
