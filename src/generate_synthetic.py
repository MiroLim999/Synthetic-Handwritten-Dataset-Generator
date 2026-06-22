"""
Synthetic data generator.

Generates fake handwriting field images + labels:
  1. choose a field type (weighted by config.FIELD_WEIGHTS)
  2. generate a realistic value for it
  3. render it in a random handwriting font
  4. degrade it to look like an old scan
  5. save the image and record (filename, label, field_type, font)

Output:
  dataset/synthetic/images/syn_000001.png ...
  dataset/synthetic/manifest.csv   (filename, label, field_type, font)
  dataset/synthetic/labels.csv     (filename, label)   <- simple format

Usage:
  python -m src.generate_synthetic --count 20000
  python -m src.generate_synthetic --count 200 --out dataset/synthetic
"""

import argparse
import csv
import random

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


def generate(count: int, out_dir=None, seed: int = config.RANDOM_SEED,
             progress_callback=None, show_bar: bool = True) -> None:
    random.seed(seed)

    out_dir = out_dir or config.SYNTHETIC_DIR
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Intro info BEFORE the bar starts, so the user immediately sees what's running.
    n_fonts = len(available_fonts())
    print(f"Generating {count:,} synthetic samples")
    print(f"  output : {img_dir}")
    print(f"  fonts  : {n_fonts} handwriting fonts")
    print()

    manifest_rows = []
    labels_rows = []

    field_types = _weighted_field_types(count)

    iterator = field_types
    progress = None
    if show_bar:
        progress = tqdm(
            field_types,
            desc="Generating",
            unit="img",
            ncols=80,
            dynamic_ncols=True,
            smoothing=0.1,
        )
        iterator = progress

    for i, field_type in enumerate(iterator, start=1):
        label = fields.make_value(field_type)
        img, font_used = render_text(label)
        img = degrade(img)

        file_name = f"syn_{i:06d}.png"
        img.save(img_dir / file_name)

        manifest_rows.append([file_name, label, field_type, font_used])
        labels_rows.append([file_name, label])

        if progress is not None and i % 50 == 0:
            progress.set_postfix_str(field_type)
        # let an external UI (e.g. the GUI) track progress
        if progress_callback is not None:
            progress_callback(i, count, field_type)

    if progress is not None:
        progress.close()

    _write_csv(out_dir / "manifest.csv",
               ["filename", "label", "field_type", "font"], manifest_rows)
    _write_csv(out_dir / "labels.csv",
               None, labels_rows)

    print(f"\nGenerated {count:,} samples -> {img_dir}")
    print(f"Manifest: {out_dir / 'manifest.csv'}")
    _print_distribution(manifest_rows)


def _write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(header)
        writer.writerows(rows)


def _print_distribution(rows):
    counts = {}
    for _, _, field_type, _ in rows:
        counts[field_type] = counts.get(field_type, 0) + 1
    print("\nField distribution:")
    for field_type, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {field_type:<14} {n}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic handwriting data.")
    parser.add_argument("--count", type=int, default=config.DEFAULT_COUNT,
                        help=f"number of samples (default {config.DEFAULT_COUNT})")
    parser.add_argument("--out", type=str, default=None,
                        help="output directory (default dataset/synthetic)")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()

    from pathlib import Path
    out_dir = Path(args.out) if args.out else None
    generate(args.count, out_dir, args.seed)


if __name__ == "__main__":
    main()
