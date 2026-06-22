"""
Split builder.

Assembles the final train / val / test sets that TrOCR fine-tuning will use.

Rules:
  - Synthetic data is split randomly (train/val) per config fractions.
  - Real (mock) data is split BY WRITER, never by image, so the same
    person's handwriting never appears in two splits. Real data also
    contributes the held-out TEST set (the honest "real handwriting" exam).

Output:
  dataset/splits/train/   dataset/splits/val/   dataset/splits/test/
  dataset/splits/labels.csv   (filename, label)  across all splits
  dataset/splits/split_manifest.csv  (filename, label, split, source)

Usage:
  python -m src.build_splits
"""

import csv
import random
import shutil
from pathlib import Path

import config


def _read_labels(csv_path: Path) -> list[tuple[str, str]]:
    """Read a (filename, label) csv. Skips a header row if present."""
    if not csv_path.exists():
        return []
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            if row[0].lower() == "filename" and row[1].lower() == "label":
                continue  # header
            rows.append((row[0], row[1]))
    return rows


def _writer_id(filename: str) -> str:
    """Extract a writer id from a real-sample filename like 'P012_name.png' -> 'P012'."""
    return filename.split("_", 1)[0]


def _copy(src: Path, dst_dir: Path, filename: str):
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_dir / filename)


def build():
    random.seed(config.RANDOM_SEED)

    splits = {"train": [], "val": [], "test": []}  # each: (filename, label, source)

    # -- synthetic: random train/val split -------------------------------
    synth_labels = _read_labels(config.SYNTHETIC_DIR / "labels.csv")
    synth_img_dir = config.SYNTHETIC_DIR / "images"
    if synth_labels:
        random.shuffle(synth_labels)
        n_train = int(len(synth_labels) * config.SYNTH_TRAIN_FRAC)
        for fn, lbl in synth_labels[:n_train]:
            splits["train"].append((fn, lbl, "synthetic"))
        for fn, lbl in synth_labels[n_train:]:
            splits["val"].append((fn, lbl, "synthetic"))
        print(f"Synthetic: {len(synth_labels)} samples "
              f"({n_train} train / {len(synth_labels) - n_train} val)")
    else:
        print("Synthetic: none found (run generate_synthetic first).")

    # -- real: split BY WRITER -------------------------------------------
    real_labels = _read_labels(config.REAL_DIR / "labels.csv")
    real_img_dir = config.REAL_DIR / "images"
    if real_labels:
        writers = sorted({_writer_id(fn) for fn, _ in real_labels})
        random.shuffle(writers)
        n_train = int(len(writers) * config.REAL_TRAIN_FRAC)
        n_val = int(len(writers) * config.REAL_VAL_FRAC)
        train_w = set(writers[:n_train])
        val_w = set(writers[n_train:n_train + n_val])
        # remainder -> test

        for fn, lbl in real_labels:
            w = _writer_id(fn)
            if w in train_w:
                splits["train"].append((fn, lbl, "real"))
            elif w in val_w:
                splits["val"].append((fn, lbl, "real"))
            else:
                splits["test"].append((fn, lbl, "real"))
        print(f"Real: {len(real_labels)} samples from {len(writers)} writers "
              f"(by-writer split)")
    else:
        print("Real: none found (collect mock sheets first). No test set built.")

    # -- copy files + write csvs -----------------------------------------
    all_rows = []
    label_rows = []
    for split_name, items in splits.items():
        split_dir = config.SPLITS_DIR / split_name
        if split_dir.exists():
            shutil.rmtree(split_dir)
        for fn, lbl, source in items:
            src = (synth_img_dir if source == "synthetic" else real_img_dir) / fn
            if not src.exists():
                continue
            _copy(src, split_dir, fn)
            all_rows.append((fn, lbl, split_name, source))
            label_rows.append((fn, lbl))

    config.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.SPLITS_DIR / "labels.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(label_rows)
    with open(config.SPLITS_DIR / "split_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "label", "split", "source"])
        w.writerows(all_rows)

    print("\nFinal splits:")
    for split_name, items in splits.items():
        print(f"  {split_name:<6} {len(items)}")
    print(f"\nWritten to {config.SPLITS_DIR}")


if __name__ == "__main__":
    build()
