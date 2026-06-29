"""
Split builder (real / mock data).

Synthetic data is now split into train/val/test at generation time
(see src/generate_synthetic.py), so there is no longer an intermediate
"synthetic" folder to assemble.

This script only handles REAL (mock) handwriting: it is split BY WRITER,
never by image, so the same person's handwriting never appears in two
splits. It is merged INTO an existing dataset folder's train/val/test
subfolders so the real exam set lives alongside the synthetic data.

Output (merged into an existing dataset folder):
  dataset/datasets/dataset_001/{train,val,test}/...
  dataset/datasets/dataset_001/labels.csv          (appends real rows)
  dataset/datasets/dataset_001/real_manifest.csv   (filename, label, split, writer)

Usage:
  python -m src.build_splits                 # merge into the latest dataset
  python -m src.build_splits --dataset 2     # merge into dataset_002
"""

import argparse
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


def build(dataset=None):
    """Split real data by writer and merge it into a dataset folder."""
    random.seed(config.RANDOM_SEED)

    # Pick the dataset folder to merge into.
    if dataset is None:
        existing = config.existing_datasets()
        if not existing:
            print("No dataset folder found. Run generate_synthetic first.")
            return None
        out_dir = existing[-1]
    else:
        out_dir = config.resolve_dataset_dir(dataset)
    if not out_dir.exists():
        print(f"Dataset folder does not exist: {out_dir}")
        return None

    real_labels = _read_labels(config.REAL_DIR / "labels.csv")
    real_img_dir = config.REAL_DIR / "images"
    if not real_labels:
        print("Real: none found (collect mock sheets first). Nothing to merge.")
        return out_dir

    writers = sorted({_writer_id(fn) for fn, _ in real_labels})
    random.shuffle(writers)
    n_train = int(len(writers) * config.REAL_TRAIN_FRAC)
    n_val = int(len(writers) * config.REAL_VAL_FRAC)
    train_w = set(writers[:n_train])
    val_w = set(writers[n_train:n_train + n_val])
    # remainder -> test

    def split_of(writer: str) -> str:
        if writer in train_w:
            return "train"
        if writer in val_w:
            return "val"
        return "test"

    manifest_rows = []
    label_rows = []
    counts = {s: 0 for s in config.SPLIT_NAMES}
    for fn, lbl in real_labels:
        src = real_img_dir / fn
        if not src.exists():
            continue
        split = split_of(_writer_id(fn))
        _copy(src, out_dir / split, fn)
        manifest_rows.append((fn, lbl, split, _writer_id(fn)))
        label_rows.append((fn, lbl, split))
        counts[split] += 1

    # append real rows to the dataset's labels.csv
    labels_csv = out_dir / "labels.csv"
    write_header = not labels_csv.exists()
    with open(labels_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["filename", "label", "split"])
        w.writerows(label_rows)

    with open(out_dir / "real_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "label", "split", "writer"])
        w.writerows(manifest_rows)

    print(f"Real: merged {len(label_rows)} samples from {len(writers)} writers "
          f"into {out_dir} (by-writer split)")
    for split in config.SPLIT_NAMES:
        print(f"  {split:<6} {counts[split]}")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="Merge real (mock) data into a dataset.")
    parser.add_argument("--dataset", type=str, default=None,
                        help="dataset name or number (default: latest)")
    args = parser.parse_args()
    build(args.dataset)


if __name__ == "__main__":
    main()
