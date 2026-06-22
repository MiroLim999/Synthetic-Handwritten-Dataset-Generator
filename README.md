# Civil Registry Handwriting Dataset Toolkit

Tools for building a handwriting-recognition dataset to fine-tune **TrOCR** for
Philippine civil registry documents (Birth, Death, Marriage certificates).

This repository is **only for creating the dataset**. Model training/inference
lives in a separate project. The output of this toolkit is a folder of cropped
field images + labels, in the format TrOCR fine-tuning expects.

---

## What it produces

```
dataset/
├── synthetic/      # FAKE handwriting (generated) — the bulk training data
├── real/           # REAL mock handwriting (collected from bond papers)
└── splits/         # final train / val / test + labels.csv
```

Each sample is one cropped field image (a name, date, place, etc.) paired with
its correct text label.

---

## Pipeline overview

1. **Synthetic generation** (`src/generate_synthetic.py`)
   Renders Filipino names / dates / places / fields using handwriting fonts,
   then degrades them to look like old scans. Auto-labeled, free, high-volume.

2. **Mock collection** (`src/make_sheets.py` + `src/process_scans.py`)
   Generate printable bond-paper sheets (prompts + boxes + corner marks),
   have people fill them, scan, then auto-crop into per-field images.

3. **Split building** (`src/build_splits.py`)
   Combine synthetic + real, split by writer, write `splits/labels.csv`.

---

## Setup

Uses the existing `trocr` conda environment, or install deps directly:

```bash
pip install -r requirements.txt
```

## Quick start

```bash
# generate 200 synthetic samples to try it out
python -m src.generate_synthetic --count 200

# build train/val/test splits from whatever data exists
python -m src.build_splits
```

See `config.py` for all settings (sizes, paths, augmentation, field weights).
