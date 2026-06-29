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
├── real/                    # REAL mock handwriting (collected from bond papers)
└── datasets/                # one numbered folder per generation run
    ├── dataset_001/
    │   ├── train/  val/  test/    # cropped field images, already split
    │   ├── labels.csv             # filename, label, split
    │   └── manifest.csv           # filename, label, split, field_type, font
    ├── dataset_002/
    └── ...
```

Each generation run writes a fresh `dataset_NNN/` folder (auto-numbered), so
runs never overwrite each other. Synthetic data is split into train/val/test
at generation time — there is no intermediate `synthetic/` folder.

Each sample is one cropped field image (a name, date, place, etc.) paired with
its correct text label.

---

## Pipeline overview

1. **Synthetic generation** (`src/generate_synthetic.py`)
   Renders Filipino names / dates / places / fields using handwriting fonts,
   degrades them to look like old scans, and writes them straight into a
   numbered `dataset_NNN/` folder split into train/val/test. Auto-labeled,
   free, high-volume.

2. **Mock collection** (`src/make_sheets.py` + `src/process_scans.py`)
   Generate printable bond-paper sheets (prompts + boxes + corner marks),
   have people fill them, scan, then auto-crop into per-field images.

3. **Merge real data** (`src/build_splits.py`)
   Split real (mock) handwriting by writer and merge it into an existing
   `dataset_NNN/` folder's train/val/test.

---

## Setup

Uses the existing `trocr` conda environment, or install deps directly:

```bash
pip install -r requirements.txt
```

## Quick start

```bash
# generate 200 synthetic samples into the next free dataset_NNN folder
python -m src.generate_synthetic --count 200

# generate into a specific dataset (by number or name)
python -m src.generate_synthetic --count 5000 --dataset 2
python -m src.generate_synthetic --count 200 --dataset my_test_run

# later: merge real data (by writer) into the latest dataset
python -m src.build_splits
```

See `config.py` for all settings (sizes, paths, augmentation, field weights).
