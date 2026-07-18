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
    │   └── manifest.csv           # filename, label, split, field_type, font, sample_mode
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

## Handwriting fonts

The toolkit ships with **24 handwriting fonts** (20 cursive, 4 print/sketch),
all tuned toward mid-1900s Philippine civil registry penmanship.

### Cursive style groups

| Group | Fonts | Best for |
|---|---|---|
| **Palmer / School cursive** | Licorice, Yellowtail, Dancing Script, Carattere, Norican | Filipino school cursive, old registry entries |
| **Elegant calligraphy** | Great Vibes, Sacramento, Tangerine, Pinyon Script, Allura, Monsieur La Doulaise, Mr De Haviland, Ruthie | Formal certificates, signatures |
| **Clean semi-formal** | Satisfy, Courgette, Alex Brush, Cookie, Yeseva One | General cursive handwriting |
| **Loose informal** | Homemade Apple, Carattere, Dancing Script | Casual or hurried entries |
| **Display / decorative** | Pacifico, Norican, Yeseva One | Headers, decorative text |

### Print / sketch fonts
Gochi Hand, Indie Flower, Kalam, Rock Salt — used only when **Font style = All fonts**.

Drop additional `.ttf` / `.otf` files into `resources/fonts/` and they will be
picked up automatically on the next run.

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

# generate damaged text for old/broken records
python -m src.generate_synthetic --count 2000 --mode semi_broken_mixed
python -m src.generate_synthetic --count 1000 --mode semi_broken_characters
python -m src.generate_synthetic --count 1000 --mode semi_broken_numerics

# restrict to cursive fonts only
python -m src.generate_synthetic --count 5000 --font-style cursive

# later: merge real data (by writer) into the latest dataset
python -m src.build_splits
```

### Font style flag

| `--font-style` | Effect |
|---|---|
| `all` (default) | All 24 handwriting fonts |
| `cursive` | Only the 20 cursive/script fonts |

Specific cursive sub-groups are selectable in the GUI only.

---

## GUI

Launch by double-clicking `Generate Images.bat` or running:

```bash
python gui.py
```

The GUI features a modern dark theme and exposes all generation options:

- **Samples** — count + quick-pick buttons (1k / 5k / 20k / 40k)
- **Dataset folder** — name, number, or auto-next
- **Names pool** — switch between name versions
- **Sample mode** — regular or semi-broken variants
- **Font style** — All fonts / Cursive only
  - When *Cursive only* is selected, a second dropdown lets you pick a specific
    cursive style group (Palmer / Elegant / Semi-formal / Loose / Decorative)
- **Options** — merge real data, package as .zip
- **Progress bar** — live count, percentage, field type, speed, and ETA

See `config.py` for all settings (sizes, paths, augmentation, field weights).
