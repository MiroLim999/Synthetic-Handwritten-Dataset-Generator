# Studio presets (Studio Recipe v2)

A preset is a JSON *studio recipe*. The same format is used for the presets in
this folder, for user presets, for `--recipe` on the command line, and for the
`studio-recipe.json` that every generated dataset contains.

- **Built-in presets** live here (`resources/presets/*.json`) and are read-only
  in the GUI.
- **User presets** live in `resources/presets/user/*.json`. The folder is
  created on demand and is not tracked by git. Filenames are slugs
  (`[a-z0-9_-]`, 1-64 characters).

## Format

```json
{
  "format": "synthetic-generator.studio-recipe",
  "schema_version": 2,
  "name": "Historical faded scan",
  "description": "Aged ledger look with light top cutoffs",
  "augmentation":  { "...": "..." },
  "stroke_damage": { "...": "..." },
  "cutoff":        { "...": "..." },
  "run":           { "...": "..." }
}
```

`format` and `schema_version` are required. `name` and `description` are
optional strings. Every section is optional, and **loading a recipe applies
only the sections it contains**. Any section may carry an informational
`based_on` string (for example the built-in profile it started from).

Validation is strict: unknown keys, values outside the bounds below, and
`min > max` are rejected with their JSON path (for example
`augmentation.blur.range: min must not exceed max`). A `schema_version` above
2 is rejected as written by a newer generator.

### Effects: a chance plus a range

Each effect has a `probability` (0-1, the chance it is applied to an image)
and one or more `[min, max]` ranges; each image draws a value from the range.
Missing entries inside a present section take the defaults (the
`historical_scan_v1` profile for augmentation, the config `BROKEN_*` values
for stroke damage, no cut for cutoff).

| Section.effect | Keys | Bounds |
|---|---|---|
| `augmentation.fade` | `probability`, `range` (contrast factor; lower = more faded) | 0.10-1.00 |
| `augmentation.brightness` | `probability`, `range` (1.0 = unchanged) | 0.30-1.60 |
| `augmentation.paper_tint` | `probability`, `range` (overlay strength) | 0-0.50 |
| `augmentation.stains` | `probability`, `count_range`, `opacity_range` (alpha) | 0-20, 0-120 |
| `augmentation.rotation` | `probability`, `range` (min/max absolute tilt, degrees; sign is random) | 0-15 |
| `augmentation.blur` | `probability`, `range` (radius, px) | 0-5 |
| `augmentation.noise` | `probability`, `range` (std-dev) | 0-50 |
| `augmentation.paper_texture` | `probability`, `range` | 0-20 |
| `augmentation.scanlines` | `probability`, `range` (alpha, whole numbers) | 0-60 |
| `augmentation.jpeg` | `probability`, `range` (quality, whole numbers) | 10-100 |
| `stroke_damage.contrast` | `probability`, `range` (contrast factor) | 0.05-1.00 |
| `stroke_damage.erode` | `probability`, `range` (blend amount) | 0-1 |
| `stroke_damage.gaps` | `probability`, `count_range`, `size_range` (px) | 1-60, 1-40 |
| `stroke_damage.scratches` | `probability`, `count_range` | 1-20 |
| `cutoff.top` / `bottom` | `probability`, `range` (fraction of the ink box height) | 0-0.40 |
| `cutoff.left` / `right` | `probability`, `range` (fraction of the ink box width) | 0-0.40 |
| `cutoff.max_ink_loss` | single number: largest share of ink a cut may remove | 0-0.50 |

Stroke damage only applies in the semi-broken sample modes. Cutoffs are real,
ink-relative cuts: labels of 3 or fewer characters get no side cuts and
capped vertical cuts, and side cuts on longer labels are capped at about a
third of one character so the image still matches its label.

### `run`

All keys are optional:

| Key | Value |
|---|---|
| `sample_mode` | `regular`, `semi_broken_mixed`, `semi_broken_words`, `semi_broken_characters`, `semi_broken_numerics` |
| `field_weights` | object of field type -> nonnegative weight; omit to follow the sample mode |
| `split_fractions` | `[train, val, test]`, summing to 1 |
| `font_style` | `all` or `cursive` |
| `cursive_group` | `""` or a cursive group name |
| `specific_font` | `""` or a font file stem |
| `names_version` | a names folder under `resources/`, for example `name1` |
| `writer_profile` | `writer_style_v1` |
| `samples_per_writer` | positive whole number |

Setting `sample_mode` without `field_weights` makes the field mix follow the
new mode.

## Precedence

Lowest to highest: built-in defaults < recipe < explicit CLI flags or later
GUI edits. For example `--recipe ledger.json --cutoff heavy` uses the
recipe's sections but the heavy cutoff.

## Old presets

Files without `schema_version` (the old flat, single-value slider presets)
are converted in memory when loaded. Each old value becomes a chance of 1
(or 0 when the effect is off) and a `[v, v]` range, with the same clamps as
before; `crop_*` percentages become ink-relative cutoff fractions. Saving
always writes version 2.
