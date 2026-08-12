# Installing handwriting fonts

No font binary is tracked or distributed by this repository. Files that happen
to exist in a local checkout under this directory are user-installed inputs and
must not be described as bundled project assets.

The generator accepts `.ttf` and `.otf` files placed directly in this directory.
If none are present, it checks the Windows fallback paths declared in
`config.py`. Those system fonts are part of the user's Windows installation,
not this project. Linux and macOS users should expect generation to stop with an
actionable `No fonts found` error until a font is installed here.

## Source and license checklist

For every font you install or distribute:

1. Download it from the type foundry, author, or another authoritative source
   over HTTPS. Avoid anonymous font-aggregation mirrors.
2. Read the license that applies to the exact downloaded version. Confirm that
   it permits your intended use, modification, model training, generated-image
   distribution, and—if applicable—font-file redistribution.
3. Keep the original license text. For a locally used font, store it outside the
   repository or under `resources/fonts/LICENSES/`. A distributor must include
   every license/attribution file required by that font's terms.
4. Compute SHA-256 after download and before use. Compare it with a checksum
   published by the source when one exists.
5. Copy `FONT_MANIFEST.example.csv` to the ignored local file
   `FONT_MANIFEST.csv` and record the exact filename, checksum, authoritative
   source URL, license identifier/URL, local license file, and installation time.
6. Run the discovery check below and inspect `run-metadata.json` after generation
   to confirm the effective font selection and resource hashes.

This workflow records evidence; it does not grant rights or certify that a font
license is suitable. If the license or source is unclear, do not use or share
the font.

## Checksums

PowerShell:

```powershell
Get-FileHash .\resources\fonts\Example-Regular.ttf -Algorithm SHA256
```

Linux or macOS:

```bash
sha256sum resources/fonts/Example-Regular.ttf
```

Record lowercase hexadecimal SHA-256 without spaces in your local manifest.
Never invent a checksum from a similarly named release.

## Verify discovery

From the repository root:

```bash
python -c "from src.render import available_fonts; print(*available_fonts(), sep='\n')"
```

The `cursive` filter recognizes configured font filename stems. A differently
named font may remain usable through the `all` pool even if it is not identified
as cursive. Restart the GUI after adding or removing fonts because discovery and
font objects are cached within the process.

## Redistribution

Font binaries remain ignored by Git by default. Do not remove that protection
or commit a font merely because it is free to download. Before redistributing a
font, obtain a documented license review, add the exact license and attribution
required for that version, and update `NOTICE.md`. The project currently makes
no representation that any local font is cleared for redistribution.
