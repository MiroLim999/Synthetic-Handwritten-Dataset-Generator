# Notices

## Project licensing status

This repository does not currently contain a project-wide source-code license.
Nothing in this notice should be interpreted as granting rights to copy,
modify, or redistribute the project code. The repository owner must choose and
add a project license before third parties can rely on such permission.

## Fonts

No font binary is tracked or intentionally distributed by this repository.
`resources/fonts/` is an ignored location for user-supplied inputs. Fonts found
in a developer's checkout or operating system remain subject to their own
licenses and are not covered by any project license. See
`resources/fonts/README.md` and `resources/fonts/FONT_MANIFEST.example.csv` for
the required source, license, attribution, and checksum workflow.

As of this notice, there are no bundled font license texts or attributions to
enumerate. If a future release distributes fonts, update this notice and add
the exact upstream license text under `resources/fonts/LICENSES/` before the
binary is committed.

## Python dependencies

Pillow, NumPy, tqdm, and optional development tools are independently licensed
third-party packages. Their names in dependency files do not incorporate them
into the project or relicense them. Review the installed package metadata and
upstream license files for the exact resolved versions when distributing an
environment or application bundle.

## Data and generated artifacts

Name lists, vocabulary, user-supplied fonts, handwriting samples, and generated
images can carry separate provenance, privacy, contractual, or license
obligations. The generator's metadata and checksums improve traceability but do
not establish ownership or permission. A distributor is responsible for
reviewing every input and satisfying consent, attribution, retention, sharing,
and deletion obligations.
