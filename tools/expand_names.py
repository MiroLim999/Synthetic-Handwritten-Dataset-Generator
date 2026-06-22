"""
One-off utility: expand the name lists to ~1000 unique entries each.

Harvests authentic names from the Faker library using Filipino (fil_PH) and
Spanish (es_ES) locales -- the right mix for old Maasin civil registry records
(Spanish-era given names + Filipino/Visayan & Spanish surnames). The curated
local surnames already in the repo are kept and prioritised.

Run:
  python -m tools.expand_names
"""

import unicodedata
from pathlib import Path

from faker import Faker

import config

TARGET = 1000
MAX_ATTEMPTS = 400_000   # safety cap on sampling loops

fil = Faker("fil_PH")
spa = Faker("es_ES")
eng = Faker("en_US")     # used only to build an Anglo-name blocklist


def _strip_accents(text: str) -> str:
    """'María' -> 'Maria'. Old PH documents usually wrote names without accents,
    and it avoids missing-glyph rendering in some handwriting fonts."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _clean(name: str) -> str:
    """Trim, collapse spaces, strip accents, drop odd entries."""
    name = _strip_accents(" ".join(name.split()))
    if not name or any(ch.isdigit() for ch in name):
        return ""
    if "," in name or "." in name:
        return ""
    return name


def build_anglo_blocklist(first: bool, draws: int = 25_000) -> set[str]:
    """Sample many English names to exclude Anglo names from the result."""
    block: set[str] = set()
    sampler = eng.first_name if first else eng.last_name
    for _ in range(draws):
        c = _clean(sampler())
        if c:
            block.add(c.lower())
    return block


def _read_existing(path: Path) -> list[str]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def harvest(samplers, target: int, seed_list: list[str], blocklist: set[str]) -> list[str]:
    """Collect `target` unique names. Curated seeds are kept unconditionally;
    Faker-harvested names are dropped if they are in the Anglo blocklist."""
    seen: dict[str, None] = {}            # preserves insertion order
    for s in seed_list:                   # keep curated names first (not filtered)
        c = _clean(s)
        if c:
            seen.setdefault(c, None)

    attempts = 0
    while len(seen) < target and attempts < MAX_ATTEMPTS:
        for sampler in samplers:
            c = _clean(sampler())
            if c and c.lower() not in blocklist:
                seen.setdefault(c, None)
        attempts += 1

    return list(seen.keys())[:target]


def write_list(path: Path, names: list[str]):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(names) + "\n")
    print(f"  {path.name:<18} {len(names)} names")


# Authentic seed bases (kept first, never filtered). Faker fills the rest.
SEED_FIRST = [
    "Maria", "Jose", "Juan", "Pedro", "Vicente", "Felix", "Eugenio", "Mariano",
    "Francisco", "Crispin", "Bonifacio", "Eustaquio", "Saturnino", "Cresencio",
    "Florencio", "Macario", "Anastacio", "Gregorio", "Ignacio", "Teodoro",
    "Antonio", "Ricardo", "Eduardo", "Roberto", "Manuel", "Ramon", "Fernando",
    "Emilio", "Andres", "Benigno", "Apolonio", "Aurelio", "Basilio", "Cipriano",
    "Dionisio", "Esteban", "Faustino", "Genaro", "Hilario", "Isidro", "Jacinto",
    "Leoncio", "Lucio", "Marcelo", "Maximo", "Melchor", "Nicanor", "Pascual",
    "Quirino", "Raymundo", "Rufino", "Catalina", "Felisa", "Concepcion",
    "Remedios", "Purificacion", "Asuncion", "Visitacion", "Soledad", "Dolores",
    "Esperanza", "Rosario", "Carmen", "Trinidad", "Consolacion", "Natividad",
    "Lourdes", "Pilar", "Filomena", "Cristina", "Teresita", "Luzviminda",
    "Corazon", "Cecilia", "Clara", "Francisca", "Gregoria", "Herminia",
    "Isabel", "Josefa", "Juliana", "Margarita", "Petronila", "Rosalia",
    "Victoria", "Lucia", "Magdalena", "Encarnacion",
]
SEED_SURNAMES = [
    "Cinco", "Maamo", "Maglasang", "Montejo", "Nierras", "Escala", "Espina",
    "Apura", "Cabug-os", "Demain", "Elumba", "Lumayno", "Mancera", "Manatad",
    "Salas", "Tibon", "Cura", "Dagatan", "Maputi", "Daligdig", "Gerona",
    "Sienes", "Lacaba", "Cabahug", "Mahinay", "Maturan", "Neri", "Tampus",
    "Tirol", "Singson", "Patalinghug", "Sabandal", "Veloso", "Pedroso",
    "Dela Cruz", "Santos", "Reyes", "Garcia", "Mendoza", "Flores", "Ramos",
    "Gonzales", "Villanueva", "Castillo", "Mercado", "del Rosario", "Aguilar",
    "Castro", "Fernandez", "Rivera", "Bautista", "Macaraeg",
]


def main():
    print("Building Anglo-name blocklists...")
    block_first = build_anglo_blocklist(first=True)
    block_last = build_anglo_blocklist(first=False)
    print(f"  blocked first names: {len(block_first)}, surnames: {len(block_last)}")

    print("Harvesting Spanish + Filipino names...")
    # First names: Spanish + Filipino given names (both sexes), Anglo filtered out.
    first_samplers = [
        spa.first_name_male, spa.first_name_female,
        fil.first_name_male, fil.first_name_female,
    ]
    # Surnames: Spanish + Filipino family names, Anglo filtered out.
    last_samplers = [spa.last_name, fil.last_name]

    first = harvest(first_samplers, TARGET, SEED_FIRST, block_first)
    last = harvest(last_samplers, TARGET, SEED_SURNAMES, block_last)
    # middle names = maternal surnames -> same pool as last names.
    middle = harvest(last_samplers, TARGET, SEED_SURNAMES + last, block_last)

    write_list(config.NAMES_DIR / "first_names.txt", first)
    write_list(config.NAMES_DIR / "middle_names.txt", middle)
    write_list(config.NAMES_DIR / "last_names.txt", last)

    print("\nDone.")


if __name__ == "__main__":
    main()
