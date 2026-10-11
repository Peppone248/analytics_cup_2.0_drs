"""Configurazione condivisa: costanti del campo e posizione dei dati.

I dati SkillCorner non stanno nella repo (~2 GB e sotto licenza), quindi il
percorso non puo' essere scritto nel codice.

La cartella puntata deve contenere matches/ con una sottocartella per match.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Campo
# ---------------------------------------------------------------------------

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
GOAL_X_LEFT = -PITCH_LENGTH / 2   # -52.5
GOAL_X_RIGHT = PITCH_LENGTH / 2   #  52.5


# ---------------------------------------------------------------------------
# Dati
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(
    os.environ.get("DRS_DATA_DIR") or (REPO_ROOT / "data")
).expanduser()
"""Cartella radice dei dati. Contiene matches/, aggregates/, bodypose/."""

MATCHES_DIR = DATA_DIR / "matches"
"""Una sottocartella per match_id, piu' i layer3_*.csv generati."""

OPENDATA_DIR = Path(
    os.environ.get("DRS_OPENDATA_DIR") or (REPO_ROOT.parent / "opendata" / "data")
).expanduser()
"""Dataset opendata separato, usato solo dagli script esplorativi."""

base = MATCHES_DIR
"""Alias storico: la maggior parte degli script importa `base`.

E' un Path, non una stringa. Gli f-string esistenti (`f"{base}\\\\{m_id}"`)
continuano a funzionare su Windows, ma la forma portabile e' `base / m_id`
oppure match_dir(m_id).
"""


def match_dir(match_id) -> Path:
    """Cartella di un singolo match."""
    return MATCHES_DIR / str(match_id)


def match_file(match_id, suffix: str) -> Path:
    """File dentro la cartella di un match.

    >>> match_file(1899585, "dynamic_events.csv")
    .../matches/1899585/1899585_dynamic_events.csv
    """
    return match_dir(match_id) / f"{match_id}_{suffix}"


def list_match_ids() -> list[str]:
    """I match_id presenti in MATCHES_DIR, ordinati."""
    require_data()
    return sorted(p.name for p in MATCHES_DIR.iterdir() if p.is_dir())


def require_data() -> Path:
    """Fallisce subito e con un messaggio utile se i dati non ci sono."""
    if not MATCHES_DIR.is_dir():
        raise FileNotFoundError(
            f"Cartella match non trovata: {MATCHES_DIR}\n"
            f"DRS_DATA_DIR = {os.environ.get('DRS_DATA_DIR') or '(non impostata)'}\n"
            "Imposta DRS_DATA_DIR sulla cartella che contiene matches/, "
            "oppure metti i dati in <repo>/data/."
        )
    return MATCHES_DIR
