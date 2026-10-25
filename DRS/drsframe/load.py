"""Asse del carico: quanta minaccia una difesa ha dovuto gestire.

Due sorgenti, in ordine di preferenza:
1. `drst` (DRS(t) di Beppe): stress medio per corsa subita, da `runs_with_stress.parquet`
   prodotto da `drst.run_pipeline(...).save(...)`. E' la misura del carico validata con il
   test di dissociazione (OR 1.24 su lead_to_shot, nullo su targeted/received).
2. fallback interno: X̂ medio, l'esposizione attesa stimata dal solo attacco. Correla 0.61
   con lo stress di drst a livello squadra-partita.

Il carico NON e' qualita' difensiva: dice cosa ti hanno chiesto di difendere. La risposta
(DRF) dice come l'hai fatto. I due assi sono quasi ortogonali per corsa (r = 0.07).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_from_drst(runs_path: str | Path) -> pd.DataFrame:
    r = pd.read_parquet(runs_path)
    team_col = "defending_team_id" if "defending_team_id" in r else "def_team"
    out = r.groupby(["match_id", team_col]).agg(load=("stress", "mean"), load_runs=("stress", "size"))
    out = out.reset_index().rename(columns={team_col: "def_team"})
    out["load_source"] = "drst stress per corsa"
    return out


def add_load(team_match: pd.DataFrame, drst_runs: str | Path | None = None) -> pd.DataFrame:
    tm = team_match.copy()
    if drst_runs is not None and Path(drst_runs).exists():
        tm = tm.merge(load_from_drst(drst_runs), on=["match_id", "def_team"], how="left")
    else:
        tm["load"] = tm["Xhat_mean"]
        tm["load_source"] = "X̂ medio (esposizione attesa dal solo attacco)"
    return tm


def team_axes(tm: pd.DataFrame) -> pd.DataFrame:
    """Una riga per squadra: carico, risposta, tiri concessi per partita."""
    cols = {"load": "mean", "DRF": "mean", "DRF_opt": "mean", "DRF_car": "mean",
            "shots_conceded": "mean", "match_id": "nunique"}
    cols = {k: v for k, v in cols.items() if k in tm}
    keys = ["def_team"] + (["team"] if "team" in tm else [])
    return tm.groupby(keys).agg(cols).rename(columns={"match_id": "matches"}).reset_index()
