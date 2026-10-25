"""Placebo: la difesa di un altro momento.

Domanda: l'effetto della risposta sui tiri viene davvero dalla posizione dei difensori,
o R sta leggendo qualcosa dell'attacco (o del contesto di squadra) per vie traverse?

Costruzione: per ogni frame, le posizioni dei giocatori della squadra che difende vengono
sostituite con le loro posizioni di `shift_s` secondi prima, nello stesso tempo di gioco
(slittamento circolare dentro il periodo). Attacco, palla e possesso restano veri. Poi si
ricalcola tutto con la stessa pipeline: geometria, esposizione, attesa out-of-fold,
risposta per evento, test sugli esiti.

Se R misura la difesa, nel placebo gli odds ratio devono tornare a ~1. Se restano lontani
da 1, una parte dell'effetto e' un artefatto (per esempio la forma media di squadra, che
sopravvive allo slittamento).
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .events import attack_events, danger_zscore, event_response
from .expected import add_expected
from .exposure import compute_exposure
from .geometry import build_frame_state
from .io import MatchData, list_matches, load_match, phase_per_frame
from .build import load_all_events
from . import validate as V


def shift_defenders(md: MatchData, shift_s: float, fps: float = 10.0) -> MatchData:
    """Copia del match con i difensori di `shift_s` secondi prima (stesso periodo, circolare)."""
    k = int(round(shift_s * fps))
    ph = phase_per_frame(md)
    att = ph["att_team"].to_numpy()
    team_of = md.players["team_id"].to_numpy()
    pos = md.pos.copy()
    shifted = np.empty_like(md.pos)
    for p in np.unique(md.period):
        idx = np.where(md.period == p)[0]
        shifted[idx] = np.roll(md.pos[idx], k, axis=0)
    # per ogni frame: True per i giocatori che in quel frame difendono
    defending = (att[:, None] >= 0) & (team_of[None, :] != att[:, None])
    pos[defending] = shifted[defending]
    return dataclasses.replace(md, pos=pos)


def run_placebo(root, cache_dir, value_surface, cfg: Config | None = None, shift_s: float = 90.0,
                match_ids=None, verbose: bool = True) -> dict:
    cfg = cfg or Config()
    root = Path(root)
    match_ids = match_ids or list_matches(root)
    F = []
    for m in match_ids:
        md = load_match(root, m, cache_dir=cache_dir)
        st = build_frame_state(shift_defenders(md, shift_s, cfg.fps), cfg)
        fr, _, _ = compute_exposure(st, value_surface, cfg)
        F.append(fr)
        if verbose:
            print(f"  placebo {shift_s:.0f}s · match {m}", flush=True)
    frames = pd.concat(F, ignore_index=True)
    add_expected(frames, cfg, verbose=False)
    atk = danger_zscore(attack_events(load_all_events(root, match_ids)))
    atk = event_response(atk, frames, cfg)
    atk = event_response(atk, frames, cfg, anchor="start", pre_s=2.0, suffix="_pre")
    response = pd.concat([V.response_matters_test(atk, "R_win"), V.response_matters_test(atk, "R_pre")])
    channels = V.channel_test(atk)
    response.insert(0, "shift_s", shift_s)
    channels.insert(0, "shift_s", shift_s)
    return dict(response=response, channels=channels, frames=frames, attacks=atk)


def compare_with_real(real: pd.DataFrame, placebo: pd.DataFrame, keys, cols) -> pd.DataFrame:
    """Tabella affiancata reale vs placebo per gli stessi test."""
    r = real.set_index(keys)[cols].add_suffix("_reale")
    p = placebo.set_index(keys)[cols].add_suffix("_placebo")
    return r.join(p).reset_index()
