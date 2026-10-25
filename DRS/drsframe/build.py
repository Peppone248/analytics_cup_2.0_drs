"""Costruzione dei dataset frame-level per tutti i match, con cache su disco."""
from __future__ import annotations

import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .events import attack_events, event_geometry
from .exposure import compute_exposure
from .geometry import FrameState, build_frame_state
from .io import list_matches, load_match
from .value import ValueSurface, fit_value_surface, possession_epv_table


def load_all_players(root: str | Path, match_ids=None) -> pd.DataFrame:
    from .io import _players_table
    import json
    root = Path(root)
    match_ids = match_ids or list_matches(root)
    out = []
    for m in match_ids:
        meta = json.loads((root / str(m) / f"{m}_match.json").read_text(encoding="utf-8"))
        t = _players_table(meta)
        t["match_id"] = m
        out.append(t)
    return pd.concat(out, ignore_index=True)


def team_names(root: str | Path, match_ids=None) -> dict:
    import json
    root = Path(root)
    names = {}
    for m in match_ids or list_matches(root):
        meta = json.loads((root / str(m) / f"{m}_match.json").read_text(encoding="utf-8"))
        for k in ("home_team", "away_team"):
            names[int(meta[k]["id"])] = meta[k].get("short_name") or meta[k]["name"]
    return names


def load_all_events(root: str | Path, match_ids=None) -> pd.DataFrame:
    root = Path(root)
    match_ids = match_ids or list_matches(root)
    out = []
    for m in match_ids:
        e = pd.read_csv(root / str(m) / f"{m}_dynamic_events.csv", low_memory=False)
        out.append(e)
    return pd.concat(out, ignore_index=True)


def save_state(st: FrameState, path: Path) -> None:
    arrs = {f.name: getattr(st, f.name) for f in fields(st)}
    for k in ("ip_phase", "oop_phase"):
        arrs[k] = arrs[k].astype(str)
    np.savez_compressed(path, **arrs)


def load_state(path: Path) -> FrameState:
    z = np.load(path, allow_pickle=False)
    kw = {k: z[k] for k in z.files}
    kw["match_id"] = int(kw["match_id"])
    kw["pitch_length"] = float(kw["pitch_length"])
    kw["pitch_width"] = float(kw["pitch_width"])
    kw["ip_phase"] = kw["ip_phase"].astype(object)
    kw["oop_phase"] = kw["oop_phase"].astype(object)
    return FrameState(**kw)


def get_state(root, match_id, cfg: Config, cache_dir: Path) -> FrameState:
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / f"{match_id}_state.npz"
    if p.exists():
        return load_state(p)
    md = load_match(root, match_id, cache_dir=cache_dir)
    st = build_frame_state(md, cfg)
    save_state(st, p)
    return st


def build_dataset(
    root: str | Path,
    cache_dir: str | Path,
    cfg: Config | None = None,
    match_ids=None,
    value_surface: ValueSurface | None = None,
    verbose: bool = True,
):
    """Ritorna (frames, defenders, attack_events_con_geometria, events, value_surface)."""
    cfg = cfg or Config()
    root, cache_dir = Path(root), Path(cache_dir)
    match_ids = match_ids or list_matches(root)
    events = load_all_events(root, match_ids)
    if value_surface is None:
        value_surface = fit_value_surface(
            possession_epv_table(events), bandwidth=cfg.value_bandwidth, step=cfg.value_grid_step
        )
    F, D, G = [], [], []
    atk = attack_events(events)
    for m in match_ids:
        t0 = time.time()
        st = get_state(root, m, cfg, cache_dir)
        fr, de, extra = compute_exposure(st, value_surface, cfg)
        F.append(fr)
        D.append(de)
        g = event_geometry(st, extra, atk[atk["match_id"] == m])
        g["match_id"] = m
        G.append(g)
        if verbose:
            print(f"  match {m}: {st.n} frame validi ({time.time() - t0:.1f}s)", flush=True)
    geo = pd.concat(G, ignore_index=True)
    atk = atk.merge(geo, on=["match_id", "event_id"], how="left")
    return (pd.concat(F, ignore_index=True), pd.concat(D, ignore_index=True), atk, events,
            value_surface)
