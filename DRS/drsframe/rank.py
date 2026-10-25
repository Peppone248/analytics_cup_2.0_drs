"""Ranking interno alle partite: episodi, squadre, difensori.

Tre livelli, tutti calcolati **dentro ogni partita** (e poi riportati anche a scala
campionato):

1. **episodi** — ogni attacco subito ha un rango di pericolosita' (z-score) e un rango di
   risposta (R medio sulla finestra). Gli "episodi critici" sono attacchi pericolosi
   (z > 1) difesi peggio dell'atteso (R < 0): sono i candidati per il video.
2. **squadre** — DRF score = media di R pesata con Phi(z): la difesa conta di piu'
   quando l'attacco subito e' pericoloso. Le due squadre di ogni partita sono ordinate.
3. **difensori** — per 90' fuori possesso: credit (copertura fornita, LOO),
   blame (esposizione residua dell'uomo assegnato), distanze da palla e uomo, quota
   di frame lato porta. Ranking per percentile dentro la partita sullo scarto dalla
   mediana del proprio gruppo di ruolo (CB, FB, MF, WM, FW).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


def add_def_team(ev: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    teams = frames.groupby("match_id")["att_team"].unique()
    def other(row):
        ts = teams.get(row.match_id, [])
        o = [t for t in ts if t != row.team_id]
        return o[0] if o else -1
    ev = ev.copy()
    ev["def_team"] = ev.apply(other, axis=1)
    return ev


def episode_ranking(ev: pd.DataFrame) -> pd.DataFrame:
    ev = ev.copy()
    g = ev.groupby(["match_id", "def_team"])
    ev["rank_danger_in_match"] = g["danger_z"].rank(ascending=False, method="first").astype(int)
    ev["rank_response_in_match"] = g["R_win"].rank(ascending=True, method="first")  # 1 = peggiore
    ev["critical"] = (ev["danger_z"] > 1.0) & (ev["R_win"] < 0)
    ev["danger_x_response"] = ev["danger_w"] * ev["R_win"]
    return ev


def team_match_scores(ev: pd.DataFrame, frames: pd.DataFrame, names: dict | None = None) -> pd.DataFrame:
    fr = frames.groupby(["match_id", "def_team"]).agg(
        frames=("X", "size"), X_mean=("X", "mean"), Xhat_mean=("Xhat", "mean"),
        R_frame_mean=("R", "mean"),
    )
    e = ev.dropna(subset=["R_win"])
    def agg(d):
        w = d["danger_w"].to_numpy()
        r = d["R_win"].to_numpy()
        return pd.Series({
            "n_attacks": len(d),
            "n_dangerous": int((d["danger_z"] > 1).sum()),
            "danger_z_mean": d["danger_z"].mean(),
            "DRF": float((w * r).sum() / max(w.sum(), 1e-9)),
            # canali: opzioni coperte (evita i tiri) e pressione sul portatore (recupera palla)
            "DRF_opt": float((w * d["R_opt_win"]).sum() / max(w.sum(), 1e-9)) if "R_opt_win" in d else np.nan,
            "DRF_car": float((w * d["R_car_win"]).sum() / max(w.sum(), 1e-9)) if "R_car_win" in d else np.nan,
            "DRF_dangerous": float(r[d["danger_z"].to_numpy() > 1].mean()) if (d["danger_z"] > 1).any() else np.nan,
            "share_dangerous_contained": float((r[d["danger_z"].to_numpy() > 1] > 0).mean())
            if (d["danger_z"] > 1).any() else np.nan,
            "n_critical": int(d["critical"].sum()) if "critical" in d else np.nan,
            "shots_conceded": int(((d["event_type"] == "player_possession") & (d["end_type"] == "shot")).sum()),
        })
    ag = e.groupby(["match_id", "def_team"]).apply(agg, include_groups=False)
    out = fr.join(ag).reset_index()
    out["rank_in_match"] = out.groupby("match_id")["DRF"].rank(ascending=False).astype(int)
    out["DRF_z_league"] = (out["DRF"] - out["DRF"].mean()) / out["DRF"].std()
    if names:
        out.insert(2, "team", out["def_team"].map(names))
    return out.sort_values(["match_id", "rank_in_match"]).reset_index(drop=True)


def team_league_table(team_match: pd.DataFrame) -> pd.DataFrame:
    g = team_match.groupby(["def_team"] + (["team"] if "team" in team_match else []))
    t = g.agg(
        matches=("match_id", "nunique"), attacks=("n_attacks", "sum"),
        DRF=("DRF", "mean"), DRF_sd=("DRF", "std"), R_frame=("R_frame_mean", "mean"),
        DRF_opt=("DRF_opt", "mean"), DRF_car=("DRF_car", "mean"),
        shots_conceded=("shots_conceded", "mean"),
        X=("X_mean", "mean"), Xhat=("Xhat_mean", "mean"),
        share_dangerous_contained=("share_dangerous_contained", "mean"),
        wins_in_match=("rank_in_match", lambda s: int((s == 1).sum())),
    ).reset_index()
    # shrinkage empirico-bayes verso la media in base al numero di partite
    between = max(t["DRF"].var() - (t["DRF_sd"].pow(2).fillna(t["DRF_sd"].pow(2).mean()) / t["matches"]).mean(), 1e-12)
    within = t["DRF_sd"].pow(2).fillna(t["DRF_sd"].pow(2).mean())
    k = between / (between + within / t["matches"])
    t["DRF_shrunk"] = t["DRF"].mean() + k * (t["DRF"] - t["DRF"].mean())
    t["shrink_k"] = k
    return t.sort_values("DRF_shrunk", ascending=False).reset_index(drop=True)


ROLE_GROUP = {
    "CB": "CB", "LCB": "CB", "RCB": "CB",
    "LB": "FB", "RB": "FB", "LWB": "FB", "RWB": "FB",
    "DM": "MF", "LDM": "MF", "RDM": "MF", "CM": "MF", "LCM": "MF", "RCM": "MF", "AM": "MF",
    "LM": "WM", "RM": "WM", "LW": "WM", "RW": "WM",
    "CF": "FW", "ST": "FW", "LF": "FW", "RF": "FW",
}


def defender_ranking(defenders: pd.DataFrame, players: pd.DataFrame, cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or Config()
    d = defenders
    g = d.groupby(["match_id", "def_team", "player_id"])
    out = g.agg(
        frames=("X", "size"),
        d_ball=("d_ball", "mean"),
        d_mark=("d_mark", "mean"),
        d_ideal=("d_ideal", "mean"),
        goal_side=("goal_side", "mean"),
        marking_share=("marking", "mean"),
        credit=("credit", "sum"),
        blame=("blame", "sum"),
    ).reset_index()
    out["minutes"] = out["frames"] / cfg.fps / 60
    out["credit_p90"] = out["credit"] / out["minutes"] * 90
    out["blame_p90"] = out["blame"] / out["minutes"] * 90
    out["net_p90"] = out["credit_p90"] - out["blame_p90"]
    out = out[out["frames"] >= cfg.min_frames_defender].copy()
    pl = players.drop_duplicates("id").set_index("id")
    out["player"] = out["player_id"].map(pl["short_name"])
    out["role"] = out["player_id"].map(pl["role"])
    # il credit dipende molto dal ruolo (un esterno copre piu' opzioni di un centrale):
    # il ranking usa lo scarto dalla mediana di campionato del proprio gruppo di ruolo
    out["role_group"] = out["role"].map(ROLE_GROUP).fillna("other")
    out["net_vs_role"] = out["net_p90"] - out.groupby("role_group")["net_p90"].transform("median")
    out["pct_in_match"] = out.groupby("match_id")["net_vs_role"].rank(pct=True)
    out["rank_in_team"] = out.groupby(["match_id", "def_team"])["net_vs_role"].rank(ascending=False).astype(int)
    return out.sort_values(["match_id", "pct_in_match"], ascending=[True, False]).reset_index(drop=True)
