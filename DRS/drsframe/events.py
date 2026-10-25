"""Attacchi subiti (eventi SkillCorner) e loro pericolosita' in z-score.

Due famiglie di attacco:
- **palla al piede**: `player_possession`, pericolo = `possession_epv_for_start`
  (EPV SkillCorner all'inizio del possesso);
- **lontano dalla palla**: `off_ball_run`, pericolo = `xthreat` (valore xT dell'opzione
  creata dalla corsa).

Le due scale non sono confrontabili, quindi il pericolo viene trasformato in log e
standardizzato **per tipo di evento sull'intero campionato**:

    z_e = (log(raw_e + eps) - mu_tipo) / sd_tipo

z = 0 e' un attacco di pericolosita' media per il suo tipo, z = +2 un attacco fra il 2-3%
piu' pericoloso. Il peso nella media difensiva e' w_e = Phi(z_e) in (0, 1).

Per ogni evento si misura la risposta difensiva sui frame [start - pre, end + post]:
media di X, Xhat, R = Xhat - X, piu' la geometria specifica (chi marcava il runner,
a che distanza, se era lato porta, quanto il runner era scoperto).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from .config import Config
from .geometry import FrameState

DANGER_SOURCES = {
    "player_possession": "possession_epv_for_start",
    "off_ball_run": "xthreat",
}
EPS = 1e-4
# colonne mediate sulla finestra di ogni evento (quelle presenti in `frames`)
RESPONSE_COLS = ("X", "Xhat", "R", "R_opt", "R_car")


def attack_events(events: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "event_id", "match_id", "event_type", "event_subtype", "frame_start", "frame_end",
        "period", "team_id", "player_id", "player_name", "x_start", "y_start", "x_end", "y_end",
        "third_start", "team_in_possession_phase_type", "team_out_of_possession_phase_type",
        "targeted", "received", "lead_to_shot", "lead_to_goal", "end_type",
        "minute_start", "second_start",
    ]
    out = []
    for et, col in DANGER_SOURCES.items():
        e = events[events["event_type"] == et].copy()
        e["danger_raw"] = e[col].astype(float)
        e = e[[c for c in keep if c in e.columns] + ["danger_raw"]]
        out.append(e.dropna(subset=["danger_raw"]))
    ev = pd.concat(out, ignore_index=True)
    return ev


def danger_zscore(ev: pd.DataFrame, reference: pd.DataFrame | None = None) -> pd.DataFrame:
    """z-score del pericolo per tipo di evento. `reference` = distribuzione di riferimento
    (default: ev stesso, cioe' il campionato)."""
    ref = ev if reference is None else reference
    ev = ev.copy()
    ev["danger_log"] = np.log(ev["danger_raw"] + EPS)
    stats = (
        np.log(ref["danger_raw"] + EPS).groupby(ref["event_type"]).agg(["mean", "std"])
    )
    mu = ev["event_type"].map(stats["mean"])
    sd = ev["event_type"].map(stats["std"])
    ev["danger_z"] = (ev["danger_log"] - mu) / sd
    ev["danger_w"] = norm.cdf(ev["danger_z"])
    return ev


def event_geometry(st: FrameState, extra: dict, ev_match: pd.DataFrame) -> pd.DataFrame:
    """Geometria della risposta difensiva per gli eventi di un match."""
    fidx = pd.Series(np.arange(st.n), index=st.frame)
    rows = []
    for _, e in ev_match.iterrows():
        sel = fidx.reindex(np.arange(int(e.frame_start), int(e.frame_end) + 1)).dropna()
        sel = sel.to_numpy(int)
        r = {"event_id": e.event_id}
        if len(sel) == 0:
            rows.append(r)
            continue
        pid = int(e.player_id)
        slot = (st.Aid[sel] == pid)
        has = slot.any(axis=1)
        if has.sum() == 0:
            rows.append(r)
            continue
        s_sel = sel[has]
        s_slot = slot[has].argmax(axis=1)
        cover = st.cover[s_sel, s_slot]
        r["att_open_start"] = 1 - cover[0]
        r["att_open_max"] = float((1 - cover).max())
        r["att_exposure_mean"] = float(extra["x_opt"][s_sel, s_slot].mean())
        if e.event_type == "player_possession":
            r["att_exposure_mean"] = float(extra["x_car"][s_sel].mean())
            r["press_mean"] = float(st.press_carrier[s_sel].mean())
        # chi marcava l'attaccante: difensore assegnato per piu' frame
        mk = st.mark[s_sel] == s_slot[:, None]                    # (n, 11d)
        cnt = mk.sum(axis=0)
        if cnt.max() > 0:
            d = int(cnt.argmax())
            on = mk[:, d]
            r["marker_id"] = int(st.Did[s_sel[on][0], d])
            r["marker_frac"] = float(on.mean())
            r["marker_d_mark"] = float(np.nanmean(st.d_mark[s_sel[on], d]))
            r["marker_goal_side"] = float(np.nanmean(st.goal_side[s_sel[on], d]))
            r["marker_d_ball"] = float(np.nanmean(st.d_ball[s_sel[on], d]))
        else:
            r["marker_id"] = -1
            r["marker_frac"] = 0.0
        # difensore piu' vicino all'attaccante a inizio evento
        dd = np.linalg.norm(st.D[s_sel[0]] - st.A[s_sel[0], s_slot[0]], axis=-1)
        dd = np.where(np.isnan(dd), np.inf, dd)
        r["nearest_def_id"] = int(st.Did[s_sel[0], dd.argmin()])
        r["nearest_def_dist"] = float(dd.min())
        rows.append(r)
    return pd.DataFrame(rows)


def event_response(ev: pd.DataFrame, frames: pd.DataFrame, cfg: Config | None = None,
                   anchor: str = "window", pre_s: float | None = None, suffix: str = "_win") -> pd.DataFrame:
    """Media di X, Xhat, R su una finestra per evento (frames deve avere Xhat e R).

    anchor="window": [start - pre, end + post]  (risposta durante l'attacco, default)
    anchor="start":  [start - pre, start]       (stato difensivo quando l'attacco parte:
                                                 non puo' contenere il tiro, test non meccanico)
    """
    cfg = cfg or Config()
    pre = int((cfg.event_pre_s if pre_s is None else pre_s) * cfg.fps)
    post = int(cfg.event_post_s * cfg.fps)
    out = []
    for mid, fm in frames.groupby("match_id", sort=False):
        fm = fm.sort_values("frame")
        fr = fm["frame"].to_numpy()
        cols = [c for c in RESPONSE_COLS if c in fm.columns]
        cs = {c: np.concatenate([[0.0], np.cumsum(fm[c].to_numpy(float))]) for c in cols}
        em = ev[ev["match_id"] == mid]
        a = np.searchsorted(fr, em["frame_start"].to_numpy() - pre, side="left")
        end = em["frame_start"].to_numpy() if anchor == "start" else em["frame_end"].to_numpy() + post
        b = np.searchsorted(fr, end, side="right")
        n = np.maximum(b - a, 0)
        res = pd.DataFrame({"event_id": em["event_id"].to_numpy(), "match_id": mid, f"n{suffix}": n})
        for c in cols:
            s = cs[c][b] - cs[c][a]
            res[f"{c}{suffix}"] = np.where(n > 0, s / np.maximum(n, 1), np.nan)
        out.append(res)
    return ev.merge(pd.concat(out, ignore_index=True), on=["event_id", "match_id"], how="left")
