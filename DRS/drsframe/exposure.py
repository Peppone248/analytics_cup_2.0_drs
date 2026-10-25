"""Esposizione difensiva X(t) e attribuzione per difensore.

Definizioni (sistema orientato, porta difesa a (+L/2, 0)):

    gain_j(t)  = max(0, V(q_j) - V(palla))            minaccia che l'opzione j aggiunge
    x_j(t)     = gain_j(t) * (1 - C_j(t))             quanta di quella minaccia resta scoperta
    x_car(t)   = (V(palla) - V_min) * (1 - Press(t))  portatore libero di progredire
    X(t)       = sum_j x_j(t) + x_car(t)

X e' in unita' EPV (probabilita' di gol del possesso): "quanto valore la difesa sta
lasciando disponibile in questo istante". Il potenziale d'attacco senza difesa e'
Pot(t) = sum_j gain_j + (V(palla) - V_min): stessa formula con copertura zero.

Attribuzione individuale (leave-one-out in forma chiusa, analoga alla LOO della tesi
Molino ma geometrica): togliendo il difensore i la copertura dell'opzione j diventa
1 - (1 - C_j)/(1 - p_ij), quindi

    credit_i(t) = sum_j gain_j (1 - C_j) p_ij / (1 - p_ij)  (+ termine portatore)

e' l'esposizione che la squadra subirebbe in piu' senza di lui (calcolata esatta con
prodotti prefisso/suffisso, non con la divisione per (1 - p_ij) che e' instabile vicino a 1). `blame_i` e' invece
l'esposizione residua dell'attaccante che i dovrebbe marcare (marcatura ungherese).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .geometry import FrameState
from .value import ValueSurface

P_CLIP = 0.999  # stesso clip usato in geometry.py per la copertura


def _prod_excluding(q: np.ndarray) -> np.ndarray:
    """Prodotto sull'ultimo asse escludendo ogni elemento a turno (prefissi x suffissi)."""
    ones = np.ones_like(q[..., :1])
    pre = np.cumprod(np.concatenate([ones, q[..., :-1]], axis=-1), axis=-1)
    suf = np.cumprod(np.concatenate([ones, q[..., :0:-1]], axis=-1), axis=-1)[..., ::-1]
    return pre * suf


def compute_exposure(st: FrameState, vs: ValueSurface, cfg: Config | None = None):
    """Ritorna (frames, defenders): tabella per frame e tabella lunga difensore x frame."""
    cfg = cfg or Config()
    T = st.n
    valid_att = (st.Aid >= 0) & ~st.Agk
    is_car = np.zeros_like(valid_att)
    has_car = st.carrier >= 0
    is_car[np.where(has_car)[0], st.carrier[has_car]] = True
    opt = valid_att & ~is_car

    Vq = vs(st.Q[..., 0], st.Q[..., 1])
    Vp = vs(st.A[..., 0], st.A[..., 1])
    Vb = vs(st.ball[:, 0], st.ball[:, 1])
    vmin = float(np.nanmin(vs.grid))
    gain = np.where(opt, np.maximum(0.0, Vq - Vb[:, None]), 0.0)
    open_ = 1.0 - st.cover
    x_opt = gain * open_
    car_val = np.where(has_car, Vb - vmin, 0.0)
    if not cfg.include_carrier:
        car_val = np.zeros(T)
    x_car = car_val * (1.0 - st.press_carrier)

    X_opt = x_opt.sum(axis=1)
    X = X_opt + x_car
    pot = gain.sum(axis=1) + car_val

    # --- attribuzione per difensore ---------------------------------------
    q = 1.0 - np.clip(st.p_int, 0, P_CLIP)                            # (T, a, d)
    open_wo = _prod_excluding(q) - q.prod(axis=2, keepdims=True)      # aumento di (1-C_j) senza d
    credit = (gain[..., None] * open_wo).sum(axis=1)                  # (T, d)
    if cfg.include_carrier:
        qc = 1.0 - np.clip(st.p_carrier, 0, P_CLIP)                   # (T, d)
        credit += car_val[:, None] * (_prod_excluding(qc[:, None, :])[:, 0] - qc.prod(axis=1, keepdims=True))
    rows = np.arange(T)[:, None]
    m = st.mark
    blame = np.where(m >= 0, x_opt[rows, np.clip(m, 0, 10)], np.nan)
    blame = np.where((m >= 0) & is_car[rows, np.clip(m, 0, 10)], x_car[:, None], blame)
    mark_val = np.where(m >= 0, Vp[rows, np.clip(m, 0, 10)], np.nan)

    marked_any = np.zeros_like(valid_att)
    for d in range(m.shape[1]):
        ok = m[:, d] >= 0
        marked_any[np.where(ok)[0], m[ok, d]] = True
    X_unmarked = np.where(opt & ~marked_any, x_opt, 0.0).sum(axis=1)

    # --- contesto d'attacco (nessuna informazione sui difensori) ----------
    A = st.A
    outfield = valid_att
    ax = np.where(outfield, A[..., 0], np.nan)
    ay = np.where(outfield, A[..., 1], np.nan)
    avx = np.where(outfield, st.Av[..., 0], np.nan)
    L = st.pitch_length
    in_box = (ax > L / 2 - 16.5) & (np.abs(ay) < 20.16)
    bspeed = np.linalg.norm(st.ball_vel, axis=1)

    frames = pd.DataFrame({
        "match_id": st.match_id,
        "frame": st.frame,
        "period": st.period,
        "t_s": st.frame / cfg.fps,
        "att_team": st.att_team,
        "def_team": st.def_team,
        "phase_index": st.phase_index,
        "ip_phase": st.ip_phase,
        "oop_phase": st.oop_phase,
        "ball_x": st.ball[:, 0],
        "ball_y": st.ball[:, 1],
        # metrica
        "X": X,
        "X_opt": X_opt,
        "X_car": x_car,
        "X_max": np.maximum(x_opt.max(axis=1), x_car),
        "X_unmarked": X_unmarked,
        "n_open": ((gain > 0.005) & (open_ > 0.5)).sum(axis=1),
        "press_carrier": np.where(has_car, st.press_carrier, np.nan),
        "def_compact_x": np.nanstd(np.where(st.Dgk, np.nan, st.D[..., 0]), axis=1),
        "def_line_x": np.nanmax(np.where(st.Dgk, np.nan, st.D[..., 0]), axis=1),
        # contesto (solo attacco)
        "c_ball_x": st.ball[:, 0],
        "c_ball_absy": np.abs(st.ball[:, 1]),
        "c_ball_z": np.nan_to_num(st.ball[:, 2], nan=0.0),
        "c_ball_speed": bspeed,
        "c_ball_vx": st.ball_vel[:, 0],
        "c_carrier": has_car.astype(int),
        "c_V_ball": Vb,
        "c_pot": pot,
        "c_gain_max": gain.max(axis=1),
        "c_gain_sum": gain.sum(axis=1),
        "c_n_gain": (gain > 0.005).sum(axis=1),
        "c_att_mean_x": np.nanmean(ax, axis=1),
        "c_att_max_x": np.nanmax(ax, axis=1),
        "c_att_width": np.nanstd(ay, axis=1),
        "c_att_in_box": in_box.sum(axis=1),
        "c_att_final_third": (ax > L / 6).sum(axis=1),
        "c_att_vx_mean": np.nanmean(avx, axis=1),
        "c_att_vx_max": np.nanmax(avx, axis=1),
    })
    frames["c_ip_phase"] = frames["ip_phase"].astype(str)

    T_, D_ = st.Did.shape
    dmask = (st.Did >= 0) & ~st.Dgk
    tt, dd = np.where(dmask)
    defenders = pd.DataFrame({
        "match_id": st.match_id,
        "frame": st.frame[tt],
        "def_team": st.def_team[tt],
        "player_id": st.Did[tt, dd],
        "d_ball": st.d_ball[tt, dd],
        "d_mark": st.d_mark[tt, dd],
        "d_ideal": st.d_ideal[tt, dd],
        "goal_side": st.goal_side[tt, dd],
        "marking": (st.mark[tt, dd] >= 0).astype(int),
        "marked_player_id": np.where(st.mark[tt, dd] >= 0,
                                     st.Aid[tt, np.clip(st.mark[tt, dd], 0, 10)], -1),
        "marked_value": mark_val[tt, dd],
        "credit": credit[tt, dd],
        "blame": blame[tt, dd],
        "X": X[tt],
    })
    extra = dict(gain=gain, x_opt=x_opt, x_car=x_car, credit=credit, blame=blame, Vb=Vb, is_car=is_car)
    return frames, defenders, extra
