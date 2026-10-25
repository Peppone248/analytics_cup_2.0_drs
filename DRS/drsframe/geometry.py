"""Geometria difensiva frame-by-frame.

Per ogni frame con una squadra in possesso (da phases_of_play):

1. **orientamento**: tutto viene ruotato di 180 gradi quando la squadra in possesso attacca
   da destra a sinistra, cosi' l'attacco va sempre verso +x e la porta difesa e' (L/2, 0);
2. **time-to-intercept** (Spearman 2018, Bekkers 2025 *Pressing Intensity*):
       T_i(q) = t_r + |p_i + v_i t_r - q| / v_max
   tempo che il difensore i impiega a raggiungere il punto q mantenendo per t_r secondi
   la velocita' attuale; trasformato in probabilita' con una logistica di scala sigma;
3. **copertura di un'opzione di passaggio** j: il difensore contende se arriva sul punto
   di ricezione q_j entro l'arrivo della palla (+ margine) oppure se taglia la linea di
   passaggio palla->q_j prima della palla. C_j = 1 - prod_i (1 - p_ij);
4. **marcatura**: assegnazione ungherese difensori di movimento -> attaccanti di
   movimento, costo = distanza dal punto di marcatura ideale (lato porta), opzione
   "a zona" oltre una distanza massima, isteresi per stabilizzare l'assegnazione nel tempo.

Le quantita' dipendenti dal valore della posizione (esposizione) sono calcolate dopo,
in exposure.py, cosi' la superficie di valore puo' essere rifittata per fold senza
ricalcolare la geometria.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.optimize import linear_sum_assignment

from .config import Config
from .io import MatchData, phase_per_frame

NSLOT = 11


@dataclass
class FrameState:
    """Stato geometrico orientato dei frame validi di un match.

    Tutte le coordinate sono nel sistema della squadra in possesso (attacca verso +x).
    Gli slot vuoti hanno id -1 e posizione NaN.
    """
    match_id: int
    pitch_length: float
    pitch_width: float
    frame: np.ndarray        # (T,)
    period: np.ndarray       # (T,)
    sign: np.ndarray         # (T,) +1 se l'attacco va verso +x nel sistema assoluto
    att_team: np.ndarray     # (T,)
    def_team: np.ndarray     # (T,)
    phase_index: np.ndarray  # (T,)
    ip_phase: np.ndarray     # (T,) object
    oop_phase: np.ndarray    # (T,) object
    ball: np.ndarray         # (T, 3) orientata
    ball_vel: np.ndarray     # (T, 2)
    A: np.ndarray            # (T, 11, 2) attaccanti
    Av: np.ndarray           # (T, 11, 2)
    Aid: np.ndarray          # (T, 11)
    Agk: np.ndarray          # (T, 11) bool
    D: np.ndarray            # (T, 11, 2) difensori
    Dv: np.ndarray
    Did: np.ndarray
    Dgk: np.ndarray
    carrier: np.ndarray      # (T,) slot attaccante in possesso, -1 se palla libera/in volo
    Q: np.ndarray            # (T, 11, 2) punto di ricezione proiettato per ogni attaccante
    p_int: np.ndarray        # (T, 11 att, 11 def) prob. che il difensore contenda l'opzione
    cover: np.ndarray        # (T, 11) copertura C_j
    p_carrier: np.ndarray    # (T, 11 def) prob. di arrivare sul portatore entro T_press
    press_carrier: np.ndarray  # (T,) pressione totale sul portatore
    mark: np.ndarray         # (T, 11 def) slot attaccante marcato, -1 = a zona / portiere
    d_ball: np.ndarray       # (T, 11 def) distanza difensore-palla
    d_mark: np.ndarray       # (T, 11 def) distanza difensore-attaccante marcato
    d_ideal: np.ndarray      # (T, 11 def) distanza dal punto di marcatura ideale
    goal_side: np.ndarray    # (T, 11 def) 1 se piu' vicino alla porta dell'attaccante marcato

    @property
    def n(self) -> int:
        return len(self.frame)

    @property
    def goal(self) -> np.ndarray:
        """Porta difesa (attaccata) nel sistema orientato."""
        return np.array([self.pitch_length / 2.0, 0.0], dtype=np.float32)


# ---------------------------------------------------------------------------
# primitive vettoriali
# ---------------------------------------------------------------------------

def logistic_time(dt: np.ndarray, sigma: float) -> np.ndarray:
    """P(arrivo in tempo) dato il margine dt = t_ball - t_defender (secondi)."""
    k = np.pi / (np.sqrt(3.0) * sigma)
    return 1.0 / (1.0 + np.exp(-k * np.clip(dt, -30, 30)))


def time_to_point(Dp: np.ndarray, Dv: np.ndarray, target: np.ndarray, cfg: Config) -> np.ndarray:
    """Tempo di arrivo dei difensori su uno o piu' punti.

    Dp, Dv: (..., nd, 2); target: (..., m, 2) -> (..., m, nd)
    """
    start = Dp + Dv * cfg.reaction_time                       # (..., nd, 2)
    diff = target[..., :, None, :] - start[..., None, :, :]   # (..., m, nd, 2)
    return cfg.reaction_time + np.linalg.norm(diff, axis=-1) / cfg.v_max


def _velocity(pos: np.ndarray, frame: np.ndarray, cfg: Config) -> np.ndarray:
    """Velocita' su posizioni assolute lisciate; NaN attraverso i buchi di frame."""
    sm = pos.copy()
    nan = np.isnan(sm)
    filled = np.where(nan, 0.0, sm)
    cnt = uniform_filter1d((~nan).astype(np.float32), cfg.smooth_frames, axis=0, mode="nearest")
    sm = uniform_filter1d(filled, cfg.smooth_frames, axis=0, mode="nearest") / np.maximum(cnt, 1e-6)
    sm[nan] = np.nan
    v = np.full_like(sm, np.nan)
    dt = 2.0 / cfg.fps
    v[1:-1] = (sm[2:] - sm[:-2]) / dt
    gap = np.zeros(len(frame), dtype=bool)
    gap[1:-1] = (frame[2:] - frame[:-2]) != 2
    v[gap] = np.nan
    speed = np.linalg.norm(v, axis=-1, keepdims=True)
    v = np.where(speed > cfg.max_speed_cap, v * cfg.max_speed_cap / np.maximum(speed, 1e-6), v)
    return np.nan_to_num(v, nan=0.0)


def _team_slots(pos, vel, players, team_ids, rows, teams_per_row):
    """Riempie gli 11 slot per la squadra indicata riga per riga."""
    T = len(rows)
    P = np.full((T, NSLOT, 2), np.nan, dtype=np.float32)
    V = np.zeros((T, NSLOT, 2), dtype=np.float32)
    I = np.full((T, NSLOT), -1, dtype=np.int64)
    G = np.zeros((T, NSLOT), dtype=bool)
    pid = players["id"].to_numpy()
    gk = players["is_gk"].to_numpy()
    for team in np.unique(teams_per_row):
        sel = np.where(teams_per_row == team)[0]
        cols = np.where(team_ids == team)[0]
        sub = pos[rows[sel]][:, cols]                       # (n, nt, 2)
        present = ~np.isnan(sub[..., 0])
        order = np.argsort(~present, axis=1, kind="stable")[:, :NSLOT]
        k = order.shape[1]
        take = np.take_along_axis(sub, order[..., None], axis=1)
        okp = np.take_along_axis(present, order, axis=1)
        P[sel, :k] = np.where(okp[..., None], take, np.nan)
        V[sel, :k] = np.take_along_axis(vel[rows[sel]][:, cols], order[..., None], axis=1)
        I[sel, :k] = np.where(okp, pid[cols][order], -1)
        G[sel, :k] = np.where(okp, gk[cols][order], False)
    return P, V, I, G


# ---------------------------------------------------------------------------
# costruzione dello stato
# ---------------------------------------------------------------------------

def build_frame_state(md: MatchData, cfg: Config | None = None, chunk: int = 4000) -> FrameState:
    cfg = cfg or Config()
    ph = phase_per_frame(md)
    ball_ok = ~np.isnan(md.ball[:, 0])
    valid = (ph["att_team"].to_numpy() >= 0) & ball_ok
    rows = np.where(valid)[0]

    vel = _velocity(md.pos, md.frame, cfg)
    bvel = _velocity(md.ball[:, None, :2], md.frame, cfg)[:, 0]

    att = ph["att_team"].to_numpy()[rows]
    deft = np.where(att == md.home_id, md.away_id, md.home_id)
    side = ph["att_side"].to_numpy()[rows]
    sign = np.where(side == "left_to_right", 1.0, -1.0).astype(np.float32)

    team_ids = md.players["team_id"].to_numpy()
    A, Av, Aid, Agk = _team_slots(md.pos, vel, md.players, team_ids, rows, att)
    D, Dv, Did, Dgk = _team_slots(md.pos, vel, md.players, team_ids, rows, deft)
    s3 = sign[:, None, None]
    A, Av, D, Dv = A * s3, Av * s3, D * s3, Dv * s3
    ball = md.ball[rows].copy()
    ball[:, :2] *= sign[:, None]
    bv = bvel[rows] * sign[:, None]

    carrier = _carrier_slots(md, rows, Aid, A, ball)

    st = FrameState(
        match_id=md.match_id, pitch_length=md.pitch_length, pitch_width=md.pitch_width,
        frame=md.frame[rows], period=md.period[rows], sign=sign,
        att_team=att, def_team=deft,
        phase_index=ph["phase_index"].to_numpy()[rows],
        ip_phase=ph["ip_phase"].to_numpy()[rows], oop_phase=ph["oop_phase"].to_numpy()[rows],
        ball=ball, ball_vel=bv, A=A, Av=Av, Aid=Aid, Agk=Agk, D=D, Dv=Dv, Did=Did, Dgk=Dgk,
        carrier=carrier,
        Q=np.full_like(A, np.nan), p_int=np.zeros((len(rows), NSLOT, NSLOT), np.float32),
        cover=np.zeros((len(rows), NSLOT), np.float32),
        p_carrier=np.zeros((len(rows), NSLOT), np.float32),
        press_carrier=np.zeros(len(rows), np.float32),
        mark=np.full((len(rows), NSLOT), -1, np.int64),
        d_ball=np.full((len(rows), NSLOT), np.nan, np.float32),
        d_mark=np.full((len(rows), NSLOT), np.nan, np.float32),
        d_ideal=np.full((len(rows), NSLOT), np.nan, np.float32),
        goal_side=np.full((len(rows), NSLOT), np.nan, np.float32),
    )
    for a in range(0, st.n, chunk):
        _coverage_chunk(st, slice(a, min(a + chunk, st.n)), cfg)
    _marking(st, cfg)
    return st


def _carrier_slots(md, rows, Aid, A, ball) -> np.ndarray:
    """Portatore: player_id del tracking se della squadra in possesso, altrimenti
    l'attaccante piu' vicino alla palla entro 2.5 m con palla sotto 1.5 m."""
    T = len(rows)
    out = np.full(T, -1, dtype=np.int64)
    pp = md.poss_player[rows]
    hit = (Aid == pp[:, None]) & (pp[:, None] >= 0)
    has = hit.any(axis=1)
    out[has] = hit[has].argmax(axis=1)
    d = np.linalg.norm(A - ball[:, None, :2], axis=-1)
    d = np.where(np.isnan(d), np.inf, d)
    near = d.argmin(axis=1)
    low = np.nan_to_num(ball[:, 2], nan=0.0) < 1.5
    fb = (~has) & (d.min(axis=1) < 2.5) & low
    out[fb] = near[fb]
    return out


def _coverage_chunk(st: FrameState, sl: slice, cfg: Config) -> None:
    A, Av, D, Dv = st.A[sl], st.Av[sl], st.D[sl], st.Dv[sl]
    b = st.ball[sl, :2]
    n = A.shape[0]
    L, W = st.pitch_length, st.pitch_width

    # punto di ricezione proiettato: l'attaccante continua a correre fino all'arrivo palla
    dist_b = np.linalg.norm(A - b[:, None, :], axis=-1)                    # (n, 11)
    t_ball = cfg.kick_time + dist_b / cfg.ball_speed
    if cfg.project_attackers:
        t_proj = np.minimum(t_ball, 1.5)[..., None]
        Q = A + Av * t_proj
    else:
        Q = A.copy()
    Q[..., 0] = np.clip(Q[..., 0], -L / 2, L / 2)
    Q[..., 1] = np.clip(Q[..., 1], -W / 2, W / 2)
    dist_q = np.linalg.norm(Q - b[:, None, :], axis=-1)
    t_ball_q = cfg.kick_time + dist_q / cfg.ball_speed

    Dz = np.where(np.isnan(D), 1e4, D)                                   # slot vuoti lontanissimi
    # 1) arrivo sul punto di ricezione
    t_def_q = time_to_point(Dz, Dv, np.nan_to_num(Q, nan=1e4), cfg)       # (n, 11a, 11d)
    p_rec = logistic_time(t_ball_q[..., None] + cfg.reception_slack - t_def_q, cfg.sigma)

    # 2) taglio della linea di passaggio
    K = cfg.lane_points
    f = (np.arange(1, K + 1) / (K + 1)).astype(np.float32)               # (K,)
    lane = b[:, None, None, :] + f[None, None, :, None] * (Q - b[:, None, :])[:, :, None, :]
    lane = np.nan_to_num(lane, nan=1e4).reshape(n, NSLOT * K, 2)
    t_def_l = time_to_point(Dz, Dv, lane, cfg).reshape(n, NSLOT, K, NSLOT)
    t_ball_l = cfg.kick_time + f[None, None, :] * dist_q[..., None] / cfg.ball_speed
    p_lane = logistic_time(t_ball_l[..., None] - t_def_l, cfg.sigma).max(axis=2)

    p = np.maximum(p_rec, p_lane).astype(np.float32)
    p[np.isnan(D[..., 0])[:, None, :].repeat(NSLOT, axis=1)] = 0.0
    p[np.isnan(A[..., 0])] = 0.0
    st.Q[sl] = Q
    st.p_int[sl] = p
    st.cover[sl] = 1.0 - np.prod(1.0 - np.clip(p, 0, 0.999), axis=2)

    # pressione sul portatore (Pressing Intensity: soglia T sul tempo di arrivo)
    t_def_b = time_to_point(Dz, Dv, b[:, None, :], cfg)[:, 0, :]         # (n, 11d)
    pc = logistic_time(cfg.carrier_pressure_T - t_def_b, cfg.sigma).astype(np.float32)
    pc[np.isnan(D[..., 0])] = 0.0
    st.p_carrier[sl] = pc
    st.press_carrier[sl] = 1.0 - np.prod(1.0 - np.clip(pc, 0, 0.999), axis=1)

    st.d_ball[sl] = np.linalg.norm(D - b[:, None, :], axis=-1)


def _marking(st: FrameState, cfg: Config) -> None:
    """Assegnazione difensore -> attaccante da marcare, frame per frame.

    Costo = distanza del difensore dal punto di marcatura ideale m_j, a cfg.mark_goal_side_offset
    metri dall'attaccante verso la porta difesa. L'opzione dummy "a zona" costa
    cfg.mark_max_distance: un difensore piu' lontano di cosi' da ogni punto ideale non marca.
    Isteresi: mantenere la coppia del frame precedente costa cfg.mark_hysteresis metri in meno.
    """
    G = st.goal
    prev = {}  # (def_id) -> att_id
    prev_frame = None
    for t in range(st.n):
        dslots = np.where((st.Did[t] >= 0) & ~st.Dgk[t])[0]
        aslots = np.where((st.Aid[t] >= 0) & ~st.Agk[t])[0]
        if len(dslots) == 0 or len(aslots) == 0:
            continue
        if prev_frame is not None and st.frame[t] - prev_frame > 1:
            prev = {}
        prev_frame = st.frame[t]
        Ap = st.A[t, aslots]
        to_goal = G[None, :] - Ap
        to_goal /= np.maximum(np.linalg.norm(to_goal, axis=1, keepdims=True), 1e-6)
        m = Ap + cfg.mark_goal_side_offset * to_goal
        Dp = st.D[t, dslots]
        cost = np.linalg.norm(Dp[:, None, :] - m[None, :, :], axis=-1)
        did = st.Did[t, dslots]
        aid = st.Aid[t, aslots]
        if prev:
            pa = np.array([prev.get(int(x), -2) for x in did])
            cost = cost - cfg.mark_hysteresis * (pa[:, None] == aid[None, :])
        nd = len(dslots)
        big = np.full((nd, nd), 1e6)
        np.fill_diagonal(big, cfg.mark_max_distance)
        full = np.hstack([cost, big])
        r, c = linear_sum_assignment(full)
        prev = {}
        for ri, ci in zip(r, c):
            ds = dslots[ri]
            if ci < len(aslots):
                asl = aslots[ci]
                st.mark[t, ds] = asl
                st.d_mark[t, ds] = np.linalg.norm(st.D[t, ds] - st.A[t, asl])
                st.d_ideal[t, ds] = np.linalg.norm(st.D[t, ds] - m[ci])
                st.goal_side[t, ds] = float(
                    np.linalg.norm(st.D[t, ds] - G) < np.linalg.norm(st.A[t, asl] - G)
                )
                prev[int(did[ri])] = int(aid[ci])
