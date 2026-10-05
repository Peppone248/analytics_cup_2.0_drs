"""One window -> long (frame, observer, target) belief-state table.

Mirrors the thesis two-stage design (Stage 1 FoV weights, Stage 2 Kalman) but
for a single contiguous window, which is how the Analytics Cup use-case works:
DRS (Beppe) or the dynamic events pick short dangerous sequences, and we run
the awareness deep-dive only there.

Window policy (same as thesis cross windows):
  [frame_lo - warmup, frame_hi], clamped to the period. The warmup lets the
  belief converge (P0 = 100 I is a deliberately vague prior); rows in the
  warmup are kept but flagged in_roi = False.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import POSE_FPS
from .fov import frame_wij
from .kalman import run_filter_batched
from .scores import add_scores

DEFAULT_WARMUP_S = 10.0
# Thesis Kalman (CV + DWNA Q, sigma_acc = 2) — kept as the reference readout.
THESIS_KW = dict(sigma_obs=1.0, sigma_acc=2.0, w_min=0.01)
# Calibrated motion prior (scripts/02_calibrate_motion_prior.py): OU velocity.
# Drives the ghost by default. See NOTES_FILIPPO.md section "motion prior".
GHOST_KW = dict(sigma_obs=1.0, w_min=0.01, init_on="seen",
                motion=dict(model="ou", tau_v=4.0, sigma_v=2.0))


def run_window(pl: pd.DataFrame, frame_lo: int, frame_hi: int,
               observer_team: int, target_team: int,
               warmup_s: float = DEFAULT_WARMUP_S, fov_kw: dict | None = None,
               kalman_kw: dict | None = None, with_thesis: bool = True) -> pd.DataFrame:
    """Run FoV + Kalman for observer_team watching target_team over a window.

    Args:
        pl: players table with orientation (ghost.orientation.add_orientation).
        frame_lo, frame_hi: 25 fps frame range of the region of interest.
        observer_team, target_team: SkillCorner team ids.
        warmup_s: seconds of burn-in before frame_lo.
        kalman_kw: overrides for the ghost filter (default GHOST_KW).
        with_thesis: also run the thesis filter on the same w_ij and add
            trP_thesis / A_cov_thesis (reproduction column).

    Returns:
        Long dataframe with one row per (frame, observer, target), including
        w_ij, ghost (x_hat, v_hat), covariance, true position, observer pose and
        all readouts from ghost.scores.add_scores.
    """
    fov_kw = fov_kw or {}
    kkw = {**GHOST_KW, **(kalman_kw or {})}
    period = int(pl.loc[pl["frame"].between(frame_lo, frame_hi), "period"].mode().iloc[0])
    per = pl[pl["period"] == period]
    lo = max(int(frame_lo - warmup_s * POSE_FPS), int(per["frame"].min()))
    win = per[per["frame"].between(lo, frame_hi)]
    frames = np.sort(win["frame"].unique())

    by_frame = {f: g for f, g in win.groupby("frame")}

    # ---- Stage 1: FoV weights per frame ------------------------------------
    rec = []
    for f in frames:
        g = by_frame[f]
        o = g[g["team_id"] == observer_team]
        t = g[g["team_id"] == target_team]
        if len(o) == 0 or len(t) == 0:
            continue
        W = frame_wij(
            obs=dict(x=o.x.values, y=o.y.values, head=o.head_angle.values,
                     speed=o.speed.values, id=o.player_id.values),
            tgt=dict(x=t.x.values, y=t.y.values, id=t.player_id.values),
            occ=dict(ls_x=g.ls_x.values, ls_y=g.ls_y.values, rs_x=g.rs_x.values,
                     rs_y=g.rs_y.values, id=g.player_id.values),
            **fov_kw)
        no, nt = W.shape
        rec.append(pd.DataFrame({
            "frame": f,
            "observer_id": np.repeat(o.player_id.values, nt),
            "target_id": np.tile(t.player_id.values, no),
            "w_ij": W.ravel(),
            "obs_x": np.repeat(o.x.values, nt), "obs_y": np.repeat(o.y.values, nt),
            "obs_head": np.repeat(o.head_angle.values, nt),
            "obs_head_source": np.repeat(o.head_source.values, nt),
            "target_x": np.tile(t.x.values, no), "target_y": np.tile(t.y.values, no),
            "target_detected": np.tile(t.is_detected.values, no),
        }))
    long = pd.concat(rec, ignore_index=True)

    # ---- Stage 2: batched Kalman over all pairs ----------------------------
    key = long["observer_id"].astype(str) + "|" + long["target_id"].astype(str)
    pcode, _ = pd.factorize(key)
    fcode = np.searchsorted(frames, long["frame"].to_numpy())
    npairs, nf = pcode.max() + 1, len(frames)
    pos = np.full((npairs, nf, 2), np.nan)
    w = np.zeros((npairs, nf))
    valid = np.zeros((npairs, nf), bool)
    pos[pcode, fcode, 0] = long["target_x"].to_numpy()
    pos[pcode, fcode, 1] = long["target_y"].to_numpy()
    w[pcode, fcode] = long["w_ij"].to_numpy()
    valid[pcode, fcode] = True

    x_hat, v_hat, P = run_filter_batched(pos, w, valid, dt=1.0 / POSE_FPS, **kkw)

    long["x_hat_x"] = x_hat[pcode, fcode, 0]
    long["x_hat_y"] = x_hat[pcode, fcode, 1]
    long["v_hat_x"] = v_hat[pcode, fcode, 0]
    long["v_hat_y"] = v_hat[pcode, fcode, 1]
    long["P_xx"] = P[pcode, fcode, 0, 0]
    long["P_yy"] = P[pcode, fcode, 1, 1]
    long["P_xy"] = P[pcode, fcode, 0, 1]
    if with_thesis:
        _, _, Pt = run_filter_batched(pos, w, valid, dt=1.0 / POSE_FPS, **THESIS_KW)
        long["trP_thesis"] = Pt[pcode, fcode, 0, 0] + Pt[pcode, fcode, 1, 1]
        long["A_cov_thesis"] = np.exp(-long["trP_thesis"] / 8.0)
    long["dist"] = np.hypot(long.target_x - long.obs_x, long.target_y - long.obs_y)
    long["in_roi"] = long["frame"] >= frame_lo
    long["t_s"] = (long["frame"] - frame_lo) / POSE_FPS
    long["period"] = period
    return add_scores(long)
