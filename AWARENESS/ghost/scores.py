"""Awareness readouts from the belief state.

The Kalman filter gives, for each (defender i, attacker j, frame t):
    x_hat  — where i *believes* j is  (the "ghost")
    P      — how uncertain i is about it
    x      — where j really is (tracking)

Three readouts, all post-hoc (never re-run the filter to change them):

1. A_cov   = exp(-tr(P) / tau)                       [thesis, decision #6]
   Pure function of the w_ij history: P does NOT depend on what the attacker
   does, only on when he was seen. It is a temporally-smoothed FoV.

2. ghost_err = ||x_hat - x||   (metres)              [NEW, Filippo's proposal]
   A_ghost  = exp(-(ghost_err / ell)^2)
   How wrong the defender's mental picture is. It is driven by the
   interaction between attention (w_ij) and attacker behaviour: an unseen
   attacker who keeps running straight stays close to his ghost (the defender
   can extrapolate him), one who changes speed/direction while unseen opens a
   gap -> exactly the blind-side runs defensive coaches care about.

3. mahal2 = e^T P^-1 e  (squared Mahalanobis distance of the truth under the
   belief). Under the model, mahal2 ~ chi2(2). Large mahal2 = the attacker is
   somewhere the defender considered unlikely -> "false belief" / surprise.

Belief quadrants (`belief_state`), using ghost_err and mahal2:
    'aware'          ghost_err <  err_thr
    'uncertain'      ghost_err >= err_thr  and mahal2 <= chi2_95   (knows he doesn't know)
    'false_belief'   ghost_err >= err_thr  and mahal2 >  chi2_95   (confident and wrong)
    'never_seen'     no sighting yet in the window -> no ghost, A_ghost = A_cov = 0
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_TAU = 8.0         # m^2, thesis value
DEFAULT_ELL = 2.0         # m, ghost-error scale: A_ghost = 0.37 at 2 m
DEFAULT_ERR_THR = 1.5     # m, ~ one stride + reach; below this the picture is "right"
CHI2_2_95 = 5.991         # 95% quantile of chi2 with 2 dof


def add_scores(df: pd.DataFrame, *, tau: float = DEFAULT_TAU, ell: float = DEFAULT_ELL,
               err_thr: float = DEFAULT_ERR_THR) -> pd.DataFrame:
    """Append A_cov, ghost_err, A_ghost, mahal2, belief_state to a pipeline output."""
    out = df.copy()
    pxx, pyy, pxy = (out[c].astype("float64") for c in ("P_xx", "P_yy", "P_xy"))
    ex = out["x_hat_x"] - out["target_x"]
    ey = out["x_hat_y"] - out["target_y"]

    out["trP"] = pxx + pyy
    out["A_cov"] = np.exp(-out["trP"] / tau)
    out["ghost_err"] = np.hypot(ex, ey)
    out["A_ghost"] = np.exp(-(out["ghost_err"] / ell) ** 2)

    det = pxx * pyy - pxy ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        out["mahal2"] = (pyy * ex ** 2 - 2 * pxy * ex * ey + pxx * ey ** 2) / det

    never = out["x_hat_x"].isna()
    out.loc[never, "A_ghost"] = 0.0          # no belief at all = fully unaware
    out.loc[never, "A_cov"] = 0.0
    state = np.where(never, "never_seen",
             np.where(out["ghost_err"] < err_thr, "aware",
              np.where(out["mahal2"] <= CHI2_2_95, "uncertain", "false_belief")))
    out["belief_state"] = pd.Categorical(state, ["aware", "uncertain", "false_belief", "never_seen"])
    return out


def ellipse_params(pxx: float, pyy: float, pxy: float, q: float = CHI2_2_95):
    """Semi-axes (a, b) and angle (rad) of the q-level covariance ellipse."""
    vals, vecs = np.linalg.eigh(np.array([[pxx, pxy], [pxy, pyy]]))
    vals = np.clip(vals, 0, None)
    a, b = np.sqrt(q * vals[1]), np.sqrt(q * vals[0])
    ang = np.arctan2(vecs[1, 1], vecs[0, 1])
    return a, b, ang


def defender_level(df: pd.DataFrame, how: str = "dist_weighted", scale_m: float = 15.0,
                   col: str = "ghost_err") -> pd.DataFrame:
    """Aggregate pair-level readouts to (frame, observer).

    how:
      'max'           worst blind spot over all attackers
      'nearest'       readout for the nearest attacker only
      'dist_weighted' mean weighted by exp(-dist / scale_m) — attackers far away
                      matter less for this defender (default)
    """
    d = df.copy()
    if how == "max":
        return d.groupby(["frame", "observer_id"], as_index=False)[col].max()
    if how == "nearest":
        idx = d.groupby(["frame", "observer_id"])["dist"].idxmin()
        return d.loc[idx, ["frame", "observer_id", "target_id", col]].reset_index(drop=True)
    d["_w"] = np.exp(-d["dist"] / scale_m)
    d["_wx"] = d["_w"] * d[col]
    g = d.groupby(["frame", "observer_id"], as_index=False)[["_w", "_wx"]].sum()
    g[col] = g["_wx"] / g["_w"]
    return g[["frame", "observer_id", col]]
