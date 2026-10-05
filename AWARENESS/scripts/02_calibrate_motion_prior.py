"""Calibrate the motion prior of the belief state on real trajectories.

Idea: the Kalman belief is a *normative* observer. Its motion model should be
the one that best predicts where players actually go when you stop looking at
them. We therefore score candidate priors on blind-prediction episodes:

  observe a player for WARM_S seconds (w = 1), then w = 0 for BLIND_S seconds,
  and evaluate at horizons H the Gaussian predictive belief N(x_hat, P) against
  the true position.

Metrics per prior and horizon:
  err_med   median ||x_hat - x|| (m)
  nll       mean negative log-likelihood of the truth (proper scoring rule)
  cover95   fraction of truths inside the 95% ellipse (ideal 0.95)

Only stretches where the player is `is_detected` throughout are used, because
SkillCorner's extrapolated positions are themselves model output (too smooth).

Usage: .venv/bin/python scripts/02_calibrate_motion_prior.py 1925299 [1996435 ...]
Writes outputs/calibration_motion_prior.csv and a figure.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ghost.data import CACHE_DIR, POSE_FPS
from ghost.kalman import run_filter_batched

OUT = Path(__file__).resolve().parents[1] / "outputs"
WARM_S, BLIND_S = 3.0, 5.0
HORIZONS = [0.5, 1.0, 2.0, 3.0, 5.0]
N_SEG = 6000
RNG = np.random.default_rng(42)

CANDIDATES = (
    [("cv_thesis", dict(sigma_acc=s), None if s == 2.0 else dict(model="cv_thesis")) for s in (1, 2, 4, 6, 8)]
    + [("cv_cont", dict(sigma_acc=s), dict(model="cv_cont")) for s in (1, 2, 4, 6)]
    + [("ou", dict(), dict(model="ou", tau_v=t, sigma_v=sv)) for t in (0.5, 1, 2, 3, 4, 6, 8, 12) for sv in (1.0, 1.5, 2.0, 2.5, 3.0)]
)


def segments(match_ids):
    segs = []
    L = int((WARM_S + BLIND_S) * POSE_FPS)
    for mid in match_ids:
        pl = pd.read_parquet(CACHE_DIR / f"{mid}_players_oriented.parquet",
                             columns=["frame", "player_id", "x", "y", "is_detected"])
        for pid, g in pl.groupby("player_id"):
            g = g.sort_values("frame")
            f = g.frame.to_numpy(); d = g.is_detected.to_numpy()
            ok = d & np.r_[True, np.diff(f) == 1]
            # run-length of consecutive ok frames
            starts = np.flatnonzero(ok & ~np.r_[False, ok[:-1]])
            ends = np.flatnonzero(ok & ~np.r_[ok[1:], False])
            xy = g[["x", "y"]].to_numpy()
            for s, e in zip(starts, ends):
                n = e - s + 1
                for k in range(s, e - L + 2, L):      # non-overlapping chunks
                    segs.append(xy[k:k + L])
    segs = np.stack(segs)
    idx = RNG.choice(len(segs), size=min(N_SEG, len(segs)), replace=False)
    return segs[idx]


def evaluate(segs, kalman_kw, motion):
    P, N, _ = segs.shape
    nw = int(WARM_S * POSE_FPS)
    w = np.zeros((P, N)); w[:, :nw] = 1.0
    x_hat, _, Ph = run_filter_batched(segs, w, np.ones((P, N), bool),
                                      dt=1 / POSE_FPS, motion=motion, **kalman_kw)
    rows = []
    for h in HORIZONS:
        t = nw - 1 + int(h * POSE_FPS)
        e = x_hat[:, t] - segs[:, t]
        S = Ph[:, t]
        Si = np.linalg.inv(S)
        m2 = np.einsum("pi,pij,pj->p", e, Si, e)
        nll = 0.5 * (m2 + np.log(np.linalg.det(S)) + 2 * np.log(2 * np.pi))
        rows.append(dict(h=h, err_med=np.median(np.linalg.norm(e, axis=1)),
                         nll=nll.mean(), cover95=(m2 <= 5.991).mean(),
                         sd_pred=np.sqrt(np.trace(S, axis1=1, axis2=2) / 2).mean()))
    return rows


def main():
    mids = [int(a) for a in sys.argv[1:]] or [1925299]
    segs = segments(mids)
    print(f"{len(segs)} blind-prediction episodes from {mids}")
    rows = []
    for fam, kkw, motion in CANDIDATES:
        label = fam + " " + ", ".join(f"{k}={v}" for k, v in {**kkw, **(motion or {})}.items() if k != "model")
        for r in evaluate(segs, kkw, motion):
            rows.append(dict(prior=label, family=fam, **r))
    df = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "calibration_motion_prior.csv", index=False)

    summ = df.groupby("prior").agg(nll=("nll", "mean"), cover95=("cover95", "mean"),
                                   err_1s=("err_med", lambda s: s.iloc[1]),
                                   err_3s=("err_med", lambda s: s.iloc[3])).sort_values("nll")
    print(summ.round(3).to_string())

    best = summ.index[0]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    for lab in [best, "cv_thesis sigma_acc=2"]:
        d = df[df.prior == lab]
        axs[0].plot(d.h, d.err_med, "o-", label=lab)
        axs[1].plot(d.h, d.cover95, "o-", label=lab)
        axs[2].plot(d.h, d.sd_pred, "o-", label=lab)
    axs[0].set_ylabel("median ghost error (m)"); axs[1].set_ylabel("coverage of 95% ellipse")
    axs[1].axhline(0.95, color="#999", ls=":"); axs[2].set_ylabel("predicted sd per axis (m)")
    for a in axs:
        a.set_xlabel("seconds without looking"); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "calibration_motion_prior.png", dpi=130)
    print("best prior:", best)


if __name__ == "__main__":
    main()
