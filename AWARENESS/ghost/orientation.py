"""Head orientation (gaze proxy), speed and shoulder geometry per player-frame.

Differences w.r.t. the thesis (Hawk-Eye):
* Hawk-Eye gives a skeleton for every player in every frame. SkillCorner pose
  comes from the broadcast camera, so only ~1/3 of player-frames carry joints
  (players off-screen have none). Head angle therefore needs gap handling.
* SkillCorner head joints are noisier (median 3.4 deg frame-to-frame change on
  the sample, occasional ~150 deg front/back flips), so we smooth.

Recipe (each step is a flag in `head_source` so it can be audited / filtered):
1. raw   : theta = atan2(nose - mid(lEar, rEar)) in the ground plane (same
           formula as the thesis, uefa_loader.parse_joints_minute).
           Gated on plausible planar nose-earmid length [5, 25] cm and joint
           error p90 <= MAX_HEAD_ERR_CM.
2. smooth: centred rolling median (kills single-frame flips) + rolling mean on
           the unit vector (cos, sin), window SMOOTH_FRAMES (0.2 s).
3. interp: gaps up to MAX_GAP_S are filled by shortest-arc linear interpolation.
4. prior : longer gaps fall back to "facing the ball" (players off camera are
           far from the ball and mostly orient towards it). This is a modelling
           assumption -> flagged 'ball_prior', and analyses on a focal defender
           should require head_source in {'pose', 'interp'}.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import POSE_FPS

MAX_HEAD_ERR_CM = 20.0
NOSE_EAR_LEN_M = (0.05, 0.25)
SMOOTH_FRAMES = 5            # 0.2 s at 25 fps
MAX_GAP_S = 1.0
SHOULDER_WIDTH_M = 0.40      # synthetic shoulders when pose is missing
SPEED_SMOOTH_FRAMES = 7
MAX_SPEED = 12.0             # m/s, above this it's a tracking glitch


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def add_orientation(players: pd.DataFrame, ball: pd.DataFrame) -> pd.DataFrame:
    """Add head_angle (rad), head_source, speed (m/s), vx, vy and filled shoulders.

    Input: compact players/ball tables from ghost.data. Rows must be unique per
    (frame, player_id). Returns a new dataframe sorted by (player_id, frame).
    """
    df = players.sort_values(["player_id", "frame"]).reset_index(drop=True).copy()

    # ---- 1. raw head angle with gating -------------------------------------
    emx = (df["lear_x"] + df["rear_x"]) / 2
    emy = (df["lear_y"] + df["rear_y"]) / 2
    hx, hy = df["nose_x"] - emx, df["nose_y"] - emy
    hlen = np.hypot(hx, hy)
    ok = (df["has_pose"]
          & hlen.between(*NOSE_EAR_LEN_M)
          & (df["head_err_cm"] <= MAX_HEAD_ERR_CM))
    ux = np.where(ok, hx / hlen, np.nan)
    uy = np.where(ok, hy / hlen, np.nan)

    # ---- 2. smoothing on the unit circle (per player) -----------------------
    g = df["player_id"]
    tmp = pd.DataFrame({"pid": g, "ux": ux, "uy": uy})
    for c in ("ux", "uy"):
        med = tmp.groupby("pid")[c].transform(
            lambda s: s.rolling(SMOOTH_FRAMES, center=True, min_periods=1).median())
        tmp[c] = med.groupby(tmp["pid"]).transform(
            lambda s: s.rolling(SMOOTH_FRAMES, center=True, min_periods=1).mean())
    tmp.loc[~ok, ["ux", "uy"]] = np.nan          # do not let smoothing invent data
    head = np.arctan2(tmp["uy"], tmp["ux"])
    source = np.where(ok, "pose", None).astype(object)

    # ---- 3. short-gap interpolation (shortest arc) --------------------------
    max_gap = int(round(MAX_GAP_S * POSE_FPS))
    head = pd.Series(head.to_numpy(), index=df.index)
    filled = head.copy()
    for pid, idx in df.groupby("player_id").groups.items():
        h = head.loc[idx].to_numpy()
        fr = df.loc[idx, "frame"].to_numpy()
        known = ~np.isnan(h)
        if known.sum() < 2:
            continue
        kf, kh = fr[known], np.unwrap(h[known])
        interp = np.interp(fr, kf, kh)
        # size of the gap each frame sits in
        pos = np.searchsorted(kf, fr)
        prev_f = np.where(pos > 0, kf[np.clip(pos - 1, 0, len(kf) - 1)], -10**9)
        next_f = np.where(pos < len(kf), kf[np.clip(pos, 0, len(kf) - 1)], 10**9)
        gap = next_f - prev_f
        use = (~known) & (gap <= max_gap + 1)
        hv = h.copy()
        hv[use] = _wrap(interp[use])
        filled.loc[idx] = hv
        src = source[idx]
        src[use] = "interp"
        source[idx] = src

    # ---- speed / velocity heading -------------------------------------------
    for c in ("x", "y"):
        sm = df.groupby("player_id")[c].transform(
            lambda s: s.rolling(SPEED_SMOOTH_FRAMES, center=True, min_periods=1).mean())
        df[f"v{c}"] = sm.groupby(df["player_id"]).diff()
    # divide by the actual frame step (half-time / sub gaps), then clip glitches
    dfr = df.groupby("player_id")["frame"].diff()
    for c in ("vx", "vy"):
        df[c] = (df[c] / dfr * POSE_FPS).where(dfr == 1, 0.0).fillna(0.0)
    sp = np.hypot(df["vx"], df["vy"])
    scale = np.where(sp > MAX_SPEED, MAX_SPEED / np.maximum(sp, 1e-9), 1.0)
    df["vx"] *= scale
    df["vy"] *= scale
    df["speed"] = np.hypot(df["vx"], df["vy"])

    # ---- 4. ball prior for long gaps -----------------------------------------
    b = ball.set_index("frame")[["ball_x", "ball_y"]]
    bx = df["frame"].map(b["ball_x"]).to_numpy()
    by = df["frame"].map(b["ball_y"]).to_numpy()
    to_ball = np.arctan2(by - df["y"].to_numpy(), bx - df["x"].to_numpy())
    vel_dir = np.arctan2(df["vy"].to_numpy(), df["vx"].to_numpy())
    prior = np.where(np.isnan(to_ball), vel_dir, to_ball)
    miss = filled.isna().to_numpy()
    filled = filled.to_numpy().copy()
    filled[miss] = prior[miss]
    source[miss] = "ball_prior"
    df["head_angle"] = filled
    df["head_source"] = source

    # ---- shoulders for the occlusion model -----------------------------------
    sh_ok = df["has_pose"] & (np.hypot(df["rs_x"] - df["ls_x"], df["rs_y"] - df["ls_y"])
                              .between(0.15, 0.60))
    # synthetic shoulders: axis perpendicular to head direction, centred on (x, y)
    half = SHOULDER_WIDTH_M / 2
    # facing direction f = (cos, sin); left shoulder is at +90 deg from f
    lx = df["x"] + half * np.cos(df["head_angle"] + np.pi / 2)
    ly = df["y"] + half * np.sin(df["head_angle"] + np.pi / 2)
    rx = df["x"] + half * np.cos(df["head_angle"] - np.pi / 2)
    ry = df["y"] + half * np.sin(df["head_angle"] - np.pi / 2)
    for c, v in (("ls_x", lx), ("ls_y", ly), ("rs_x", rx), ("rs_y", ry)):
        df[c] = np.where(sh_ok, df[c], v)
    df["shoulder_source"] = np.where(sh_ok, "pose", "synthetic")
    return df
