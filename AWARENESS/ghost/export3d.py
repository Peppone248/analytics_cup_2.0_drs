"""Export one analysed window to the self-contained 3D viewer (viewer/pov3d.html).

The viewer is a single HTML file with the data inlined (so it opens from disk,
no server needed). `build_viewer` reads viewer/pov3d_template.html, replaces
the /*__DATA__*/ placeholder with the JSON and writes the result.

Per frame we ship: ball xyz, every player's (x, y) + 29-joint skeleton when
available (z re-based so the lowest foot joint touches the grass), and every
defender's beliefs about every attacker (ghost position, 95% ellipse, error,
belief state, w_ij) so the viewer can switch the focal defender.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import POSE_FPS, load_window_joints, players_meta
from .scores import ellipse_params

JOINTS = ("nose", "neck", "lEye", "rEye", "lEar", "rEar", "lShoulder", "rShoulder",
          "lElbow", "rElbow", "lWrist", "rWrist", "lThumb", "rThumb", "lPinky", "rPinky",
          "midHip", "lHip", "rHip", "lKnee", "rKnee", "lAnkle", "rAnkle", "lHeel", "rHeel",
          "lBigToe", "rBigToe", "lSmallToe", "rSmallToe")
SKELETON = (("nose", "neck"), ("neck", "lShoulder"), ("neck", "rShoulder"), ("neck", "midHip"),
            ("lShoulder", "lElbow"), ("lElbow", "lWrist"), ("rShoulder", "rElbow"),
            ("rElbow", "rWrist"), ("midHip", "lHip"), ("midHip", "rHip"), ("lHip", "lKnee"),
            ("lKnee", "lAnkle"), ("rHip", "rKnee"), ("rKnee", "rAnkle"), ("lAnkle", "lHeel"),
            ("lAnkle", "lBigToe"), ("rAnkle", "rHeel"), ("rAnkle", "rBigToe"),
            ("nose", "lEye"), ("nose", "rEye"), ("lEye", "lEar"), ("rEye", "rEar"))
FOOT = [JOINTS.index(j) for j in ("lAnkle", "rAnkle", "lHeel", "rHeel", "lBigToe", "rBigToe")]
STATE_CODE = {"aware": 0, "uncertain": 1, "false_belief": 2, "never_seen": 3}
VIEWER_DIR = Path(__file__).resolve().parents[1] / "viewer"


def _r(v, n=2):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), n)


def _beliefs(r: pd.DataFrame | None):
    """(G, H) for one observer at one frame: ghosts of every target + head angle."""
    G, H = [], None
    if r is None or not len(r):
        return G, H
    H = [_r(r.obs_head.iloc[0], 3), r.obs_head_source.iloc[0]]
    for q in r.itertuples():
        if np.isnan(q.x_hat_x):
            G.append([int(q.target_id), None, None, 0, 0, 0, None, 3, _r(q.w_ij, 3)])
            continue
        a, b, ang = ellipse_params(q.P_xx, q.P_yy, q.P_xy)
        G.append([int(q.target_id), _r(q.x_hat_x), _r(q.x_hat_y), _r(a), _r(b), _r(ang, 3),
                  _r(q.ghost_err), STATE_CODE[str(q.belief_state)], _r(q.w_ij, 3)])
    return G, H


def _series(pair: pd.DataFrame, f0: int):
    if pair is None or not len(pair):
        return None
    return dict(t=[round((f - f0) / POSE_FPS, 2) for f in pair.frame],
                w=[_r(v, 3) for v in pair.w_ij],
                err=[_r(v) for v in pair.ghost_err],
                acov=[_r(v, 3) for v in pair.A_cov_thesis],
                aghost=[_r(v, 3) for v in pair.A_ghost],
                state=[STATE_CODE[str(s)] for s in pair.belief_state])


def export_window(match_id: int, res: pd.DataFrame, pl: pd.DataFrame, ball: pd.DataFrame,
                  focal: int, target: int | None, title: str = "", subtitle: str = "",
                  step: int = 1, all_defenders: bool = True) -> dict:
    """Export a window. Frames carry `g`/`h` (default focal defender) and, with
    all_defenders, `G`/`H`: the same for every defender, aligned with
    clip["defs"] — so the viewer can switch the focal defender."""
    roi = res[res.in_roi]
    frames = np.sort(roi.frame.unique())[::step]
    f0, f1 = int(frames[0]), int(frames[-1])
    joints = load_window_joints(match_id, f0, f1)
    meta = players_meta(match_id).set_index("player_id")
    obs_team = int(pl.loc[pl.player_id == focal, "team_id"].iloc[0])

    win = pl[pl.frame.between(f0, f1)]
    by_f = {f: g for f, g in win.groupby("frame")}
    defs = sorted(int(d) for d in roi.observer_id.unique()) if all_defenders else [int(focal)]
    rf_all = {(int(o), f): g for (o, f), g in roi.groupby(["observer_id", "frame"])}
    bl = ball.set_index("frame")

    out_frames = []
    for f in frames:
        g = by_f.get(f)
        if g is None:
            continue
        P = []
        for p in g.itertuples():
            j = joints.get(int(f), {}).get(int(p.player_id))
            flat = None
            if j:
                arr = np.array([j[k][:3] if k in j else (np.nan,) * 3 for k in JOINTS], float)
                zmin = np.nanmin(arr[FOOT, 2])
                if np.isfinite(zmin):
                    arr[:, 2] -= zmin - 0.03
                flat = [None if np.isnan(v) else round(float(v), 2) for v in arr.ravel()]
            P.append([int(p.player_id), _r(p.x), _r(p.y), flat, int(bool(p.is_detected))])
        G, H = _beliefs(rf_all.get((int(focal), f)))
        GA, HA = zip(*[_beliefs(rf_all.get((d, f))) for d in defs]) if all_defenders else ((), ())
        bx = bl.loc[f] if f in bl.index else None
        B = None if bx is None or pd.isna(bx.ball_x) else [_r(bx.ball_x), _r(bx.ball_y), _r(bx.ball_z)]
        fr = dict(f=int(f), t=round((f - f0) / POSE_FPS, 2), b=B, p=P, g=G, h=H)
        if all_defenders:
            fr["G"], fr["H"] = list(GA), list(HA)
        out_frames.append(fr)

    pairs = roi[roi.target_id == target].sort_values("frame") if target else roi.iloc[0:0]
    series = _series(pairs[pairs.observer_id == focal], f0) if target else None
    series_by = {str(d): _series(pairs[pairs.observer_id == d], f0) for d in defs} if target else {}

    players = {}
    for pid in win.player_id.unique():
        m = meta.loc[pid]
        players[int(pid)] = dict(n=int(m.number), name=m.short_name, role=m.role,
                                 d=int(m.team_id == obs_team))
    return dict(title=title, subtitle=subtitle, fps=POSE_FPS / step, focal=int(focal),
                target=int(target) if target else None, joints=JOINTS,
                skeleton=[[JOINTS.index(a), JOINTS.index(b)] for a, b in SKELETON],
                players=players, frames=out_frames, series=series,
                defs=defs, series_by=series_by)


def build_viewer(data: dict | list[dict], out_path: str | Path) -> Path:
    """Inline one clip (dict) or several clips (list) into the viewer template."""
    clips = data if isinstance(data, list) else [data]
    tpl = (VIEWER_DIR / "pov3d_template.html").read_text()
    payload = json.dumps(clips, separators=(",", ":"))
    out = Path(out_path)
    out.write_text(tpl.replace("/*__DATA__*/[]", payload))
    return out
