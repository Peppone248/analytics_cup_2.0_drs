"""Load SkillCorner body-pose matches into compact tables.

A full pose match is ~3.3 GB of JSON (153k frames at 25 fps). Parsing it is the
slow part, so we do it ONCE per match and keep only what the awareness model
needs, in a long parquet table (one row per player per frame):

    frame, period, clock_s, player_id, x, y, is_detected, has_pose,
    nose_x/y/z, lear_x/y, rear_x/y, ls_x/y, rs_x/y, head_err_cm, sh_err_cm

plus a per-frame ball/possession table. Full 29-joint skeletons are only needed
for the 3D viewer, so they are pulled lazily for short windows
(`load_window_joints`), skipping all other lines with a cheap frame-number peek.

Conventions
-----------
* Coordinates: SkillCorner pitch-centred metres, x along the length
  (+-52.5), y across (+-34). Joints are in the same frame as x/y (checked on
  the sample: midHip - (x, y) has median |d| ~ 5 cm). z of the joints behaves
  like a height (nose ~1.6 m, feet ~0.15 m) even if the README warns it is
  only centroid-relative -> fine for drawing, not used in the model.
* Time: we stay on the native 25 fps pose clock (dt = 0.04 s, identical to the
  Hawk-Eye setup of the thesis). tracking_frame (10 fps) = frame / 2.5.
* `clock_s` is the game clock in seconds (period 2 starts at 45:00), parsed
  from the record timestamp; it is what match.json start/end times use, so it
  is how we decide who is on the pitch (the pose file lists the whole squad,
  including benched / substituted players with stale x/y).
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Iterator

import numpy as np
import orjson
import pandas as pd

AW_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = AW_DIR.parent
DATA_DIR = REPO_DIR / "data"
LOCAL_DIR = AW_DIR / "data_local"
CACHE_DIR = AW_DIR / "outputs" / "cache"

POSE_FPS = 25
DT = 1.0 / POSE_FPS

# Joints kept in the compact table (head + shoulders = what FoV needs).
_HEAD_JOINTS = ("nose", "lEar", "rEar")
_SH_JOINTS = ("lShoulder", "rShoulder")


# --------------------------------------------------------------------------- #
# raw record streaming
# --------------------------------------------------------------------------- #
def _archive_path(match_id: int) -> Path:
    return LOCAL_DIR / f"{match_id}.jsonl.zip"


def iter_raw_lines(match_id: int) -> Iterator[bytes]:
    """Yield raw JSON lines of a downloaded pose archive (or the repo sample)."""
    arch = _archive_path(match_id)
    if arch.exists():
        with zipfile.ZipFile(arch) as zf, zf.open(f"{match_id}.jsonl") as fh:
            yield from fh
        return
    import gzip
    sample = DATA_DIR / "bodypose" / f"sample_{match_id}_phase406.jsonl.gz"
    if sample.exists():
        with gzip.open(sample, "rb") as fh:
            yield from fh
        return
    raise FileNotFoundError(
        f"No pose data for {match_id}. Download it with "
        "src.data.pose_loading.download_match(match_id, dest_dir='AWARENESS/data_local')."
    )


def _peek_frame(line: bytes) -> int:
    # every line starts with b'{"frame": <int>,'
    i = line.index(b":") + 1
    j = line.index(b",", i)
    return int(line[i:j])


def _clock_to_s(ts: str | None) -> float:
    if not ts:
        return np.nan
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


# --------------------------------------------------------------------------- #
# match metadata
# --------------------------------------------------------------------------- #
def match_meta(match_id: int) -> dict:
    return json.loads((DATA_DIR / "matches" / str(match_id) / f"{match_id}_match.json").read_text())


def players_meta(match_id: int) -> pd.DataFrame:
    """One row per squad player: id, name, team, role, on/off game-clock seconds."""
    m = match_meta(match_id)
    rows = []
    for p in m["players"]:
        role = p.get("player_role") or {}
        rows.append({
            "player_id": p["id"],
            "short_name": p["short_name"],
            "number": p["number"],
            "team_id": p["team_id"],
            "role": role.get("acronym"),
            "position_group": role.get("position_group"),
            "on_s": _clock_to_s(p["start_time"]) if p.get("start_time") else np.nan,
            "off_s": _clock_to_s(p["end_time"]) if p.get("end_time") else np.inf,
        })
    df = pd.DataFrame(rows)
    df["side"] = np.where(df["team_id"] == m["home_team"]["id"], "home", "away")
    return df


# --------------------------------------------------------------------------- #
# compact cache
# --------------------------------------------------------------------------- #
def build_compact_cache(match_id: int, frame_lo: int | None = None,
                        frame_hi: int | None = None, force: bool = False
                        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse a pose match once into (players, ball) parquet tables.

    Args:
        match_id: SkillCorner match id (1925299 or 1996435).
        frame_lo, frame_hi: optional 25 fps frame range (inclusive). None = all.
        force: rebuild even if the cache exists.

    Returns:
        players: long table, one row per (frame, player) that is on the pitch.
        ball: one row per frame with ball xyz and possession.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tag = "full" if frame_lo is None and frame_hi is None else f"{frame_lo}_{frame_hi}"
    p_path = CACHE_DIR / f"{match_id}_players_{tag}.parquet"
    b_path = CACHE_DIR / f"{match_id}_ball_{tag}.parquet"
    if p_path.exists() and b_path.exists() and not force:
        return pd.read_parquet(p_path), pd.read_parquet(b_path)

    meta = players_meta(match_id)
    on = dict(zip(meta.player_id, meta.on_s))
    off = dict(zip(meta.player_id, meta.off_s))

    cols = {k: [] for k in (
        "frame", "period", "clock_s", "player_id", "x", "y", "is_detected", "has_pose",
        "nose_x", "nose_y", "nose_z", "lear_x", "lear_y", "rear_x", "rear_y",
        "ls_x", "ls_y", "rs_x", "rs_y", "head_err_cm", "sh_err_cm")}
    bcols = {k: [] for k in (
        "frame", "period", "clock_s", "ball_x", "ball_y", "ball_z", "ball_detected",
        "poss_player_id", "poss_group")}
    nan = np.nan

    for line in iter_raw_lines(match_id):
        f = _peek_frame(line)
        if frame_lo is not None and f < frame_lo:
            continue
        if frame_hi is not None and f > frame_hi:
            break
        rec = orjson.loads(line)
        per = rec.get("period")
        if per is None:          # pre-match / half-time frames carry no data
            continue
        clk = _clock_to_s(rec.get("timestamp"))
        b = rec.get("ball_data") or {}
        poss = rec.get("possession") or {}
        bcols["frame"].append(f); bcols["period"].append(per); bcols["clock_s"].append(clk)
        bcols["ball_x"].append(b.get("x")); bcols["ball_y"].append(b.get("y"))
        bcols["ball_z"].append(b.get("z")); bcols["ball_detected"].append(b.get("is_detected"))
        bcols["poss_player_id"].append(poss.get("player_id")); bcols["poss_group"].append(poss.get("group"))

        for p in rec.get("player_data") or []:
            pid = p["player_id"]
            # on-pitch filter: the file lists the full squad with stale x/y
            if not (on.get(pid, nan) <= clk < off.get(pid, np.inf)):
                continue
            if p.get("x") is None:
                continue
            J = p.get("joints")
            cols["frame"].append(f); cols["period"].append(per); cols["clock_s"].append(clk)
            cols["player_id"].append(pid); cols["x"].append(p["x"]); cols["y"].append(p["y"])
            cols["is_detected"].append(bool(p.get("is_detected")))
            if J and all(k in J for k in _HEAD_JOINTS + _SH_JOINTS):
                n, le, re = J["nose"]["xyz"], J["lEar"]["xyz"], J["rEar"]["xyz"]
                ls, rs = J["lShoulder"]["xyz"], J["rShoulder"]["xyz"]
                cols["has_pose"].append(True)
                cols["nose_x"].append(n[0]); cols["nose_y"].append(n[1]); cols["nose_z"].append(n[2])
                cols["lear_x"].append(le[0]); cols["lear_y"].append(le[1])
                cols["rear_x"].append(re[0]); cols["rear_y"].append(re[1])
                cols["ls_x"].append(ls[0]); cols["ls_y"].append(ls[1])
                cols["rs_x"].append(rs[0]); cols["rs_y"].append(rs[1])
                cols["head_err_cm"].append(max(J[k].get("p90_mae_cm") or nan for k in _HEAD_JOINTS))
                cols["sh_err_cm"].append(max(J[k].get("p90_mae_cm") or nan for k in _SH_JOINTS))
            else:
                cols["has_pose"].append(False)
                for k in ("nose_x", "nose_y", "nose_z", "lear_x", "lear_y", "rear_x", "rear_y",
                          "ls_x", "ls_y", "rs_x", "rs_y", "head_err_cm", "sh_err_cm"):
                    cols[k].append(nan)

    players = pd.DataFrame(cols)
    float_cols = [c for c in players.columns if c not in
                  ("frame", "period", "player_id", "is_detected", "has_pose")]
    players[float_cols] = players[float_cols].astype("float32")
    players = players.merge(meta[["player_id", "team_id", "side", "role", "short_name", "number"]],
                            on="player_id", how="left")
    ball = pd.DataFrame(bcols)
    for c in ("ball_x", "ball_y", "ball_z"):
        ball[c] = ball[c].astype("float64")
    players.to_parquet(p_path, index=False)
    ball.to_parquet(b_path, index=False)
    return players, ball


def load_compact(match_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full-match compact tables (builds the cache on first call, ~2-4 min)."""
    return build_compact_cache(match_id)


# --------------------------------------------------------------------------- #
# full skeletons for a short window (3D viewer)
# --------------------------------------------------------------------------- #
def load_window_joints(match_id: int, frame_lo: int, frame_hi: int,
                       player_ids: set[int] | None = None) -> dict[int, dict[int, dict]]:
    """Return {frame: {player_id: {joint: (x, y, z, err_cm)}}} for a short window."""
    out: dict[int, dict[int, dict]] = {}
    for line in iter_raw_lines(match_id):
        f = _peek_frame(line)
        if f < frame_lo:
            continue
        if f > frame_hi:
            break
        rec = orjson.loads(line)
        fr = {}
        for p in rec.get("player_data") or []:
            if player_ids is not None and p["player_id"] not in player_ids:
                continue
            J = p.get("joints")
            if J:
                fr[p["player_id"]] = {k: (*v["xyz"], v.get("p90_mae_cm")) for k, v in J.items()}
        out[f] = fr
    return out


# --------------------------------------------------------------------------- #
# events
# --------------------------------------------------------------------------- #
def dynamic_events(match_id: int) -> pd.DataFrame:
    de = pd.read_csv(DATA_DIR / "matches" / str(match_id) / f"{match_id}_dynamic_events.csv",
                     low_memory=False)
    # 10 fps tracking frames -> 25 fps pose frames
    for c in ("frame_start", "frame_end"):
        de[f"pose_{c}"] = (de[c] * 2.5).round().astype("Int64")
    return de
