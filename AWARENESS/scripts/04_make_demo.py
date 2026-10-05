"""Build the demo material for the Tuesday meeting.

* picks the most interesting off-ball runs from outputs/offball_runs_{id}.parquet
  (reliable focal head angle, focal defender close to the runner, longest
  false-belief time), plus the committed sample phase (Brisbane quick break)
* for each clip: 2D GIF (pitch + time series) and a PNG snapshot at the worst moment,
  in outputs/media/clips/ (git-ignored)
* one self-contained 3D viewer with all clips: outputs/ghost_viewer.html

Usage: .venv/bin/python scripts/04_make_demo.py 1925299 [n_clips]
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ghost.data import CACHE_DIR, POSE_FPS, load_compact, players_meta
from ghost.export3d import build_viewer, export_window
from ghost.pipeline import run_window
from ghost.viz2d import animate_window, snapshot

OUT = Path(__file__).resolve().parents[1] / "outputs"
PRE_S, POST_S = 3.0, 1.5


def pick_runs(mid, n):
    runs = pd.read_parquet(OUT / f"offball_runs_{mid}.parquet")
    rel = runs[(runs.head_ok >= 0.6) & (runs.dist_start < 12) & (runs.seen_frac < 0.6)]
    rel = rel.sort_values(["false_belief_s", "ghost_err_end"], ascending=False)
    return rel.head(n)


def main():
    mid = int(sys.argv[1]) if len(sys.argv) > 1 else 1925299
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    pl = pd.read_parquet(CACHE_DIR / f"{mid}_players_oriented.parquet")
    _, ball = load_compact(mid)
    meta = players_meta(mid).set_index("player_id")
    teams = list(pl.team_id.dropna().unique())
    media = OUT / "media" / "clips"
    media.mkdir(parents=True, exist_ok=True)

    clips, order = [], []
    for ev in pick_runs(mid, n).itertuples():
        f0 = int(ev.frame_start - PRE_S * POSE_FPS)
        f1 = int(ev.frame_end + POST_S * POSE_FPS)
        dfn = int([t for t in teams if t != ev.team_id][0])
        res = run_window(pl, f0, f1, observer_team=dfn, target_team=int(ev.team_id))
        foc, run = int(ev.focal_def_id), int(ev.runner_id)
        fname, rname = meta.loc[foc, "short_name"], meta.loc[run, "short_name"]
        title = f"{ev.minute}' {ev.subtype.replace('_', ' ')} — {rname}"
        sub = (f"Defender {fname} (#{meta.loc[foc, 'number']}) saw the runner "
               f"{ev.seen_frac:.0%} of the run · {ev.false_belief_s:.1f}s in false belief · "
               f"{'dangerous' if ev.dangerous else 'not dangerous'}, "
               f"{'targeted' if ev.targeted else 'not targeted'}")
        tag = f"{mid}_{ev.event_id}"
        pair = res[res.in_roi & (res.observer_id == foc) & (res.target_id == run)]
        worst = int(pair.loc[pair.ghost_err.idxmax(), "frame"]) if pair.ghost_err.notna().any() else int(pair.frame.iloc[len(pair) // 2])
        snapshot(res, pl, ball, foc, worst, target=run, out_path=media / f"{tag}.png",
                 title=f"{title}\n{sub}")
        animate_window(res, pl, ball, foc, target=run, out_path=media / f"{tag}.gif",
                       title=title)
        clip = export_window(mid, res, pl, ball, foc, run, title=title, subtitle=sub)
        (CACHE_DIR / f"clip_{tag}.json").write_text(json.dumps(clip, separators=(",", ":")))
        clips.append(clip)
        order.append(f"clip_{tag}.json")
        res.to_parquet(CACHE_DIR / f"clip_{tag}.parquet", index=False)
        print("clip", tag, "|", title)

    (CACHE_DIR / "clips_order.json").write_text(json.dumps(order))
    out = build_viewer(clips, OUT / "ghost_viewer.html")
    print("viewer ->", out, f"{out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
