"""Awareness of defenders during SkillCorner off-ball runs (bridge to DRS).

For every off_ball_run of the attacking-type subtypes in a pose match:
  * window = [run start, run end + POST_S], warmup 10 s (belief convergence)
  * observers = opponents of the runner's team, targets = runner's team
  * focal defender = opponent nearest to the runner at run start
  * per-run metrics on the (focal defender, runner) pair, over the run:
      seen_frac        fraction of frames with w_ij > w_min (runner observed)
      ghost_err_max    max ||ghost - runner|| (m)
      ghost_err_end    ghost error at run end (moment the ball could arrive)
      false_belief_s   seconds in 'false_belief' state
      A_cov_thesis     mean thesis awareness (reproduction)
      head_ok          share of frames where the focal head angle comes from pose
  * team-level: how many defenders had the runner in a false-belief state.

Output: outputs/offball_runs_{match}.parquet (+ the long pair tables of the top runs
in outputs/cache/ for visualisation).

Usage: .venv/bin/python scripts/03_offball_runs.py 1925299
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ghost.data import CACHE_DIR, POSE_FPS, dynamic_events
from ghost.pipeline import run_window

OUT = Path(__file__).resolve().parents[1] / "outputs"
SUBTYPES = {"behind", "run_ahead_of_the_ball", "cross_receiver", "overlap",
            "underlap", "pulling_half_space", "pulling_wide", "support", "coming_short"}
POST_S = 1.0


def main():
    mid = int(sys.argv[1]) if len(sys.argv) > 1 else 1925299
    pl = pd.read_parquet(CACHE_DIR / f"{mid}_players_oriented.parquet")
    teams = pl.team_id.dropna().unique()
    de = dynamic_events(mid)
    runs = de[(de.event_type == "off_ball_run") & de.event_subtype.isin(SUBTYPES)].copy()
    print(f"{len(runs)} runs")

    rows = []
    for k, ev in enumerate(runs.itertuples()):
        f0, f1 = int(ev.pose_frame_start), int(ev.pose_frame_end + POST_S * POSE_FPS)
        att = int(ev.team_id)
        dfn = int([t for t in teams if t != att][0])
        try:
            res = run_window(pl, f0, f1, observer_team=dfn, target_team=att)
        except Exception as e:                       # noqa: BLE001
            print("skip", ev.event_id, e)
            continue
        roi = res[res.in_roi & (res.target_id == ev.player_id)]
        if roi.empty:
            continue
        start = roi[roi.frame == roi.frame.min()]
        focal = int(start.loc[start.dist.idxmin(), "observer_id"])
        pair = roi[roi.observer_id == focal].sort_values("frame")
        end_row = pair[pair.frame <= int(ev.pose_frame_end)].iloc[-1]
        fb_def = (roi.groupby("observer_id").belief_state
                  .apply(lambda s: (s == "false_belief").mean() > 0.2).sum())
        rows.append(dict(
            event_id=ev.event_id, subtype=ev.event_subtype, period=ev.period,
            minute=ev.minute_start, second=ev.second_start,
            runner_id=int(ev.player_id), runner=ev.player_name, team_id=att,
            focal_def_id=focal, frame_start=f0, frame_end=int(ev.pose_frame_end),
            dist_start=float(start.dist.min()),
            seen_frac=float((pair.w_ij > 0.01).mean()),
            ghost_err_max=float(pair.ghost_err.max()),
            ghost_err_end=float(end_row.ghost_err),
            false_belief_s=float((pair.belief_state == "false_belief").sum() / POSE_FPS),
            never_seen_s=float((pair.belief_state == "never_seen").sum() / POSE_FPS),
            run_s=float(len(pair) / POSE_FPS),
            A_cov_mean=float(pair.A_cov.mean()),
            A_cov_thesis_mean=float(pair.A_cov_thesis.mean()),
            A_ghost_mean=float(pair.A_ghost.mean()),
            head_ok=float(pair.obs_head_source.isin(["pose", "interp"]).mean()),
            n_def_false_belief=int(fb_def),
            targeted=ev.targeted, received=ev.received, dangerous=ev.dangerous,
            lead_to_shot=ev.lead_to_shot, xthreat=ev.xthreat,
        ))
        if k % 25 == 0:
            print(k, ev.event_subtype, rows[-1]["ghost_err_max"])
    df = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    df.to_parquet(OUT / f"offball_runs_{mid}.parquet", index=False)
    rel = df[(df.head_ok >= 0.6) & (df.dist_start < 12)]
    print(f"\n{len(df)} runs processed, {len(rel)} with reliable focal head angle "
          "(>=60% pose) and focal defender within 12 m at run start")
    cols = ["event_id", "subtype", "minute", "runner", "focal_def_id", "dist_start", "seen_frac",
            "ghost_err_end", "false_belief_s", "never_seen_s", "A_cov_thesis_mean",
            "A_ghost_mean", "targeted", "dangerous"]
    print(rel.sort_values(["false_belief_s", "ghost_err_end"], ascending=False)[cols]
          .head(15).round(2).to_string())
    print("\nbelief readouts by run outcome (reliable subset):")
    print(rel.groupby("dangerous")[["seen_frac", "ghost_err_end", "false_belief_s",
                                    "A_cov_thesis_mean", "A_ghost_mean"]].mean().round(3))


if __name__ == "__main__":
    main()
