"""Griglia di screening: cosa alzerebbe l'OR senza far scendere la split-half.

Per ogni candidato tre numeri indipendenti:
  corr_y    correlazione con lead_to_shot          -> spinge l'OR
  pcorr_y   la stessa al netto di fase x subtype   -> informazione NUOVA
  sh        split-half della media per squadra     -> non deve crollare
  r_depth   corr(media squadra, deep_phase_share)  -> e' solo "difendo basso"?
"""
import numpy as np, pandas as pd

d = pd.read_csv("out/ref/_layer3_tracking.csv")
y = d.lead_to_shot.astype(int).astype(float)
PH, ST, TEAM, MATCH = ("team_out_of_possession_phase_type", "event_subtype",
                       "defending_team_id", "match_id")

# quota di corse in fase profonda, per squadra (il baseline da battere)
DEEP = {"low_block", "defending_transition", "defending_quick_break"}
d["_deep"] = d[PH].isin(DEEP).astype(float)
depth = d.groupby(TEAM)._deep.mean()

grp = d.groupby([PH, ST])
def resid(s):                      # residuo sulle medie di cella fase x subtype
    return s - s.groupby([d[PH], d[ST]]).transform("mean")
y_r = resid(y)

def split_half(col, n_iter=200, seed=0, min_m=3):
    t = pd.DataFrame({"t": d[TEAM], "m": d[MATCH], "v": col}).dropna()
    k = t.groupby("t").m.nunique(); t = t[t.t.isin(k[k >= min_m].index)]
    ms = t.m.unique(); h = len(ms) // 2
    rng = np.random.default_rng(seed); o = []
    for _ in range(n_iter):
        p = rng.permutation(ms)
        a = t[t.m.isin(p[:h])].groupby("t").v.mean()
        b = t[t.m.isin(p[h:])].groupby("t").v.mean()
        c = a.index.intersection(b.index)
        if len(c) >= 6: o.append(np.corrcoef(a[c], b[c])[0, 1])
    return float(np.median(o)) if o else np.nan

SKIP = {"defender_id", "player_id", "team_id", "match_id", "defending_team_id",
        "frame_start", "frame_end", "pp_frame", "skipped_reason", "_deep",
        "sep_n_frames", "lead_to_shot", "targeted", "received"}
cand = [c for c in d.columns
        if d[c].dtype.kind in "if" and c not in SKIP and d[c].notna().mean() > 0.5]

rows = []
for c in cand:
    v = d[c].astype(float)
    ok = v.notna()
    if v[ok].std() == 0: continue
    tm = pd.DataFrame({"t": d[TEAM], "v": v}).dropna().groupby("t").v.mean()
    common = tm.index.intersection(depth.index)
    rows.append(dict(
        feature=c,
        corr_y=np.corrcoef(v[ok], y[ok])[0, 1],
        pcorr_y=np.corrcoef(resid(v.fillna(v.mean()))[ok], y_r[ok])[0, 1],
        sh=split_half(v),
        r_depth=np.corrcoef(tm[common], depth[common])[0, 1]))

r = pd.DataFrame(rows)
r["abs_p"] = r.pcorr_y.abs()
r = r.sort_values("abs_p", ascending=False)

print("tutti i candidati, ordinati per informazione NUOVA (|pcorr_y|)\n")
print(f"{'feature':34s} {'corr_y':>8s} {'pcorr_y':>8s} {'sh':>7s} {'r_depth':>8s}")
for _, x in r.iterrows():
    print(f"{x.feature:34s} {x.corr_y:+8.3f} {x.pcorr_y:+8.3f} "
          f"{x.sh:+7.3f} {x.r_depth:+8.3f}")

print("\n\n=== IL QUADRANTE BUONO: |pcorr_y|>=0.05, sh>=0.60, |r_depth|<=0.70 ===")
g = r[(r.abs_p >= 0.05) & (r.sh >= 0.60) & (r.r_depth.abs() <= 0.70)]
print(g[["feature", "corr_y", "pcorr_y", "sh", "r_depth"]].round(3).to_string(index=False)
      if len(g) else "  nessuno")

print("\n=== le due proposte ===")
for c in ["ball_dist_runner_end", "ball_dist_goal_end",
          "ball_dist_runner_start", "ball_dist_goal_start",
          "geo_dist_end", "geo_dist_min", "geo_dist_start"]:
    x = r[r.feature == c]
    if len(x): 
        x = x.iloc[0]
        print(f"{c:26s} corr {x.corr_y:+.3f}  pcorr {x.pcorr_y:+.3f}  "
              f"sh {x.sh:+.3f}  r_depth {x.r_depth:+.3f}")
