"""Fonti di anisotropia mai testate: relative e direzionali, non posizionali."""
import numpy as np, pandas as pd

d = pd.read_csv("out/ref/_layer3_tracking.csv")
y = d.lead_to_shot.astype(int).astype(float)
PH, ST, TEAM, MATCH = ("team_out_of_possession_phase_type", "event_subtype",
                       "defending_team_id", "match_id")
DEEP = {"low_block", "defending_transition", "defending_quick_break"}
d["_deep"] = d[PH].isin(DEEP).astype(float)
depth = d.groupby(TEAM)._deep.mean()

def resid(s): return s - s.groupby([d[PH], d[ST]]).transform("mean")
y_r = resid(y)

def sh(col, n_iter=200, seed=0, min_m=3):
    t = pd.DataFrame({"t": d[TEAM], "m": d[MATCH], "v": col}).dropna()
    k = t.groupby("t").m.nunique(); t = t[t.t.isin(k[k >= min_m].index)]
    ms = t.m.unique(); h = len(ms)//2; rng = np.random.default_rng(seed); o = []
    for _ in range(n_iter):
        p = rng.permutation(ms)
        a = t[t.m.isin(p[:h])].groupby("t").v.mean()
        b = t[t.m.isin(p[h:])].groupby("t").v.mean()
        c = a.index.intersection(b.index)
        if len(c) >= 6: o.append(np.corrcoef(a[c], b[c])[0,1])
    return float(np.median(o)) if o else np.nan

# --- triangolo palla / corridore / porta: angoli dai soli tre lati
bg, br = d.ball_dist_goal_end.astype(float), d.ball_dist_runner_end.astype(float)
rg = d.geo_dist_end.astype(float)
def ang(a, b, c):   # angolo opposto a c, fra i lati a e b
    v = (a**2 + b**2 - c**2) / (2*a*b)
    return np.arccos(np.clip(v, -1, 1))
ang_ball   = ang(bg, br, rg)   # al vertice PALLA: corridore in linea col gol?
ang_runner = ang(br, rg, bg)   # al vertice CORRIDORE
ang_goal   = ang(bg, rg, br)   # al vertice PORTA: ampiezza dell'attacco

# --- direzione della corsa rispetto alla porta
dx = d.geo_x_end - d.geo_x_start
dy = d.geo_y_end - d.geo_y_start
L = np.hypot(dx, dy).replace(0, np.nan)
gx, gy = 52.5 - d.geo_x_start, 0.0 - d.geo_y_start
GL = np.hypot(gx, gy)
head = (dx*gx + dy*gy) / (L*GL)      # cos: +1 corre verso la porta

C = {
 "|y_start| (laterale ass.)":  d.y_start.abs(),
 "|y_end| (laterale ass.)":    d.geo_y_end.abs(),
 "ang. al vertice PALLA":      ang_ball,
 "ang. al vertice CORRIDORE":  ang_runner,
 "ang. al vertice PORTA":      ang_goal,
 "heading verso porta (cos)":  head,
 "angle_of_engagement":        d.angle_of_engagement.astype(float),
 "goal_side_start":            d.goal_side_start.astype(float),
 "goal_side_end":              d.goal_side_end.astype(float),
 "T_r = exp(-br/20)  [D-14]":  np.exp(-br/20),
}
print(f"{'candidato':30s} {'cop.':>5s} {'corr_y':>8s} {'pcorr_y':>8s} "
      f"{'p/c':>5s} {'sh':>7s} {'r_depth':>8s}")
for k, v in C.items():
    v = pd.Series(np.asarray(v, dtype=float), index=d.index)
    ok = v.notna()
    if ok.mean() < 0.3 or v[ok].std() == 0:
        print(f"{k:30s} {ok.mean():5.0%}  copertura/varianza insufficiente"); continue
    c0 = np.corrcoef(v[ok], y[ok])[0,1]
    cp = np.corrcoef(resid(v.fillna(v.mean()))[ok], y_r[ok])[0,1]
    tm = pd.DataFrame({"t": d[TEAM], "v": v}).dropna().groupby("t").v.mean()
    cm = tm.index.intersection(depth.index)
    print(f"{k:30s} {ok.mean():5.0%} {c0:+8.3f} {cp:+8.3f} "
          f"{abs(cp/c0) if c0 else 0:5.2f} {sh(v):+7.3f} "
          f"{np.corrcoef(tm[cm], depth[cm])[0,1]:+8.3f}")
