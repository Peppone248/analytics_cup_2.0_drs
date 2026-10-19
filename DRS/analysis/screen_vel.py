import numpy as np, pandas as pd
d = pd.read_csv("out/ref/_layer3_tracking.csv")
v = pd.read_csv("data/velocity_features.csv")
d = d.merge(v, on=["match_id", "event_id"], how="left")
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
    k = t.groupby("t").m.nunique(); t = t[t.t.isin(k[k>=min_m].index)]
    ms = t.m.unique(); h = len(ms)//2; rng = np.random.default_rng(seed); o=[]
    for _ in range(n_iter):
        p = rng.permutation(ms)
        a = t[t.m.isin(p[:h])].groupby("t").v.mean()
        b = t[t.m.isin(p[h:])].groupby("t").v.mean()
        c = a.index.intersection(b.index)
        if len(c) >= 6: o.append(np.corrcoef(a[c], b[c])[0,1])
    return float(np.median(o)) if o else np.nan

CAND = ["vel_run_cos_goal", "vel_run_cos_goal_centr", "vel_runner_speed",
        "vel_def_speed", "vel_rel_speed", "vel_closing",
        "vel_def_cos_runner", "vel_def_cos_owngoal"]
print(f"{'candidato':26s} {'cop.':>5s} {'corr_y':>8s} {'pcorr_y':>8s} "
      f"{'p/c':>5s} {'sh':>7s} {'r_depth':>8s}")
res = {}
for c in CAND:
    s = d[c].astype(float); ok = s.notna()
    c0 = np.corrcoef(s[ok], y[ok])[0,1]
    cp = np.corrcoef(resid(s.fillna(s.mean()))[ok], y_r[ok])[0,1]
    tm = pd.DataFrame({"t": d[TEAM], "v": s}).dropna().groupby("t").v.mean()
    cm = tm.index.intersection(depth.index)
    S = sh(s); rd = np.corrcoef(tm[cm], depth[cm])[0,1]
    res[c] = (c0, cp, S, rd)
    print(f"{c:26s} {ok.mean():5.0%} {c0:+8.3f} {cp:+8.3f} "
          f"{abs(cp/c0) if c0 else 0:5.2f} {S:+7.3f} {rd:+8.3f}")
print("\n=== quadrante buono: |pcorr|>=0.05, sh>=0.60, |r_depth|<=0.70 ===")
g = [c for c,(a,b,S,rd) in res.items() if abs(b)>=0.05 and S>=0.60 and abs(rd)<=0.70]
print("  " + (", ".join(g) if g else "nessuno"))
print("\n=== riferimenti ===")
print(f"{'T_r exp(-br/20) [D-14]':26s} corr +0.097 pcorr +0.080 p/c 0.83 sh +0.685 r_depth +0.542")
print(f"{'ang. vertice PALLA [D-15]':26s} corr +0.115 pcorr +0.112 p/c 0.98 sh +0.141 r_depth -0.216")
