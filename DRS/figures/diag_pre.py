import sys
from dataclasses import replace
import numpy as np, pandas as pd, statsmodels.api as sm
sys.path.insert(0, "DRS/DRS_t")
import drst as D

cfg = replace(D.DEFAULT, prior_kind="geometric", geo_use_endpoint=False,
              use_tracking_structure=True, containment_kind="separation_gain")
df = pd.read_csv("out/ref/_layer3_tracking.csv")
pri = D.geometric_priors(df, cfg)
df = D.compute_stress(df, pri, cfg=cfg, structural=True)
HZ = cfg.tracking_hz; w = int(3*HZ)

pre = pd.Series(np.nan, index=df.index)
for (mid, team), g in df.groupby(["match_id", "defending_team_id"]):
    curve, _ = D.build_curve(g, cfg)
    for i, r in g.iterrows():
        f0 = int(r.frame_start)
        seg = curve[max(0, f0-w):f0]
        pre[i] = seg.mean() if seg.size else 0.0
df["pre"] = pre
y = df.lead_to_shot.astype(int)

def logit(mask, col, label):
    d = df[mask]
    v = d[col]
    z = (v - v.mean())/v.std() if v.std() else v*0
    X = sm.add_constant(pd.concat(
        [pd.Series(z, name="x"),
         pd.get_dummies(d.event_subtype, prefix="st", drop_first=True).astype(float)], axis=1))
    r = sm.Logit(y[mask].astype(int), X).fit(disp=0)
    b, se = r.params["x"], r.bse["x"]
    print(f"{label:46s} n={mask.sum():5d}  OR {np.exp(b):.3f} "
          f"[{np.exp(b-1.96*se):.3f}, {np.exp(b+1.96*se):.3f}]  p={r.pvalues['x']:.2e}")

nz = df.pre > 0
print(f"quota con DRS(t) pre-corsa > 0: {nz.mean():.1%}")
print(f"shot rate   pre=0: {y[~nz].mean():.3f}    pre>0: {y[nz].mean():.3f}\n")
logit(pd.Series(True, index=df.index), "pre", "tutte le corse (riferimento)")
logit(nz, "pre", "SOLO corse con pre > 0  (effetto graduato?)")
df["prebin"] = nz.astype(float)
logit(pd.Series(True, index=df.index), "prebin", "solo binaria pre>0 vs pre=0")
