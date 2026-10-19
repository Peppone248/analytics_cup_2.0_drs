import sys, json
from dataclasses import replace
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "DRS/DRS_t")
import drst as D

cfg0 = replace(D.DEFAULT, prior_kind="geometric", geo_prior_scale=25.0,
               geo_use_endpoint=False, geo_ball_gate=False,
               use_tracking_structure=True, use_curve_containment=False,
               containment_kind="separation_gain")
layer3 = Path("out/ref/_layer3_tracking.csv")
base_geo = D.geometric_priors

def patched(scale):
    def f(df, cfg=cfg0):
        g = base_geo(df, cfg)
        t = np.exp(-df["ball_dist_runner_end"].astype(float) / scale)
        t = t.fillna(t.mean())
        out = g * t
        return out / out.max()
    return f

for scale in (30.0, 45.0, 70.0):
    D.geometric_priors = patched(scale)
    out = Path(f"out/tr{int(scale)}")
    res = D.run_pipeline(layer3, cfg=cfg0, structural=None, with_oos=True,
                         verbose=False)
    res.save(out)
    v = pd.read_csv(out / "validation.csv")
    s = pd.read_csv(out / "stability.csv")
    print(f"\n######## T_r, scala {scale:.0f} m ########")
    print(v.to_string(index=False))
    print(s.to_string(index=False))
D.geometric_priors = base_geo
