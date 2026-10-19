import sys
from dataclasses import replace
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "DRS/DRS_t")
import drst as D

cfg = replace(D.DEFAULT, prior_kind="geometric", geo_prior_scale=25.0,
              geo_use_endpoint=False, geo_ball_gate=False,
              use_tracking_structure=True, use_curve_containment=False,
              containment_kind="separation_gain")
layer3 = Path("out/ref/_layer3_tracking.csv")

# containment NEUTRO: 1.0 ovunque -> stress = prior x struttura
D.separation_gain_containment = lambda df, c=cfg: pd.Series(1.0, index=df.index)

res = D.run_pipeline(layer3, cfg=cfg, structural=None, with_oos=True, verbose=False)
res.save(Path("out/nocf"))
print(pd.read_csv("out/nocf/validation.csv")[
    ["outcome","odds_ratio","ci_lo","ci_hi","p_value"]].round(4).to_string(index=False))
print()
print(pd.read_csv("out/nocf/stability.csv")[
    ["metric","median","spearman_brown"]].round(3).to_string(index=False))
