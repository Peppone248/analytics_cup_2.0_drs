import sys
from dataclasses import replace
from pathlib import Path
import pandas as pd
sys.path.insert(0, "DRS/DRS_t")
import drst as D

OUT = Path("out/ref_learned")
cfg = replace(D.DEFAULT, prior_kind="geometric", geo_prior_scale=25.0,
              geo_use_endpoint=False, geo_ball_gate=False,
              use_tracking_structure=True, use_curve_containment=False,
              containment_kind="learned")

merged = D.merge_into_layer3(
    D.load_layer3("data/matches/layer3_enriched.csv", cfg),
    pd.read_csv("data/tracking_features.csv"), cfg)
OUT.mkdir(parents=True, exist_ok=True)
layer3 = OUT / "_layer3_tracking.csv"
merged.to_csv(layer3, index=False)
print(f"copertura tracking: {merged['sep_min'].notna().mean():.1%}")

res = D.run_pipeline(layer3, cfg=cfg, structural=None, with_oos=True)
out = res.save(OUT)
if res.two_axis is not None:
    cols = ["defending_team_id", "n_matches", "n_runs", "esposizione",
            "cedimento", "deep_phase_share"]
    print("\nProfilo a due assi")
    print(res.two_axis[cols].round(3).to_string(index=False))
print(f"\nsalvato in {out}")
