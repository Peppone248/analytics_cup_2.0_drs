import sys
from dataclasses import replace
import numpy as np, pandas as pd
sys.path.insert(0, "DRS/DRS_t")
import drst as D

cfg = replace(D.DEFAULT, prior_kind="geometric", geo_use_endpoint=False,
              use_tracking_structure=True, containment_kind="separation_gain")
df = pd.read_csv("out/ref/_layer3_tracking.csv")
pri = D.geometric_priors(df, cfg)
df = D.compute_stress(df, pri, cfg=cfg, structural=True)
HZ, TAIL = cfg.tracking_hz, int(cfg.tail_multiplier * cfg.tau_decay_frames)

ov_sup, ov_all, cov_team, cov_match, pre_zero, pre_n = [], [], [], [], 0, 0
corr_prev = []
for mid, mg in df.groupby("match_id"):
    n = int(mg.frame_end.max()) + cfg.padding_frames
    # copertura a livello di MATCH (tutte le corse insieme)
    sup_m = np.zeros(n, bool)
    for f0, f1 in zip(mg.frame_start.astype(int), mg.frame_end.astype(int)):
        sup_m[max(0,f0):min(n,f1+TAIL)] = True
    cov_match.append(sup_m.mean())

    for team, g in mg.groupby("defending_team_id"):
        cnt = np.zeros(n, int); sup = np.zeros(n, bool)
        for f0, f1 in zip(g.frame_start.astype(int), g.frame_end.astype(int)):
            a, b = max(0,int(f0)), min(n,int(f1)+TAIL)
            cnt[a:b] += 1; sup[a:b] = True
        cov_team.append(sup.mean())
        if sup.sum():
            ov_sup.append((cnt[sup] >= 2).mean())
        ov_all.append((cnt >= 2).mean())

        # il DRS(t) nei 3 s PRIMA di ogni corsa: quante volte e' zero?
        curve, idx = D.build_curve(g, cfg)
        w = int(3 * HZ)
        for _, r in g.iterrows():
            f0 = int(r.frame_start)
            pre = curve[max(0, f0-w):f0]
            pre_n += 1
            if pre.size == 0 or np.allclose(pre, 0): pre_zero += 1

print(f"copertura media per MATCH   {np.mean(cov_match):.1%}")
print(f"copertura media per SQUADRA {np.mean(cov_team):.1%}   (e' questa che")
print( "   alimenta le metriche di squadra)")
print()
print(f"frame supportati con >=2 corse attive (sovrapposizione): {np.mean(ov_sup):.1%}")
print(f"  sul totale dei frame del match:                        {np.mean(ov_all):.1%}")
print()
print(f"corse il cui DRS(t) nei 3 s precedenti e' IDENTICAMENTE ZERO: "
      f"{pre_zero}/{pre_n} = {pre_zero/pre_n:.1%}")
