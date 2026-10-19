"""Cerca un hero case nei 2 match con body pose: DRS alto + decoy + falsa credenza."""
import sys
from dataclasses import replace
import numpy as np, pandas as pd
sys.path.insert(0,"DRS/DRS_t"); import drst as D

cfg=replace(D.DEFAULT,prior_kind="geometric",geo_use_endpoint=False,
            use_tracking_structure=True,containment_kind="separation_gain")
full=pd.read_csv("out/ref/_layer3_tracking.csv")
full=D.compute_stress(full,D.geometric_priors(full,cfg),cfg=cfg,structural=True)
aw=pd.concat([pd.read_parquet(f"/tmp/aw/offball_runs_{m}.parquet").assign(match_id=m)
              for m in (1925299,1996435)],ignore_index=True)
AWC=["seen_frac","ghost_err_max","ghost_err_end","false_belief_s","never_seen_s",
     "A_ghost_mean","n_def_false_belief","focal_def_id","runner","run_s"]
d=full.merge(aw[["match_id","event_id"]+AWC],on=["match_id","event_id"],how="left")
HZ=cfg.tracking_hz
WIN=int(12*HZ)   # episodio = finestra di 12 s

rows=[]
for (mid,team),g in d[d.match_id.isin([1925299,1996435])].groupby(["match_id","defending_team_id"]):
    curve,idx=D.build_curve(g,cfg)
    shots=g[g.lead_to_shot.astype(bool)]
    for _,s in shots.iterrows():
        f=int(s.frame_start)
        ep=g[(g.frame_start>=f-WIN//2)&(g.frame_start<=f+WIN//2)]
        if ep.lead_to_shot.sum()==0: continue
        a=f-WIN//2; b=min(f+WIN, idx.n_frames)
        rows.append(dict(
            match=mid, team=team, min=round(f/HZ/60,2), frame=f,
            n_run=len(ep), n_shot=int(ep.lead_to_shot.sum()),
            n_tgt=int(ep.targeted.sum()),
            decoy=int((ep.lead_to_shot.astype(bool)&~ep.targeted.astype(bool)).sum()),
            drs_peak=float(curve[a:b].max()),
            aw_cov=float(ep.seen_frac.notna().mean()),
            seen_frac=float(ep.seen_frac.mean(skipna=True)),
            never_seen=float(ep.never_seen_s.mean(skipna=True)),
            false_belief=float(ep.false_belief_s.mean(skipna=True)),
            ghost_err=float(ep.ghost_err_max.mean(skipna=True)),
            n_fb=float(ep.n_def_false_belief.mean(skipna=True))))
e=pd.DataFrame(rows).drop_duplicates(subset=["match","team","frame"])
# un episodio per cluster: tieni il picco piu' alto entro 12 s
e=e.sort_values("drs_peak",ascending=False)
keep=[]
for _,r in e.iterrows():
    if not any((r.match==k.match)and(r.team==k.team)and abs(r.frame-k.frame)<WIN for k in keep):
        keep.append(r)
e=pd.DataFrame(keep)

e=e[e.aw_cov>0.5]
print(f"episodi con tiro e awareness disponibile: {len(e)}\n")
print("--- TOP per picco DRS ---")
print(e.head(8)[["match","team","min","n_run","n_shot","n_tgt","decoy","drs_peak",
                 "seen_frac","never_seen","false_belief","ghost_err","n_fb"]]
      .round(3).to_string(index=False))
print("\n--- TOP per DECOY (tutte non mirate) + DRS alto ---")
dec=e[(e.decoy>=2)&(e.n_tgt==0)].sort_values("drs_peak",ascending=False)
print(dec.head(6)[["match","team","min","n_run","n_shot","decoy","drs_peak",
                   "seen_frac","never_seen","false_belief","ghost_err","n_fb"]]
      .round(3).to_string(index=False) if len(dec) else "  nessuno con >=2 decoy e 0 mirate")
print("\n--- TOP per FALSA CREDENZA, fra gli episodi con tiro ---")
print(e.sort_values("false_belief",ascending=False).head(6)[
    ["match","team","min","n_run","n_shot","n_tgt","decoy","drs_peak",
     "seen_frac","never_seen","false_belief","ghost_err"]].round(3).to_string(index=False))
e.to_csv("out/figures/hero_candidates.csv",index=False)
print("\nsalvato out/figures/hero_candidates.csv")
