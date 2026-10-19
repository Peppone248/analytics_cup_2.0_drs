"""Ricostruzione del DRS(t) per un match: curva, supporto, corse ed esiti."""
import sys
from dataclasses import replace
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "DRS/DRS_t")
import drst as D

MID = int(sys.argv[1]) if len(sys.argv) > 1 else 1996436
cfg = replace(D.DEFAULT, prior_kind="geometric", geo_use_endpoint=False,
              use_tracking_structure=True, containment_kind="separation_gain")
df = pd.read_csv("out/ref/_layer3_tracking.csv")
m = df[df.match_id == MID].copy()

pri = D.geometric_priors(m, cfg)
m = D.compute_stress(m, pri, cfg=cfg, structural=True)
# una curva per squadra difendente
teams = sorted(m.defending_team_id.unique())
curves = {t: D.build_curve(m[m.defending_team_id == t], cfg) for t in teams}

HZ = cfg.tracking_hz
for t, (c, idx) in curves.items():
    print(f"team {t}: n_frames {idx.n_frames} ({idx.n_frames/HZ/60:.1f} min) "
          f"copertura {idx.coverage():.1%}  corse {len(m[m.defending_team_id==t])}  "
          f"picco {c.max():.3f}  media sui frame supportati {idx.masked(c).mean():.4f}")

T = teams[0]
curve, idx = curves[T]
g = m[m.defending_team_id == T]
sh = g[g.lead_to_shot.astype(bool)]
# finestra attorno al picco piu' alto che contiene un lead_to_shot
peak = int(np.argmax(curve))
W = int(90 * HZ)
lo, hi = max(0, peak - W//2), min(idx.n_frames, peak + W//2)
tt = np.arange(idx.n_frames) / HZ / 60

INK, MUT = "#1a1a1a", "#6b6b6b"
fig, axes = plt.subplots(2, 1, figsize=(12, 7.2), dpi=190,
                         gridspec_kw={"height_ratios": [1, 1.25]})
fig.patch.set_facecolor("white")

for ax, (a, b), title in zip(axes, [(0, idx.n_frames), (lo, hi)],
        [f"Match {MID} — DRS(t) completo, squadra {T}",
         f"Zoom di {(hi-lo)/HZ:.0f} s attorno al picco"]):
    ax.set_facecolor("white")
    sl = slice(a, b)
    # zone senza supporto
    s = idx.support[sl].astype(int)
    edges = np.flatnonzero(np.diff(np.r_[0, 1 - s, 0]))
    for i in range(0, len(edges), 2):
        ax.axvspan(tt[a + edges[i]], tt[a + min(edges[i+1], b-a-1)],
                   color="#f2f2f2", zorder=0, lw=0)
    ax.plot(tt[sl], curve[sl], color="#2f6fab", lw=1.5, zorder=3)
    ax.fill_between(tt[sl], 0, curve[sl], color="#2f6fab", alpha=.12, zorder=2)
    for _, r in g.iterrows():
        f0, f1 = int(r.frame_start), int(r.frame_end)
        if f1 < a or f0 > b: continue
        shot = bool(r.lead_to_shot)
        ax.plot([f0/HZ/60, f1/HZ/60], [-curve.max()*.055]*2,
                color="#c23b22" if shot else "#9bb8d1",
                lw=3.4 if shot else 2.2, solid_capstyle="butt", zorder=4)
    ax.set_ylim(-curve.max()*.11, curve.max()*1.08)
    ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=7)
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
    for sp in ("left","bottom"): ax.spines[sp].set_color("#cfcfcf")
    ax.tick_params(colors=MUT, labelsize=8.5)
    ax.grid(True, axis="y", color="#ededed", lw=.7); ax.set_axisbelow(True)
axes[0].axvspan(tt[lo], tt[hi-1], color="#ffd9a0", alpha=.45, zorder=1, lw=0)
axes[1].set_xlabel("minuti", fontsize=9.5, color=INK)
axes[0].set_ylabel("DRS(t)", fontsize=9.5, color=INK)
axes[1].set_ylabel("DRS(t)", fontsize=9.5, color=INK)
fig.text(.005, .005,
    f"Grigio = frame senza supporto (il DRS(t) non e' definito, lo zero non "
    f"significa 'difesa sotto controllo'). Copertura {idx.coverage():.1%}.  "
    f"Barre in basso: corse; rosse = lead_to_shot.",
    fontsize=8.3, color=MUT)
fig.tight_layout(rect=[0, .028, 1, 1])
Path("out/figures").mkdir(parents=True, exist_ok=True)
fig.savefig(f"out/figures/curve_{MID}.png", bbox_inches="tight", facecolor="white")
print(f"\nsalvato out/figures/curve_{MID}.png")
print(f"corse {len(g)}, di cui lead_to_shot {len(sh)}; picco al minuto {peak/HZ/60:.1f}")
