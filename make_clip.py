"""Clip del DRS(t) con le tre correzioni di D-21.

1. DICHIARA L'ESITO. Ogni corsa porta il suo esito (`NON MIRATA`, `-> TIRO`);
   intestazione persistente con il riassunto dell'episodio. Non si inventa
   l'istante del tiro: il dato e' un flag per corsa, non un frame.
2. NIENTE FUTURO. La curva e' disegnata solo fino al cursore: a 29.31 il
   modello non sa che salira' a 0.92, e la striscia non deve dirlo.
3. BANDA DI SUPPORTO. Dove il DRS(t) non e' definito lo zero non significa
   "difesa sotto controllo" (D-20: 78.6% del tempo per squadra).
"""
import sys
from dataclasses import replace
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
sys.path.insert(0, "DRS/DRS_t")
import drst as D

MID, TEAM = 1996436, 1804
F0, F1 = 17570, 17700          # ~13 s attorno al picco del minuto 29.4
FPS, SIGMA, CELL = 10, 5.0, 2.0
MODE = "gap"

cfg = replace(D.DEFAULT, prior_kind="geometric", geo_use_endpoint=False,
              use_tracking_structure=True, containment_kind="separation_gain")

df = pd.read_csv("out/ref/_layer3_tracking.csv")
df = D.compute_stress(df, D.geometric_priors(df, cfg), cfg=cfg, structural=True)
g = df[(df.match_id == MID) & (df.defending_team_id == TEAM)].copy()
curve, idx = D.build_curve(g, cfg)

tr = D.load_tracking(f"data/tracking_raw/{MID}/{MID}_tracking_extrapolated.jsonl",
                     (F0 - 200, F1 + 200))
p2t = D.infer_player_teams(g, tr, cfg)

ep = g[(g.frame_start >= F0) & (g.frame_start <= F1)]
n_shot, n_tgt = int(ep.lead_to_shot.sum()), int(ep.targeted.sum())
print(f"episodio: {len(ep)} corse, {n_shot} portano a tiro, {n_tgt} mirate")

step = max(1, cfg.tracking_hz // FPS)
frames = list(range(F0, F1, step))
xs = np.arange(-D.PITCH_X/2, D.PITCH_X/2 + CELL, CELL)
ys = np.arange(-D.PITCH_Y/2, D.PITCH_Y/2 + CELL, CELL)
grid = np.meshgrid(xs, ys)
acts = [D.active_weights(g, f, cfg) for f in frames]
fields = [D.stress_field(f, a, tr, grid, cfg, MODE, SIGMA) for f, a in zip(frames, acts)]
vmax = max(1e-6, max(f.max() for f in fields))

def outcome_tag(r):
    t = [] if r["targeted"] else ["NON MIRATA"]
    if r["lead_to_shot"]: t.append("-> TIRO")
    return "  ".join(t)

fig = plt.figure(figsize=(15.5, 8.9))
gs = fig.add_gridspec(2, 2, height_ratios=[4, 1.15], width_ratios=[3.1, 1],
                      hspace=0.14, wspace=0.04)
axp, axr, axc = fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1]), fig.add_subplot(gs[1,:])
fig.patch.set_facecolor("#1b4332"); axp.set_facecolor("#2d6a4f")

heat = axp.imshow(fields[0], extent=[xs[0], xs[-1], ys[0], ys[-1]], origin="lower",
                  cmap="inferno", alpha=0.75, vmin=0, vmax=vmax, zorder=1,
                  interpolation="bilinear")
D.draw_pitch(axp)
sc_a = axp.scatter([], [], s=120, c=D.ATT, ec="white", lw=1.2, zorder=5, label="in possesso")
sc_d = axp.scatter([], [], s=120, c=D.DEF_, ec="white", lw=1.2, zorder=5, label="difende")
sc_b = axp.scatter([], [], s=55, c=D.BALL, ec="black", lw=.8, zorder=6, label="palla")
axp.legend(loc="lower right", fontsize=8, framealpha=.25, facecolor="#1b4332",
           edgecolor="#4a7c59", labelcolor="white").set_zorder(9)

# --- 1. intestazione che dichiara l'esito dell'episodio
axp.set_title(f"Match {MID} · min 29.4 — {len(ep)} corse, "
              f"{'nessuna mirata' if n_tgt==0 else f'{n_tgt} mirate'}, "
              f"l'azione finisce in TIRO",
              color="#f1c40f", fontsize=13.5, weight="bold", pad=12)
fig.text(.012, .962, "effetto decoy: la difesa viene aperta da corse a cui "
         "nessuno passa la palla", color="#d5e8dd", fontsize=9.5, style="italic")

txt = axp.text(.012, .975, "", transform=axp.transAxes, color="white", fontsize=10.5,
               va="top", family="monospace", zorder=8,
               bbox=dict(boxstyle="round,pad=0.45", fc="#1b4332", ec="#4a7c59", alpha=.8))

# --- 3. banda di supporto nella striscia
t = np.arange(F0, F1) / cfg.tracking_hz / 60
sup = idx.support[F0:F1]
edges = np.flatnonzero(np.diff(np.r_[0, (~sup).astype(int), 0]))
for i in range(0, len(edges), 2):
    axc.axvspan(t[edges[i]], t[min(edges[i+1], len(t)-1)],
                color="#0f2b1f", zorder=0, lw=0)
if (~sup).any():
    axc.text(.012, .86, "fascia scura = DRS(t) non definito", transform=axc.transAxes,
             color="#7f9e8e", fontsize=7.5, style="italic", va="top")

# --- 2. la curva si disegna solo fino al cursore
line, = axc.plot([], [], color=D.HEAT, lw=1.9, zorder=3)
fillc = [axc.fill_between(t[:1], curve[F0:F0+1], color=D.HEAT, alpha=.25, zorder=2)]
cursor = axc.axvline(t[0], color="white", lw=1.4, zorder=4)
axc.set_xlim(t[0], t[-1]); axc.set_ylim(0, curve[F0:F1].max()*1.12)
axc.set_facecolor("#1b4332"); axc.tick_params(colors="white", labelsize=8)
for sp in axc.spines.values(): sp.set_color("#4a7c59")
axc.set_ylabel("DRS(t)", color="white", fontsize=9)
axc.set_xlabel("minuti", color="white", fontsize=9)

overlay = []
def update(i):
    f, act = frames[i], acts[i]
    heat.set_data(fields[i])
    pos = tr.at(tr.nearest_frame(f) or -1)
    a = np.array([p for q,p in pos.items() if p2t.get(q) != TEAM]) if pos else np.empty((0,2))
    d = np.array([p for q,p in pos.items() if p2t.get(q) == TEAM]) if pos else np.empty((0,2))
    sc_a.set_offsets(a if len(a) else np.empty((0,2)))
    sc_d.set_offsets(d if len(d) else np.empty((0,2)))
    b = tr.ball.get(tr.nearest_frame(f) or -1)
    sc_b.set_offsets([[b[0], b[1]]] if b else np.empty((0,2)))

    for art in overlay: art.remove()
    overlay.clear()
    ordered = act.sort_values("w", ascending=False)
    for k, (_, r) in enumerate(ordered.iterrows()):
        rid = int(r["player_id"])
        if rid not in pos: continue
        rx, ry = pos[rid]
        shot = bool(r["lead_to_shot"])
        fl = D.stress_field(f, ordered.iloc[[k]], tr, grid, cfg, MODE, SIGMA)
        if fl.max() > .04 * vmax:
            cs = axp.contour(grid[0], grid[1], fl, levels=[.28*fl.max()],
                             colors="#ff5252" if shot else "#f1c40f",
                             linewidths=1.5 if shot else 1.1, alpha=.8, zorder=4)
            overlay.extend(cs.collections if hasattr(cs, "collections") else [cs])
        did = r[cfg.col_defender]
        if pd.notna(did) and int(did) in pos:
            dx, dy = pos[int(did)]
            overlay.append(axp.plot([dx, rx], [dy, ry], color="white", lw=1.0, ls="--",
                                    alpha=.3 + .55*min(r["w"]/max(vmax,1e-6),1), zorder=4)[0])
        if k < 5:
            overlay.append(axp.text(rx, ry+2.6, f"{k+1}", color="black", fontsize=8.5,
                family="monospace", weight="bold", ha="center", va="center", zorder=8,
                bbox=dict(boxstyle="circle,pad=0.26",
                          fc="#ff5252" if shot else "#f1c40f", ec="black", lw=.6)))
            tag = outcome_tag(r)
            overlay.append(axp.text(rx, ry-3.4, D._run_label(r, cfg),
                color="#ff5252" if shot else "#f1c40f", fontsize=7.5, family="monospace",
                ha="center", va="center", zorder=8,
                bbox=dict(boxstyle="round,pad=0.22", fc="#1b4332", ec="none", alpha=.75)))
            if tag:
                overlay.append(axp.text(rx, ry-6.0, tag, color="#ff8a80", fontsize=6.8,
                    family="monospace", weight="bold", ha="center", va="center", zorder=8,
                    bbox=dict(boxstyle="round,pad=0.20", fc="#1b4332", ec="none", alpha=.75)))

    D.draw_run_panel(axr, act, cfg, vmax)
    # badge d'esito accanto a ogni riga del pannello
    for j, (_, r) in enumerate(ordered.head(5).iterrows()):
        tag = outcome_tag(r)
        if tag:
            # sulla riga del nome giocatore, allineato a destra: non collide
            # con il subtype ne' con il valore dello stress
            axr.text(1.0, .90 - j*.175 - .030, tag, color="#ff8a80", fontsize=6.8,
                     family="monospace", weight="bold", va="center", ha="right",
                     transform=axr.transAxes)

    j = f - F0 + 1
    line.set_data(t[:j], curve[F0:F0+j])
    fillc[0].remove()
    fillc[0] = axc.fill_between(t[:j], curve[F0:F0+j], color=D.HEAT, alpha=.25, zorder=2)
    cursor.set_xdata([f/cfg.tracking_hz/60]*2)
    txt.set_text(f"frame {f}   {f/cfg.tracking_hz/60:5.2f}'\nDRS(t) {curve[f]:.3f}")
    return [heat, sc_a, sc_d, sc_b, txt, cursor, line, *overlay]

anim = FuncAnimation(fig, update, frames=len(frames), interval=1000/FPS, blit=False)
out = Path("out/figures/clip_v3_min29.mp4")
out.parent.mkdir(parents=True, exist_ok=True)
anim.save(out, writer=FFMpegWriter(fps=FPS, bitrate=3000))
plt.close(fig)
print("salvato", out)
