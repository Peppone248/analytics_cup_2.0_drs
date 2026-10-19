"""Figura 1 — profilo difensivo a due assi, 13 squadre A-League 24/25.

Assi in z-score sulle 13 squadre (D-17: `cedimento` e' compresso per
costruzione — media di una sigmoide — quindi i valori grezzi non si leggono).
CI al 90% da bootstrap a blocchi sui match: l'unita' di campionamento e' la
partita, non la corsa (Q-05).
"""
import json, glob
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SRC, OUT = "out/ref_learned", "out/figures"
EXPO, CED = "stress_per_run", "containment_phase_adj"
rng = np.random.default_rng(7)

names = {}
for p in glob.glob("data/matches/*/*_match.json"):
    d = json.load(open(p, encoding="utf-8"))
    for k in ("home_team", "away_team"):
        t = d.get(k) or {}
        if t.get("id"):
            names[int(t["id"])] = t.get("short_name") or t.get("name")

tm = pd.read_csv(f"{SRC}/team_match_metrics.csv")

def wmean(g, c):
    """Media SEMPLICE sulle partite, non pesata sulle corse.

    E' la definizione di aggregate.team_two_axis: ogni partita pesa uguale,
    coerente col fatto che l'unita' di campionamento e' la partita (Q-05).
    La versione pesata sulle corse differisce in modo trascurabile
    (corr 0.9996, scarto massimo 0.0005) ma non riprodurrebbe il repo."""
    return float(g[c].mean())

rows = []
for tid, g in tm.groupby("defending_team_id"):
    boot = np.array([[wmean(s, EXPO), wmean(s, CED)]
                     for s in (g.sample(len(g), replace=True, random_state=int(i))
                               for i in rng.integers(0, 1e6, 600))])
    rows.append(dict(tid=tid, club=names.get(tid, str(tid)), k=len(g),
                     n=g.n_runs.sum(), expo=wmean(g, EXPO), ced=wmean(g, CED),
                     deep=wmean(g, "deep_phase_share"),
                     e_lo=np.percentile(boot[:, 0], 5), e_hi=np.percentile(boot[:, 0], 95),
                     c_lo=np.percentile(boot[:, 1], 5), c_hi=np.percentile(boot[:, 1], 95)))
d = pd.DataFrame(rows)

# z-score sulle 13 squadre; i CI seguono la stessa trasformazione
for a, lo, hi in (("expo", "e_lo", "e_hi"), ("ced", "c_lo", "c_hi")):
    mu, sd = d[a].mean(), d[a].std(ddof=0)
    for c in (a, lo, hi):
        d[c + "_z"] = (d[c] - mu) / sd

INK, MUT, GRID = "#1a1a1a", "#6b6b6b", "#e4e4e4"
fig, ax = plt.subplots(figsize=(9.2, 7.4), dpi=200)
fig.patch.set_facecolor("white"); ax.set_facecolor("white")

lim = 2.45
ax.axhline(0, color="#c8c8c8", lw=1, zorder=1)
ax.axvline(0, color="#c8c8c8", lw=1, zorder=1)
for q, (x, y, ha, va) in {
    "ESPOSTA E PERMISSIVA":  (lim-.08,  lim-.08, "right", "top"),
    "ESPOSTA MA RESISTENTE": (lim-.08, -lim+.08, "right", "bottom"),
    "PROTETTA MA FRAGILE":   (-lim+.08, lim-.08, "left",  "top"),
    "SOLIDA":                (-lim+.08,-lim+.08, "left",  "bottom"),
}.items():
    ax.text(x, y, q, ha=ha, va=va, fontsize=8.5, color="#b4b4b4",
            fontweight="bold")

cmap = plt.get_cmap("Blues")
norm = plt.Normalize(d.deep.min(), d.deep.max())
MIN_K = 3   # stessa soglia della split-half

# --- etichette: repulsione semplice per evitare sovrapposizioni
lab = d[["expo_z", "ced_z"]].to_numpy(float) + np.array([.14, .10])
for _ in range(260):
    for i in range(len(lab)):
        for j in range(i + 1, len(lab)):
            dxy = lab[i] - lab[j]
            dist = np.hypot(*dxy)
            if dist < .46:
                push = (dxy / (dist + 1e-9)) * (.46 - dist) * .5
                lab[i] += push; lab[j] -= push
    lab[:, 0] = np.clip(lab[:, 0], -lim + .3, lim - .92)
    lab[:, 1] = np.clip(lab[:, 1], -lim + .2, lim - .22)

for i, (_, r) in enumerate(d.iterrows()):
    col = cmap(0.38 + 0.56 * norm(r.deep))
    solid = r.k >= MIN_K
    if r.k >= 2:   # con 1 solo match il bootstrap ricampiona sempre la stessa
                   # partita: l'intervallo sarebbe un falso zero, non si disegna
        ax.plot([r.e_lo_z, r.e_hi_z], [r.ced_z, r.ced_z], color=col,
                lw=1.3, alpha=.5, zorder=2, solid_capstyle="round")
        ax.plot([r.expo_z, r.expo_z], [r.c_lo_z, r.c_hi_z], color=col,
                lw=1.3, alpha=.5, zorder=2, solid_capstyle="round")
    ax.scatter(r.expo_z, r.ced_z, s=40 + 26 * r.k,
               facecolor=col if solid else "white",
               edgecolor="white" if solid else col,
               linewidth=1.7, zorder=3)
    lx, ly = lab[i]
    ax.plot([r.expo_z, lx], [r.ced_z, ly], color="#bdbdbd", lw=.6, zorder=2)
    ax.text(lx, ly, r.club, fontsize=9, color=INK if solid else MUT,
            ha="left", va="center", zorder=4)
    note = f"{r.k} match" + ("" if r.k >= 2 else " · CI non stimabile")
    ax.text(lx, ly - .155, note, fontsize=7.3, color=MUT, ha="left",
            va="center", zorder=4)

ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_xlabel("ESPOSIZIONE  →  volume di stress difensivo concesso per corsa\n"
              "z-score sulle 13 squadre (grezzo: 0.027 – 0.051)",
              fontsize=9.5, color=INK, labelpad=9)
ax.set_ylabel("CEDIMENTO  →  quanto male viene gestita, dato il contesto\n"
              "z-score (grezzo: 0.476 – 0.520)", fontsize=9.5, color=INK, labelpad=9)
ax.set_title("Profilo difensivo a due assi — A-League 2024/25, 8 877 off-ball run",
             fontsize=12.5, color=INK, pad=16, loc="left", fontweight="bold")
ax.text(0, 1.012, "Barre: intervallo al 90% da bootstrap a blocchi sui match. "
        "Dimensione: partite osservate.", transform=ax.transAxes,
        fontsize=8.6, color=MUT, va="bottom")

for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color("#cfcfcf")
ax.tick_params(colors=MUT, labelsize=8.5)
ax.grid(True, color=GRID, lw=.7, zorder=0)
ax.set_axisbelow(True)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
cb = fig.colorbar(sm, ax=ax, fraction=.03, pad=.015)
cb.set_label("quota di corse subite in fase profonda", fontsize=8.6, color=MUT)
cb.ax.tick_params(colors=MUT, labelsize=7.6); cb.outline.set_visible(False)

ax.legend(handles=[
    Line2D([], [], marker="o", ls="", mfc="#4a7fb5", mec="white", mew=1.6,
           ms=9, label=f"\u2265{MIN_K} match (soglia della split-half)"),
    Line2D([], [], marker="o", ls="", mfc="white", mec="#4a7fb5", mew=1.6,
           ms=8, label="<3 match — stima indicativa"),
], loc="upper left", frameon=False, fontsize=8.2, labelcolor=MUT,
   handletextpad=.6, borderpad=.1, bbox_to_anchor=(.008, .905))

fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(f"{OUT}/fig1_two_axis.{ext}", bbox_inches="tight", facecolor="white")
print(d[["club","k","expo","ced","deep","expo_z","ced_z"]].round(3).to_string(index=False))
print("\nsalvato in out/figures/fig1_two_axis.{png,svg}")
