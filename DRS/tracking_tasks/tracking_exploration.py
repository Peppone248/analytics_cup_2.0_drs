import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from DRS.config import base
from DRS.utils import load_tracking

pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)

def plot_separation(curve, run, ax=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["t_seconds"], df["sep"], marker='o', markersize=3)
    ax.axhline(2, color='gray', linestyle='--', alpha=0.4, label='2m (attaccato)')
    ax.set_xlabel("tempo dall'inizio della run (s)")
    ax.set_ylabel("separation (m)")
    ax.set_title(f"{run['event_subtype']} | received={run['received']} | match {run['match_id']} | event {run['event_id']}")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def compute_separation(frames, run, fps=10):
    """
    Dato il tracking (lista di dict frame) e una riga del Layer 3,
    restituisce la curva di separation nel tempo per quella run.
    
    Returns:
        pd.Series indicizzata per frame, con una colonna 'sep' e 't_seconds',
        oppure None se la run non è calcolabile (buchi di tracking).
    """
    # 1. Estrai metadata della run (ids, frame_start/end)
    # 2. Slice del tracking sui frame della run
    # 3. Esplodi player_data in DataFrame lungo
    # 4. Filtra sui due player_id
    # 5. Check: ci sono abbastanza frame per runner e difensore?
    #    Se no → return None
    # 6. Pivot su frame con x,y per i due giocatori
    # 7. Calcola sep via np.hypot
    # 8. Costruisci t_seconds dall'index
    # 9. Return DataFrame con [t_seconds, sep] indicizzato per frame

    # players metadata
    defender_id = int(run["defender_id"])
    player_id = int(run["player_id"])
    frame_start = run["frame_start"]
    frame_end = run["frame_end"]

    frames_subset = [frames[frame_start:frame_end+1]]

    df_players_data = pd.json_normalize(
        frames_subset[0],
        record_path="player_data",
        meta=["frame", "timestamp", "period"]
    )

    if df_players_data.empty or "player_id" not in df_players_data.columns:
        return None

    between_frames = df_players_data.loc[(df_players_data["player_id"].isin([defender_id, player_id]))]

    present_runner = (between_frames["player_id"] == player_id).sum()
    present_defender = (between_frames["player_id"] == defender_id).sum()
    expected = frame_end - frame_start + 1

    if present_runner < expected * 0.7 or present_defender < expected * 0.7:
        return None
    
    pvt = between_frames.pivot_table(
                        index="frame",
                        columns="player_id",
                        values=["x", "y"]
                        )
    
    ts = (between_frames.groupby("frame")["timestamp"]
                     .first()
                     .pipe(pd.to_timedelta)
                     .dt.total_seconds())
    pvt["t_seconds"] = ts - ts.iloc[0]

    # computing euclidean distance between def and players
    dx = pvt[("x", player_id)] - pvt[("x", defender_id)]
    dy = pvt[("y", player_id)] - pvt[("y", defender_id)]

    sep = np.hypot(dx, dy)
    sep = sep.sort_index()

    sep_df = pd.DataFrame({
        "t_seconds": pvt["t_seconds"],
        "sep": sep
    })

    return sep_df

m_id = 1925299
path_tracking = f"{base}\\{m_id}\\{m_id}_tracking_extrapolated.jsonl"
path_layer3 = f"{base}\\layer3_all_matches.csv"

frames = load_tracking(path_tracking)
layer3 = pd.read_csv(path_layer3)

# prendi una singola riga di tipo run cross_rec
run = layer3.loc[(layer3['event_subtype'] == "cross_receiver") & 
                 (layer3['match_id'] == m_id)].iloc[19]

df = compute_separation(frames, run)
plot_separation(df, run)

drop_off_runs = layer3.loc[(layer3['event_subtype'] == "dropping_off") & 
                 (layer3['match_id'] == m_id)]

sub = drop_off_runs
rec = sub[sub['received']]
notrec = sub[~sub['received']]

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(rec['separation_start'], bins=15, alpha=0.5, color='green', 
        label=f'received (n={len(rec)})', density=True)
ax.hist(notrec['separation_start'], bins=15, alpha=0.5, color='red', 
        label=f'not received (n={len(notrec)})', density=True)
ax.set_xlabel("separation_start (m)")
ax.set_ylabel("density")
ax.set_title(f"dropping_off | match {m_id}")
ax.legend()
plt.tight_layout()
plt.show()

runs_subset = drop_off_runs.sample(min(60, len(drop_off_runs)), random_state=42)
t_grid = np.arange(0, 1.2, 0.1)
curves_rec = []
curves_notrec = []

fig, ax = plt.subplots(figsize=(10, 6))

for _, r in runs_subset.iterrows():
    curve = compute_separation(frames, r)
    if curve is None:
        continue
    # plot curva sottile
    color = 'green' if r['received'] else 'red'
    ax.plot(curve['t_seconds'], curve['sep'], color=color, alpha=0.25)
    # accumula per mediana
    if curve['t_seconds'].max() < 2.5:
        continue
    sep_interp = np.interp(t_grid, curve['t_seconds'], curve['sep'])
    (curves_rec if r['received'] else curves_notrec).append(sep_interp)

arr_rec = np.vstack(curves_rec) if curves_rec else np.empty((0, len(t_grid)))
arr_notrec = np.vstack(curves_notrec) if curves_notrec else np.empty((0, len(t_grid)))

if len(arr_rec):
    med_rec = np.median(arr_rec, axis=0)
    ax.plot(t_grid, med_rec, color='darkgreen', linewidth=3, linestyle='-',
            label=f'median received (n={len(arr_rec)})', zorder=10)
if len(arr_notrec):
    med_notrec = np.median(arr_notrec, axis=0)
    ax.plot(t_grid, med_notrec, color='darkred', linewidth=3, linestyle='--',
            label=f'median not received (n={len(arr_notrec)})', zorder=10)

ax.axhline(2, color='gray', linestyle='--', alpha=0.4)
ax.set_xlabel("tempo dall'inizio della run (s)")
ax.set_ylabel("separation (m)")
ax.set_title(f"dropping_off | match {m_id}")
ax.set_xlim(0, 3)
ax.set_ylim(0, 10)
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()