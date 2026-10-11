"""
enrich_layer3.py

Aggiunge al Layer 3 base (layer3_all_matches.csv) tre livelli di contesto:
  Blocco 1 — fasi difensive e di possesso (join temporale su phases_of_play)
  Blocco 2 — feature geometriche/contestuali statiche dell'off_ball_run
             (merge diretto su event_id dal raw dynamic_events)
  Blocco 3 — feature dinamiche dal player_possession attivo al frame_start
             (merge_asof backward)

Output: layer3_enriched.csv
"""
import pandas as pd
import os
from config import base
from utils import load_phases

# ---------------------------------------------------------------------
# INPUT
# ---------------------------------------------------------------------
path_layer3 = f"{base}\\layer3_all_matches.csv"
layer3 = pd.read_csv(path_layer3)
print(f"Layer 3 base: {layer3.shape}")

# ---------------------------------------------------------------------
# BLOCCO 1 — Fasi (join temporale IntervalIndex)
# ---------------------------------------------------------------------
print("\n[Blocco 1] Fasi difensive e di possesso")
enriched_pieces = []

for m_id in layer3["match_id"].unique():
    path_phases = f"{base}\\{m_id}\\{m_id}_phases_of_play.csv"
    phases_m = load_phases(path_phases)

    intervals = pd.IntervalIndex.from_arrays(
        phases_m["frame_start"], phases_m["frame_end"], closed="left"
    )

    layer3_m = layer3.loc[layer3["match_id"] == m_id].copy()
    positions = intervals.get_indexer(layer3_m["frame_start"].values)
    valid = positions >= 0

    out_phase = pd.Series(index=layer3_m.index, dtype="object")
    in_phase = pd.Series(index=layer3_m.index, dtype="object")
    out_phase.iloc[valid] = phases_m["team_out_of_possession_phase_type"].iloc[positions[valid]].values
    in_phase.iloc[valid] = phases_m["team_in_possession_phase_type"].iloc[positions[valid]].values

    layer3_m["out_of_possession_phase"] = out_phase.values
    layer3_m["in_possession_phase"] = in_phase.values
    enriched_pieces.append(layer3_m)

layer3_enriched = pd.concat(enriched_pieces, ignore_index=True)
print(f"  dopo fasi: {layer3_enriched.shape} | NaN out_phase: {layer3_enriched['out_of_possession_phase'].isna().sum()}")

# ---------------------------------------------------------------------
# BLOCCO 2 — Feature statiche dell'off_ball_run (merge diretto su event_id)
# ---------------------------------------------------------------------
# Colonne contestuali popolate su off_ball_run in dynamic_events.
# Join su (match_id, event_id) — 1-a-1, no duplicati.
print("\n[Blocco 2] Feature geometriche/contestuali statiche")

static_cols = [
    "event_id",                             # chiave merge
    # --- zona di campo
    "third_start",                          # defensive / middle / attacking third
    "channel_start",                        # center / half_space / wide
    "x_start", "y_start",                   # coordinate (metri)
    "penalty_area_start",                   # dentro/fuori area
    "attacking_side",                       # verso quale lato attacca
    # --- geometria difensiva locale
    "goal_side_start",                      # difensore è tra runner e porta (inizio)
    "goal_side_end",                        # idem (fine run)
    "angle_of_engagement",                  # angolo con cui difensore si confronta
    "interplayer_distance_start",           # distanza runner-difensore (confronto con separation)
    "inside_defensive_shape_start",         # runner dentro il blocco difensivo
    "delta_to_last_defensive_line_start",   # runner davanti/dietro la linea
    "last_defensive_line_x_start",          # x della linea difensiva
    # --- contesto collettivo difensivo
    "organised_defense",                    # difesa organizzata (binaria)
    "defensive_structure",                  # struttura macro
    "n_opponents_ahead_start",              # avversari tra runner e porta
    "n_teammates_ahead_start",              # compagni davanti al runner
    "n_simultaneous_runs",                  # altre run attive contemporaneamente
    "team_out_of_possession_phase_type"
]

static_pieces = []
for m_id in layer3_enriched["match_id"].unique():
    ev = pd.read_csv(f"{base}\\{m_id}\\{m_id}_dynamic_events.csv")
    obr = ev[ev["event_type"] == "off_ball_run"][static_cols].copy()
    obr["match_id"] = int(m_id)
    static_pieces.append(obr)

static_all = pd.concat(static_pieces, ignore_index=True)

# merge 1-a-1 su (match_id, event_id)
before = len(layer3_enriched)
layer3_enriched = layer3_enriched.merge(
    static_all, on=["match_id", "event_id"], how="left", validate="one_to_one"
)
assert len(layer3_enriched) == before, "merge ha duplicato righe!"
print(f"  dopo statiche: {layer3_enriched.shape}")

# ---------------------------------------------------------------------
# BLOCCO 3 — Feature dinamiche dal player_possession (merge_asof)
# ---------------------------------------------------------------------
# pp_* = contesto del portatore attivo al frame_start dell'off_ball_run
print("\n[Blocco 3] Feature dinamiche dal player_possession")

pp_cols = [
    "frame_start",                   # chiave merge_asof
    "overall_pressure_start",        # pressione sul portatore (ordinale)
    "reception_difficulty_start",    # difficoltà di ricezione (ordinale)
    "n_off_ball_runs",               # run attive (duplicato logico di n_simultaneous_runs, ok tenere)
    "n_passing_options",             # ricchezza opzioni passaggio
    "n_opponents_ahead_start",       # avversari davanti al portatore
    "game_state",                    # vincente/pareggio/perdente
]

pp_pieces = []
for m_id in layer3_enriched["match_id"].unique():
    ev = pd.read_csv(f"{base}\\{m_id}\\{m_id}_dynamic_events.csv")
    pp_df = ev[ev["event_type"] == "player_possession"][pp_cols].copy()
    pp_df = pp_df.rename(columns={
        "frame_start": "pp_frame",
        "overall_pressure_start": "pp_pressure",
        "reception_difficulty_start": "pp_reception_difficulty",
        "n_off_ball_runs": "pp_n_runs",
        "n_passing_options": "pp_n_passing_options",
        "n_opponents_ahead_start": "pp_n_opponents_ahead",
        "game_state": "pp_game_state",
    })
    pp_df["match_id"] = int(m_id)
    pp_df = pp_df.sort_values("pp_frame")
    pp_pieces.append(pp_df)

pp_all = pd.concat(pp_pieces, ignore_index=True)

# merge_asof per match (richiede sorted)
merged_pieces = []
for m_id in layer3_enriched["match_id"].unique():
    layer3_m = layer3_enriched[layer3_enriched["match_id"] == m_id].sort_values("frame_start").copy()
    pp_m = pp_all[pp_all["match_id"] == m_id].sort_values("pp_frame")
    merged = pd.merge_asof(
        layer3_m,
        pp_m.drop(columns=["match_id"]),
        left_on="frame_start",
        right_on="pp_frame",
        direction="backward",
    )
    merged_pieces.append(merged)

layer3_enriched = pd.concat(merged_pieces, ignore_index=True)
print(f"  dopo pp_*: {layer3_enriched.shape}")

# ---------------------------------------------------------------------
# SAVE + DIAGNOSTICA NaN
# ---------------------------------------------------------------------
out_path = f"{base}\\layer3_enriched.csv"
layer3_enriched.to_csv(out_path, index=False)
print(f"\nSalvato: {out_path}")
print(f"Shape finale: {layer3_enriched.shape}")

# NaN rate per colonna, ordinato: colonne nuove in alto se problematiche
print("\n[Diagnostica NaN per colonna, ordinata per % NaN decrescente]")
nan_rate = (layer3_enriched.isna().mean() * 100).round(2).sort_values(ascending=False)
nan_rate = nan_rate[nan_rate > 0]  # mostra solo colonne con qualche NaN
print(nan_rate.to_string())

# Spot check sulle NUOVE colonne
print("\n[Spot check tipi delle nuove colonne]")
new_cols = ["goal_side_start", "goal_side_end", "angle_of_engagement",
            "interplayer_distance_start", "channel_start", "penalty_area_start",
            "n_simultaneous_runs", "organised_defense", "defensive_structure",
            "n_teammates_ahead_start"]
for c in new_cols:
    if c in layer3_enriched.columns:
        s = layer3_enriched[c]
        print(f"  {c}: dtype={s.dtype}, nunique={s.nunique()}, "
              f"NaN%={s.isna().mean()*100:.1f}, "
              f"sample={s.dropna().unique()[:5].tolist()}")
    else:
        print(f"  {c}: MISSING (non presente nel raw)")