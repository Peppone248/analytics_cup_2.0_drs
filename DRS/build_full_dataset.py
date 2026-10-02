import pandas as pd
import numpy as np
import os
from config import base
from utils import load_events, load_tracking, load_match, build_lookup, extract_team_ids, build_player_to_team
from nearest_defender import nearest_defender
from context_features import compute_goal_side, detect_goal_positions
from scipy.stats import pearsonr

match_ids = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]

raw_counts = []
for m_id in match_ids:
    events = load_events(f"{base}\\{m_id}\\{m_id}_dynamic_events.csv")
    match_data = load_match(f"{base}\\{m_id}\\{m_id}_match.json")
    home_id = match_data["home_team"]["id"]
    away_id = match_data["away_team"]["id"]
    
    runs = events[events["event_type"] == "off_ball_run"]
    dropping = runs[runs["event_subtype"] == "dropping_off"]
    
    raw_counts.append({
        "match_id": int(m_id),
        "home_id": home_id,
        "away_id": away_id,
        "n_dropping_total": len(dropping),
        "n_dropping_home_attack": (dropping["team_id"] == home_id).sum(),
        "n_dropping_away_attack": (dropping["team_id"] == away_id).sum(),
    })

raw = pd.DataFrame(raw_counts)
print(raw)

df_nn_matches = { }
results = []

outcome_cols = ["event_id", "event_subtype", "player_id", "player_name",
                "team_id", "frame_start", "frame_end",
                "targeted", "received", "lead_to_shot",
                "separation_start", "separation_end", "separation_gain"]

for m_id in match_ids:
    try:
        path_event = f"{base}\\{m_id}\\{m_id}_dynamic_events.csv"
        path_tracking = f"{base}\\{m_id}\\{m_id}_tracking_extrapolated.jsonl"
        path_matches = f"{base}\\{m_id}\\{m_id}_match.json"

        match_data = load_match(path_matches)
        events = load_events(path_event)
        frames = load_tracking(path_tracking)

        runs = events[events["event_type"] == "off_ball_run"]
        runs = runs.dropna(subset=["separation_start", "frame_start", "player_id", "team_id"])
        frame_lookup = build_lookup(frames)
        player_to_team = build_player_to_team(match_data)
        home_id, away_id = extract_team_ids(match_data)
        print(f"[{match_ids.index(m_id)+1}/{len(match_ids)}] match {m_id}: {len(runs)} runs")

        for _, row in runs.iterrows():   
            out = nearest_defender(row, frame_lookup, player_to_team)
            out["event_id"] = row["event_id"]
            out["match_id"] = row["match_id"]
            out["separation_start"] = row["separation_start"]
            out["defending_team_id"] = away_id if row["team_id"] == home_id else home_id
            for col in outcome_cols:
                out[col] = row[col]
            results.append(out)
    except Exception as e:
        print(f"SKIP match {m_id}: {e}")
        continue


df_nn_matches = pd.DataFrame(results).sort_values(by="match_id")
print(df_nn_matches.value_counts())

valid = df_nn_matches[df_nn_matches["skipped_reason"].isna()]
print(f"\nRun valide: {len(valid)}")

r, p = pearsonr(valid["distance"], valid["separation_start"])
print(f"Pearson r globale = {r:.4f} (n={len(valid)})")

valid = df_nn_matches[df_nn_matches["skipped_reason"].isna()].copy()
valid.to_csv(os.path.join(base, "layer3_all_matches.csv"), index=False)
print(f"Salvato: {len(valid)} run, {valid['match_id'].nunique()} match")
print(f"\nSubtype distribution:\n{valid['event_subtype'].value_counts().to_string()}")
print(f"\nReceived rate per subtype:\n{valid.groupby('event_subtype')['received'].mean().sort_values(ascending=False).to_string()}")