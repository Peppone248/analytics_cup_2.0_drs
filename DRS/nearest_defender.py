import pandas as pd
import json
import numpy as np
from scipy.stats import pearsonr


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

path_to_matches = r"E:\Projects\opendata\data\matches\1899585\1899585_match.json"
path_to_events = r"E:\Projects\opendata\data\matches\1899585\1899585_dynamic_events.csv"
path_to_tracking = r"E:\Projects\opendata\data\matches\1899585\1899585_tracking_extrapolated.jsonl"

def nearest_defender(run_row, frame_lookup, player_to_team):
    frame_start = int(run_row["frame_start"])
    runner_id = int(run_row["player_id"])
    runner_team_id = int(run_row["team_id"])

    empty = {
        "defender_id": None,
        "distance": None,
        "n_opponents_available": 0,
        "skipped_reason": None,
    }

    # Check 1: frame esiste nel tracking?
    if frame_start not in frame_lookup:
        return {**empty, "skipped_reason": "frame_not_in_tracking"}

    frame = frame_lookup[frame_start]

    # Check 2: player_data non vuoto?
    if not frame["player_data"]:
        return {**empty, "skipped_reason": "empty_frame"}

    # Check 3: runner presente e detected?
    runner = next((p for p in frame["player_data"] 
                   if p["player_id"] == runner_id), None)
    if runner is None:
        return {**empty, "skipped_reason": "runner_not_in_frame"}
    if not runner["is_detected"]:
        return {**empty, "skipped_reason": "runner_not_detected"}

    # Check 4: opponents detected?
    opponents = [p for p in frame["player_data"] 
                 if player_to_team.get(p["player_id"]) != runner_team_id]
    if len(opponents) == 0:
        return {**empty, "skipped_reason": "no_opponents_detected", 
                "n_opponents_available": 0}

    # Calcolo vero
    runner_xy = np.array([runner["x"], runner["y"]])
    opp_xy = np.array([[p["x"], p["y"]] for p in opponents])
    opp_ids = np.array([p["player_id"] for p in opponents])

    distances = np.linalg.norm(opp_xy - runner_xy, axis=1)
    idx = np.argmin(distances)

    return {
        "defender_id": int(opp_ids[idx]),
        "distance": float(distances[idx]),   # ← nota: float, non int!
        "n_opponents_available": len(opponents),
        "skipped_reason": None,
    }

with open(path_to_matches, "r") as f:
    match_data = json.load(f)
events = pd.read_csv(path_to_events)

frames = []
with open(path_to_tracking, "r") as f:
    for line in f:
        frames.append(json.loads(line))


runs = events[events["event_type"] == "off_ball_run"]
runs = runs.dropna(subset=["separation_start", "frame_start", "player_id", "team_id"])
frame_lookup = {f["frame"]: f for f in frames}
player_to_team = {p["id"]: p["team_id"] for p in match_data["players"]}

results = []

for _, row in runs.iterrows():
    out = nearest_defender(
        run_row=row,
        frame_lookup=frame_lookup,
        player_to_team=player_to_team,
    )
    out["event_id"] = row["event_id"]
    out["separation_start"] = row["separation_start"]
    results.append(out)

df_results = pd.DataFrame(results)
#print(df_results)
print(df_results["skipped_reason"].value_counts(dropna=False))
print(len(df_results), "totali,",
      df_results["skipped_reason"].isna().sum(), "processate ok")

valid = df_results[df_results["skipped_reason"].isna()]
print(valid["distance"].describe())
delta = valid["distance"] - valid["separation_start"]
print(delta.describe())

r, p = pearsonr(valid["distance"], valid["separation_start"])
print(f"Pearson r = {r:.3f}")

valid = df_results[df_results["skipped_reason"].isna()].copy()
valid["delta"] = valid["distance"] - valid["separation_start"]

outliers = valid[valid["delta"] > 3]
#print(len(outliers))
#print('outliers: ', outliers)

outliers_with_context = outliers.merge(runs[["event_subtype", "frame_start", "player_name", "event_id"]],
                                       on="event_id",
                                       how="left")
# --- Layer 3: join esiti ---
outcome_cols = ["targeted", "received", "lead_to_shot",
                "separation_start", "separation_end", "separation_gain"]
context_cols = ["event_id", "event_subtype", "player_id", "player_name",
                "team_id", "frame_start", "frame_end"]

# check disponibilità
for col in outcome_cols:
    if col in runs.columns:
        non_null = runs[col].notna().sum()
        print(f"  {col}: {non_null} non-null ({non_null/len(runs)*100:.1f}%)")
    else:
        print(f"  {col}: ASSENTE")

# join: prendi solo le run processate ok
keep_cols = context_cols + [c for c in outcome_cols if c in runs.columns]
layer3 = valid.merge(runs[keep_cols], on="event_id", how="left")

print(f"\nLayer 3 shape: {layer3.shape}")
print(layer3.columns.tolist())
print(layer3.head(3).to_string())


# pulizia: tieni solo separation_start da SkillCorner
layer3 = layer3.drop(columns=["separation_start_x", "skipped_reason"])
layer3 = layer3.rename(columns={"separation_start_y": "separation_start"})

# --- Profilo difensivo per defender ---
profile = layer3.groupby("defender_id").agg(
    n_runs=("event_id", "count"),
    targeted_rate=("targeted", "mean"),
    received_rate=("received", "mean"),
    lead_to_shot_rate=("lead_to_shot", "mean"),
    avg_separation_start=("separation_start", "mean"),
    avg_separation_gain=("separation_gain", "mean"),
).round(3)

profile = profile.sort_values("n_runs", ascending=False)
print(profile.to_string())


match_data["teams"] = [
    match_data["home_team"],
    match_data["away_team"]
]

# mappa defender_id → nome e squadra
id_to_name = {p["id"]: p["short_name"] for p in match_data["players"]}
id_to_team = {p["id"]: p["team_id"] for p in match_data["players"]}
team_names = {t["id"]: t["short_name"] for t in match_data["teams"]}

profile = profile.reset_index()
profile["defender_name"] = profile["defender_id"].astype(int).map(id_to_name)
profile["team"] = profile["defender_id"].astype(int).map(id_to_team).map(team_names)

cols = ["defender_name", "team", "n_runs", "targeted_rate", 
        "received_rate", "lead_to_shot_rate", "avg_separation_start", 
        "avg_separation_gain"]
print(profile[cols].to_string(index=False))
