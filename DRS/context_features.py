# range delle coordinate — serve per capire dove sono le porte
import numpy as np
import pandas as pd
import json
from utils import unique_teams_dict
from scipy.stats import fisher_exact
from config import GOAL_X_LEFT,GOAL_X_RIGHT


match_id = 1899585
base = rf"E:\Projects\analytics_cup_2.0_drs\data\matches\{match_id}"

# dati
layer3 = pd.read_csv(f"{base}/layer3_{match_id}.csv")
with open(f"{base}/{match_id}_match.json", "r") as f:
    match_data = json.load(f)

events = pd.read_csv(f"{base}/{match_id}_dynamic_events.csv")
runs = events[events["event_type"] == "off_ball_run"].dropna(
    subset=["separation_start", "frame_start", "player_id", "team_id"])
frames = []
with open(f"{base}/{match_id}_tracking_extrapolated.jsonl", "r") as f:
    for line in f:
        frames.append(json.loads(line))

def compute_goal_side(layer3, frame_lookup, player_to_team, goal_map):
    results = []
    for _, row in layer3.iterrows():
        frame = frame_lookup.get(int(row["frame_start"]))
        if frame is None:
            results.append(None)
            continue

        defender = next((p for p in frame["player_data"]
                         if p["player_id"] == int(row["defender_id"])), None)
        runner = next((p for p in frame["player_data"]
                       if p["player_id"] == int(row["player_id"])), None)

        if defender is None or runner is None:
            results.append(None)
            continue

        defender_team = player_to_team[defender["player_id"]]
        own_goal = goal_map[(defender_team, frame["period"])]

        if own_goal < 0:
            is_goal_side = defender["x"] <= runner["x"]
        else:
            is_goal_side = defender["x"] >= runner["x"]

        results.append(is_goal_side)

    return results


def detect_goal_positions(frames, player_to_team, period=1):
    """Determina quale porta difende ogni squadra nel periodo dato."""
    team_xs = {}
    for f in frames:
        if f.get("period") != period:
            continue
        for p in f.get("player_data", []):
            if p["is_detected"]:
                tid = player_to_team.get(p["player_id"])
                if tid:
                    team_xs.setdefault(tid, []).append(p["x"])

    team_means = {tid: np.mean(xs) for tid, xs in team_xs.items()}
    deepest_team = min(team_means, key=team_means.get)

    goal_map = {}
    for tid in team_means:
        defends_left = (tid == deepest_team)
        goal_map[(tid, period)] = GOAL_X_LEFT if defends_left else GOAL_X_RIGHT
        # periodo opposto: si scambiano
        other_period = 2 if period == 1 else 1
        goal_map[(tid, other_period)] = GOAL_X_RIGHT if defends_left else GOAL_X_LEFT

    return goal_map

all_x, all_y = [], []
ball_x, ball_y = [], []

for f in frames[:5000]:  # campione, non serve tutto
    if f["ball_data"] and f["ball_data"].get("x") is not None:
        ball_x.append(f["ball_data"]["x"])
        ball_y.append(f["ball_data"]["y"])
    for p in f.get("player_data", []):
        all_x.append(p["x"])
        all_y.append(p["y"])

print(f"Player X: min={min(all_x):.1f}  max={max(all_x):.1f}")
print(f"Player Y: min={min(all_y):.1f}  max={max(all_y):.1f}")
print(f"Ball   X: min={min(ball_x):.1f}  max={max(ball_x):.1f}")
print(f"Ball   Y: min={min(ball_y):.1f}  max={max(ball_y):.1f}")

# carica roster per sapere chi è in quale squadra
with open(f"{base}/{match_id}_match.json", "r") as f:
    match_data = json.load(f)

match_data = unique_teams_dict(match_data)

player_to_team = {p["id"]: p["team_id"] for p in match_data["players"]}
team_ids = list(set(player_to_team.values()))


for team in match_data["teams"]:
    tid = team["id"]
    team_players = {pid for pid, t in player_to_team.items() if t == tid}
    xs = [p["x"] for f in frames[:5000]
          if f.get("period") == 1
          for p in f.get("player_data", [])
          if p["player_id"] in team_players and p["is_detected"]]
    print(f"{team['short_name']} (id={tid}): mean X 1st half = {np.mean(xs):.1f}")

goal_map = detect_goal_positions(frames, player_to_team, 1)
print(goal_map)
frame_lookup = {f["frame"]: f for f in frames}
layer3["goal_side"] = compute_goal_side(layer3, frame_lookup, player_to_team, goal_map)

# check rapido: distribuzione e correlazione con received
print(f"goal_side non-null: {layer3['goal_side'].notna().sum()}")
print(f"goal_side True: {layer3['goal_side'].sum()}")
print()
print(layer3.groupby("goal_side")["received"].mean())

# compute_goal_side(layer3, frame_lookup, player_to_team, goal_map)

gs = layer3[layer3["goal_side"] == True]
ngs = layer3[layer3["goal_side"] == False]

table = [[gs["received"].sum(), len(gs) - gs["received"].sum()],
         [ngs["received"].sum(), len(ngs) - ngs["received"].sum()]]

odds, p = fisher_exact(table)
print(f"Fisher exact p = {p:.3f}")

layer3["close_and_goal_side"] = (layer3["goal_side"] == True) & (layer3["separation_start"] < 5)
print(layer3.groupby("close_and_goal_side")["received"].mean())