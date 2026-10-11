"""Feature di contesto geometrico ricavate dal tracking.

Due funzioni pure, importate da build_full_dataset.py:

    detect_goal_positions(frames, player_to_team, period)
        -> quale porta difende ogni squadra in ciascun periodo

    compute_goal_side(layer3, frame_lookup, player_to_team, goal_map)
        -> per ogni run, il difensore e' dal lato porta rispetto al runner?

L'esplorazione che era qui a livello di modulo (range delle coordinate,
verifica delle porte, test di Fisher su goal_side) e' ora sotto main() e
gira solo lanciando il file direttamente:

    python context_features.py              # match di default
    python context_features.py 1996436      # un altro match
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import GOAL_X_LEFT, GOAL_X_RIGHT

DEFAULT_EXPLORE_MATCH = 1899585


# ---------------------------------------------------------------------------
# Funzioni pure — nessun IO, nessun effetto all'import
# ---------------------------------------------------------------------------

def detect_goal_positions(frames, player_to_team, period=1):
    """Determina quale porta difende ogni squadra nel periodo dato.

    La squadra con la X media piu' bassa nel periodo difende la porta di
    sinistra; nel periodo opposto le porte si scambiano.
    """
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


def compute_goal_side(layer3, frame_lookup, player_to_team, goal_map):
    """Per ogni run: il difensore e' fra il runner e la propria porta?

    Restituisce una lista allineata alle righe di layer3, con None dove il
    frame o uno dei due giocatori non sono disponibili nel tracking.
    """
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


# ---------------------------------------------------------------------------
# Esplorazione — gira solo da riga di comando
# ---------------------------------------------------------------------------

def _describe_phases(match_id):
    from config import match_file

    events = pd.read_csv(match_file(match_id, "dynamic_events.csv"))
    print(events.columns.tolist())
    print(events.head())

    phases = pd.read_csv(match_file(match_id, "phases_of_play.csv"))
    print(phases.columns.tolist())
    print(f"n fasi: {len(phases)}")
    print(f"durata media (frame): {(phases['frame_end'] - phases['frame_start']).mean():.1f}")
    print(f"durata media (sec): {phases['duration'].mean():.1f}")
    return events


def _describe_coordinates(frames, n_frames=5000):
    """Range delle coordinate — serve per capire dove sono le porte."""
    all_x, all_y = [], []
    ball_x, ball_y = [], []

    for f in frames[:n_frames]:  # campione, non serve tutto
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


def _describe_team_sides(match_data, frames, player_to_team, n_frames=5000):
    for team in match_data["teams"]:
        tid = team["id"]
        team_players = {pid for pid, t in player_to_team.items() if t == tid}
        xs = [p["x"] for f in frames[:n_frames]
              if f.get("period") == 1
              for p in f.get("player_data", [])
              if p["player_id"] in team_players and p["is_detected"]]
        print(f"{team['short_name']} (id={tid}): mean X 1st half = {np.mean(xs):.1f}")


def _test_goal_side(layer3):
    """Il goal_side discrimina la ricezione? Fisher exact sulla 2x2."""
    from scipy.stats import fisher_exact

    print(f"goal_side non-null: {layer3['goal_side'].notna().sum()}")
    print(f"goal_side True: {layer3['goal_side'].sum()}")
    print()
    print(layer3.groupby("goal_side")["received"].mean())

    gs = layer3[layer3["goal_side"] == True]       # noqa: E712
    ngs = layer3[layer3["goal_side"] == False]     # noqa: E712

    table = [[gs["received"].sum(), len(gs) - gs["received"].sum()],
             [ngs["received"].sum(), len(ngs) - ngs["received"].sum()]]

    odds, p = fisher_exact(table)
    print(f"Fisher exact p = {p:.3f}")

    layer3["close_and_goal_side"] = (
        (layer3["goal_side"] == True) & (layer3["separation_start"] < 5)   # noqa: E712
    )
    print(layer3.groupby("close_and_goal_side")["received"].mean())


def main(match_id=DEFAULT_EXPLORE_MATCH):
    from config import match_dir, match_file, require_data
    from utils import (build_lookup, build_player_to_team, load_match,
                       load_tracking, unique_teams_dict)

    require_data()
    print(f"Match {match_id} — {match_dir(match_id)}\n")

    layer3 = pd.read_csv(match_dir(match_id) / f"layer3_{match_id}.csv")
    match_data = unique_teams_dict(load_match(match_file(match_id, "match.json")))
    frames = load_tracking(match_file(match_id, "tracking_extrapolated.jsonl"))
    player_to_team = build_player_to_team(match_data)

    _describe_phases(match_id)
    _describe_coordinates(frames)
    _describe_team_sides(match_data, frames, player_to_team)

    goal_map = detect_goal_positions(frames, player_to_team, period=1)
    print(goal_map)

    frame_lookup = build_lookup(frames)
    layer3["goal_side"] = compute_goal_side(
        layer3, frame_lookup, player_to_team, goal_map
    )
    _test_goal_side(layer3)


if __name__ == "__main__":
    import sys

    main(int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_EXPLORE_MATCH)
