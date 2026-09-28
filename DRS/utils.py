import pandas as pd
import json

def unique_teams_dict(match_data):
    match_data["teams"] = [
    match_data["home_team"],
    match_data["away_team"]
    ]

    return match_data

def load_events(path):
    events = pd.read_csv(path, encoding="utf-8")
    return events

def load_match(path):
    with open(path, "r", encoding="utf-8") as f:
        match_data = json.load(f)
    return match_data

def load_tracking(path):
    frames = []
    with open(path, "r") as f:
        for line in f:
            frames.append(json.loads(line))
    return frames

def build_player_to_team(match_data):
    player_to_team = {p["id"]: p["team_id"] for p in match_data["players"]}

    return player_to_team

def build_lookup(frames):
    frame_lookup = {f["frame"]: f for f in frames}
    return frame_lookup