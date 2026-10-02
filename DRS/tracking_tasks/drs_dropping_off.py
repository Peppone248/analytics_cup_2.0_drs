import pandas as pd
import numpy as np
import json
from DRS.config import base

path_layer3 = f"{base}\\layer3_all_matches.csv"

layer3 = pd.read_csv(path_layer3)
dropping = layer3[layer3.event_subtype == "dropping_off"].copy()

dropping = layer3[layer3["event_subtype"] == "dropping_off"]

print(dropping[dropping["match_id"] == 1886347].groupby("defending_team_id").size())

print(dropping.shape)
print(f"received rate: {dropping['received'].mean()}")
print(f"squadre difendenti: {dropping['defending_team_id'].nunique()}")

# ho abbastanza run per squadra da fare un ranking?
team_counts = dropping.groupby("defending_team_id").agg(
    n=("event_id", "count"),
    received_rate=("received", "mean")
).sort_values("n")

print(team_counts)

print(f"match totali: {dropping['match_id'].nunique()}")
print(f"match per squadra difendente:")
print(dropping.groupby("defending_team_id")["match_id"].nunique().sort_values())

# Ground truth vs expected
X = dropping[["separation_start", "n_opponents_available"]]
y = dropping["received"].astype(int)
groups = dropping["match_id"]

print(X.isna().sum())