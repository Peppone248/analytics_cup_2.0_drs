import json
import pandas as pd

base = "data/matches/1953632/1953632_"

with open(base + "match.json", encoding="utf-8") as f:
    match = json.load(f)

print("Chiavi match.json:", list(match.keys()))
players = match.get("players", [])
print("N giocatori:", len(players))
print(json.dumps(players[:2], indent=2, ensure_ascii=False))

ev = pd.read_csv(base + "dynamic_events.csv")
print("\nColonne dynamic_events:")
print(ev.columns.tolist())
if "event_type" in ev.columns:
    print("\n", ev["event_type"].value_counts())