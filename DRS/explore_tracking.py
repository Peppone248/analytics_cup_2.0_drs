import json
from pathlib import Path

DATA_DIR = Path(r"E:\Projects\opendata\data")
MATCH_ID = 1953632
match_dir = DATA_DIR / "matches" / str(MATCH_ID)

with open(match_dir / f"{MATCH_ID}_match.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

home_id = meta["home_team"]["id"]
away_id = meta["away_team"]["id"]
print(f"HOME: {meta['home_team']['short_name']} (id={home_id})")
print(f"AWAY: {meta['away_team']['short_name']} (id={away_id})")
print(f"home_team_side: {meta.get('home_team_side')}")
print()

# 1. Mappa per id "classico"
player_to_team_by_id = {p["id"]: p["team_id"] for p in meta["players"]}
# 2. Mappa per trackable_object (potrebbe essere quello usato dal tracking)
player_to_team_by_trk = {p["trackable_object"]: p["team_id"]
                         for p in meta["players"]
                         if p.get("trackable_object") is not None}

print(f"# players in roster: {len(meta['players'])}")
print(f"# unique 'id': {len(player_to_team_by_id)}")
print(f"# unique 'trackable_object': {len(player_to_team_by_trk)}")

# 3. Prendo il set di player_id visti nel tracking (basta un pass sui primi ~1000 frame vivi)
tracking_path = match_dir / f"{MATCH_ID}_tracking_extrapolated.jsonl"
seen_in_tracking = set()
live_scanned = 0
with open(tracking_path, "r", encoding="utf-8") as f:
    for line in f:
        fr = json.loads(line)
        if fr["player_data"]:
            for p in fr["player_data"]:
                seen_in_tracking.add(p["player_id"])
            live_scanned += 1
            if live_scanned >= 500:  # 500 live frame bastano a coprire tutti i titolari
                break

print(f"\n# distinct player_id seen in tracking (first 500 live frames): {len(seen_in_tracking)}")

match_by_id = seen_in_tracking & set(player_to_team_by_id.keys())
match_by_trk = seen_in_tracking & set(player_to_team_by_trk.keys())
print(f"  match against roster 'id':               {len(match_by_id)} / {len(seen_in_tracking)}")
print(f"  match against roster 'trackable_object': {len(match_by_trk)} / {len(seen_in_tracking)}")

# 4. Decido quale chiave usare e costruisco i lookup finali
if len(match_by_trk) > len(match_by_id):
    key_used = "trackable_object"
    player_to_team = player_to_team_by_trk
    id_lookup = {p["trackable_object"]: p for p in meta["players"]
                 if p.get("trackable_object") is not None}
else:
    key_used = "id"
    player_to_team = player_to_team_by_id
    id_lookup = {p["id"]: p for p in meta["players"]}

print(f"\nKEY USED for tracking join: {key_used}")

# 5. Sanity check: 11 titolari per squadra visti nei primi frame vivi
from collections import Counter
team_counts = Counter(player_to_team.get(pid) for pid in seen_in_tracking)
print(f"\nPlayers seen per team_id (first 500 live frames):")
for tid, c in team_counts.most_common():
    label = "HOME" if tid == home_id else "AWAY" if tid == away_id else "UNKNOWN"
    print(f"  team_id={tid} ({label}): {c}")

# 6. Nome + ruolo dei primi 3 player_id incontrati, come sanity check leggibile
print(f"\nSample players (first 3 from tracking):")
for pid in list(seen_in_tracking)[:3]:
    rec = id_lookup.get(pid)
    if rec:
        print(f"  {pid} -> {rec['short_name']} ({rec['player_role']['acronym']}, team={rec['team_id']})")
    else:
        print(f"  {pid} -> NOT IN ROSTER")