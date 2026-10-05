"""Parse a pose match once and add orientation -> outputs/cache/{id}_players_oriented.parquet.

Usage:  .venv/bin/python scripts/01_prepare_match.py 1925299
(needs AWARENESS/data_local/{id}.jsonl.zip, see README)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ghost.data import CACHE_DIR, load_compact
from ghost.orientation import add_orientation

match_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1925299
players, ball = load_compact(match_id)
pl = add_orientation(players, ball)
pl.to_parquet(CACHE_DIR / f"{match_id}_players_oriented.parquet", index=False)
print(pl.head_source.value_counts(normalize=True).round(3).to_string())
print(f"speed p99 = {pl.speed.quantile(0.99):.2f} m/s, max = {pl.speed.max():.2f}")
