"""Rebuild outputs/ghost_viewer.html from the cached clip JSONs (after editing the template).

Usage: .venv/bin/python scripts/05_rebuild_viewer.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ghost.data import CACHE_DIR
from ghost.export3d import build_viewer

order_file = CACHE_DIR / "clips_order.json"          # written by 04_make_demo.py
names = (json.loads(order_file.read_text()) if order_file.exists()
         else sorted(p.name for p in CACHE_DIR.glob("clip_*.json")))
clips = [json.loads((CACHE_DIR / n).read_text()) for n in names]
out = build_viewer(clips, Path(__file__).resolve().parents[1] / "outputs" / "ghost_viewer.html")
print(f"{len(clips)} clips -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
