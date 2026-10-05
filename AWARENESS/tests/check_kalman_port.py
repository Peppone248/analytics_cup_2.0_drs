"""The ported batched filter (tau_v=None) must match the thesis reference
implementation (filterpy, one pair at a time) on random gap-free sequences."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, "/Users/filippomolino/Documents/PySport/master_thesis")  # thesis repo (reference)
from ghost.kalman import run_filter_batched
from src.awareness.kalman.filter import run_filter

rng = np.random.default_rng(0)
P, N = 20, 300
pos = np.cumsum(rng.normal(0, 0.2, (P, N, 2)), axis=1)
w = np.clip(rng.uniform(-0.5, 1, (P, N)), 0, 1)
valid = np.ones((P, N), bool)
xb, _, Pb = run_filter_batched(pos, w, valid)
for i in range(P):
    xr, Pr = run_filter(pos[i], w[i])
    assert np.allclose(xr, xb[i], atol=1e-8) and np.allclose(Pr, Pb[i], atol=1e-8), i
print("OK: ported filter == thesis run_filter on", P, "pairs x", N, "frames")
