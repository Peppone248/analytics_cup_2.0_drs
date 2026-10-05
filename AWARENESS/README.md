# AWARENESS — the "Ghost" model (Filippo) · v0

> **v0** — works end-to-end, still under review (method, thresholds, file sizes, code cleanup).

Belief-state awareness of defenders on the SkillCorner open **body-pose** matches
(1925299 Brisbane–Perth, 1996435 Sydney–Adelaide), ported from Filippo's master
thesis (Euro 2024 Hawk-Eye) and extended with a new readout: the **ghost**,
i.e. where the defender *believes* each attacker is.

```
pose (25 fps, 29 joints) ─► head angle + shoulders ─► FoV weight w_ij (Bekkers SSAC26)
                                                        │
             true attacker position ──────────────► adaptive-R Kalman ─► ghost x̂, covariance P
                                                        │
                    readouts: ghost error ‖x̂ − x‖ · Mahalanobis (false belief) · A_cov (thesis)
```

Activity log (dated entries, newest first: what was done, methodological choices, results, open
points): **[NOTES_FILIPPO.md](NOTES_FILIPPO.md)** (in Italian, for the team).

## Layout

| path | what |
|---|---|
| `ghost/data.py` | stream pose archives → compact parquet (players / ball), lazy full skeletons |
| `ghost/orientation.py` | head angle (gating, circular smoothing, gap fill, ball prior), speed, shoulders |
| `ghost/fov.py` | FoV + occlusion weights, copied from the thesis (Bekkers) |
| `ghost/kalman.py` | batched adaptive-R Kalman (thesis), + motion prior option + init-on-first-sighting |
| `ghost/motion.py` | motion priors (thesis CV, continuous CV, Ornstein–Uhlenbeck) via exact Van Loan discretisation |
| `ghost/scores.py` | A_cov (thesis), ghost error, A_ghost, Mahalanobis², belief states |
| `ghost/pipeline.py` | one window → long (frame, defender, attacker) table |
| `ghost/viz2d.py` | 2D pitch snapshot / GIF with ghosts, FoV cone, time series |
| `ghost/export3d.py` + `viewer/pov3d_template.html` | self-contained three.js viewer: volumetric bodies, holographic ghosts, broadcast / top / follow / **POV** cameras, selectable focal defender |
| `scripts/01..06` | prepare match → calibrate prior → off-ball runs → demo → rebuild viewer → screenshots |
| `tests/check_kalman_port.py` | ported filter == thesis filter (numerically) |

## Run

```bash
cd AWARENESS
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
# download pose (≈600 MB each, from the Hugging Face dataset; see data/bodypose/README.md)
.venv/bin/python -c "import sys; sys.path.insert(0,'..'); from src.data.pose_loading import download_match; download_match(1925299, dest_dir='data_local')"
.venv/bin/python scripts/01_prepare_match.py 1925299          # ≈4 min, once
.venv/bin/python scripts/02_calibrate_motion_prior.py 1925299 1996435
.venv/bin/python scripts/03_offball_runs.py 1925299           # ≈4 min
.venv/bin/python scripts/04_make_demo.py 1925299 4            # outputs/ghost_viewer.html (+ GIFs/PNGs in outputs/media/)
open outputs/ghost_viewer.html
```

`.venv/`, `data_local/`, `outputs/cache/` and `outputs/media/` (GIFs, PNG snapshots, screenshots) are git-ignored (see `.gitignore`).
