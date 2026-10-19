"""Velocita' di corridore e difensore al frame di PARTENZA della corsa.

Q-06: serve una fonte di anisotropia direzionale ancorata alla partenza.
Tutte le direzionali testate finora usano il punto di arrivo, quindi sono
proprieta' dell'episodio (D-15).

Differenze finite a 10 Hz. Due varianti:
  backward  v = (p(t0) - p(t0-w)) / (w/hz)     solo frame PRE-corsa
  centrale  v = (p(t0+k) - p(t0-k)) / (2k/hz)  usa i primi decimi di corsa

La backward e' il default: strettamente pre-evento, come il DRS(t) misurato
nei 3 s prima. La centrale e' il controllo di sensibilita'.

I coseni sono invarianti all'orientamento del campo, quindi si lavora in
coordinate grezze definendo la porta attaccata per corsa da `attacking_side`.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "DRS/DRS_t")
import drst as D

cfg = D.DEFAULT
HZ = cfg.tracking_hz          # 10
W_BACK = 5                    # 0.5 s
K_CENTR = 2                   # +/- 0.2 s
GK_SPEED_GATE = 1.0           # m/s: sotto, la direzione del difensore e' indefinita


def vel(players, frames_sorted, pid, f_from, f_to, dt):
    """Velocita' fra due frame, None se il giocatore manca in uno dei due."""
    a = players.get(f_from, {}).get(pid)
    b = players.get(f_to, {}).get(pid)
    if a is None or b is None:
        return None
    return np.array([(b[0] - a[0]) / dt, (b[1] - a[1]) / dt])


def cosang(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-9 or nv < 1e-9:
        return np.nan
    return float(np.dot(u, v) / (nu * nv))


def run_features(r, tr):
    f0 = int(r[cfg.col_frame_start])
    pid = int(r["player_id"])
    did = int(r[cfg.col_defender]) if pd.notna(r[cfg.col_defender]) else None
    ltr = str(r.get("attacking_side", "")) == "left_to_right"
    goal = np.array([cfg.goal_x if ltr else -cfg.goal_x, 0.0])

    P = tr.players
    pr = P.get(f0, {}).get(pid)
    out = {cfg.col_match: int(r[cfg.col_match]), cfg.col_event: r[cfg.col_event]}
    if pr is None:
        return out
    pr = np.array(pr)

    vr_b = vel(P, tr.frames, pid, f0 - W_BACK, f0, W_BACK / HZ)
    vr_c = vel(P, tr.frames, pid, f0 - K_CENTR, f0 + K_CENTR, 2 * K_CENTR / HZ)

    if vr_b is not None:
        out["vel_runner_speed"] = float(np.linalg.norm(vr_b))
        out["vel_run_cos_goal"] = cosang(vr_b, goal - pr)
    if vr_c is not None:
        out["vel_run_cos_goal_centr"] = cosang(vr_c, goal - pr)
        out["vel_runner_speed_centr"] = float(np.linalg.norm(vr_c))

    if did is None:
        return out
    pdf = P.get(f0, {}).get(did)
    if pdf is None:
        return out
    pdf = np.array(pdf)
    vd_b = vel(P, tr.frames, did, f0 - W_BACK, f0, W_BACK / HZ)
    if vd_b is None:
        return out
    sd = float(np.linalg.norm(vd_b))
    out["vel_def_speed"] = sd
    # proxy del cono visivo: in 2D l'orientamento si stima dalla direzione di
    # moto, quindi "quanto il corridore e' fuori asse rispetto a dove vado".
    # Sotto il gate di velocita' la direzione e' indefinita -> NaN, non 0.
    if sd >= GK_SPEED_GATE:
        out["vel_def_cos_runner"] = cosang(vd_b, pr - pdf)
        out["vel_def_cos_owngoal"] = cosang(vd_b, -goal - pdf)
    if vr_b is not None:
        out["vel_rel_speed"] = float(np.linalg.norm(vr_b)) - sd
        sep = pr - pdf
        n = np.linalg.norm(sep)
        if n > 1e-9:
            # derivata della distanza corridore-difensore: <0 il difensore chiude
            out["vel_closing"] = float(np.dot(vr_b - vd_b, sep) / n)
    return out


def main(layer3_path, tracking_root, out_path):
    runs = D.load_layer3(layer3_path, cfg)
    root = Path(tracking_root)
    parts = []
    for mid in sorted(runs[cfg.col_match].unique()):
        m = runs[runs[cfg.col_match] == mid]
        p = root / f"{mid}/{mid}_tracking_extrapolated.jsonl"
        if not p.exists():
            print(f"[vel] tracking assente: {mid}"); continue
        lo = int(m[cfg.col_frame_start].min()) - 50
        hi = int(m[cfg.col_frame_end].max()) + 50
        tr = D.load_tracking(p, (lo, hi))
        rows = [run_features(r, tr) for _, r in m.iterrows()]
        df = pd.DataFrame(rows)
        cov = df.get("vel_run_cos_goal", pd.Series(dtype=float)).notna().mean()
        print(f"[vel] match {mid}: {len(m)} corse, copertura cos_goal {cov:.1%}",
              flush=True)
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(out_path, index=False)
    print(f"[vel] {len(out)} righe x {out.shape[1]} colonne -> {out_path}")
    print(out.notna().mean().round(3).to_string())


if __name__ == "__main__":
    main("data/matches/layer3_enriched.csv", "data/tracking_raw",
         "data/velocity_features.csv")
