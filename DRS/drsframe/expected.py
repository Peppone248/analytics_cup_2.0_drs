"""Esposizione attesa dal solo contesto d'attacco, out-of-fold (GroupKFold su match_id).

Idea ("ghosting" statistico, cfr. Le et al. 2017; Teranishi et al. 2025): data la
configurazione dell'attacco in un frame (palla, posizioni e velocita' degli attaccanti,
fase di possesso) quanta esposizione concede in media una difesa del campionato?
Le feature NON contengono posizioni dei difensori, quindi

    R(t) = Xhat(t) - X(t)

e' la parte di esposizione spiegata dalla difesa: positivo = la difesa concede meno
del difensore medio nella stessa situazione d'attacco.

Il modello e' allenato su un frame ogni `train_stride` (riduce l'autocorrelazione) e
predice tutti i frame del fold escluso: nessun match contribuisce alla propria attesa.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from .config import Config

PHASES = ["build_up", "create", "finish", "direct", "quick_break", "transition",
          "chaotic", "set_play"]


def context_matrix(frames: pd.DataFrame, use_phase: bool = False) -> pd.DataFrame:
    """Feature di contesto: solo palla e attaccanti. La fase SkillCorner e' esclusa di
    default perche' etichettata a posteriori (una fase diventa "finish" anche perche' la
    difesa ha lasciato arrivare l'attacco in zona di tiro)."""
    cols = [c for c in frames.columns if c.startswith("c_") and c != "c_ip_phase"]
    X = frames[cols].astype(float).copy()
    if use_phase:
        ph = frames["c_ip_phase"].astype(str)
        for p in PHASES:
            X[f"ph_{p}"] = (ph == p).astype(float)
    return X


def fit_expected(frames: pd.DataFrame, cfg: Config | None = None, target: str = "X",
                 verbose: bool = True) -> tuple[np.ndarray, pd.DataFrame]:
    cfg = cfg or Config()
    Xc = context_matrix(frames, cfg.context_phase)
    y = frames[target].to_numpy(float)
    groups = frames["match_id"].to_numpy()
    oof = np.full(len(frames), np.nan)
    rep = []
    n_groups = len(np.unique(groups))
    gkf = GroupKFold(n_splits=min(cfg.n_folds, n_groups))
    for k, (tr, te) in enumerate(gkf.split(Xc, y, groups)):
        tr_s = tr[:: cfg.train_stride]
        mdl = HistGradientBoostingRegressor(
            loss="poisson", max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
            min_samples_leaf=80, l2_regularization=1.0, random_state=cfg.random_state,
        )
        mdl.fit(Xc.iloc[tr_s], y[tr_s])
        oof[te] = mdl.predict(Xc.iloc[te])
        r2 = 1 - np.mean((y[te] - oof[te]) ** 2) / np.var(y[te])
        sp = pd.Series(y[te]).corr(pd.Series(oof[te]), method="spearman")
        rep.append(dict(fold=k, n_train=len(tr_s), n_test=len(te),
                        test_matches=len(np.unique(groups[te])), r2=r2, spearman=sp))
        if verbose:
            print(f"  fold {k}: R2={r2:.3f} spearman={sp:.3f}", flush=True)
    return oof, pd.DataFrame(rep)


CHANNELS = {"opt": "X_opt", "car": "X_car"}


def add_expected(frames: pd.DataFrame, cfg: Config | None = None, verbose: bool = True) -> pd.DataFrame:
    """Aggiunge Xhat/R totali e per canale. Ritorna il report dei fold del modello totale.

    Canali della risposta (ortogonali per costruzione dei dati, corr ~0.01):
      R_opt = Xhat_opt - X_opt   copertura delle opzioni di passaggio
      R_car = Xhat_car - X_car   pressione sul portatore
    Ogni canale ha la sua attesa, stimata out-of-fold dallo stesso contesto d'attacco.
    """
    cfg = cfg or Config()
    frames["Xhat"], rep = fit_expected(frames, cfg, verbose=verbose)
    frames["R"] = frames["Xhat"] - frames["X"]
    for k, col in CHANNELS.items():
        frames[f"Xhat_{k}"], _ = fit_expected(frames, cfg, target=col, verbose=False)
        frames[f"R_{k}"] = frames[f"Xhat_{k}"] - frames[col]
    return rep
