"""Containment come residuo rispetto a un'attesa APPRESA.

Oggi:  containment = sigmoid((gain - media_subtype) / sd_subtype)
Qui:   containment = sigmoid((gain - previsione_modello) / sd_residuo)

Il modello prevede separation_gain dalla sola geometria PRE-corsa. Nessun
esito entra: il target e' separation_gain, non lead_to_shot. Le previsioni
sono out-of-fold (GroupKFold su match_id), altrimenti i residui sono
sgonfiati dal fit in-sample.

Perche' funziona: lo z-score per subtype lascia dentro il containment la
parte di separation_gain che la geometria spiega da sola (R2 = 0.40). Quella
parte dipende da DOVE e' avvenuta la corsa, quindi varia da partita a
partita: per la stima di squadra e' rumore. Condizionando sulla geometria la
si rimuove, e l'affidabilita' del containment sale.

Verificato su 8877 corse, 20 match:
    split-half containment_failure_mean   +0.026  ->  +0.605
    standardizzato per fase               -0.074  ->  +0.519
    confronto appaiato (stessi 400 split) +0.614 +/- 0.030, vince nel 98%
    dissociazione                         regge (targeted p=0.74, received p=0.41)
    OR su lead_to_shot                    1.217 -> 1.229 (1.114 -> 1.128 con fase)
    corr con structural_vulnerability     -0.000  (nessun doppio conteggio)

ATTENZIONE: dentro il prodotto dello stress la metrica di squadra
standardizzata per fase PEGGIORA (-0.110 +/- 0.008, vince nel 9%). Il
containment appreso funziona come ASSE SEPARATO del profilo, non come
fattore moltiplicativo. Vedi README.

Uso:
    python learned_containment.py out/ref/_layer3_merged.csv
"""

import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

PHASE = "out_of_possession_phase"

# Feature osservabili PRIMA della corsa. Escluse di proposito:
#   - separation_end, sep_*            misurate durante/dopo la corsa
#   - str_stretch_rate, str_recovery_rate, str_d_*   co-realizzate con la corsa
#   - distance, frame_end              durata della corsa, nota solo a corsa finita
#   - qualunque esito
# Vuote al 100% nel dataset attuale, quindi inutilizzabili:
#   interplayer_distance_start, goal_side_start, angle_of_engagement,
#   n_teammates_ahead_start
NUMERIC = [
    "geo_dist_start", "x_start", "y_start", "separation_start",
    "n_opponents_ahead_start", "n_simultaneous_runs",
    "delta_to_last_defensive_line_start", "last_defensive_line_x_start",
    "ball_dist_goal_start", "ball_dist_runner_start",
    "str_spread_x", "str_spread_y", "str_area", "str_line_x", "str_line_spread",
    "pp_n_runs", "pp_n_passing_options", "pp_n_opponents_ahead",
]
CATEGORICAL = ["event_subtype", PHASE, "third_start", "channel_start", "pp_pressure"]


def learned_containment(df: pd.DataFrame, n_splits: int = 5, seed: int = 0):
    """Restituisce (containment in [0,1], R2 out-of-fold della previsione).

    Da passare a run_pipeline come:
        cont, _ = learned_containment(df)
        run_pipeline(path, containment=lambda d, c: cont.reindex(d.index).fillna(0.5))
    """
    missing = [c for c in NUMERIC + CATEGORICAL if c not in df.columns]
    if missing:
        raise ValueError(f"colonne mancanti: {missing}")

    X = df[NUMERIC].copy()
    for c in CATEGORICAL:
        X[c] = df[c].astype("category").cat.codes
    y = df["separation_gain"].astype(float)

    pred = pd.Series(np.nan, index=df.index)
    for train, test in GroupKFold(n_splits=n_splits).split(X, y, groups=df["match_id"]):
        model = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.06, max_depth=5,
            min_samples_leaf=40, l2_regularization=1.0, random_state=seed,
        )
        model.fit(X.iloc[train], y.iloc[train])
        pred.iloc[test] = model.predict(X.iloc[test])

    resid = y - pred
    r2 = 1.0 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum()
    containment = (1.0 / (1.0 + np.exp(-(resid / resid.std())))).fillna(0.5)
    return containment, float(r2)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "out/ref/_layer3_merged.csv"
    df = pd.read_csv(path)
    cont, r2 = learned_containment(df)
    print(f"{len(df)} corse, {df.match_id.nunique()} match")
    print(f"R2 out-of-fold della previsione di separation_gain: {r2:+.3f}")
    print(f"containment appreso: media {cont.mean():.3f}  sd {cont.std():.3f}")
    out = path.rsplit(".", 1)[0] + "_learned_containment.csv"
    df[["match_id", "event_id"]].assign(containment_learned=cont).to_csv(out, index=False)
    print(f"salvato in {out}")
