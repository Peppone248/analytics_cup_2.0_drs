"""Stress per singola corsa.

    stress = danger_prior(subtype) x containment_failure

Moltiplicativo: una corsa poco pericolosa gestita male resta poco pericolosa,
una corsa pericolosa gestita bene e' attenuata.

Il containment_failure e' l'unico punto in cui la DIFESA entra nel modello.
Oggi usa solo la dinamica della separation (uno scalare). E' la limitazione
principale del prototipo: vedi `ContainmentModel` per i punti di estensione.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

from .config import DRSConfig, DEFAULT


class ContainmentModel(Protocol):
    """Interfaccia di un modello di containment failure.

    Qualunque implementazione prende il Layer 3 (eventualmente arricchito) e
    restituisce una Series in [0, 1] allineata all'indice: 0 = la difesa ha
    gestito la corsa meglio del tipico, 1 = molto peggio.

    Implementazioni previste:
      - SeparationGainContainment  (questo file, attiva)
      - SeparationCurveContainment (stub: usa la curva separation(t) intera)
      - AwarenessContainment       (stub: integra A(i,j,t) di Filippo)
    """

    def __call__(self, df: pd.DataFrame, cfg: DRSConfig) -> pd.Series: ...


def separation_gain_containment(
    df: pd.DataFrame, cfg: DRSConfig = DEFAULT
) -> pd.Series:
    """z-score di separation_gain normalizzato PER SUBTYPE, mappato in [0,1].

    La normalizzazione per subtype e' obbligatoria: il gain atteso e'
    strutturalmente diverso fra subtype.
      cross_receiver  gain medio -1.25  (va in area affollata)
      dropping_off    gain medio +0.52  (si allontana dalla porta)
    Senza normalizzare, il modello premierebbe la difesa ogni volta che
    l'attaccante corre verso un'area congestionata.

    z > 0  -> il runner si e' liberato piu' del tipico per quel subtype
    z < 0  -> il difensore ha accorciato piu' del tipico

    La sigmoide mappa z in (0,1) con 0.5 = comportamento tipico.
    """
    stats = df.groupby(cfg.col_subtype)[cfg.col_sep_gain].agg(["mean", "std"])

    mu = df[cfg.col_subtype].map(stats["mean"])
    sd = df[cfg.col_subtype].map(stats["std"]).replace(0, np.nan)

    z = (df[cfg.col_sep_gain] - mu) / sd
    out = 1.0 / (1.0 + np.exp(-z))
    # subtype a varianza nulla o gain mancante -> comportamento tipico
    return out.fillna(0.5)


# Feature della geometria PRE-corsa per il containment appreso.
# Escluse di proposito:
#   separation_end, sep_*                  misurate durante o dopo la corsa
#   str_stretch_rate, str_recovery_rate    co-realizzate con la corsa
#   str_d_*                                idem
#   distance, frame_end                    durata, nota solo a corsa finita
#   qualunque esito
# Vuote al 100% sui 20 match, quindi inutilizzabili anche se pre-corsa:
#   interplayer_distance_start, goal_side_start, angle_of_engagement,
#   n_teammates_ahead_start
LEARNED_NUM = (
    "geo_dist_start", "x_start", "y_start", "separation_start",
    "n_opponents_ahead_start", "n_simultaneous_runs",
    "delta_to_last_defensive_line_start", "last_defensive_line_x_start",
    "ball_dist_goal_start", "ball_dist_runner_start",
    "str_spread_x", "str_spread_y", "str_area", "str_line_x", "str_line_spread",
    "pp_n_runs", "pp_n_passing_options", "pp_n_opponents_ahead",
)
LEARNED_CAT = ("event_subtype", "out_of_possession_phase", "third_start",
               "channel_start", "pp_pressure")


def has_learned_columns(df: pd.DataFrame) -> bool:
    """True se il containment appreso e' calcolabile (richiede il merge con
    tracking_features: 8 delle 18 colonne numeriche vengono da la')."""
    return all(c in df.columns for c in LEARNED_NUM + LEARNED_CAT)


def learned_containment(df: pd.DataFrame, cfg: DRSConfig = DEFAULT) -> pd.Series:
    """Containment come residuo rispetto a un'attesa APPRESA dalla geometria.

        separation_gain_containment:  sigmoid((gain - media_subtype) / sd_subtype)
        learned_containment:          sigmoid((gain - previsione) / sd_residuo)

    L'attesa dello z-score e' la media del subtype, un riferimento grossolano:
    sa solo che per un `behind` il gain tipico e' quello, non che quel
    `behind` partiva a 40 m con la linea gia' allungata. Qui l'attesa e' la
    previsione di un gradient boosting che vede tutta la geometria pre-corsa.

    NESSUN ESITO: il target e' `separation_gain`, non `lead_to_shot`. Lo
    verifica test_learned_containment_ignores_outcomes.

    OUT-OF-FOLD. Le previsioni sono prodotte con GroupKFold su match_id, con
    `cfg.learned_n_splits` fold. In-sample i residui sarebbero sgonfiati dal
    fit e il containment sembrerebbe piu' informativo di quanto sia.

    Nota non ovvia: GroupKFold e' deterministico e dissociation_oos usa gli
    stessi gruppi e lo stesso numero di fold, quindi le partizioni coincidono
    e la previsione per una corsa del fold k viene sempre da un modello
    addestrato sui fold diversi da k. Un containment precalcolato cosi' e
    passato a dissociation_oos resta quindi out-of-sample per davvero, non
    per approssimazione. Se cambi learned_n_splits, cambia anche n_splits
    della validazione.

    Verificato su 8877 corse, 20 match:
        R2 out-of-fold della previsione di separation_gain   +0.401
        corr con il containment per subtype                  +0.729
        split-half di containment_failure_mean    +0.026 -> +0.605
        standardizzato per fase                   -0.074 -> +0.519
        R2 spiegata dalla sola fase difensiva              0.003
        corr con structural_vulnerability                 -0.000
        dissociazione                             regge (targeted p=0.74)

    Il 60% di separation_gain resta non spiegato, e dentro c'e' ancora sia la
    scelta tattica sia l'errore: questo non risolve il problema dello
    sguardo, rende solo misurabile il fattore su cui l'awareness puo' agire.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold

    missing = [c for c in LEARNED_NUM + LEARNED_CAT if c not in df.columns]
    if missing:
        raise ValueError(
            f"containment appreso: colonne mancanti {missing}. "
            "Richiede il merge con tracking_features.csv."
        )

    X = df.loc[:, list(LEARNED_NUM)].copy()
    for c in LEARNED_CAT:
        X[c] = df[c].astype("category").cat.codes
    y = df[cfg.col_sep_gain].astype(float)

    pred = pd.Series(np.nan, index=df.index, dtype=float)
    splitter = GroupKFold(n_splits=cfg.learned_n_splits)
    for train, test in splitter.split(X, y, groups=df[cfg.col_match]):
        model = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.06, max_depth=5,
            min_samples_leaf=40, l2_regularization=1.0, random_state=0,
        )
        model.fit(X.iloc[train], y.iloc[train])
        pred.iloc[test] = model.predict(X.iloc[test])

    resid = y - pred
    sd = resid.std()
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.5, index=df.index)
    return (1.0 / (1.0 + np.exp(-(resid / sd)))).fillna(0.5)


def fixed_containment(values: pd.Series):
    """Avvolge un containment GIA' calcolato in un ContainmentModel.

    Serve per non rifittare il modello appreso a ogni chiamata di
    compute_stress, e per passare lo stesso containment out-of-fold a
    dissociation_oos, che altrimenti lo ricalcolerebbe su un solo fold.
    """
    def _fixed(df: pd.DataFrame, cfg: DRSConfig = DEFAULT) -> pd.Series:
        return values.reindex(df.index).fillna(0.5)
    return _fixed


CURVE_COLS = ("sep_min", "sep_late", "sep_min_frac", "sep_time_close")


def has_curve_columns(df: pd.DataFrame) -> bool:
    """True se il Layer 3 e' stato arricchito con tracking_features."""
    return all(c in df.columns for c in CURVE_COLS)


def separation_curve_containment(
    df: pd.DataFrame, cfg: DRSConfig = DEFAULT
) -> pd.Series:
    """Containment dalla forma della curva separation(t) invece che dal solo
    `separation_gain = end - start`.

    Richiede le colonne prodotte da `tracking_features.extract_all` e
    unite con `merge_into_layer3`.

    L'idea: il gain confonde due situazioni diverse. Un difensore che perde
    3 m all'inizio e li recupera ha gain 0 come uno che resta sempre
    attaccato; uno che resta attaccato e poi si fa bruciare all'ultimo
    secondo ha gain positivo come uno staccato da subito. La fuga TARDIVA
    e' quella che conta, perche' e' quando arriva la palla.

        escape = sep_late - sep_min

    quanto il runner si e' liberato DOPO il momento di massima prossimita'.
    Combinato con il gain classico, entrambi z-scorati per subtype e mappati
    in [0,1] con sigmoide, poi media pesata da `w_late_escape`.

    ATTENZIONE — RISULTATO PRELIMINARE NEGATIVO. Su un singolo match
    (1996436, 387 corse) le feature di curva NON migliorano il fit rispetto
    al solo `separation_gain`: AIC 301.0 contro 297.6 del baseline, cioe'
    sei parametri in piu' per niente. Le correlazioni parziali al netto di
    fase e subtype sono tutte sotto 0.08. Per questo la funzione esiste ma
    NON e' il default: va rivalutata sui 20 match prima di adottarla.
    Le feature di struttura, al contrario, migliorano (AIC 291.4).
    """
    missing = [c for c in CURVE_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"colonne di curva mancanti: {missing}. Estrarle con "
            "tracking_features.extract_all e unirle con merge_into_layer3."
        )

    def _z01(values: pd.Series) -> pd.Series:
        st = values.groupby(df[cfg.col_subtype]).agg(["mean", "std"])
        mu = df[cfg.col_subtype].map(st["mean"])
        sd = df[cfg.col_subtype].map(st["std"]).replace(0, np.nan)
        return (1.0 / (1.0 + np.exp(-(values - mu) / sd))).fillna(0.5)

    gain_part = _z01(df[cfg.col_sep_gain])
    escape = (df["sep_late"] - df["sep_min"]).fillna(0.0)
    late_part = _z01(escape)

    w = cfg.w_late_escape
    out = (1.0 - w) * gain_part + w * late_part
    # corse senza curva calcolabile: ricadi sul containment classico
    no_curve = df["sep_min"].isna()
    if no_curve.any():
        out[no_curve] = separation_gain_containment(df[no_curve], cfg)
    return out


def awareness_containment(df, awareness_df, cfg: DRSConfig = DEFAULT):
    """STUB — containment che integra l'awareness (modello Kalman di Filippo).

    Formato atteso per `awareness_df`, concordato come interfaccia di scambio:
        frame | defender_id | attacker_id | awareness | in_view

    Risolve l'ambiguita' che la separation da sola non puo' sciogliere: due
    corse con identico separation_gain sono tatticamente diverse se in una il
    difensore stava guardando il runner (scelta: lo lascio andare) e
    nell'altra no (errore: non l'ho visto).

    Dove serve di piu' (diagnosi dal prototipo): il controllo per subtype
    regge su behind (1.50x), run_ahead (1.37x), support (1.43x),
    coming_short (1.80x) ma CADE su cross_receiver (0.99x) e dropping_off
    (0.57x) — esattamente i casi in cui la separation non discrimina.

    Disponibile su 2 match A-League con body pose; sui 18 match 2D serve una
    versione degradata (direzione di moto del difensore come proxy del cono
    visivo).
    """
    raise NotImplementedError(
        "awareness_containment: in attesa dell'export A(i,j,t) da Filippo."
    )


STRUCTURAL_COLS = (
    "delta_to_last_defensive_line_start",
    "n_simultaneous_runs",
    "penalty_area_start",
)

TRACKING_STRUCT_COLS = ("str_stretch_rate", "str_recovery_rate")


def has_structural_columns(df: pd.DataFrame) -> bool:
    """True se il DataFrame e' un layer3_enriched con le colonne strutturali."""
    return all(c in df.columns for c in STRUCTURAL_COLS)


def structural_vulnerability(
    df: pd.DataFrame, cfg: DRSConfig = DEFAULT, reference: pd.DataFrame | None = None
) -> pd.Series:
    """Vulnerabilita' strutturale della difesa in [0, 1]. Richiede enriched.

    Terzo fattore dello stress, accanto al prior di pericolosita' e al
    containment. E' il primo pezzo in cui la DIFESA entra nel modello oltre
    alla separation.

    Tre componenti, tutte osservabili PRIMA della corsa (nessun esito):

      v_delta  posizione del runner rispetto alla linea difensiva.
               delta basso = la corsa parte sulla linea o oltre.
               Rango percentile invertito.
      v_over   carico difensivo: quante corse simultanee.
      v_zone   la corsa parte in area di rigore.

    Combinate con media geometrica pesata: una componente vicina a zero
    abbassa il risultato, cosa che una media aritmetica non farebbe.

    LIMITE IMPORTANTE. Queste colonne sono campionate al `frame_start` della
    corsa, quindi la vulnerabilita' strutturale e' un attributo della corsa,
    NON un segnale di fondo continuo. Per un vero fondo geometrico frame per
    frame serve il tracking XY: vedi indexing.DenseTrackingIndex.

    Parameters
    ----------
    reference : se fornito, i ranghi percentili sono calcolati rispetto a
        questo DataFrame invece che a `df`. Serve per la validazione
        out-of-sample: i ranghi del fold di test vanno calcolati sulla
        distribuzione del train, non su se stessi.
    """
    missing = [c for c in STRUCTURAL_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"colonne strutturali mancanti: {missing}. "
            "Richiede layer3_enriched.csv, non layer3_all_matches.csv."
        )

    col = "delta_to_last_defensive_line_start"
    if reference is None:
        v_delta = 1.0 - df[col].rank(pct=True)
    else:
        ref = np.sort(reference[col].values)
        v_delta = 1.0 - pd.Series(
            np.searchsorted(ref, df[col].values, side="right") / len(ref),
            index=df.index,
        )

    v_over = (df["n_simultaneous_runs"].clip(upper=4) / 4).clip(0.05, 1.0)
    v_zone = 0.5 + 0.5 * df["penalty_area_start"].astype(float)

    comps = [(v_delta.clip(0.01, 1.0), cfg.w_delta_to_line),
             (v_over, cfg.w_overload),
             (v_zone, cfg.w_zone)]

    # --- componenti dal tracking XY, se presenti
    # Sono le uniche due feature di tracking la cui correlazione con l'esito
    # SALE controllando per fase e subtype (+0.071 e +0.063 contro +0.087 e
    # +0.029 grezze): informazione genuinamente nuova, non un proxy della
    # fase come si e' rivelata la compattezza assoluta (str_spread_x passa
    # da -0.224 grezzo a +0.010 parziale).
    if cfg.use_tracking_structure and all(c in df.columns for c in TRACKING_STRUCT_COLS):
        ref = reference if reference is not None else df
        for col, weight in (("str_stretch_rate", cfg.w_stretch),
                            ("str_recovery_rate", cfg.w_recovery)):
            if weight <= 0:
                continue
            base = ref[col].dropna()
            if base.empty:
                continue
            srt = np.sort(base.values)
            # rango percentile: valore alto = la difesa si allunga / non si
            # ricompone = piu' vulnerabile
            v = pd.Series(
                np.searchsorted(srt, df[col].values, side="right") / len(srt),
                index=df.index,
            ).fillna(0.5).clip(0.05, 1.0)
            comps.append((v, weight))

    W = sum(w for _, w in comps)
    out = np.ones(len(df), dtype=float)
    for v, w in comps:
        out = out * np.asarray(v, dtype=float) ** w
    return pd.Series(out ** (1.0 / W), index=df.index)


def compute_stress(
    df: pd.DataFrame,
    priors: dict | pd.Series,
    containment: ContainmentModel = separation_gain_containment,
    cfg: DRSConfig = DEFAULT,
    structural: bool | pd.Series = False,
) -> pd.DataFrame:
    """Aggiunge le colonne `danger_prior`, `containment_failure`,
    eventualmente `structural_vulnerability`, e `stress`.

    Parameters
    ----------
    priors : dict subtype -> danger, oppure una Series gia' allineata a `df`
        (usata per i prior spaziali, che dipendono da subtype x zona).
    structural : False per lo stress a due fattori; True per calcolare la
        vulnerabilita' strutturale da `df`; una Series per passarla gia'
        calcolata (serve out-of-sample, con i ranghi del train).

    Non modifica `df` in place: restituisce una copia.
    """
    out = df.copy()
    if isinstance(priors, pd.Series):
        out["danger_prior"] = priors.values
    else:
        out["danger_prior"] = out[cfg.col_subtype].map(priors)
    out["containment_failure"] = containment(out, cfg)
    out["stress"] = out["danger_prior"] * out["containment_failure"]

    if structural is not False:
        sv = (
            structural
            if isinstance(structural, pd.Series)
            else structural_vulnerability(out, cfg)
        )
        out["structural_vulnerability"] = np.asarray(sv)
        out["stress"] = out["stress"] * out["structural_vulnerability"]

    n_missing = out["danger_prior"].isna().sum()
    if n_missing:
        subs = sorted(out.loc[out["danger_prior"].isna(), cfg.col_subtype].unique())
        raise ValueError(
            f"{n_missing} run con subtype privo di prior: {subs}. "
            "Aggiornare FAMILY_MAP/priors o filtrare a monte."
        )
    return out
