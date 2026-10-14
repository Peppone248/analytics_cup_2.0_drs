"""
drst.py — DRS(t): Defensive Response Signature come curva temporale.
Versione single-file. Equivalente al package drst/, nessuna installazione.

    stress = danger_prior             (geometrico: exp(-dist/25m), non usa
                                       gli esiti; oppure da subtype/zona)
           x containment_failure      (separation gain)
           x structural_vulnerability (enriched, + tracking XY se c'e')

Lo stress NON usa mai gli esiti (targeted/received/lead_to_shot): restano
fuori per la validazione.

USO IN VS CODE
    1. metti questo file accanto ai dati
    2. imposta i path nel blocco CONFIGURA QUI
    3. F5 (o: python drst.py)

Da notebook o altro script nella stessa cartella:
    from drst import run_pipeline, extract_all, animate_match, compare

FLUSSO COMPLETO
    1. feature di tracking, una volta sola (~4 s per match):
         extract_all("data/layer3_enriched.csv",
                     "E:/Projects/opendata/data/matches",
                     "data/tracking_features.csv")
    2. imposta TRACKING_FEATURES qui sotto e rilancia
    3. per confrontare le configurazioni:
         compare("data/layer3_enriched.csv", "data/tracking_features.csv",
                 "out/upgrade")

DIPENDENZE
    obbligatorie  pandas numpy pyarrow
    validazione   statsmodels scikit-learn
    figure        matplotlib    (+ ffmpeg per gli MP4, altrimenti GIF)
    pip install pandas numpy pyarrow statsmodels scikit-learn matplotlib

INDICE (Ctrl+F sui marcatori ===)
     1. CONFIGURAZIONE
     2. PRIOR DI PERICOLOSITA' (geometrico, spaziale, subtype)
     3. STRESS per corsa — containment + vulnerabilita' strutturale
     4. FRAME-BASED INDEXING STRATEGY
     5. CURVA DRS(t)
     6. AGGREGAZIONE per squadra e difensore
     7. VALIDAZIONE
     8. FEATURE DAL TRACKING XY — curva sep(t) e struttura
     9. PIPELINE end-to-end
    10. FIGURE statiche
    11. ANIMAZIONE 2D su campo
    12. CONFRONTO fra configurazioni
    13. MAIN

Risultati di riferimento, limiti noti ed estensioni: vedi README.md
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, asdict, field, replace
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

# ====== CONFIGURA QUI ======================================================
LAYER3_PATH = Path("data/layer3_enriched.csv")
TRACKING_FEATURES = None      # Path("data/tracking_features.csv"), o None
OUT_DIR = Path("out/drst_v2")
PLOT_MATCH = None             # match_id della figura statica, o None
RUN_OOS = False               # dissociazione out-of-sample (piu' lenta)

PRIOR_KIND = "geometric"      # "geometric" | "spatial" | "subtype"
GEO_SCALE = 25.0              # metri, solo per il prior geometrico
GEO_USE_ENDPOINT = False      # prior su min(partenza, arrivo) invece della
                              # sola partenza. OFF: migliora la singola
                              # corsa (corr +0.403 -> +0.469) e peggiora la
                              # squadra (split-half +0.749 -> +0.628).
                              # Accenderlo per animazioni e case study,
                              # non per le metriche di squadra. Vedi README
GEO_BALL_GATE = False         # OFF: attenua il prior se la palla e' lontana,
                              # ma aggiunge +0.001 di correlazione = rumore
STRUCTURAL = None             # None = auto
USE_TRACKING_STRUCTURE = True # str_stretch_rate e str_recovery_rate
USE_CURVE_CONTAINMENT = False # OFF: sui 20 match peggiora (vedi README)
CONTAINMENT_KIND = "separation_gain"
                              # "separation_gain" | "learned". Il learned e'
                              # il residuo rispetto a un'attesa appresa dalla
                              # geometria: porta la split-half del containment
                              # da +0.026 a +0.605. Richiede TRACKING_FEATURES
                              # e scikit-learn. Resta OFF di default perche'
                              # dentro il prodotto dello stress peggiora la
                              # metrica standardizzata per fase: va letto
                              # come asse separato in team_two_axis.csv,
                              # che lo usa comunque. Vedi README

# --- animazione 2D (opzionale)
ANIMATE_MATCH = None          # match_id, o None per saltare
TRACKING_PATH = Path("data/tracking.jsonl")
N_WINDOWS = 5
SPATIAL_MODE = "gap"          # "runner" | "projected" | "gap"
FPS = 10
# ===========================================================================


# ===========================================================================
# 1. CONFIGURAZIONE
# ===========================================================================

# Famiglie tattiche dei subtype di off_ball_run.
# Fonte: off_ball_run_subtypes.md (definizioni ufficiali SkillCorner).
FAMILY_MAP: dict[str, str] = {
    "coming_short": "costruzione",
    "dropping_off": "costruzione",
    "run_ahead_of_the_ball": "progressione",
    "support": "progressione",
    "cross_receiver": "finalizzazione",
    "behind": "finalizzazione",
    "pulling_wide": "laterali",
    "pulling_half_space": "laterali",
    "overlap": "laterali",
    "underlap": "laterali",
}


@dataclass(frozen=True)
class DRSConfig:
    """Configurazione del modello.

    frozen=True: immutabile e hashable, cosi' puo' essere serializzata nei
    metadati di un run ed e' sempre chiaro con quali parametri e' stato
    prodotto un risultato.
    """

    # --- dati
    tracking_hz: int = 10
    """Frequenza del tracking SkillCorner A-League. DA VERIFICARE sui parquet:
    il recap assume 10 Hz, la tesi lavorava a 25 Hz su Hawk-Eye."""

    # --- prior di pericolosita'
    prior_shrinkage_k: float = 50.0
    """Pseudo-osservazioni di shrinkage Empirical Bayes verso la media globale.
    Piu' alto = piu' shrinkage. Protegge i subtype piccoli (overlap n=270,
    underlap n=226)."""

    prior_normalize: bool = True
    """Normalizza i prior a max=1 per leggibilita' della scala DRS."""

    prior_kind: str = "geometric"
    """Quale prior di pericolosita' usare: 'geometric' (default, dalla sola
    geometria, non tocca gli esiti), 'spatial' (subtype x zona, da shot
    rate) o 'subtype' (solo subtype, da shot rate).

    Il geometrico e' il default perche' out-of-sample batte gli altri su
    tutti i fronti e elimina la circolarita': OR 1.235 contro 1.165 del
    prior per subtype x third, con dissociazione piu' pulita."""

    geo_prior_scale: float = 25.0
    """Lunghezza di decadimento del prior geometrico, in metri. Vedi lo
    sweep documentato in priors.geometric_priors."""

    geo_use_endpoint: bool = False
    """Usa il punto della corsa piu' vicino alla porta — min(partenza,
    arrivo) — invece della sola partenza, quando `geo_dist_min` e'
    disponibile da tracking_features.

    DEFAULT OFF dopo la verifica a copertura piena (11 ottobre). Il punto
    di arrivo migliora la singola corsa e PEGGIORA la squadra:

        corr per corsa con lead_to_shot   +0.403 -> +0.469
        OR su lead_to_shot (8877 corse)    1.241 ->  1.255
        split-half di stress_per_run      +0.749 -> +0.628
        split-half di drs_mean_all        +0.722 -> +0.598

    Le ipotesi sbagliate, scartate con i dati:
      - endogeneita' rispetto al containment: corr(geo_progress,
        containment) = -0.004. Il punto di arrivo NON e' dove la difesa
        ha fermato la corsa.
      - doppio conteggio col terzo fattore: corr(prior, struttura) passa
        da +0.537 a +0.532. Nessuna sovrapposizione aggiunta.
      - riducibile a una scala piu' larga: la correlazione fra il prior
        col punto di arrivo e il prior sulla partenza satura a 0.843 per
        qualunque scala fra 25 e 45 m. E' informazione nuova, non una
        riparametrizzazione.

    Quello che resta e' la spiegazione giusta: DOVE una corsa finisce e'
    una proprieta' dell'episodio — se l'attaccante ha insistito, se la
    corsa di un compagno ha portato via un marcatore — mentre DOVE PARTE
    e' una proprieta' di come la squadra si dispone, e quella si ripete.
    Il punto di arrivo porta segnale sulla corsa e rumore sulla squadra.

    QUANDO ACCENDERLO. La claim principale del DRS e' la split-half per
    squadra, quindi per le metriche di squadra va tenuto spento. Per
    l'animazione, i case study e qualunque lettura per-corsa e' il prior
    migliore: accenderlo li' e' corretto.

    Ricade automaticamente sulla partenza se la colonna non c'e'."""

    geo_ball_gate: bool = False
    """Attenua il prior quando la palla e' lontana dalla porta: una corsa a
    dieci metri dalla porta mentre il pallone e' nella meta' campo opposta
    non e' una minaccia immediata.

    DEFAULT OFF: `ball_dist_goal_start` correla -0.52 con la distanza della
    corsa, quindi il gate duplica in gran parte il prior stesso invece di
    aggiungere informazione. Attivabile per sperimentare."""

    geo_ball_scale: float = 40.0
    """Scala del gate della palla, in metri. Solo se geo_ball_gate=True."""

    goal_x: float = 52.5
    """x della porta attaccata nel sistema di coordinate dell'enriched, che
    e' gia' orientato per direzione d'attacco."""

    goal_y: float = 0.0

    cell_shrinkage_k: float = 30.0
    """Shrinkage delle celle (subtype, zona) verso il prior del subtype.
    Usato solo da priors.spatial_priors."""

    # --- vulnerabilita' strutturale (richiede layer3_enriched)
    w_delta_to_line: float = 1.0
    """Peso di delta_to_last_defensive_line_start nella vulnerabilita'
    strutturale. Il segnale piu' forte fra le colonne enriched: shot rate
    19.9% nel quintile piu' basso (corsa che parte sulla linea) contro 3.2%
    nel piu' alto."""

    w_overload: float = 1.0
    """Peso di n_simultaneous_runs. Effetto overload: shot rate 7.4% con 0
    corse simultanee, 37.8% con 4 — mentre il targeted rate SCENDE da 42.3%
    a 10.8%. Piu' corse insieme = ogni singola meno mirata, situazione
    collettiva molto piu' pericolosa."""

    w_zone: float = 0.5
    """Peso di penalty_area_start. Shot rate 34.0% in area contro 11.4%
    fuori, ma solo 265 run: peso ridotto."""

    # --- componenti dal tracking XY (richiedono tracking_features)
    use_tracking_structure: bool = True
    """Include str_stretch_rate e str_recovery_rate nella vulnerabilita'
    strutturale quando le colonne sono presenti."""

    w_stretch: float = 0.75
    """Peso di str_stretch_rate: quanto la difesa si allunga DURANTE la
    corsa (m/s). Correlazione parziale +0.071 al netto di fase e subtype."""

    w_recovery: float = 0.75
    """Peso di str_recovery_rate: quanto la difesa si ricompone nei 2s DOPO
    la corsa. Correlazione parziale +0.063, superiore alla grezza (+0.029):
    e' informazione che la fase non contiene. Non ottenibile dal Layer 3
    enriched in nessun modo, serve il tracking."""

    containment_kind: str = "separation_gain"
    """Quale modello di containment usare: 'separation_gain' (z-score del gain
    per subtype, leggibile) o 'learned' (residuo rispetto a un'attesa appresa
    dalla geometria pre-corsa).

    Il learned e' molto piu' AFFIDABILE come tratto di squadra:

        split-half di containment_failure_mean   +0.026 -> +0.605
        standardizzato per fase                  -0.074 -> +0.519
        confronto appaiato, stessi 400 split     +0.614 +/- 0.030, vince nel 98%

    La ragione: lo z-score per subtype lascia dentro il containment la parte
    di separation_gain che la geometria spiega da sola (R2 = 0.40). Quella
    parte dipende da DOVE e' avvenuta la corsa, quindi per la stima di
    squadra e' rumore. Condizionando sulla geometria la si rimuove.

    DEFAULT 'separation_gain' per due motivi, non per inerzia:
      - dentro il prodotto dello stress il learned PEGGIORA la metrica di
        squadra standardizzata per fase (-0.110 +/- 0.008, vince nel 9%).
        Va usato come ASSE SEPARATO del profilo, ed e' quello che fa
        aggregate.team_two_axis, che lo calcola da se'.
      - lo z-score e' spiegabile in una riga a un allenatore, il residuo di
        un gradient boosting no. Per animazioni e case study resta il
        riferimento.

    Nessuno dei due usa gli esiti: il target del learned e' separation_gain."""

    learned_n_splits: int = 5
    """Fold di GroupKFold per le previsioni out-of-fold del containment
    appreso. Deve coincidere con `n_splits` di dissociation_oos, altrimenti
    le partizioni non sono le stesse e l'out-of-fold non e' piu' allineato
    al train fold della validazione."""

    relevance_threshold: float = 0.02
    """Soglia di rilevanza per i confronti appaiati fra configurazioni.

    Con 8 squadre e 40 righe squadra-match una differenza di split-half sotto
    questa soglia puo' avere l'intervallo che esclude lo zero ed essere
    praticamente irrilevante. paired_split_half la riporta come
    'trascurabile' invece di 'migliora'."""

    use_curve_containment: bool = False
    """Usa separation_curve_containment quando le colonne di curva ci sono.

    DEFAULT OFF dopo la verifica sui 20 match (11 ottobre). La curva
    sep(t) PEGGIORA il modello:

        containment_failure_mean   +0.026 -> -0.134  (32% di split positivi)
        stress_per_run              0.628 ->  0.607

    Un containment a -0.134 significa anticorrelazione fra le due meta':
    le squadre con containment alto in una meta' lo hanno basso nell'altra.
    Piu' risoluzione temporale sul duello non aiuta — il problema non e'
    che la separation sia misurata male, e' che la separation non
    distingue una scelta tattica da un errore. Vedi README.

    Resta attivabile per sperimentare."""

    w_late_escape: float = 0.5
    """Peso della fuga tardiva (sep_late - sep_min) rispetto al gain classico
    in separation_curve_containment. Solo se quel containment viene usato."""

    # --- kernel temporale
    tau_decay_frames: int = 25
    """Costante di decadimento esponenziale dopo la fine della corsa.
    25 frame = 2.5s a 10 Hz. Lo spazio aperto resta aperto per qualche secondo.
    VERIFICATO: il risultato e' insensibile a tau tra 5 e 60 frame
    (OR su lead_to_shot resta in [1.23, 1.27])."""

    tail_multiplier: float = 4.0
    """Lunghezza della coda di decadimento in multipli di tau. 4*tau copre
    il 98% dell'integrale esponenziale."""

    use_ramp: bool = True
    """Rampa lineare 0->stress durante la corsa (la minaccia si sviluppa).
    False = gradino. Entrambe funzionano, la rampa e' piu' plausibile."""

    aggregation: str = "sum"
    """Come combinare stress di corse simultanee: 'sum' | 'max' | 'noisy_or'.
    'sum' e' il default. 'noisy_or' = 1 - prod(1 - s_i), satura e modella
    la ridondanza (due corse che minacciano la stessa zona).
    DA TESTARE: il decoy e' il 68% dei lead_to_shot, l'overload potrebbe
    non essere additivo."""

    # --- validazione
    lookback_s: float = 3.0
    """Finestra pre-run su cui si misura il DRS(t) per la validazione.
    Guardare PRIMA di frame_start e' cio' che rende la validazione non
    circolare: la curva in quel punto non contiene la run valutata."""

    padding_frames: int = 300
    """Frame di coda aggiunti alla curva oltre l'ultimo frame_end."""

    # --- colonne attese nel Layer 3
    col_match: str = "match_id"
    col_event: str = "event_id"
    col_subtype: str = "event_subtype"
    col_frame_start: str = "frame_start"
    col_frame_end: str = "frame_end"
    col_sep_gain: str = "separation_gain"
    col_team: str = "team_id"
    col_defender: str = "defender_id"
    col_phase: str = "out_of_possession_phase"
    """Fase difensiva. Serve alla standardizzazione per fase, che e' il modo
    di togliere da una metrica di squadra l'altezza del blocco.

    Perche' e' necessaria: il danger_prior e' per il 52% prevedibile dalla
    sola fase e per il 78% dal terzo di partenza, mentre la squadra che
    difende aggiunge 1.3 punti dopo la fase. Al livello di squadra
    `prior_mean` da solo vale +0.721 di split-half contro i +0.724 dello
    stress completo, e correla 0.985 con la quota di corse concesse in fase
    profonda. Senza standardizzare, la claim principale del DRS e' in buona
    parte "quanto profondo difende questa squadra"."""

    deep_phases: tuple[str, ...] = (
        "low_block", "defending_set_play", "defending_quick_break",
    )
    """Fasi in cui le corse partono vicino alla porta difesa (distanza media
    32-34 m, contro 53 m del medium_block e 76 m dell'high_block). La loro
    quota e' la misura grezza di altezza del blocco con cui si confronta il
    prior: split-half +0.793, piu' stabile del DRS stesso."""
    outcomes: tuple[str, ...] = ("targeted", "received", "lead_to_shot")
    danger_outcome: str = "lead_to_shot"

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT = DRSConfig()

# ===========================================================================
# 2. PRIOR DI PERICOLOSITA' (geometrico, spaziale, subtype)
# ===========================================================================

def outcome_priors(
    df: pd.DataFrame, cfg: DRSConfig = DEFAULT
) -> tuple[dict[str, float], pd.DataFrame]:
    """Shot rate per subtype con shrinkage Empirical Bayes.

    p_shrunk(s) = (successi_s + k * p_globale) / (n_s + k)

    Returns
    -------
    priors : dict subtype -> danger in (0, 1]
    table  : DataFrame diagnostico con sum, size, shrunk, danger
    """
    y = cfg.danger_outcome
    glob = df[y].mean()

    g = df.groupby(cfg.col_subtype)[y].agg(["sum", "size"])
    g["shrunk"] = (g["sum"] + cfg.prior_shrinkage_k * glob) / (
        g["size"] + cfg.prior_shrinkage_k
    )
    g["danger"] = g["shrunk"] / g["shrunk"].max() if cfg.prior_normalize else g["shrunk"]

    return g["danger"].to_dict(), g


def spatial_priors(
    df: pd.DataFrame,
    cfg: DRSConfig = DEFAULT,
    zone_col: str = "third_start",
) -> tuple[dict, dict]:
    """Prior gerarchico danger(subtype, zona), con shrinkage a due livelli.

    Una run_ahead_of_the_ball a centrocampo e una al limite dell'area hanno
    lo stesso prior 0.438 con `outcome_priors`. Qui no.

    Shrinkage gerarchico: ogni cella (subtype, zona) viene tirata verso il
    prior del proprio subtype, non verso la media globale. Protegge le celle
    piccole senza appiattirle sulla media di tutto.

        p_subtype(s) = (succ_s + k_sub * p_glob) / (n_s + k_sub)
        p_cella(s,z) = (succ_sz + k_cell * p_subtype(s)) / (n_sz + k_cell)

    Spread empirico su layer3_enriched (8877 run, zone = third_start):
        prior per subtype      0.022 - 0.284
        prior per subtype x zona  molto piu' ampio, con celle a shot rate
        quasi nullo (costruzione nel terzo difensivo, n=521) che il prior
        per subtype non distingue

    IMPORTANTE PER LA VALIDAZIONE: usando questo prior, la logistica di
    dissociazione deve controllare ANCHE per la zona, non solo per il
    subtype. Altrimenti parte dell'effetto misurato e' il prior stesso.
    `validate.dissociation(..., extra_controls=["third_start"])`.

    Returns
    -------
    cells : dict (subtype, zona) -> danger
    fallback : dict subtype -> danger, per celle non viste
    """
    y = cfg.danger_outcome
    glob = df[y].mean()

    sub = df.groupby(cfg.col_subtype)[y].agg(["sum", "size"])
    sub["p"] = (sub["sum"] + cfg.prior_shrinkage_k * glob) / (
        sub["size"] + cfg.prior_shrinkage_k
    )

    cell = df.groupby([cfg.col_subtype, zone_col])[y].agg(["sum", "size"])
    cell["parent"] = cell.index.get_level_values(0).map(sub["p"])
    cell["p"] = (cell["sum"] + cfg.cell_shrinkage_k * cell["parent"]) / (
        cell["size"] + cfg.cell_shrinkage_k
    )

    mx = cell["p"].max() if cfg.prior_normalize else 1.0
    return (
        {k: v / mx for k, v in cell["p"].items()},
        {k: v / mx for k, v in sub["p"].items()},
    )


def map_spatial_prior(
    df: pd.DataFrame,
    cells: dict,
    fallback: dict,
    cfg: DRSConfig = DEFAULT,
    zone_col: str = "third_start",
) -> pd.Series:
    """Applica un prior spaziale a un DataFrame, con fallback al livello
    subtype per le celle non viste in training."""
    return pd.Series(
        [
            cells.get((s, z), fallback.get(s, 0.5))
            for s, z in zip(df[cfg.col_subtype], df[zone_col])
        ],
        index=df.index,
    )


# ---------------------------------------------------------------------------
# Prior geometrico — non deriva dagli esiti
# ---------------------------------------------------------------------------

def geometric_priors(
    df: pd.DataFrame, cfg: DRSConfig = DEFAULT
) -> pd.Series:
    """Pericolosita' dalla sola geometria: distanza del punto di partenza
    della corsa dalla porta attaccata.

        danger = exp(-dist / scale),  dist = |(goal_x, 0) - (x_start, y_start)|

    PERCHE' E' MEGLIO DEL PRIOR DA ESITI. `outcome_priors` e
    `spatial_priors` derivano dallo shot rate, lo stesso esito su cui poi
    si valida: la circolarita' va neutralizzata controllando per subtype e
    zona nella logistica, e questo consuma parte del segnale. Il prior
    geometrico non tocca gli esiti, quindi la validazione non ha bisogno di
    controlli di zona e l'odds ratio e' direttamente interpretabile.

    VERIFICATO su 8877 corse, out-of-sample (GroupKFold su match):

        prior da esiti (subtype x third)   OR 1.165   p 1.3e-08
        prior geometrico (scala 25 m)      OR 1.235   p 1.2e-15

    e la dissociazione regge meglio: p(received) passa da 0.35 a 0.71.
    Split-half di stress_per_run da +0.644 a +0.735 (scala 25 m).

    SCALA. Governa quanto rapidamente il pericolo decade allontanandosi
    dalla porta. Lo sweep su 8877 corse mostra un compromesso:

        scala    OR      split-half   split-half residualizzato
        12 m     1.179     0.581            0.551
        16 m     1.206     0.582            0.558
        20 m     1.223     0.583            0.562
        25 m     1.235     0.576            0.563
        30 m     1.243     0.562            0.554
        45 m     1.252     0.683            0.322

    Scale larghe alzano l'OR ma appiattiscono le differenze fra squadre.
    Il default 25 m e' il punto in cui la stabilita' residualizzata e'
    massima; sotto i 20 m il prior diventa quasi binario "in area o no".

    PUNTO DI ARRIVO (cfg.geo_use_endpoint, default False). Se la colonna
    `geo_dist_min` e' presente — la porta `tracking_features` — il prior
    usa il punto della corsa PIU' VICINO alla porta invece della sola
    partenza. Motivo: l'enriched porta solo le coordinate di partenza, ma
    una corsa in media avvicina alla porta di 7.3 m, quindi il prior sulla
    partenza sottostima sistematicamente chi attacca la profondita'.

    VERIFICATO sul match 1996436 (387 corse, sanity check
    corr(geo_x_start, x_start) = 1.000), correlazione di exp(-dist/25) con
    lead_to_shot e shot rate per quartile:

        punto usato            corr     Q1 -> Q4
        partenza               +0.403   3.1% -> 38.1%
        arrivo                 +0.458   0.0% -> 41.2%
        media delle due        +0.445   0.0% -> 39.2%
        min(partenza, arrivo)  +0.469   0.0% -> 41.2%   <- con endpoint ON

    Il minimo batte l'arrivo da solo perche' copre due casi opposti con la
    stessa regola: la corsa che parte larga e si chiude, e quella che parte
    in area e poi si allarga per ricevere. In entrambi conta il momento di
    massima minaccia, non il punto finale.

    Nessuna circolarita' aggiunta: sono coordinate, non esiti.

    MA IL DEFAULT E' OFF. A copertura piena sui 20 match il punto di arrivo
    migliora la singola corsa e PEGGIORA la squadra: OR 1.217 -> 1.254,
    split-half di stress_per_run +0.724 -> +0.628. Dove una corsa finisce
    e' una proprieta' dell'episodio, dove parte e' una proprieta' di come
    la squadra si dispone, e solo quest'ultima si ripete. Vedi la
    docstring di DRSConfig.geo_use_endpoint e il README.

    NB: il prior geometrico ignora il TIPO di corsa. Un dropping_off e un
    behind che partono dallo stesso punto hanno lo stesso prior. Perde
    l'informazione tattica dei subtype in cambio di indipendenza dagli
    esiti. La variante ibrida (geometrico x prior del subtype) e' stata
    provata e sta in mezzo: OR 1.216, split-half 0.671 — nessuno dei due
    vantaggi pieni, quindi non e' il default.
    """
    need = ("x_start", "y_start")
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(
            f"colonne geometriche mancanti: {missing}. "
            "Richiede layer3_enriched.csv."
        )
    # x_start e' gia' orientato per direzione d'attacco nell'enriched:
    # attacking_third ha media +27.8, defensive_third -26.7. La porta
    # attaccata sta quindi a x = +meta' campo.
    dist = np.hypot(cfg.goal_x - df["x_start"], cfg.goal_y - df["y_start"])

    if cfg.geo_use_endpoint and "geo_dist_min" in df.columns:
        cov = df["geo_dist_min"].notna().mean()
        if 0.0 < cov < 0.9:
            # COPERTURA PARZIALE = POPOLAZIONI MISTE. Le corse col punto di
            # arrivo ricevono prior sistematicamente piu' alti (la distanza
            # minima e' <= quella di partenza per costruzione), quindi i
            # match con tracking estratto sembrano piu' pericolosi degli
            # altri. Verificato: con copertura al 4.4% (1 match su 20) la
            # split-half di stress_per_run scende da +0.765 a +0.731,
            # mentre l'OR sale da 1.241 a 1.248. Non e' un miglioramento
            # del modello, e' un effetto di match.
            print(f"[!] geo_dist_min copre solo {cov:.1%} delle corse: il "
                  "prior mescola due popolazioni e le metriche di squadra "
                  "non sono confrontabili fra match. Estrarre le feature "
                  "di tracking su tutti i match, oppure "
                  "geo_use_endpoint=False.")
        # geo_dist_min e' gia' min(partenza, arrivo) e gia' orientata, ma
        # copre solo i match con tracking estratto: dove manca si ricade
        # sulla distanza dalla partenza, cosi' il prior resta definito su
        # tutte le corse e le due popolazioni sono sulla stessa scala.
        dist = np.fmin(dist, df["geo_dist_min"].fillna(np.inf))

    g = np.exp(-dist / cfg.geo_prior_scale)

    if cfg.geo_ball_gate and "ball_dist_goal_start" in df.columns:
        gate = np.exp(-df["ball_dist_goal_start"] / cfg.geo_ball_scale)
        g = g * gate.fillna(gate.mean())

    return (g / g.max()) if cfg.prior_normalize else g

# ===========================================================================
# 3. STRESS per corsa — containment + vulnerabilita' strutturale
# ===========================================================================

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

# ===========================================================================
# 4. FRAME-BASED INDEXING STRATEGY
# ===========================================================================

@dataclass
class FrameIndex:
    """L'asse temporale di un match, con l'informazione su dove il DRS(t)
    e' definito.

    Attributes
    ----------
    match_id : int
    n_frames : int
        Lunghezza dell'asse.
    support : np.ndarray di bool, shape (n_frames,)
        True dove il DRS(t) ha supporto (qualcosa lo determina).
        Dove e' False, un valore 0 significa "non misurato", non "difesa
        sotto controllo". Distinguere i due casi e' essenziale per ogni
        aggregazione onesta.
    baseline : np.ndarray di float, shape (n_frames,)
        Livello di fondo su cui gli eventi depositano stress. Zero nel
        prototipo event-anchored; sara' il segnale geometrico continuo
        (vulnerabilita' strutturale della difesa) nella versione tracking.
    """

    match_id: int
    n_frames: int
    support: np.ndarray
    baseline: np.ndarray

    def coverage(self) -> float:
        """Frazione di frame con supporto. Metrica diagnostica da riportare
        sempre accanto a qualunque aggregato."""
        return float(self.support.mean())

    def masked(self, curve: np.ndarray) -> np.ma.MaskedArray:
        """La curva con i frame non supportati mascherati, cosi' `.mean()` e
        compagnia ignorano i buchi invece di contarli come zeri."""
        return np.ma.masked_array(curve, mask=~self.support)


class IndexingStrategy(ABC):
    """Come si costruisce il FrameIndex di un match."""

    name: str = "abstract"

    @abstractmethod
    def build(self, match_runs: pd.DataFrame, cfg: DRSConfig) -> FrameIndex:
        """Costruisce l'indice per un singolo match a partire dalle sue run."""
        ...


class EventAnchoredIndex(IndexingStrategy):
    """Strategia del prototipo: il supporto e' dove ci sono eventi.

    Un frame ha supporto se cade dentro una corsa o nella sua coda di
    decadimento. Baseline identicamente zero: senza eventi non sappiamo nulla
    sullo stato della difesa.

    LIMITE da dichiarare nel paper: copertura 35-46% dei frame. Gli aggregati
    temporali vanno calcolati sui frame supportati (`FrameIndex.masked`),
    altrimenti una squadra che subisce poche corse sembra solida solo perche'
    ha piu' zeri.
    """

    name = "event_anchored"

    def build(self, match_runs: pd.DataFrame, cfg: DRSConfig = DEFAULT) -> FrameIndex:
        if match_runs.empty:
            raise ValueError("match_runs vuoto")

        mid = int(match_runs[cfg.col_match].iloc[0])
        n = int(match_runs[cfg.col_frame_end].max()) + cfg.padding_frames

        support = np.zeros(n, dtype=bool)
        tail = int(cfg.tail_multiplier * cfg.tau_decay_frames)
        for f0, f1 in zip(
            match_runs[cfg.col_frame_start].astype(int),
            match_runs[cfg.col_frame_end].astype(int),
        ):
            support[max(0, f0) : min(n, f1 + tail)] = True

        return FrameIndex(
            match_id=mid,
            n_frames=n,
            support=support,
            baseline=np.zeros(n, dtype=float),
        )


class DenseTrackingIndex(IndexingStrategy):
    """STUB — supporto su ogni frame di gioco effettivo, con fondo geometrico.

    E' il passo che trasforma il DRS(t) da treno di impulsi a segnale
    continuo, e l'intervento che cambia di piu' la natura della metrica:
    oggi la difesa entra nel modello solo tramite `separation_gain`, quindi
    due squadre che subiscono le stesse corse hanno DRS(t) identico anche se
    una difendeva compatta e alta e l'altra sfaldata.

    Cosa deve calcolare, frame per frame, dal tracking XY (gia' in locale,
    repo opendata, scaricato via git lfs):

      - altezza della linea difensiva, normalizzata per lato
        L_t = media delle x dei 4 difensori piu' arretrati
        (cfr. Teranishi et al., arXiv:2511.06191, back-four spatial control)
      - altezza relativa alla palla: R_t = x_palla - L_t
      - compattezza longitudinale e laterale dei 10 di movimento
      - numero di difensori fra la palla e la porta
      - velocita' di recupero della linea

    Queste diventano la `baseline`: la vulnerabilita' strutturale della
    difesa indipendentemente dagli eventi. Lo stress degli eventi si somma
    sopra, e la formula passa da

        stress = pericolo_attacco x (il runner si e' liberato?)
    a
        stress = pericolo(subtype, zona) x dinamica_duello x vulnerabilita'

    con due fattori su tre che descrivono la difesa.

    `support` sara' True su tutti i frame con 22 giocatori tracciati
    (il ~74% dei frame: la distribuzione e' bimodale, 22 giocatori o 0,
    i frame vuoti sono gli stoppages).
    """

    name = "dense_tracking"

    def build(self, match_runs: pd.DataFrame, cfg: DRSConfig = DEFAULT) -> FrameIndex:
        raise NotImplementedError(
            "DenseTrackingIndex: richiede il loader del tracking XY. "
            "Vedi README, sezione 'Estensioni previste'."
        )


class PhaseWindowIndex(IndexingStrategy):
    """STUB — indicizzazione per fase di gioco invece che per frame.

    Le fasi di SkillCorner (medium_block, low_block, high_block,
    defending_transition, chaotic, ...) durano in media 6.8s e sono gia'
    nel dataset (438 fasi/match, join temporale gia' implementato in
    enrich_layer3.py).

    Unita' di analisi = la fase, non il frame. Il DRS della fase e'
    l'integrale del DRS(t) sulla sua durata, normalizzato.

    Perche' puo' valere: la fase e' il predittore dominante dello stadio 3
    (shot rate per fase: spread 24x, da high_block 1.6% a
    defending_quick_break 38%), e un'unita' di analisi piu' grande del frame
    riduce il problema di campione che ha affondato la fingerprint per
    squadra.
    """

    name = "phase_window"

    def build(self, match_runs: pd.DataFrame, cfg: DRSConfig = DEFAULT) -> FrameIndex:
        raise NotImplementedError(
            "PhaseWindowIndex: richiede phases_of_play del match."
        )


STRATEGIES: dict[str, type[IndexingStrategy]] = {
    "event_anchored": EventAnchoredIndex,
    "dense_tracking": DenseTrackingIndex,
    "phase_window": PhaseWindowIndex,
}


def get_strategy(name: str) -> IndexingStrategy:
    if name not in STRATEGIES:
        raise KeyError(f"strategia '{name}' sconosciuta. Disponibili: {list(STRATEGIES)}")
    return STRATEGIES[name]()

# ===========================================================================
# 5. CURVA DRS(t)
# ===========================================================================

def run_kernel(
    n_frames: int, f0: int, f1: int, stress: float, cfg: DRSConfig = DEFAULT
) -> np.ndarray:
    """Contributo di una singola corsa all'asse dei frame."""
    k = np.zeros(n_frames, dtype=float)
    f0, f1 = int(f0), int(min(f1, n_frames - 1))
    if f1 <= f0 or f0 >= n_frames:
        return k

    dur = f1 - f0
    if cfg.use_ramp and dur > 0:
        k[f0:f1] = stress * np.linspace(0.0, 1.0, dur)
    else:
        k[f0:f1] = stress

    tail_end = min(f1 + int(cfg.tail_multiplier * cfg.tau_decay_frames), n_frames)
    if tail_end > f1:
        t = np.arange(tail_end - f1)
        k[f1:tail_end] = stress * np.exp(-t / cfg.tau_decay_frames)
    return k


def _combine(acc: np.ndarray, new: np.ndarray, how: str) -> np.ndarray:
    if how == "sum":
        return acc + new
    if how == "max":
        return np.maximum(acc, new)
    if how == "noisy_or":
        # satura: due corse che minacciano la stessa zona non raddoppiano
        return 1.0 - (1.0 - np.clip(acc, 0, 1)) * (1.0 - np.clip(new, 0, 1))
    raise ValueError(f"aggregation '{how}' sconosciuta")


def build_curve(
    match_runs: pd.DataFrame,
    cfg: DRSConfig = DEFAULT,
    strategy: IndexingStrategy | None = None,
) -> tuple[np.ndarray, FrameIndex]:
    """Costruisce DRS(t) per un match.

    Richiede la colonna `stress` (vedi stress.compute_stress).

    Returns
    -------
    curve : np.ndarray, shape (n_frames,)
    index : FrameIndex con supporto e copertura
    """
    if "stress" not in match_runs.columns:
        raise ValueError("colonna 'stress' mancante: chiamare compute_stress() prima")

    strategy = strategy or EventAnchoredIndex()
    idx = strategy.build(match_runs, cfg)

    curve = idx.baseline.copy()
    for _, r in match_runs.iterrows():
        k = run_kernel(
            idx.n_frames,
            r[cfg.col_frame_start],
            r[cfg.col_frame_end],
            float(r["stress"]),
            cfg,
        )
        curve = _combine(curve, k, cfg.aggregation)
    return curve, idx


def build_all_curves(
    df: pd.DataFrame,
    cfg: DRSConfig = DEFAULT,
    strategy: IndexingStrategy | None = None,
    by_defending_team: bool = False,
) -> dict:
    """Curve per tutti i match.

    by_defending_team=False -> chiave: match_id, curva del match intero
    by_defending_team=True  -> chiave: (match_id, defending_team_id), curva
                               dello stress subito da quella difesa.
                               Richiede la colonna `defending_team_id`.
    """
    out = {}
    if by_defending_team:
        if "defending_team_id" not in df.columns:
            raise ValueError(
                "colonna 'defending_team_id' mancante: chiamare "
                "aggregate.add_defending_team() prima"
            )
        # l'asse del match e' comune alle due squadre, cosi' le curve sono
        # confrontabili frame a frame
        lens = df.groupby(cfg.col_match)[cfg.col_frame_end].max() + cfg.padding_frames
        for (mid, team), g in df.groupby([cfg.col_match, "defending_team_id"]):
            g = g.copy()
            g[cfg.col_frame_end] = g[cfg.col_frame_end].clip(upper=lens[mid] - 1)
            curve, idx = build_curve(g, cfg, strategy)
            # pad all'asse comune del match
            if len(curve) < lens[mid]:
                pad = int(lens[mid]) - len(curve)
                curve = np.concatenate([curve, np.zeros(pad)])
                idx.support = np.concatenate([idx.support, np.zeros(pad, bool)])
                idx.baseline = np.concatenate([idx.baseline, np.zeros(pad)])
                idx.n_frames = int(lens[mid])
            out[(int(mid), int(team))] = (curve, idx)
    else:
        for mid, g in df.groupby(cfg.col_match):
            out[int(mid)] = build_curve(g, cfg, strategy)
    return out

# ===========================================================================
# 6. AGGREGAZIONE per squadra e difensore
# ===========================================================================

def add_defending_team(df: pd.DataFrame, cfg: DRSConfig = DEFAULT) -> pd.DataFrame:
    """Aggiunge `defending_team_id` come complementare di team_id nel match.

    `team_id` nel Layer 3 e' la squadra che FA la corsa (in possesso). La
    squadra che difende e' l'altra delle due presenti nel match.

    Se la colonna esiste gia' (prodotta da build_full_dataset.py dai
    metadati di match.json) viene lasciata com'e'.
    """
    if "defending_team_id" in df.columns:
        return df

    out = df.copy()
    mapping = {}
    for mid, teams in out.groupby(cfg.col_match)[cfg.col_team].unique().items():
        if len(teams) != 2:
            raise ValueError(
                f"match {mid}: trovate {len(teams)} squadre, attese 2. "
                "Derivare defending_team_id da match.json."
            )
        mapping[mid] = {teams[0]: teams[1], teams[1]: teams[0]}

    out["defending_team_id"] = [
        mapping[m][t] for m, t in zip(out[cfg.col_match], out[cfg.col_team])
    ]
    return out


def phase_weights(df: pd.DataFrame, cfg: DRSConfig = DEFAULT) -> pd.Series:
    """Mix di fase medio di lega, da usare come pesi fissi per tutte le squadre."""
    return df[cfg.col_phase].value_counts(normalize=True)


def phase_standardized(
    g: pd.DataFrame, col: str, weights: pd.Series, cfg: DRSConfig = DEFAULT
) -> float:
    """Media di `col` entro ciascuna fase, ripesata col mix di lega.

    E' una standardizzazione DIRETTA, la stessa tecnica dei tassi di
    mortalita' standardizzati per eta': confronta due squadre come se
    difendessero con la stessa distribuzione di fasi.

    Togliere la fase e' necessario perche' senza di essa la metrica e' in
    buona parte altezza del blocco: `prior_mean` da solo correla 0.985 con
    la quota di corse concesse in fase profonda, e quella quota ha split-half
    +0.793, piu' del DRS completo. Standardizzando, la correlazione di
    stress_per_run col mix di fase scende da 0.880 a 0.553.

    Costo: lo split-half scende (stress_per_run +0.724 -> +0.534). Non e' un
    peggioramento del modello, e' la rimozione di una componente che era
    informazione gia' nota.
    """
    per_phase = g.groupby(cfg.col_phase, observed=True)[col].mean()
    w = weights.reindex(per_phase.index)
    tot = w.sum()
    if not np.isfinite(tot) or tot == 0:
        return float("nan")
    return float((per_phase * w).sum() / tot)


def team_match_metrics(
    df: pd.DataFrame, curves: dict, cfg: DRSConfig = DEFAULT
) -> pd.DataFrame:
    """Una riga per (match, squadra difendente).

    `curves` nel formato di build_all_curves(by_defending_team=True):
    chiave (match_id, team_id) -> (curve, FrameIndex).

    Gli aggregati temporali usano solo i frame con supporto (FrameIndex.masked):
    contare gli zeri dei frame non misurati premierebbe le difese che
    subiscono poche corse.
    """
    has_phase = cfg.col_phase in df.columns
    weights = phase_weights(df, cfg) if has_phase else None

    rows = []
    for (mid, team), (curve, idx) in curves.items():
        g = df[(df[cfg.col_match] == mid) & (df["defending_team_id"] == team)]
        m = idx.masked(curve)
        row = {
            cfg.col_match: mid,
            "defending_team_id": team,
            # --- VOLUME (non usare per profili di squadra)
            "n_runs": len(g),
            "drs_mean_all": float(curve.mean()),
            "drs_auc": float(curve.sum() / cfg.tracking_hz),
            # --- QUALITA' (normalizzate per volume)
            "stress_per_run": float(g["stress"].mean()),
            "drs_mean_supported": float(m.mean()) if m.count() else np.nan,
            "drs_p95_supported": (
                float(np.percentile(m.compressed(), 95)) if m.count() else np.nan
            ),
            "containment_failure_mean": float(g["containment_failure"].mean()),
            "prior_mean": float(g["danger_prior"].mean()),
            # --- diagnostica
            "coverage": idx.coverage(),
            "n_frames": idx.n_frames,
        }
        if has_phase and len(g):
            # --- QUALITA' al netto dell'altezza del blocco
            row["stress_per_run_phase_adj"] = phase_standardized(g, "stress", weights, cfg)
            row["containment_phase_adj"] = phase_standardized(
                g, "containment_failure", weights, cfg)
            row["deep_phase_share"] = float(g[cfg.col_phase].isin(cfg.deep_phases).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def team_two_axis(
    df: pd.DataFrame,
    team_match: pd.DataFrame,
    cfg: DRSConfig = DEFAULT,
    containment: pd.Series | None = None,
) -> pd.DataFrame:
    """Profilo di squadra a DUE ASSI, invece di un numero solo.

        asse 1  ESPOSIZIONE   quanto stress subisce la difesa, dato come gioca
        asse 2  CEDIMENTO     quanto cede nei duelli, a parita' di fase

    ATTENZIONE AL SEGNO: su entrambi gli assi ALTO = PEGGIO. `cedimento` e' la
    media di `containment_failure`, cioe' di quanto il runner si libera piu'
    del previsto: una difesa che contiene bene ha cedimento BASSO. Si chiama
    cosi' e non "contenimento" proprio per non invertire la lettura.

    Perche' due assi e non il prodotto a tre fattori. Il prodotto e' dominato
    dal prior, che e' per il 52% fase difensiva: `prior_mean` da solo ha
    split-half +0.721 contro i +0.724 dello stress completo. Il containment
    appreso, da solo, ha +0.605 (+0.519 standardizzato per fase) ed e' il
    secondo tratto piu' affidabile dopo il volume — ma messo DENTRO il
    prodotto peggiora la metrica standardizzata per fase di 0.110 +/- 0.008,
    perche' viene diluito. Come asse separato no.

    I due assi rispondono a domande diverse e vanno letti insieme:
    l'esposizione e' in larga parte una scelta dell'allenatore (dove e quanto
    profondo difendere), il contenimento e' esecuzione.

    `containment`: se passi il containment appreso, l'asse 2 usa quello al
    posto di `containment_failure` della pipeline. E' la configurazione
    consigliata, e lo stress non va ricalcolato.
    """
    d = df
    if containment is not None:
        d = df.copy()
        d["containment_failure"] = containment.reindex(df.index).fillna(0.5).values

    weights = phase_weights(d, cfg)
    rows = []
    for (mid, team), g in d.groupby([cfg.col_match, "defending_team_id"]):
        rows.append({
            cfg.col_match: mid,
            "defending_team_id": team,
            "n_runs": len(g),
            "esposizione": float(g["stress"].mean()),
            "cedimento": phase_standardized(g, "containment_failure", weights, cfg),
            "deep_phase_share": float(g[cfg.col_phase].isin(cfg.deep_phases).mean()),
        })
    per_match = pd.DataFrame(rows)

    prof = per_match.groupby("defending_team_id").agg(
        n_matches=(cfg.col_match, "nunique"),
        n_runs=("n_runs", "sum"),
        esposizione=("esposizione", "mean"),
        cedimento=("cedimento", "mean"),
        deep_phase_share=("deep_phase_share", "mean"),
    ).reset_index()

    # percentili dentro il campione: un profilo si legge in relativo, e con 8-13
    # squadre il valore assoluto non dice niente a un allenatore
    for c in ("esposizione", "cedimento"):
        prof[f"{c}_pct"] = (prof[c].rank(pct=True) * 100).round(0)
    return prof.sort_values("esposizione", ascending=False).reset_index(drop=True)


def team_profile(
    team_match: pd.DataFrame,
    metric: str = "stress_per_run",
    n_boot: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Profilo per squadra con CI bootstrap sui match.

    Il bootstrap e' sui MATCH, non sulle run: l'unita' di campionamento
    indipendente e' la partita. Con mediana 3 match per squadra i CI sono
    larghi, ed e' corretto che lo siano.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for team, g in team_match.groupby("defending_team_id"):
        v = g[metric].dropna().values
        if len(v) == 0:
            continue
        boot = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
        rows.append(
            {
                "defending_team_id": team,
                "n_matches": len(v),
                metric: v.mean(),
                "ci95_lo": float(np.percentile(boot, 2.5)),
                "ci95_hi": float(np.percentile(boot, 97.5)),
            }
        )
    return pd.DataFrame(rows).sort_values(metric, ascending=False).reset_index(drop=True)


def defender_profile(
    df: pd.DataFrame, min_runs: int = 30, cfg: DRSConfig = DEFAULT
) -> pd.DataFrame:
    """Profilo per difensore: stress medio delle corse in cui era il
    difensore piu' vicino.

    ATTRIBUZIONE: usa `defender_id` del Layer 3, cioe' nearest defender.
    E' l'approssimazione piu' semplice fra le tre valutate nel progetto
    (nearest / marking bidirezionale / leave-one-out). Il nearest defender
    e' validato (Pearson 0.9937 contro il benchmark) ma attribuisce a un
    solo giocatore una responsabilita' che e' spesso condivisa.
    """
    g = df.groupby(cfg.col_defender).agg(
        n_runs=("stress", "size"),
        stress_mean=("stress", "mean"),
        containment_failure_mean=("containment_failure", "mean"),
        n_matches=(cfg.col_match, "nunique"),
    )
    g = g[g.n_runs >= min_runs]
    return g.sort_values("stress_mean", ascending=False).reset_index()

# ===========================================================================
# 7. VALIDAZIONE
# ===========================================================================

def pre_run_drs(
    df: pd.DataFrame,
    curves: dict,
    cfg: DRSConfig = DEFAULT,
    carry: list[str] | None = None,
) -> pd.DataFrame:
    """Per ogni corsa, il DRS(t) nei `lookback_s` secondi PRIMA del suo inizio.

    Guardare prima di frame_start e' cio' che rende la validazione non
    circolare: la curva in quella finestra non contiene il contributo della
    corsa che stiamo valutando, solo quello delle corse precedenti.
    """
    lookback = int(cfg.lookback_s * cfg.tracking_hz)
    rows = []

    for mid, m in df.groupby(cfg.col_match):
        curve = curves[int(mid)][0] if isinstance(curves[int(mid)], tuple) else curves[int(mid)]
        for _, r in m.iterrows():
            f0 = int(r[cfg.col_frame_start])
            lo = max(0, f0 - lookback)
            if f0 <= lo:
                continue
            pre = curve[lo:f0]
            rec = {
                cfg.col_match: mid,
                cfg.col_event: r[cfg.col_event],
                "subtype": r[cfg.col_subtype],
                "drs_pre_mean": pre.mean(),
                "drs_pre_max": pre.max(),
            }
            for y in cfg.outcomes:
                rec[y] = r[y]
            for c in carry or []:
                rec[c] = r[c]
            rows.append(rec)
    return pd.DataFrame(rows)


def dissociation(
    pre: pd.DataFrame,
    cfg: DRSConfig = DEFAULT,
    extra_controls: list[str] | None = None,
) -> pd.DataFrame:
    """Logistica di `outcome ~ DRS_pre + subtype [+ extra]` per ciascun esito.

    Il controllo per subtype e' essenziale: senza, l'effetto potrebbe essere
    generato dal prior (che deriva dallo shot rate per subtype). Con il
    controllo, l'odds ratio misurato e' DENTRO ogni subtype.

    `extra_controls` elenca altre colonne categoriche di `pre` da includere.
    OBBLIGATORIO passare la colonna di zona quando si usa un prior spaziale
    (priors.spatial_priors): senza, parte dell'effetto misurato sarebbe il
    prior stesso. L'OR cala — e' corretto che cali, il test e' piu' severo.
    """
    import statsmodels.api as sm

    p = pre.copy()
    p["drs_z"] = (p.drs_pre_mean - p.drs_pre_mean.mean()) / p.drs_pre_mean.std()
    parts = [pd.get_dummies(p["subtype"], prefix="st", drop_first=True).astype(float)]
    for c in extra_controls or []:
        if c not in p.columns:
            raise ValueError(f"controllo '{c}' assente da `pre`")
        parts.append(pd.get_dummies(p[c], prefix=c[:6], drop_first=True).astype(float))
    X = sm.add_constant(pd.concat([p[["drs_z"]]] + parts, axis=1))

    rows = []
    for y in cfg.outcomes:
        res = sm.Logit(p[y].astype(int), X).fit(disp=0)
        rows.append(
            {
                "outcome": y,
                "n": len(p),
                "base_rate": p[y].mean(),
                "beta": res.params["drs_z"],
                "se": res.bse["drs_z"],
                "odds_ratio": np.exp(res.params["drs_z"]),
                "ci_lo": np.exp(res.params["drs_z"] - 1.96 * res.bse["drs_z"]),
                "ci_hi": np.exp(res.params["drs_z"] + 1.96 * res.bse["drs_z"]),
                "p_value": res.pvalues["drs_z"],
            }
        )
    return pd.DataFrame(rows)


def dissociation_oos(
    df: pd.DataFrame,
    cfg: DRSConfig = DEFAULT,
    n_splits: int = 5,
    strategy=None,
    spatial_prior: bool | None = None,
    structural: bool = False,
    zone_col: str = "third_start",
    prior_kind: str | None = None,
    containment: "ContainmentModel | None" = None,
) -> pd.DataFrame:
    """Dissociazione out-of-sample: prior, normalizzazione della separation e
    ranghi strutturali stimati SOLO sul train fold, DRS(t) valutato sul fold
    tenuto fuori.

    GroupKFold su match_id. E' la versione da citare nel paper: la variante
    in-sample stima tutto sulle stesse run su cui valida.
    """
    from sklearn.model_selection import GroupKFold


    kind = prior_kind or cfg.prior_kind
    if spatial_prior is True:
        kind = "spatial"
    elif spatial_prior is False and prior_kind is None and cfg.prior_kind == "spatial":
        kind = "subtype"
    if kind == "geometric" and not {"x_start", "y_start"} <= set(df.columns):
        kind = "spatial" if zone_col in df.columns else "subtype"

    gkf = GroupKFold(n_splits=n_splits)
    parts = []
    for tr, va in gkf.split(df, df[cfg.danger_outcome], df[cfg.col_match]):
        train, valid = df.iloc[tr], df.iloc[va]

        # il prior geometrico non dipende dal train: non deriva dagli esiti,
        # ed e' esattamente il motivo per cui e' preferibile
        if kind == "geometric":
            pri = geometric_priors(valid, cfg)
        elif kind == "spatial":
            cells, fb = spatial_priors(train, cfg, zone_col)
            pri = map_spatial_prior(valid, cells, fb, cfg, zone_col)
        else:
            pri, _ = outcome_priors(train, cfg)

        st = train.groupby(cfg.col_subtype)[cfg.col_sep_gain].agg(["mean", "std"])
        if containment is separation_curve_containment:
            if not {"sep_late", "sep_min"} <= set(train.columns):
                raise ValueError(
                    "separation_curve_containment richiede sep_late e sep_min: "
                    "unire tracking_features prima di validare."
                )
            _st_escape = (train["sep_late"] - train["sep_min"]).groupby(
                train[cfg.col_subtype]).agg(["mean", "std"])

        # Il containment va ricostruito con le statistiche del TRAIN, ma
        # deve rispettare il modello scelto. Prima questa closure era sempre
        # quella basata sul gain, quindi con --oos la configurazione del
        # containment veniva ignorata in silenzio: le righe "+ curva sep(t)"
        # uscivano identiche al riferimento.
        def _containment(d, c=cfg, _st=st, _model=containment):
            mu = d[c.col_subtype].map(_st["mean"])
            sd = d[c.col_subtype].map(_st["std"]).replace(0, np.nan)
            gain01 = (1.0 / (1.0 + np.exp(-(d[c.col_sep_gain] - mu) / sd))).fillna(0.5)
            if _model is None or _model is separation_gain_containment:
                return gain01
            if _model is separation_curve_containment:
                # fuga tardiva, z-scorata sulle statistiche del train
                esc = (d["sep_late"] - d["sep_min"])
                est = (_st_escape["mean"], _st_escape["std"])
                mu_e = d[c.col_subtype].map(est[0])
                sd_e = d[c.col_subtype].map(est[1]).replace(0, np.nan)
                late01 = (1.0 / (1.0 + np.exp(-(esc - mu_e) / sd_e))).fillna(0.5)
                out = (1.0 - c.w_late_escape) * gain01 + c.w_late_escape * late01
                return out.where(d["sep_min"].notna(), gain01)
            # modello custom: lo applichiamo cosi' com'e'
            return _model(d, c)

        # ranghi strutturali calcolati sulla distribuzione del TRAIN
        sv = structural_vulnerability(valid, cfg, reference=train) if structural else False

        v = compute_stress(valid, pri, containment=_containment, cfg=cfg, structural=sv)
        curves = build_all_curves(v, cfg, strategy)
        parts.append(pre_run_drs(v, curves, cfg,
                                 carry=[zone_col] if kind == "spatial" else None))

    return dissociation(
        pd.concat(parts, ignore_index=True),
        cfg,
        extra_controls=[zone_col] if kind == "spatial" else None,
    )


def split_half(
    team_match: pd.DataFrame,
    metric: str,
    n_iter: int = 400,
    seed: int = 0,
    min_matches: int = 3,
    team_col: str = "defending_team_id",
    match_col: str = "match_id",
) -> dict:
    """Stabilita' split-half di una metrica aggregata per squadra.

    Divide i match in due meta' casuali, aggrega la metrica per squadra in
    ciascuna meta', correla. Ripete n_iter volte.

    Lettura:
        mediana > 0.5    tratto di squadra stabile
        mediana 0.3-0.5  segnale debole ma presente
        mediana < 0.3    rumore di campione
    """
    corrs, n_common, n_teams = split_half_curves(
        team_match, metric, n_iter=n_iter, seed=seed,
        min_matches=min_matches, team_col=team_col, match_col=match_col,
    )
    ok = corrs[~np.isnan(corrs)]
    med = float(np.median(ok)) if len(ok) else np.nan
    return {
        "metric": metric,
        "n_teams": n_teams,
        "n_valid_iter": len(ok),
        "median": med,
        "iqr_lo": float(np.percentile(ok, 25)) if len(ok) else np.nan,
        "iqr_hi": float(np.percentile(ok, 75)) if len(ok) else np.nan,
        "frac_positive": float((ok > 0).mean()) if len(ok) else np.nan,
        "median_teams_per_split": int(np.median(n_common)) if n_common else 0,
        "spearman_brown": spearman_brown(med),
    }


def split_half_curves(
    team_match: pd.DataFrame,
    metric: str,
    n_iter: int = 400,
    seed: int = 0,
    min_matches: int = 3,
    team_col: str = "defending_team_id",
    match_col: str = "match_id",
) -> tuple[np.ndarray, list[int], int]:
    """Le n_iter correlazioni split-half, non solo la loro mediana.

    Esposte perche' servono per il confronto appaiato: con lo stesso seed due
    configurazioni vedono le STESSE divisioni, e la differenza va valutata
    divisione per divisione. Vedi paired_split_half.
    """
    rng = np.random.default_rng(seed)
    cnt = team_match.groupby(team_col)[match_col].nunique()
    ok = cnt[cnt >= min_matches].index
    d = team_match[team_match[team_col].isin(ok)]
    mids = d[match_col].unique()

    corrs, n_common = [], []
    for _ in range(n_iter):
        perm = rng.permutation(mids)
        h1 = set(perm[: len(perm) // 2])
        a = d[d[match_col].isin(h1)].groupby(team_col)[metric].mean()
        b = d[~d[match_col].isin(h1)].groupby(team_col)[metric].mean()
        common = a.index.intersection(b.index)
        n_common.append(len(common))
        if len(common) < 4:
            corrs.append(np.nan)
            continue
        corrs.append(a.loc[common].corr(b.loc[common]))

    # NaN conservati, non scartati: le due configurazioni di un confronto
    # appaiato devono restare allineate posizione per posizione.
    return np.array(corrs, dtype=float), n_common, len(ok)


def spearman_brown(r: float, n_parts: float = 2.0) -> float:
    """Affidabilita' del campione INTERO, da quella misurata su una meta'.

        r_full = n*r / (1 + (n-1)*r)

    La split-half divide i 20 match in due meta' da 10, quindi misura
    l'affidabilita' di una stima su meta' campione. Quella che interessa e'
    l'affidabilita' della stima sui 20 match, che e' piu' alta. Spearman-Brown
    e' la correzione classica per il raddoppio della lunghezza del test.

    Sulle metriche del progetto:

        stress_per_run                      +0.724 -> +0.840
        stress_per_run std. per fase        +0.534 -> +0.696
        containment appreso                 +0.605 -> +0.754
        containment appreso std. per fase   +0.519 -> +0.683
        containment per subtype             +0.026 -> +0.051

    Non e' un modo di gonfiare i numeri: e' la stima corretta della quantita'
    che interessa, e infatti non salva il containment per subtype.
    """
    if not np.isfinite(r):
        return float("nan")
    den = 1.0 + (n_parts - 1.0) * r
    if den == 0:
        return float("nan")
    return float(n_parts * r / den)


def paired_split_half(
    team_match_a: pd.DataFrame,
    team_match_b: pd.DataFrame,
    metric: str,
    cfg: DRSConfig = DEFAULT,
    n_iter: int = 400,
    seed: int = 0,
    min_matches: int = 3,
) -> dict:
    """Confronto APPAIATO fra due configurazioni sulla stessa metrica.

    `team_match_b` e' la variante, `team_match_a` il riferimento: il delta e'
    b - a, quindi positivo significa che la variante migliora.

    PERCHE' APPAIATO. Le due configurazioni girano sulle stesse divisioni
    (stesso seed), quindi i due risultati sono correlati e la differenza
    elimina il rumore comune. Confrontare due mediane indipendenti butta via
    quel guadagno e fa sembrare reali differenze che non lo sono: le mediane
    di stress_per_run con e senza la struttura da tracking sono +0.749 e
    +0.724, mentre il delta appaiato e' +0.008 +/- 0.010, cioe' nulla.

    SOGLIA DI RILEVANZA. Con 8 squadre un delta sotto cfg.relevance_threshold
    puo' avere l'intervallo che esclude lo zero ed essere irrilevante, quindi
    il verdetto lo chiama 'trascurabile'.

    Verdetti osservati sul progetto, metrica stress_per_run:
        senza struttura da tracking   +0.008 +/- 0.010   trascurabile
        senza terzo fattore           +0.007 +/- 0.011   trascurabile
        geo_use_endpoint=True         -0.141 +/- 0.018   peggiora
    """
    a, _, _ = split_half_curves(team_match_a, metric, n_iter=n_iter, seed=seed,
                                min_matches=min_matches)
    b, _, _ = split_half_curves(team_match_b, metric, n_iter=n_iter, seed=seed,
                                min_matches=min_matches)
    n = min(len(a), len(b))
    diff = b[:n] - a[:n]
    valid = diff[~np.isnan(diff)]
    if len(valid) == 0:
        return {"metric": metric, "delta_mean": np.nan, "ci95": np.nan,
                "frac_b_wins": np.nan, "verdict": "non calcolabile",
                "median_a": np.nan, "median_b": np.nan, "n_valid_iter": 0}

    mean = float(valid.mean())
    ci = float(1.96 * valid.std(ddof=1) / np.sqrt(len(valid)))
    if abs(mean) < cfg.relevance_threshold:
        verdict = "trascurabile"
    elif mean - ci > 0:
        verdict = "migliora"
    elif mean + ci < 0:
        verdict = "peggiora"
    else:
        verdict = "indistinguibile"

    return {
        "metric": metric,
        "median_a": float(np.nanmedian(a[:n])),
        "median_b": float(np.nanmedian(b[:n])),
        "delta_mean": mean,
        "ci95": ci,
        "frac_b_wins": float((valid > 0).mean()),
        "verdict": verdict,
        "n_valid_iter": len(valid),
    }


def kernel_sensitivity(
    df: pd.DataFrame,
    priors: dict,
    taus=(5, 15, 25, 40, 60),
    ramps=(True, False),
    cfg: DRSConfig = DEFAULT,
) -> pd.DataFrame:
    """Sensibilita' dell'odds ratio alla parametrizzazione del kernel.

    Da rifare ogni volta che si cambia il modello di containment: serve a
    dimostrare che il risultato non e' un artefatto di tuning.
    """


    d = compute_stress(df, priors, cfg=cfg)
    rows = []
    for tau in taus:
        for ramp in ramps:
            c = replace(cfg, tau_decay_frames=tau, use_ramp=ramp)
            curves = build_all_curves(d, c)
            res = dissociation(pre_run_drs(d, curves, c), c)
            r = res[res.outcome == c.danger_outcome].iloc[0]
            rows.append(
                {
                    "tau_frames": tau,
                    "tau_s": tau / c.tracking_hz,
                    "ramp": ramp,
                    "odds_ratio": r.odds_ratio,
                    "p_value": r.p_value,
                }
            )
    return pd.DataFrame(rows)

# ===========================================================================
# 8. FEATURE DAL TRACKING XY — curva sep(t) e struttura
# ===========================================================================

# ---------------------------------------------------------------------------
# Gruppo 1 — curva separation(t)
# ---------------------------------------------------------------------------

def separation_curve(
    run: pd.Series, tracking: Tracking, cfg: DRSConfig = DEFAULT,
    min_coverage: float = 0.70
) -> np.ndarray | None:
    """separation(t) frame per frame fra runner e difensore assegnato.

    None se la copertura del tracking nella finestra e' sotto `min_coverage`
    (buchi: il 29% dei frame e' vuoto negli stoppage).
    """
    rid = int(run["player_id"])
    did = run[cfg.col_defender]
    if pd.isna(did):
        return None
    did = int(did)

    f0, f1 = int(run[cfg.col_frame_start]), int(run[cfg.col_frame_end])
    vals = []
    for f in range(f0, f1 + 1):
        nf = tracking.nearest_frame(f, max_gap=3)
        if nf is None:
            continue
        pos = tracking.players[nf]
        if rid not in pos or did not in pos:
            continue
        (rx, ry), (dx, dy) = pos[rid], pos[did]
        vals.append(np.hypot(rx - dx, ry - dy))

    n_expected = max(f1 - f0 + 1, 1)
    if len(vals) / n_expected < min_coverage or len(vals) < 4:
        return None
    return np.asarray(vals)


def separation_curve_features(sep: np.ndarray, cfg: DRSConfig = DEFAULT) -> dict:
    """Feature della forma della curva. Tutte pre-esito.

    sep_min         separation minima raggiunta
    sep_min_frac    QUANDO avviene il minimo, 0 = inizio, 1 = fine corsa.
                    Perdere terreno all'ultimo secondo e' peggio che
                    perderlo e recuperarlo.
    sep_slope_min   pendenza nel momento critico (m/s al minimo): negativa
                    = il difensore stava ancora accorciando, positiva = il
                    runner si stava gia' staccando
    sep_time_close  frazione di corsa passata sotto 3 m dal difensore
    sep_inversions  quante volte la derivata cambia segno: un difensore che
                    recupera e riperde oscilla
    sep_auc         area sotto la curva normalizzata per durata = separation
                    media sostenuta (non solo agli estremi)
    sep_late        separation media nell'ultimo 30% della corsa, che e' il
                    momento in cui tipicamente arriva la palla
    sep_range       escursione totale
    """
    n = len(sep)
    d = np.diff(sep) * cfg.tracking_hz          # m/s
    i_min = int(np.argmin(sep))
    late = sep[int(0.7 * n):]

    # inversioni robuste: ignora oscillazioni sotto 0.3 m/s
    sign = np.sign(np.where(np.abs(d) < 0.3, 0.0, d))
    sign = sign[sign != 0]
    inversions = int((np.diff(sign) != 0).sum()) if len(sign) > 1 else 0

    return {
        "sep_min": float(sep.min()),
        "sep_min_frac": float(i_min / max(n - 1, 1)),
        "sep_slope_min": float(d[min(i_min, len(d) - 1)]) if len(d) else 0.0,
        "sep_time_close": float((sep < 3.0).mean()),
        "sep_inversions": inversions,
        "sep_auc": float(sep.mean()),
        "sep_late": float(late.mean()) if len(late) else float(sep[-1]),
        "sep_range": float(sep.max() - sep.min()),
        "sep_n_frames": n,
    }


# ---------------------------------------------------------------------------
# Gruppo 1b — geometria della corsa: punto di arrivo e palla
# ---------------------------------------------------------------------------

def run_geometry(
    run: pd.Series, tracking: Tracking, cfg: DRSConfig = DEFAULT
) -> dict:
    """Dove la corsa FINISCE, e dov'e' la palla.

    PERCHE' SERVE. Il Layer 3 enriched porta `x_start`/`y_start` ma NON le
    coordinate di arrivo. Il prior geometrico guarda quindi dove la corsa
    parte, mentre il pericolo e' dove arriva: in media una corsa avvicina
    alla porta di 7.3 m.

    VERIFICATO sul match 1996436 (387 corse), correlazione con lead_to_shot
    di exp(-dist/25) calcolato su punti diversi:

        partenza (attuale)   +0.403
        arrivo               +0.458     (+14%)
        media delle due      +0.445
        min(partenza,arrivo) +0.469     (+16%)   <- il migliore

    Il minimo e' anche il piu' difendibile: cattura sia la corsa che si
    avvicina sia quella che parte gia' vicina e poi si allarga. In entrambi
    i casi conta il momento in cui l'attaccante e' stato piu' minaccioso.

    ORIENTAMENTO. Il tracking e' in coordinate assolute e le squadre
    cambiano campo all'intervallo; `x_start` dell'enriched e' gia'
    orientato. Qui si riallinea con `attacking_side`, e il sanity check e'
    che `geo_x_start` coincida con `x_start` (verificato: r = 1.000).

    La palla: una corsa a dieci metri dalla porta mentre il pallone e'
    nella meta' campo opposta non e' una minaccia immediata. `ball_dist_*`
    serve a distinguerla.
    """
    pid = int(run["player_id"])
    f0, f1 = int(run[cfg.col_frame_start]), int(run[cfg.col_frame_end])
    n0, n1 = tracking.nearest_frame(f0, 5), tracking.nearest_frame(f1, 5)
    if n0 is None or n1 is None:
        return {}
    p0, p1 = tracking.players[n0], tracking.players[n1]
    if pid not in p0 or pid not in p1:
        return {}

    flip = str(run.get("attacking_side", "left_to_right")) != "left_to_right"
    sgn = -1.0 if flip else 1.0
    sx, sy = p0[pid]
    ex, ey = p1[pid]
    sx, ex = sgn * sx, sgn * ex

    d_start = float(np.hypot(cfg.goal_x - sx, cfg.goal_y - sy))
    d_end = float(np.hypot(cfg.goal_x - ex, cfg.goal_y - ey))

    out = {
        "geo_x_start": sx, "geo_y_start": sy,
        "geo_x_end": ex, "geo_y_end": ey,
        "geo_dist_start": d_start,
        "geo_dist_end": d_end,
        "geo_dist_min": min(d_start, d_end),
        "geo_progress": d_start - d_end,      # quanto la corsa avvicina
    }

    b0 = tracking.ball.get(n0)
    b1 = tracking.ball.get(n1)
    if b0 and b0[0] is not None:
        bx, by = sgn * b0[0], b0[1]
        out["ball_dist_goal_start"] = float(np.hypot(cfg.goal_x - bx, cfg.goal_y - by))
        out["ball_dist_runner_start"] = float(np.hypot(bx - sx, by - sy))
    if b1 and b1[0] is not None:
        bx, by = sgn * b1[0], b1[1]
        out["ball_dist_goal_end"] = float(np.hypot(cfg.goal_x - bx, cfg.goal_y - by))
        out["ball_dist_runner_end"] = float(np.hypot(bx - ex, by - ey))
    return out


# ---------------------------------------------------------------------------
# Gruppo 2 — struttura difensiva
# ---------------------------------------------------------------------------

def _outfield(
    pos: dict, team_ids: dict, team: int, gk: int | None, flip: bool = False
) -> np.ndarray:
    """Giocatori di movimento della squadra, opzionalmente con x orientata.

    ORIENTAMENTO. Il tracking e' in coordinate assolute e le squadre cambiano
    campo all'intervallo, mentre `last_defensive_line_x_start` del Layer 3 e'
    gia' normalizzato per direzione d'attacco. Senza orientare, l'altezza
    della linea calcolata qui e' scorrelata da quella di SkillCorner
    (verificato: r = -0.17 senza flip).

    Convenzione: dopo l'orientamento, x alta = vicino alla PROPRIA porta,
    cioe' linea arretrata. Coerente con il Layer 3.
    """
    pts = [p for q, p in pos.items() if team_ids.get(q) == team and q != gk]
    if not pts:
        return np.empty((0, 2))
    a = np.asarray(pts, dtype=float)
    if flip:
        a = a.copy()
        a[:, 0] = -a[:, 0]
    return a


def identify_goalkeepers(
    tracking: Tracking, team_ids: dict, n_sample: int = 300
) -> dict[int, int]:
    """Il portiere di ogni squadra = il giocatore con |x| medio piu' estremo
    e varianza di posizione piu' bassa. Va escluso da compattezza e linea:
    un portiere in uscita falserebbe entrambe.
    """
    sample = tracking.frames[:: max(1, len(tracking.frames) // n_sample)]
    acc: dict[int, list] = {}
    for f in sample:
        for q, (x, y) in tracking.players[f].items():
            acc.setdefault(q, []).append(x)

    out = {}
    for team in set(team_ids.values()):
        cand = {q: np.asarray(v) for q, v in acc.items()
                if team_ids.get(q) == team and len(v) > 10}
        if not cand:
            continue
        # punteggio: estremita' media alta, dispersione bassa
        score = {q: np.abs(v.mean()) - 0.6 * v.std() for q, v in cand.items()}
        out[team] = max(score, key=score.get)
    return out


def structure_at_frame(
    pos: dict, team_ids: dict, defending_team: int, gk: int | None,
    flip: bool = False
) -> dict | None:
    """Forma della difesa in un singolo frame.

    `flip` orienta la x secondo la direzione d'attacco (vedi `_outfield`).
    spread_x, spread_y e area sono invarianti all'orientamento; line_x e
    centroid_x no, e senza flip non sono confrontabili fra i due tempi.
    """
    pts = _outfield(pos, team_ids, defending_team, gk, flip)
    if len(pts) < 7:
        return None
    x, y = pts[:, 0], pts[:, 1]
    back4 = np.sort(x)[-4:]       # x gia' orientato: alto = vicino propria porta
    return {
        "spread_x": float(x.max() - x.min()),
        "spread_y": float(y.max() - y.min()),
        "line_x": float(back4.mean()),
        "line_spread": float(back4.max() - back4.min()),
        "centroid_x": float(x.mean()),
        "centroid_y": float(y.mean()),
        "n_outfield": len(pts),
    }


def structure_features(
    run: pd.Series, tracking: Tracking, team_ids: dict, gks: dict,
    cfg: DRSConfig = DEFAULT, recovery_s: float = 2.0
) -> dict:
    """Struttura difensiva all'inizio della corsa e sua evoluzione.

    str_spread_x / str_spread_y  compattezza longitudinale e laterale dei
                                 giocatori di movimento al frame_start
    str_area                     area del rettangolo che li contiene, proxy
                                 dell'ampiezza occupata
    str_line_x                   altezza della linea (media dei 4 piu'
                                 arretrati)
    str_line_spread              quanto e' slabbrata la linea a quattro
    str_d_spread_x               variazione della compattezza longitudinale
                                 durante la corsa: positiva = la squadra si
                                 e' allungata, negativa = si e' compattata
    str_recovery_rate            velocita' di ricompattamento nei
                                 `recovery_s` secondi dopo la fine della
                                 corsa (m/s). Negativa = la difesa si
                                 ricompone, positiva = continua a sfaldarsi.
                                 E' la feature che il layer3_enriched non
                                 puo' dare in nessun modo.
    """
    gk = gks.get(defending := int(run["defending_team_id"]))
    f0, f1 = int(run[cfg.col_frame_start]), int(run[cfg.col_frame_end])
    f2 = f1 + int(recovery_s * cfg.tracking_hz)

    # l'attacco va verso +x quando attacking_side == 'left_to_right': in quel
    # caso la porta della squadra che difende e' a +x e la x grezza va bene.
    # Altrimenti va ribaltata, cosi' x alta = linea arretrata in entrambi i casi.
    flip = str(run.get("attacking_side", "left_to_right")) != "left_to_right"

    def at(f):
        nf = tracking.nearest_frame(f, max_gap=5)
        return structure_at_frame(tracking.players[nf], team_ids, defending, gk, flip) \
            if nf is not None else None

    s0, s1, s2 = at(f0), at(f1), at(f2)
    if s0 is None:
        return {}

    out = {
        "str_spread_x": s0["spread_x"],
        "str_spread_y": s0["spread_y"],
        "str_area": s0["spread_x"] * s0["spread_y"],
        "str_line_x": s0["line_x"],
        "str_line_spread": s0["line_spread"],
    }
    if s1:
        dur_s = max((f1 - f0) / cfg.tracking_hz, 0.1)
        out["str_d_spread_x"] = s1["spread_x"] - s0["spread_x"]
        out["str_d_spread_y"] = s1["spread_y"] - s0["spread_y"]
        out["str_d_line_x"] = s1["line_x"] - s0["line_x"]
        out["str_stretch_rate"] = out["str_d_spread_x"] / dur_s
    if s1 and s2:
        out["str_recovery_rate"] = (s2["spread_x"] - s1["spread_x"]) / recovery_s
        out["str_line_recovery"] = (s2["line_x"] - s1["line_x"]) / recovery_s
    return out


# ---------------------------------------------------------------------------
# Estrazione per match
# ---------------------------------------------------------------------------

def extract_match(
    match_id: int, runs: pd.DataFrame, tracking_path: str | Path,
    cfg: DRSConfig = DEFAULT, verbose: bool = True
) -> pd.DataFrame:
    """Tutte le feature di tracking per le corse di un match."""
    m = runs[runs[cfg.col_match] == match_id]
    if m.empty:
        raise ValueError(f"nessuna corsa per il match {match_id}")

    lo = int(m[cfg.col_frame_start].min()) - 50
    hi = int(m[cfg.col_frame_end].max()) + 50
    tracking = load_tracking(tracking_path, (lo, hi))
    team_ids = infer_player_teams(m, tracking, cfg)
    gks = identify_goalkeepers(tracking, team_ids)
    if verbose:
        print(f"[tracking] match {match_id}: {len(m)} corse, "
              f"{len(tracking.frames)} frame, portieri {list(gks.values())}")

    rows, no_curve = [], 0
    for _, r in m.iterrows():
        rec = {cfg.col_match: match_id, cfg.col_event: r[cfg.col_event]}
        rec.update(run_geometry(r, tracking, cfg))
        sep = separation_curve(r, tracking, cfg)
        if sep is None:
            no_curve += 1
        else:
            rec.update(separation_curve_features(sep, cfg))
        rec.update(structure_features(r, tracking, team_ids, gks, cfg))
        rows.append(rec)

    out = pd.DataFrame(rows)
    if verbose:
        print(f"[tracking] curva separation non calcolabile su {no_curve}/{len(m)} "
              f"corse ({no_curve/len(m):.1%}, buchi nel tracking)")
    return out


def extract_all(
    layer3_path: str | Path, tracking_root: str | Path,
    out_path: str | Path, cfg: DRSConfig = DEFAULT,
    pattern: str = "{mid}/{mid}_tracking_extrapolated.jsonl",
) -> pd.DataFrame:
    """Estrae le feature per tutti i match presenti nel Layer 3.

    `tracking_root` e' la cartella `data/matches` della repo opendata.
    I match il cui file di tracking non esiste vengono saltati con un avviso.
    """

    runs = load_layer3(layer3_path, cfg)
    root = Path(tracking_root)
    parts, missing = [], []

    for mid in sorted(runs[cfg.col_match].unique()):
        p = root / pattern.format(mid=mid)
        if not p.exists():
            missing.append(mid)
            continue
        parts.append(extract_match(int(mid), runs, p, cfg))

    if missing:
        print(f"[tracking] tracking assente per {len(missing)} match: {missing}")
    if not parts:
        raise FileNotFoundError(f"nessun file di tracking trovato sotto {root}")

    out = pd.concat(parts, ignore_index=True)
    out.to_csv(out_path, index=False)
    print(f"[tracking] {len(out)} righe x {out.shape[1]} colonne -> {out_path}")
    return out


def merge_into_layer3(
    layer3: pd.DataFrame, features: pd.DataFrame, cfg: DRSConfig = DEFAULT
) -> pd.DataFrame:
    """Merge delle feature di tracking sul Layer 3, su (match_id, event_id)."""
    return layer3.merge(features, on=[cfg.col_match, cfg.col_event], how="left")

# ===========================================================================
# 9. PIPELINE end-to-end
# ===========================================================================

@dataclass
class DRSResult:
    """Tutto quello che un run produce. Serializzabile per riproducibilita'."""

    config: DRSConfig
    runs: pd.DataFrame                 # Layer 3 + danger_prior, containment, stress
    priors: dict | pd.Series
    prior_table: pd.DataFrame
    curves_by_match: dict              # match_id -> (curve, FrameIndex)
    curves_by_team: dict = field(default_factory=dict)
    team_match: pd.DataFrame | None = None
    validation: pd.DataFrame | None = None
    stability: pd.DataFrame | None = None
    two_axis: pd.DataFrame | None = None

    def save(self, outdir: str | Path) -> Path:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        self.runs.to_parquet(outdir / "runs_with_stress.parquet")
        self.prior_table.to_csv(outdir / "priors.csv")
        if self.team_match is not None:
            self.team_match.to_csv(outdir / "team_match_metrics.csv", index=False)
        if self.validation is not None:
            self.validation.to_csv(outdir / "validation.csv", index=False)
        if self.stability is not None:
            self.stability.to_csv(outdir / "stability.csv", index=False)
        if self.two_axis is not None:
            self.two_axis.to_csv(outdir / "team_two_axis.csv", index=False)
        np.savez_compressed(
            outdir / "curves.npz",
            **{str(k): v[0] for k, v in self.curves_by_match.items()},
        )
        pd.Series(self.config.to_dict()).to_json(outdir / "config.json", indent=2)
        return outdir


def load_layer3(path: str | Path, cfg: DRSConfig = DEFAULT) -> pd.DataFrame:
    """Carica il Layer 3 e applica i filtri minimi.

    Funziona sia con layer3_all_matches.csv (18 colonne) sia con
    layer3_enriched.csv (28+): le colonne in piu' vengono ignorate dal
    modello attuale ma restano disponibili a valle.
    """
    df = pd.read_csv(path)

    unmapped = set(df[cfg.col_subtype].dropna().unique()) - set(FAMILY_MAP)
    if unmapped:
        n = df[cfg.col_subtype].isin(unmapped).sum()
        print(f"[warn] {n} run con subtype non mappato, scartate: {sorted(unmapped)}")
        df = df[df[cfg.col_subtype].isin(FAMILY_MAP)]

    df = df.copy()
    df["subtype_family"] = df[cfg.col_subtype].map(FAMILY_MAP)

    required = [
        cfg.col_match, cfg.col_event, cfg.col_subtype,
        cfg.col_frame_start, cfg.col_frame_end, cfg.col_sep_gain, cfg.col_team,
        *cfg.outcomes,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"colonne mancanti nel Layer 3: {missing}")

    for c in cfg.outcomes:
        if df[c].dtype == object:
            df[c] = df[c].map({"True": True, "False": False, True: True, False: False})
    return df


def run_pipeline(
    layer3_path: str | Path,
    cfg: DRSConfig = DEFAULT,
    containment: ContainmentModel | None = None,
    indexing: str = "event_anchored",
    spatial_prior: bool | None = None,
    structural: bool | None = None,
    zone_col: str = "third_start",
    with_validation: bool = True,
    with_oos: bool = False,
    verbose: bool = True,
) -> DRSResult:
    """Esegue la pipeline completa.

    Parameters
    ----------
    containment : modello di containment failure. None = automatico, che
        significa separation_gain_containment salvo che
        `cfg.use_curve_containment` sia True e le colonne di curva ci siano.
        La curva e' disattivata di default perche' sui 20 match peggiora il
        containment (vedi DRSConfig.use_curve_containment).
    indexing : strategia di indicizzazione dei frame.
        'event_anchored' (attiva) | 'dense_tracking' | 'phase_window' (stub).
    spatial_prior : forza il prior per subtype x zona (True) o per solo
        subtype (False). None = usa `cfg.prior_kind`, che di default e'
        'geometric': il prior dalla sola geometria, che non tocca gli esiti
        e out-of-sample batte gli altri (OR 1.235 contro 1.165).
    structural : terzo fattore di vulnerabilita' strutturale nello stress.
        None = automatico (attivo se le colonne enriched esistono).
    with_oos : esegue anche la dissociazione out-of-sample (5-fold su match).
        Piu' lenta, ma e' la versione da citare nel paper.
    """
    log = print if verbose else (lambda *a, **k: None)
    strategy: IndexingStrategy = get_strategy(indexing)

    df = load_layer3(layer3_path, cfg)
    log(f"Layer 3: {len(df)} run su {df[cfg.col_match].nunique()} match")

    learned = None
    if containment is None:
        if cfg.containment_kind == "learned":
            if not has_learned_columns(df):
                raise ValueError(
                    "containment_kind='learned' richiede il merge con "
                    "tracking_features.csv (8 colonne numeriche vengono da la')."
                )
            # calcolato una volta e congelato: altrimenti compute_stress e
            # dissociation_oos rifitterebbero il modello a ogni chiamata, e
            # l'OOS lo rifitterebbe su un fold solo
            learned = learned_containment(df, cfg)
            containment = fixed_containment(learned)
            containment.__name__ = "learned_containment"
        elif cfg.use_curve_containment and has_curve_columns(df):
            containment = separation_curve_containment
        else:
            containment = separation_gain_containment
    has_trk = cfg.use_tracking_structure and "str_recovery_rate" in df.columns
    log(f"Containment: {getattr(containment, '__name__', 'custom')}"
        + ("   struttura: + tracking XY" if has_trk else ""))

    kind = cfg.prior_kind
    if spatial_prior is False:
        kind = "subtype"
    elif spatial_prior is True:
        kind = "spatial"
    if kind == "geometric" and not {"x_start", "y_start"} <= set(df.columns):
        kind = "spatial" if zone_col in df.columns else "subtype"
    spatial_prior = kind == "spatial"
    if structural is None:
        structural = has_structural_columns(df)

    if kind == "geometric":
        priors = geometric_priors(df, cfg)
        prior_tab = priors.rename("danger").to_frame()
        log(f"Prior geometrico (scala {cfg.geo_prior_scale:.0f} m, non usa gli "
            f"esiti), max={priors.max():.3f} min={priors.min():.3f}")
    elif spatial_prior:
        cells, fb = spatial_priors(df, cfg, zone_col)
        priors = map_spatial_prior(df, cells, fb, cfg, zone_col)
        prior_tab = pd.Series(cells).rename("danger").to_frame()
        log(f"Prior spaziale: {len(cells)} celle (subtype x {zone_col}), "
            f"max={priors.max():.3f} min={priors.min():.3f}")
    else:
        priors, prior_tab = outcome_priors(df, cfg)
        log(f"Prior: {len(priors)} subtype, "
            f"max={max(priors.values()):.3f} min={min(priors.values()):.3f}")

    df = compute_stress(df, priors, containment, cfg, structural=structural)
    df = add_defending_team(df, cfg)
    log(f"Stress: media {df.stress.mean():.3f}, max {df.stress.max():.3f}"
        + ("  [+ vulnerabilita' strutturale]" if structural else ""))

    curves_match = build_all_curves(df, cfg, strategy)
    cov = np.mean([idx.coverage() for _, idx in curves_match.values()])
    log(f"Curve: {len(curves_match)} match, copertura media {cov:.1%} "
        f"(strategia '{strategy.name}')")

    curves_team = build_all_curves(df, cfg, strategy, by_defending_team=True)
    tm = team_match_metrics(df, curves_team, cfg)

    res = DRSResult(
        config=cfg, runs=df, priors=priors, prior_table=prior_tab,
        curves_by_match=curves_match, curves_by_team=curves_team, team_match=tm,
    )

    if cfg.col_phase in df.columns:
        res.two_axis = team_two_axis(df, tm, cfg)
        log(f"Profilo a due assi: {len(res.two_axis)} squadre "
            f"(esposizione + contenimento standardizzato per fase)")

    if with_validation:
        try:
            import statsmodels.api as _sm  # noqa: F401
        except ImportError:
            log("\n[!] statsmodels non installato: validazione saltata.")
            log("    Curve e metriche di squadra sono comunque calcolate e salvate.")
            log("    Per la validazione completa:  pip install statsmodels")
            with_validation = False

    if with_validation:
        extra = [zone_col] if kind == "spatial" else None
        if with_oos:
            res.validation = dissociation_oos(
                df, cfg, strategy=strategy, structural=structural,
                zone_col=zone_col, prior_kind=kind, containment=containment,
            )
        else:
            pre = pre_run_drs(df, curves_match, cfg, carry=extra)
            res.validation = dissociation(pre, cfg, extra_controls=extra)
        log("\nDissociazione" + (" (out-of-sample)" if with_oos else " (in-sample)"))
        for _, r in res.validation.iterrows():
            log(f"  {r.outcome:14s} OR={r.odds_ratio:.3f} "
                f"[{r.ci_lo:.3f}, {r.ci_hi:.3f}]  p={r.p_value:.2e}")

        metrics = ["n_runs", "drs_mean_all", "stress_per_run",
                   "drs_mean_supported", "containment_failure_mean", "prior_mean"]
        metrics += [m for m in ("stress_per_run_phase_adj", "containment_phase_adj",
                                "deep_phase_share") if m in tm.columns]
        res.stability = pd.DataFrame([split_half(tm, m) for m in metrics])
        log("\nSplit-half stability (squadre con >=3 match)")
        log(f"  {'metrica':28s} {'mediana':>8s} {'IQR':>17s} {'pos':>5s} {'Sp.-Brown':>10s}")
        for _, r in res.stability.iterrows():
            log(f"  {r.metric:28s} {r['median']:+8.3f} "
                f"[{r.iqr_lo:+.3f},{r.iqr_hi:+.3f}] {r.frac_positive:5.0%} "
                f"{r.spearman_brown:+10.3f}")
        log("  Spearman-Brown = affidabilita' sui 20 match interi; la mediana e'")
        log("  misurata su meta' campione, quindi sottostima quella che interessa.")

    return res

# ===========================================================================
# 10. FIGURE statiche
# ===========================================================================

RED = "#c0392b"
DARK = "#2c3e50"
GREY = "#95a5a6"


def plot_match_curve(
    curve: np.ndarray,
    match_runs: pd.DataFrame,
    cfg: DRSConfig = DEFAULT,
    zoom_window_s: float = 90.0,
    title: str | None = None,
    ax=None,
):
    """Curva DRS(t) di un match, con zoom sul picco.

    Pannello alto: partita intera, triangoli sulle corse che portano a tiro.
    Pannello basso: finestra attorno al picco, bande = corse attive.
    """
    import matplotlib.pyplot as plt

    t = np.arange(len(curve)) / cfg.tracking_hz / 60.0
    fig, axes = plt.subplots(2, 1, figsize=(15, 7), height_ratios=[2, 1])

    ax = axes[0]
    ax.fill_between(t, curve, color=RED, alpha=0.25, lw=0)
    ax.plot(t, curve, color=RED, lw=0.7)
    sh = match_runs[match_runs[cfg.danger_outcome]]
    if len(sh):
        ax.scatter(
            sh[cfg.col_frame_start] / cfg.tracking_hz / 60,
            [curve[min(int(f), len(curve) - 1)] for f in sh[cfg.col_frame_start]],
            s=28, color=DARK, marker="v", zorder=5,
            label=f"run → tiro (n={len(sh)})",
        )
        ax.legend(fontsize=8, loc="upper right")
    p95 = np.percentile(curve, 95)
    ax.axhline(p95, ls="--", lw=0.8, color=GREY)
    ax.text(0.2, p95 * 1.05, "p95", fontsize=8, color=GREY)
    ax.set_ylabel("DRS(t)")
    ax.set_xlabel("minuti")
    ax.set_xlim(0, t.max())
    ax.set_title(title or f"DRS(t) — {len(match_runs)} off-ball run", fontsize=11)

    pk = int(np.argmax(curve))
    half = int(zoom_window_s * cfg.tracking_hz)
    lo, hi = max(0, pk - half), min(len(curve), pk + half)
    ax = axes[1]
    ax.fill_between(t[lo:hi], curve[lo:hi], color=RED, alpha=0.25, lw=0)
    ax.plot(t[lo:hi], curve[lo:hi], color=RED, lw=1.5)
    w = match_runs[
        (match_runs[cfg.col_frame_start] >= lo) & (match_runs[cfg.col_frame_start] <= hi)
    ]
    denom = cfg.tracking_hz * 60
    for _, r in w.iterrows():
        col = DARK if r[cfg.danger_outcome] else GREY
        ax.axvspan(
            r[cfg.col_frame_start] / denom, r[cfg.col_frame_end] / denom,
            alpha=0.12, color=col, lw=0,
        )
    ax.set_ylabel("DRS(t)")
    ax.set_xlabel("minuti")
    ax.set_title(
        f"Zoom ±{zoom_window_s:.0f}s sul picco (min {t[pk]:.1f}) — "
        "bande = corse attive, scure = corse che portano a tiro",
        fontsize=9,
    )
    fig.tight_layout()
    return fig


def plot_team_profile(profile: pd.DataFrame, metric: str = "stress_per_run"):
    """Profilo per squadra con CI bootstrap, ordinato."""
    import matplotlib.pyplot as plt

    p = profile.sort_values(metric)
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(p) + 1.5))
    y = np.arange(len(p))
    ax.errorbar(
        p[metric], y,
        xerr=[p[metric] - p.ci95_lo, p.ci95_hi - p[metric]],
        fmt="o", color=DARK, ecolor=GREY, capsize=3, ms=5,
    )
    ax.axvline(p[metric].mean(), ls="--", lw=0.8, color=RED)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{int(t)} (n={n})" for t, n in zip(p.defending_team_id, p.n_matches)])
    ax.set_xlabel(metric)
    ax.set_title(f"Profilo difensivo per squadra — {metric}, CI95 bootstrap sui match",
                 fontsize=10)
    fig.tight_layout()
    return fig

# ===========================================================================
# 11. ANIMAZIONE 2D su campo
# ===========================================================================

# dimensioni campo SkillCorner: coordinate centrate
PITCH_X = 105.0
PITCH_Y = 68.0

ATT = "#e74c3c"      # squadra in possesso
DEF_ = "#3498db"     # squadra che difende
BALL = "#f1c40f"
HEAT = "#c0392b"


# ---------------------------------------------------------------------------
# Caricamento tracking
# ---------------------------------------------------------------------------

@dataclass
class Tracking:
    """Tracking di un match, indicizzato per frame."""
    players: dict          # frame -> {player_id: (x, y)}
    ball: dict             # frame -> (x, y, z) oppure None
    frames: np.ndarray     # frame disponibili con 22 giocatori

    def at(self, frame: int) -> dict:
        return self.players.get(frame, {})

    def nearest_frame(self, frame: int, max_gap: int = 10) -> int | None:
        """Frame tracciato piu' vicino. Il 29% dei frame e' vuoto (stoppages)."""
        if frame in self.players:
            return frame
        i = np.searchsorted(self.frames, frame)
        cands = [c for c in (i - 1, i) if 0 <= c < len(self.frames)]
        if not cands:
            return None
        best = min((self.frames[c] for c in cands), key=lambda f: abs(f - frame))
        return int(best) if abs(best - frame) <= max_gap else None


def load_tracking(
    path: str | Path, frame_range: tuple[int, int] | None = None
) -> Tracking:
    """Legge il JSONL. `frame_range` limita la lettura, utile per un singolo
    clip: il file completo e' ~90 MB per match."""
    players, ball = {}, {}
    lo, hi = frame_range if frame_range else (-np.inf, np.inf)

    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            fr = rec["frame"]
            if fr < lo or fr > hi:
                continue
            pd_ = rec.get("player_data") or []
            if not pd_:
                continue
            players[fr] = {p["player_id"]: (p["x"], p["y"]) for p in pd_}
            b = rec.get("ball_data") or {}
            ball[fr] = (b.get("x"), b.get("y"), b.get("z")) if b.get("x") is not None else None

    return Tracking(players, ball, np.array(sorted(players)))


def infer_player_teams(
    runs: pd.DataFrame, tracking: Tracking, cfg: DRSConfig = DEFAULT, n_sample: int = 400
) -> dict[int, int]:
    """Mappa player_id -> team_id.

    Il tracking SkillCorner non porta il team_id, e il match.json non e'
    sempre a portata. La ricaviamo dal Layer 3: ogni runner appartiene a
    `team_id`, ogni difensore a `defending_team_id`.

    Restano fuori i giocatori che non compaiono mai nel Layer 3 — tipicamente
    i due portieri (non fanno off-ball run e non sono quasi mai il difensore
    piu' vicino). Quelli vengono assegnati alla squadra ai cui compagni gia'
    noti sono mediamente piu' vicini.

    Se hai il match.json dell'opendata, una mappa esatta da
    `players[i].id` -> `players[i].team_id` e' preferibile: passala
    direttamente al posto di questa funzione.
    """
    p2t: dict[int, int] = {}
    for _, r in runs.iterrows():
        p2t[int(r["player_id"])] = int(r[cfg.col_team])
        if pd.notna(r[cfg.col_defender]):
            p2t[int(r[cfg.col_defender])] = int(r["defending_team_id"])

    seen = set()
    for fr in tracking.frames[:: max(1, len(tracking.frames) // n_sample)]:
        seen |= set(tracking.players[fr])
    unknown = seen - set(p2t)
    if not unknown:
        return p2t

    teams = sorted(set(p2t.values()))
    sample = tracking.frames[:: max(1, len(tracking.frames) // n_sample)]
    for pid in unknown:
        dists = {t: [] for t in teams}
        for fr in sample:
            pos = tracking.players[fr]
            if pid not in pos:
                continue
            x0, y0 = pos[pid]
            for t in teams:
                pts = [pos[q] for q in pos if p2t.get(q) == t]
                if pts:
                    a = np.asarray(pts)
                    dists[t].append(np.hypot(a[:, 0] - x0, a[:, 1] - y0).mean())
        if any(dists[t] for t in teams):
            p2t[pid] = min(teams, key=lambda t: np.mean(dists[t]) if dists[t] else np.inf)
    print(f"[animate] {len(unknown)} giocatori assegnati per prossimita' "
          f"(probabili portieri): {sorted(unknown)}")
    return p2t


# ---------------------------------------------------------------------------
# Campo di stress
# ---------------------------------------------------------------------------

def _gauss(gx, gy, cx, cy, amp, sigma):
    return amp * np.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * sigma ** 2))


def stress_field(
    frame: int,
    active_runs: pd.DataFrame,
    tracking: Tracking,
    grid: tuple[np.ndarray, np.ndarray],
    cfg: DRSConfig = DEFAULT,
    spatial_mode: str = "gap",
    sigma: float = 5.0,
    project_s: float = 1.5,
) -> np.ndarray:
    """Campo di stress su griglia per un singolo frame.

    `active_runs` sono le corse il cui contributo temporale al frame e' > 0,
    con una colonna `w` gia' pesata dal kernel temporale.
    """
    gx, gy = grid
    field = np.zeros_like(gx, dtype=float)
    pos = tracking.at(tracking.nearest_frame(frame) or -1)
    if not pos:
        return field

    for _, r in active_runs.iterrows():
        rid = int(r["player_id"])
        if rid not in pos:
            continue
        rx, ry = pos[rid]
        amp = float(r["w"])

        if spatial_mode == "runner":
            field += _gauss(gx, gy, rx, ry, amp, sigma)

        elif spatial_mode == "projected":
            prev = tracking.nearest_frame(frame - cfg.tracking_hz)
            vx = vy = 0.0
            if prev is not None and rid in tracking.at(prev):
                px, py = tracking.at(prev)[rid]
                vx, vy = (rx - px), (ry - py)   # metri al secondo
            field += _gauss(gx, gy, rx + vx * project_s, ry + vy * project_s, amp, sigma)

        elif spatial_mode == "gap":
            did = r.get(cfg.col_defender)
            if pd.isna(did) or int(did) not in pos:
                field += _gauss(gx, gy, rx, ry, amp, sigma)
                continue
            dx, dy = pos[int(did)]
            # deposita lungo il segmento difensore-runner, piu' denso sul runner
            for t in np.linspace(0.0, 1.0, 5):
                cx, cy = dx + t * (rx - dx), dy + t * (ry - dy)
                field += _gauss(gx, gy, cx, cy, amp * (0.3 + 0.7 * t) / 2.5, sigma * 0.8)
        else:
            raise ValueError(f"spatial_mode '{spatial_mode}' sconosciuto")

    return field


def active_weights(
    runs: pd.DataFrame, frame: int, cfg: DRSConfig = DEFAULT
) -> pd.DataFrame:
    """Corse attive al frame, con il peso del kernel temporale applicato.

    Replica esattamente il kernel di curve.run_kernel, cosi' la somma del
    campo coincide con il DRS(t) scalare.
    """
    tail = int(cfg.tail_multiplier * cfg.tau_decay_frames)
    m = runs[(runs[cfg.col_frame_start] <= frame) & (runs[cfg.col_frame_end] + tail > frame)].copy()
    if m.empty:
        m["w"] = []
        return m

    f0 = m[cfg.col_frame_start].astype(int).values
    f1 = m[cfg.col_frame_end].astype(int).values
    s = m["stress"].values
    w = np.zeros(len(m))

    during = frame < f1
    if cfg.use_ramp:
        # replica esattamente np.linspace(0, 1, dur) di curve.run_kernel:
        # l'elemento i-esimo vale i/(dur-1), non i/dur
        denom = np.maximum(f1 - f0 - 1, 1)
        w[during] = s[during] * np.clip((frame - f0[during]) / denom[during], 0, 1)
    else:
        w[during] = s[during]
    after = ~during
    w[after] = s[after] * np.exp(-(frame - f1[after]) / cfg.tau_decay_frames)

    m["w"] = w
    return m[m.w > 1e-6]


# ---------------------------------------------------------------------------
# Selezione finestre
# ---------------------------------------------------------------------------

def find_peak_windows(
    curve: np.ndarray, n: int = 5, pad_s: float = 5.0, cfg: DRSConfig = DEFAULT
) -> list[tuple[int, int, int]]:
    """Le n finestre con DRS piu' alto, non sovrapposte.

    Returns lista di (frame_picco, frame_inizio, frame_fine).
    """
    pad = int(pad_s * cfg.tracking_hz)
    c = curve.copy()
    out = []
    for _ in range(n):
        pk = int(np.argmax(c))
        if c[pk] <= 0:
            break
        out.append((pk, max(0, pk - pad), min(len(curve), pk + pad)))
        c[max(0, pk - 2 * pad): pk + 2 * pad] = 0   # sopprimi l'intorno
    return out


# ---------------------------------------------------------------------------
# Disegno
# ---------------------------------------------------------------------------

def draw_pitch(ax, lw=1.2, color="#ffffff", alpha=0.55):
    """Campo centrato, 105x68."""
    hx, hy = PITCH_X / 2, PITCH_Y / 2
    k = dict(color=color, lw=lw, alpha=alpha, zorder=2)
    ax.plot([-hx, hx, hx, -hx, -hx], [-hy, -hy, hy, hy, -hy], **k)
    ax.plot([0, 0], [-hy, hy], **k)
    th = np.linspace(0, 2 * np.pi, 80)
    ax.plot(9.15 * np.cos(th), 9.15 * np.sin(th), **k)
    for s in (-1, 1):
        ax.plot([s * hx, s * (hx - 16.5), s * (hx - 16.5), s * hx],
                [-20.16, -20.16, 20.16, 20.16], **k)
        ax.plot([s * hx, s * (hx - 5.5), s * (hx - 5.5), s * hx],
                [-9.16, -9.16, 9.16, 9.16], **k)
        ax.plot([s * hx, s * hx], [-3.66, 3.66], color=color, lw=lw * 2.5,
                alpha=0.9, zorder=3)
    ax.set_xlim(-hx - 3, hx + 3)
    ax.set_ylim(-hy - 3, hy + 3)
    ax.set_aspect("equal")
    ax.axis("off")

def _run_label(row, cfg: DRSConfig) -> str:
    """Etichetta compatta di una corsa attiva: subtype abbreviato + peso."""
    st = str(row[cfg.col_subtype])
    short = {
        "run_ahead_of_the_ball": "run ahead", "pulling_half_space": "pull HS",
        "pulling_wide": "pull wide", "cross_receiver": "cross rec",
        "coming_short": "short", "dropping_off": "drop",
    }.get(st, st)
    return f"{short} {row['w']:.2f}"


def draw_run_panel(ax, act: pd.DataFrame, cfg: DRSConfig, vmax: float):
    """Pannello laterale: una riga per corsa attiva, con la scomposizione dei
    tre fattori che ne determinano lo stress.

        stress = danger_prior x containment_failure x structural_vulnerability

    E' il pezzo che risponde a "perche' QUESTA corsa pesa piu' di quella":
    due corse con lo stesso peso possono averlo per ragioni opposte — una
    perche' e' intrinsecamente pericolosa, l'altra perche' il difensore si e'
    fatto staccare.
    """
    ax.clear()
    ax.set_facecolor("#1b4332")
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    if act.empty:
        ax.text(0.5, 0.5, "nessuna corsa attiva", color="#95a5a6",
                fontsize=10, ha="center", va="center", style="italic")
        return

    act = act.sort_values("w", ascending=False).head(5)
    ax.text(0.0, 0.975, "CORSE ATTIVE", color="white", fontsize=10,
            family="monospace", weight="bold", va="top")

    bars = [("pericolo", "danger_prior", "#e67e22"),
            ("duello", "containment_failure", "#e74c3c"),
            ("struttura", "structural_vulnerability", "#9b59b6")]
    row_h = 0.175
    for i, (_, r) in enumerate(act.iterrows()):
        y = 0.90 - i * row_h
        frac = min(r["w"] / max(vmax, 1e-6), 1.0)
        # pastiglia numerata, stesso numero del cerchio sul campo
        ax.text(0.0, y, f"{i+1}", color="black", fontsize=9, family="monospace",
                weight="bold", va="center", ha="center",
                bbox=dict(boxstyle="circle,pad=0.28", fc="#f1c40f", ec="none"))
        ax.text(0.09, y, f"{r[cfg.col_subtype]}", color="white", fontsize=9,
                family="monospace", va="center")
        ax.text(1.0, y, f"{r['w']:.3f}", color="#f1c40f", fontsize=10,
                family="monospace", weight="bold", va="center", ha="right")
        ax.text(0.09, y - 0.030, str(r.get("player_name", ""))[:22],
                color="#95a5a6", fontsize=7.5, family="monospace", va="center")

        for j, (lbl, col, c) in enumerate(bars):
            by = y - 0.055 - j * 0.026
            v = float(r[col]) if col in r and pd.notna(r[col]) else 0.0
            ax.text(0.09, by, lbl, color="#7f8c8d", fontsize=6.5,
                    family="monospace", va="center")
            ax.add_patch(plt_rect(0.40, by - 0.008, 0.60, 0.016, "#2d3436"))
            ax.add_patch(plt_rect(0.40, by - 0.008, 0.60 * v, 0.016, c))
            ax.text(1.005, by, f"{v:.2f}", color="#95a5a6", fontsize=6.5,
                    family="monospace", va="center", ha="left")


def plt_rect(x, y, w, h, color):
    from matplotlib.patches import Rectangle
    return Rectangle((x, y), w, h, facecolor=color, edgecolor="none")


def animate_window(
    window: tuple[int, int, int],
    runs: pd.DataFrame,
    curve: np.ndarray,
    tracking: Tracking,
    player_teams: dict[int, int],
    defending_team: int,
    out_path: str | Path,
    cfg: DRSConfig = DEFAULT,
    spatial_mode: str = "gap",
    cell: float = 2.0,
    sigma: float = 5.0,
    fps: int = 10,
    vmax: float | None = None,
    title: str = "",
    show_contours: bool = True,
    label_runners: bool = True,
):
    """Anima una finestra e la salva (.mp4 se ffmpeg c'e', altrimenti .gif).

    Layout: campo con heatmap e colorbar a sinistra, pannello delle corse
    attive a destra, curva DRS(t) con cursore in basso.

    `vmax` fissa la scala della heatmap: PASSALO SEMPRE uguale fra i clip
    dello stesso match (lo fa `animate_match`), altrimenti un momento
    tranquillo normalizzato su se stesso sembra un assedio.

    `show_contours` disegna il perimetro del contributo di OGNI singola
    corsa prima della somma: senza, due corse vicine si fondono in una
    macchia unica e non si distinguono.
    `label_runners` numera i runner attivi sul campo, con gli stessi numeri
    del pannello laterale.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

    pk, f0, f1 = window
    step = max(1, cfg.tracking_hz // fps)
    frames = list(range(f0, f1, step))

    xs = np.arange(-PITCH_X / 2, PITCH_X / 2 + cell, cell)
    ys = np.arange(-PITCH_Y / 2, PITCH_Y / 2 + cell, cell)
    grid = np.meshgrid(xs, ys)

    acts = [active_weights(runs, f, cfg) for f in frames]
    fields = [stress_field(f, a, tracking, grid, cfg, spatial_mode, sigma)
              for f, a in zip(frames, acts)]
    if vmax is None:
        vmax = max(1e-6, max(fl.max() for fl in fields))

    fig = plt.figure(figsize=(15.5, 8.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[4, 1], width_ratios=[3.1, 1],
                          hspace=0.10, wspace=0.04)
    axp = fig.add_subplot(gs[0, 0])
    axr = fig.add_subplot(gs[0, 1])
    axc = fig.add_subplot(gs[1, :])
    fig.patch.set_facecolor("#1b4332")
    axp.set_facecolor("#2d6a4f")

    heat = axp.imshow(fields[0], extent=[xs[0], xs[-1], ys[0], ys[-1]],
                      origin="lower", cmap="inferno", alpha=0.75,
                      vmin=0, vmax=vmax, zorder=1, interpolation="bilinear")
    draw_pitch(axp)

    # --- colorbar: la legenda della heatmap
    cax = axp.inset_axes([1.012, 0.06, 0.020, 0.60])
    cb = fig.colorbar(heat, cax=cax)
    cb.ax.tick_params(colors="white", labelsize=7, pad=2)
    cb.outline.set_edgecolor("#4a7c59")
    cb.set_ticks([0, vmax / 2, vmax])
    cb.set_ticklabels(["0", f"{vmax/2:.2f}", f"{vmax:.2f}"])
    cax.set_title("stress\nlocale", color="white", fontsize=7.5,
                  family="monospace", pad=6)
    # il nero non e' "difesa in controllo": e' assenza di misura
    cax.text(0.5, -0.055, "nero =\nnon misurato", transform=cax.transAxes,
             color="#95a5a6", fontsize=6, ha="center", va="top",
             style="italic", linespacing=1.4)

    sc_att = axp.scatter([], [], s=120, c=ATT, ec="white", lw=1.2, zorder=5,
                         label="in possesso")
    sc_def = axp.scatter([], [], s=120, c=DEF_, ec="white", lw=1.2, zorder=5,
                         label="difende")
    sc_ball = axp.scatter([], [], s=55, c=BALL, ec="black", lw=0.8, zorder=6,
                          label="palla")
    leg = axp.legend(loc="lower right", fontsize=8, framealpha=0.25,
                     facecolor="#1b4332", edgecolor="#4a7c59", labelcolor="white")
    leg.set_zorder(9)

    overlay = []   # artisti ridisegnati a ogni frame
    txt = axp.text(0.012, 0.975, "", transform=axp.transAxes, color="white",
                   fontsize=10.5, va="top", family="monospace", zorder=8,
                   bbox=dict(boxstyle="round,pad=0.45", fc="#1b4332", ec="#4a7c59",
                             alpha=0.80))
    axp.set_title(title or f"DRS(t) — picco al frame {pk}", color="white", fontsize=12)

    t = np.arange(f0, f1) / cfg.tracking_hz / 60
    axc.plot(t, curve[f0:f1], color=HEAT, lw=1.6)
    axc.fill_between(t, curve[f0:f1], color=HEAT, alpha=0.25)
    cursor = axc.axvline(t[0], color="white", lw=1.4)
    axc.set_facecolor("#1b4332")
    axc.tick_params(colors="white", labelsize=8)
    for sp in axc.spines.values():
        sp.set_color("#4a7c59")
    axc.set_xlabel("minuti", color="white", fontsize=9)
    axc.set_ylabel("DRS(t)", color="white", fontsize=9)
    axc.set_xlim(t[0], t[-1])

    def update(i):
        f, act = frames[i], acts[i]
        heat.set_data(fields[i])
        pos = tracking.at(tracking.nearest_frame(f) or -1)
        a = np.array([p for q, p in pos.items() if player_teams.get(q) != defending_team]) \
            if pos else np.empty((0, 2))
        d = np.array([p for q, p in pos.items() if player_teams.get(q) == defending_team]) \
            if pos else np.empty((0, 2))
        sc_att.set_offsets(a if len(a) else np.empty((0, 2)))
        sc_def.set_offsets(d if len(d) else np.empty((0, 2)))
        b = tracking.ball.get(tracking.nearest_frame(f) or -1)
        sc_ball.set_offsets([[b[0], b[1]]] if b else np.empty((0, 2)))

        for art in overlay:
            art.remove()
        overlay.clear()

        ordered = act.sort_values("w", ascending=False)
        for k, (_, r) in enumerate(ordered.iterrows()):
            rid, did = int(r["player_id"]), r[cfg.col_defender]
            if rid not in pos:
                continue
            rx, ry = pos[rid]

            # contorno del contributo della SINGOLA corsa, prima della somma
            if show_contours:
                fl = stress_field(f, ordered.iloc[[k]], tracking, grid,
                                  cfg, spatial_mode, sigma)
                if fl.max() > 0.04 * vmax:
                    cs = axp.contour(grid[0], grid[1], fl,
                                     levels=[0.28 * fl.max()],
                                     colors="#f1c40f", linewidths=1.1,
                                     alpha=0.75, zorder=4)
                    overlay.extend(cs.collections if hasattr(cs, "collections") else [cs])

            if pd.notna(did) and int(did) in pos:
                dx, dy = pos[int(did)]
                overlay.append(axp.plot(
                    [dx, rx], [dy, ry], color="white", lw=1.0, ls="--",
                    alpha=0.30 + 0.55 * min(r["w"] / max(vmax, 1e-6), 1),
                    zorder=4)[0])

            # pastiglia numerata sul runner, stesso numero del pannello
            if label_runners and k < 5:
                overlay.append(axp.text(
                    rx, ry + 2.6, f"{k+1}", color="black", fontsize=8.5,
                    family="monospace", weight="bold", ha="center", va="center",
                    zorder=8,
                    bbox=dict(boxstyle="circle,pad=0.26", fc="#f1c40f", ec="black", lw=0.6)))
                overlay.append(axp.text(
                    rx, ry - 3.4, _run_label(r, cfg), color="#f1c40f", fontsize=7.5,
                    family="monospace", ha="center", va="center", zorder=8,
                    bbox=dict(boxstyle="round,pad=0.22", fc="#1b4332", ec="none", alpha=0.75)))

        draw_run_panel(axr, act, cfg, vmax)
        txt.set_text(f"frame {f}   {f/cfg.tracking_hz/60:5.2f}'\n"
                     f"DRS(t) {curve[f]:.3f}")
        cursor.set_xdata([f / cfg.tracking_hz / 60] * 2)
        return [heat, sc_att, sc_def, sc_ball, txt, cursor, *overlay]

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=False)
    out_path = Path(out_path)
    try:
        anim.save(out_path.with_suffix(".mp4"), writer=FFMpegWriter(fps=fps, bitrate=2800))
        saved = out_path.with_suffix(".mp4")
    except Exception:
        saved = out_path.with_suffix(".gif")
        anim.save(saved, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return saved

def animate_match(
    match_id: int,
    runs_parquet: str | Path,
    curves_npz: str | Path,
    tracking_path: str | Path,
    out_dir: str | Path,
    n_windows: int = 5,
    cfg: DRSConfig = DEFAULT,
    spatial_mode: str = "gap",
    pad_s: float = 5.0,
    fps: int = 10,
):
    """Entrypoint: genera i clip delle n finestre piu' calde di un match.

    La scala della heatmap (`vmax`) e' calcolata una volta su tutte le
    finestre e riusata, cosi' i clip sono confrontabili fra loro.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = pd.read_parquet(runs_parquet)
    runs = runs[runs[cfg.col_match] == match_id].copy()
    if runs.empty:
        raise ValueError(f"nessuna corsa per il match {match_id} in {runs_parquet}")
    curve = np.load(curves_npz)[str(match_id)]

    wins = find_peak_windows(curve, n_windows, pad_s, cfg)
    lo = min(w[1] for w in wins) - cfg.tracking_hz
    hi = max(w[2] for w in wins) + cfg.tracking_hz
    print(f"[animate] {len(wins)} finestre, frame {lo}-{hi}; carico il tracking…")
    tracking = load_tracking(tracking_path, (lo, hi))
    print(f"[animate] {len(tracking.frames)} frame tracciati nel range")

    teams = infer_player_teams(runs, tracking, cfg)
    defending = int(runs["defending_team_id"].mode().iloc[0])

    # scala comune a tutti i clip
    xs = np.arange(-PITCH_X / 2, PITCH_X / 2 + 2.0, 2.0)
    ys = np.arange(-PITCH_Y / 2, PITCH_Y / 2 + 2.0, 2.0)
    grid = np.meshgrid(xs, ys)
    peak_vals = [stress_field(pk, active_weights(runs, pk, cfg), tracking, grid,
                              cfg, spatial_mode).max() for pk, _, _ in wins]
    vmax = max(peak_vals) if peak_vals else 1.0
    print(f"[animate] scala heatmap comune vmax={vmax:.3f}")

    saved = []
    for i, w in enumerate(wins, 1):
        pk = w[0]
        d = runs[(runs[cfg.col_frame_start] <= pk) & (runs[cfg.col_frame_end] >= pk)]
        lbl = "+".join(sorted(d[cfg.col_subtype].unique())) or "—"
        p = animate_window(
            w, runs, curve, tracking, teams, defending,
            out_dir / f"drst_{match_id}_w{i}_min{pk/cfg.tracking_hz/60:.0f}",
            cfg, spatial_mode, fps=fps, vmax=vmax,
            title=f"DRS(t) {match_id} — min {pk/cfg.tracking_hz/60:.1f}  "
                  f"DRS={curve[pk]:.2f}  [{lbl}]",
        )
        print(f"[animate] {i}/{len(wins)}  {p.name}")
        saved.append(p)
    return saved

# ===========================================================================
# 12. CONFRONTO fra configurazioni
# ===========================================================================

CONFIGS = [
    ("A  riferimento", separation_gain_containment, False),
    ("B  + struttura tracking", separation_gain_containment, True),
    ("C  + curva sep(t)", separation_curve_containment, False),
    ("D  completo", separation_curve_containment, True),
]

# ESITO sui 20 match (11 ottobre), con prior spaziale:
#   A  OR 1.165   stress_per_run 0.628   containment +0.026
#   B  OR 1.148   stress_per_run 0.637   containment +0.026   <- scelta
#   C  OR 1.165   stress_per_run 0.607   containment -0.134
#   D  OR 1.148   stress_per_run 0.622   containment -0.134
# La curva sep(t) e' stata quindi disattivata di default. La struttura da
# tracking resta, con un guadagno piccolo (+0.009 su stress_per_run).


def compare(
    layer3_path: str | Path,
    tracking_features_path: str | Path,
    out_dir: str | Path,
    cfg: DRSConfig = DEFAULT,
    with_oos: bool = False,
) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = load_layer3(layer3_path, cfg)
    feats = pd.read_csv(tracking_features_path)
    merged = merge_into_layer3(base, feats, cfg)

    cov = merged["sep_min"].notna().mean() if "sep_min" in merged else 0.0
    print(f"Layer 3: {len(merged)} corse, feature di tracking su {cov:.1%}")
    if cov < 0.5:
        print("[!] copertura sotto il 50%: le corse senza curva ricadono sul "
              "containment classico, il confronto e' indebolito. Estrarre le "
              "feature su piu' match prima di decidere.")
    merged_path = out_dir / "_layer3_tracking.csv"
    merged.to_csv(merged_path, index=False)

    rows = []
    for label, containment, use_trk in CONFIGS:
        print(f"\n{'='*64}\n{label}\n{'='*64}")
        res = run_pipeline(
            merged_path,
            cfg=replace(cfg, use_tracking_structure=use_trk),
            containment=containment,
            with_validation=True,
            with_oos=with_oos,
            verbose=True,
        )
        res.save(out_dir / label.split()[0])

        v = res.validation.set_index("outcome")
        s = res.stability.set_index("metric")
        rows.append({
            "config": label,
            "OR_shot": v.loc[cfg.danger_outcome, "odds_ratio"],
            "p_shot": v.loc[cfg.danger_outcome, "p_value"],
            "OR_targeted": v.loc["targeted", "odds_ratio"],
            "p_targeted": v.loc["targeted", "p_value"],
            "OR_received": v.loc["received", "odds_ratio"],
            "p_received": v.loc["received", "p_value"],
            "sh_stress_per_run": s.loc["stress_per_run", "median"],
            "sh_containment": s.loc["containment_failure_mean", "median"],
            "sh_drs_supported": s.loc["drs_mean_supported", "median"],
        })

    tab = pd.DataFrame(rows)
    tab.to_csv(out_dir / "confronto.csv", index=False)

    print(f"\n\n{'='*72}\nCONFRONTO\n{'='*72}")
    print(tab.round(4).to_string(index=False))
    print("\nLettura:")
    print("  sh_stress_per_run  deve salire sopra il riferimento di A")
    print("  sh_containment     fermo a ~0.03 in A: e' quello che la curva")
    print("                     sep(t) dovrebbe sbloccare")
    print("  p_targeted e p_received DEVONO restare non significativi:")
    print("                     se scendono sotto 0.05 la configurazione sta")
    print("                     reintroducendo la scelta dell'attaccante")
    print(f"\nArtefatti in {out_dir.resolve()}")
    return tab

# ===========================================================================
# 13. MAIN
# ===========================================================================

def main():
    cfg = replace(DEFAULT, prior_kind=PRIOR_KIND, geo_prior_scale=GEO_SCALE,
                  geo_use_endpoint=GEO_USE_ENDPOINT,
                  geo_ball_gate=GEO_BALL_GATE,
                  use_tracking_structure=USE_TRACKING_STRUCTURE,
                  use_curve_containment=USE_CURVE_CONTAINMENT,
                  containment_kind=CONTAINMENT_KIND)

    layer3 = LAYER3_PATH
    if TRACKING_FEATURES is not None:
        merged = merge_into_layer3(load_layer3(LAYER3_PATH, cfg),
                                   pd.read_csv(TRACKING_FEATURES), cfg)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        layer3 = OUT_DIR / "_layer3_tracking.csv"
        merged.to_csv(layer3, index=False)
        cov = merged["sep_min"].notna().mean() if "sep_min" in merged else 0.0
        print(f"Feature di tracking unite: copertura {cov:.1%}")
        if GEO_USE_ENDPOINT:
            gc = (merged["geo_dist_min"].notna().mean()
                  if "geo_dist_min" in merged else 0.0)
            if gc == 0.0:
                print("[!] geo_dist_min assente: il prior usa la sola "
                      "partenza. Ri-estrarre le feature di tracking con la "
                      "versione aggiornata per attivare il punto di arrivo.")
            else:
                print(f"Punto di arrivo su {gc:.1%} delle corse "
                      f"(avvicinamento medio alla porta "
                      f"{merged['geo_progress'].mean():+.1f} m)")

    res = run_pipeline(layer3, cfg=cfg, structural=STRUCTURAL, with_oos=RUN_OOS)
    out = res.save(OUT_DIR)
    if res.two_axis is not None:
        print("\nProfilo a due assi (alto = peggio su entrambi gli assi)")
        cols = ["defending_team_id", "n_matches", "n_runs", "esposizione",
                "cedimento", "deep_phase_share"]
        print(res.two_axis[cols].round(3).to_string(index=False))
    print(f"\nSalvato in {out.resolve()}")

    if PLOT_MATCH is not None:
        try:
            import matplotlib
        except ImportError:
            print("[!] matplotlib non installato: figura saltata.")
        else:
            matplotlib.use("Agg")
            curve, _ = res.curves_by_match[PLOT_MATCH]
            runs = res.runs[res.runs[cfg.col_match] == PLOT_MATCH]
            fig = plot_match_curve(curve, runs, cfg, title=f"DRS(t) - match {PLOT_MATCH}")
            p = out / f"curve_{PLOT_MATCH}.png"
            fig.savefig(p, dpi=140, bbox_inches="tight")
            print(f"Figura: {p.resolve()}")

    if ANIMATE_MATCH is not None:
        animate_match(ANIMATE_MATCH, out / "runs_with_stress.parquet",
                      out / "curves.npz", TRACKING_PATH, out / "clips",
                      n_windows=N_WINDOWS, cfg=cfg,
                      spatial_mode=SPATIAL_MODE, fps=FPS)
    return res


if __name__ == "__main__":
    main()
