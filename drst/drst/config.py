"""Parametri del modello DRS(t). Un unico punto di verita'.

Ogni parametro qui e' una scelta di modellazione, non un dettaglio tecnico:
va citato nel paper e versionato insieme ai risultati.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field


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
