"""
drst.py — DRS(t): Defensive Response Signature come curva temporale.
Versione single-file. Equivalente al package drst/, nessuna installazione.

    v0  stress = danger_prior(subtype) x containment_failure
    v1  stress = danger_prior(subtype, zona) x containment_failure
                 x structural_vulnerability          [richiede enriched]

run_pipeline rileva le colonne e attiva la v1 da sola su layer3_enriched.csv.
Lo stress NON usa mai gli esiti (targeted/received/lead_to_shot): restano
fuori per la validazione.

USO IN VS CODE
    1. metti questo file accanto ai dati
    2. imposta LAYER3_PATH qui sotto
    3. F5 (o: python drst.py)

Oppure da un notebook / altro script nella stessa cartella:
    from drst import run_pipeline
    res = run_pipeline("data/layer3_enriched.csv")
    res.save("out/drst_v1")

Dipendenze: pandas numpy scipy statsmodels scikit-learn pyarrow
            matplotlib (solo per le figure)

INDICE (Ctrl+F sui marcatori ===)
    1. CONFIGURAZIONE — ogni parametro e' una scelta di modellazione
    2. PRIOR DI PERICOLOSITA' — per subtype e spaziale (subtype x zona)
    3. STRESS per singola corsa — containment + vulnerabilita' strutturale
    4. FRAME-BASED INDEXING STRATEGY — punto di estensione centrale
    5. COSTRUZIONE DELLA CURVA DRS(t)
    6. AGGREGAZIONE per squadra e per difensore
    7. VALIDAZIONE — dissociazione, split-half, sensibilita'
    8. PIPELINE end-to-end
    9. FIGURE
    10. MAIN

Risultati di riferimento, limiti noti ed estensioni previste: vedi README.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field, replace
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

# ====== CONFIGURA QUI ======================================================
LAYER3_PATH = Path("data\matches\layer3_enriched.csv")
OUT_DIR = Path("out/drst_v1")
PLOT_MATCH = 2007721          # match_id della figura, None per saltarla
RUN_OOS = False               # dissociazione out-of-sample (piu' lenta)
SPATIAL_PRIOR = None          # None = auto, False = forza prior per subtype
STRUCTURAL = None             # None = auto, False = disattiva la struttura
# ===========================================================================


# ===========================================================================
# 1. CONFIGURAZIONE — ogni parametro e' una scelta di modellazione
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
    outcomes: tuple[str, ...] = ("targeted", "received", "lead_to_shot")
    danger_outcome: str = "lead_to_shot"

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT = DRSConfig()

# ===========================================================================
# 2. PRIOR DI PERICOLOSITA' — per subtype e spaziale (subtype x zona)
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

# ===========================================================================
# 3. STRESS per singola corsa — containment + vulnerabilita' strutturale
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


def separation_curve_containment(df, frames_loader, cfg: DRSConfig = DEFAULT):
    """STUB — containment dalla curva separation(t) intera.

    Oggi usiamo `separation_gain = end - start`: due punti su una traiettoria
    di 25-35 frame. La curva completa e' gia' calcolabile con la pipeline
    `compute_separation(frames, run)` esistente nella repo.

    Feature che `gain` nasconde e che questa implementazione deve estrarre:
      - separation minima e IN QUALE FRAME avviene (perdere 3 m all'inizio e
        recuperarli != perderli all'ultimo secondo)
      - pendenza nel momento critico (derivata al minimo)
      - area sotto la curva sotto una soglia ("secondi passati a < 3 m")
      - numero di inversioni (recuperato e riperso?)

    Massimo guadagno per minimo sforzo fra le estensioni previste: il codice
    di calcolo della curva esiste gia'.
    """
    raise NotImplementedError(
        "separation_curve_containment: collegare compute_separation() dalla repo."
    )


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

    w = (cfg.w_delta_to_line, cfg.w_overload, cfg.w_zone)
    W = sum(w)
    return (
        (v_delta.clip(0.01, 1.0) ** w[0]) * (v_over ** w[1]) * (v_zone ** w[2])
    ) ** (1.0 / W)


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
# 4. FRAME-BASED INDEXING STRATEGY — punto di estensione centrale
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
# 5. COSTRUZIONE DELLA CURVA DRS(t)
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
# 6. AGGREGAZIONE per squadra e per difensore
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
    rows = []
    for (mid, team), (curve, idx) in curves.items():
        g = df[(df[cfg.col_match] == mid) & (df["defending_team_id"] == team)]
        m = idx.masked(curve)
        rows.append(
            {
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
                # --- diagnostica
                "coverage": idx.coverage(),
                "n_frames": idx.n_frames,
            }
        )
    return pd.DataFrame(rows)


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
# 7. VALIDAZIONE — dissociazione, split-half, sensibilita'
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
    spatial_prior: bool = False,
    structural: bool = False,
    zone_col: str = "third_start",
) -> pd.DataFrame:
    """Dissociazione out-of-sample: prior, normalizzazione della separation e
    ranghi strutturali stimati SOLO sul train fold, DRS(t) valutato sul fold
    tenuto fuori.

    GroupKFold su match_id. E' la versione da citare nel paper: la variante
    in-sample stima tutto sulle stesse run su cui valida.
    """
    from sklearn.model_selection import GroupKFold

    from .priors import outcome_priors, spatial_priors, map_spatial_prior
    from .stress import compute_stress, structural_vulnerability
    from .curve import build_all_curves

    gkf = GroupKFold(n_splits=n_splits)
    parts = []
    for tr, va in gkf.split(df, df[cfg.danger_outcome], df[cfg.col_match]):
        train, valid = df.iloc[tr], df.iloc[va]

        if spatial_prior:
            cells, fb = spatial_priors(train, cfg, zone_col)
            pri = map_spatial_prior(valid, cells, fb, cfg, zone_col)
        else:
            pri, _ = outcome_priors(train, cfg)

        st = train.groupby(cfg.col_subtype)[cfg.col_sep_gain].agg(["mean", "std"])

        def _containment(d, c=cfg, _st=st):
            mu = d[c.col_subtype].map(_st["mean"])
            sd = d[c.col_subtype].map(_st["std"]).replace(0, np.nan)
            z = (d[c.col_sep_gain] - mu) / sd
            return (1.0 / (1.0 + np.exp(-z))).fillna(0.5)

        # ranghi strutturali calcolati sulla distribuzione del TRAIN
        sv = structural_vulnerability(valid, cfg, reference=train) if structural else False

        v = compute_stress(valid, pri, containment=_containment, cfg=cfg, structural=sv)
        curves = build_all_curves(v, cfg, strategy)
        parts.append(pre_run_drs(v, curves, cfg, carry=[zone_col] if spatial_prior else None))

    return dissociation(
        pd.concat(parts, ignore_index=True),
        cfg,
        extra_controls=[zone_col] if spatial_prior else None,
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
            continue
        c = a.loc[common].corr(b.loc[common])
        if not np.isnan(c):
            corrs.append(c)

    corrs = np.array(corrs)
    return {
        "metric": metric,
        "n_teams": len(ok),
        "n_valid_iter": len(corrs),
        "median": float(np.median(corrs)) if len(corrs) else np.nan,
        "iqr_lo": float(np.percentile(corrs, 25)) if len(corrs) else np.nan,
        "iqr_hi": float(np.percentile(corrs, 75)) if len(corrs) else np.nan,
        "frac_positive": float((corrs > 0).mean()) if len(corrs) else np.nan,
        "median_teams_per_split": int(np.median(n_common)) if n_common else 0,
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
    from dataclasses import replace

    from .stress import compute_stress
    from .curve import build_all_curves

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
# 8. PIPELINE end-to-end
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
    containment: ContainmentModel = separation_gain_containment,
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
    containment : modello di containment failure. Sostituire qui per passare
        a separation_curve_containment o awareness_containment.
    indexing : strategia di indicizzazione dei frame.
        'event_anchored' (attiva) | 'dense_tracking' | 'phase_window' (stub).
    spatial_prior : prior danger(subtype, zona) invece che per solo subtype.
        None = automatico (attivo se la colonna di zona esiste).
        Richiede layer3_enriched.
    structural : terzo fattore di vulnerabilita' strutturale nello stress.
        None = automatico (attivo se le colonne enriched esistono).
    with_oos : esegue anche la dissociazione out-of-sample (5-fold su match).
        Piu' lenta, ma e' la versione da citare nel paper.
    """
    log = print if verbose else (lambda *a, **k: None)
    strategy: IndexingStrategy = get_strategy(indexing)

    df = load_layer3(layer3_path, cfg)
    log(f"Layer 3: {len(df)} run su {df[cfg.col_match].nunique()} match")

    if spatial_prior is None:
        spatial_prior = zone_col in df.columns
    if structural is None:
        structural = has_structural_columns(df)

    if spatial_prior:
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

    if with_validation:
        extra = [zone_col] if spatial_prior else None
        if with_oos:
            res.validation = dissociation_oos(
                df, cfg, strategy=strategy, spatial_prior=spatial_prior,
                structural=structural, zone_col=zone_col,
            )
        else:
            pre = pre_run_drs(df, curves_match, cfg, carry=extra)
            res.validation = dissociation(pre, cfg, extra_controls=extra)
        log("\nDissociazione" + (" (out-of-sample)" if with_oos else " (in-sample)"))
        for _, r in res.validation.iterrows():
            log(f"  {r.outcome:14s} OR={r.odds_ratio:.3f} "
                f"[{r.ci_lo:.3f}, {r.ci_hi:.3f}]  p={r.p_value:.2e}")

        res.stability = pd.DataFrame([
            split_half(tm, m)
            for m in ["n_runs", "drs_mean_all", "stress_per_run",
                      "drs_mean_supported", "containment_failure_mean"]
        ])
        log("\nSplit-half stability (squadre con >=3 match)")
        for _, r in res.stability.iterrows():
            log(f"  {r.metric:26s} mediana={r['median']:+.3f} "
                f"IQR=[{r.iqr_lo:+.3f},{r.iqr_hi:+.3f}] pos={r.frac_positive:.0%}")

    return res

# ===========================================================================
# 9. FIGURE
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
# 10. MAIN
# ===========================================================================

def main():
    res = run_pipeline(LAYER3_PATH, cfg=DEFAULT, spatial_prior=SPATIAL_PRIOR,
                       structural=STRUCTURAL, with_oos=RUN_OOS)
    out = res.save(OUT_DIR)
    print(f"\nSalvato in {out.resolve()}")

    if PLOT_MATCH is not None:
        import matplotlib
        matplotlib.use("Agg")
        curve, _ = res.curves_by_match[PLOT_MATCH]
        runs = res.runs[res.runs[DEFAULT.col_match] == PLOT_MATCH]
        fig = plot_match_curve(curve, runs, DEFAULT, title=f"DRS(t) - match {PLOT_MATCH}")
        p = out / f"curve_{PLOT_MATCH}.png"
        fig.savefig(p, dpi=140, bbox_inches="tight")
        print(f"Figura: {p.resolve()}")
    return res


if __name__ == "__main__":
    main()
