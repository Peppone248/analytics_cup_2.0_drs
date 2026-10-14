"""Aggregazione del DRS(t) a livello di squadra e di difensore.

AVVERTENZA METODOLOGICA, da rispettare in ogni nuova metrica aggiunta qui.

Il DRS(t) medio di un match correla 0.83 con il numero di corse subite. Una
metrica non normalizzata per volume misura "quanto ti attaccano", non "quanto
difendi bene". Verifica empirica:

    n_runs                      split-half +0.70
    drs_mean                    split-half +0.53   <- quasi tutto volume
    drs_mean residualizzato     split-half -0.15   <- non resta niente
    stress_per_run              split-half +0.44   <- qualita' vera

Ogni metrica qui e' etichettata VOLUME o QUALITA'. Per i profili di squadra
usare solo le QUALITA'.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DRSConfig, DEFAULT


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
