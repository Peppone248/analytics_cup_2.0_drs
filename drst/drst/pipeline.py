"""Pipeline end-to-end: dal Layer 3 alle curve e alle metriche.

Punto di ingresso unico. `run_pipeline()` esegue in ordine:
    prior -> stress -> curve -> metriche di squadra -> validazione
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DRSConfig, DEFAULT, FAMILY_MAP
from .priors import (outcome_priors, spatial_priors, map_spatial_prior,
                     geometric_priors)
from .stress import (compute_stress, separation_gain_containment,
                     structural_vulnerability, has_structural_columns,
                     separation_curve_containment, has_curve_columns,
                     learned_containment, has_learned_columns,
                     fixed_containment, ContainmentModel)
from .curve import build_all_curves
from .indexing import IndexingStrategy, get_strategy
from .aggregate import (add_defending_team, team_match_metrics, team_profile,
                        team_two_axis)
from . import validate as V


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
            res.validation = V.dissociation_oos(
                df, cfg, strategy=strategy, structural=structural,
                zone_col=zone_col, prior_kind=kind, containment=containment,
            )
        else:
            pre = V.pre_run_drs(df, curves_match, cfg, carry=extra)
            res.validation = V.dissociation(pre, cfg, extra_controls=extra)
        log("\nDissociazione" + (" (out-of-sample)" if with_oos else " (in-sample)"))
        for _, r in res.validation.iterrows():
            log(f"  {r.outcome:14s} OR={r.odds_ratio:.3f} "
                f"[{r.ci_lo:.3f}, {r.ci_hi:.3f}]  p={r.p_value:.2e}")

        metrics = ["n_runs", "drs_mean_all", "stress_per_run",
                   "drs_mean_supported", "containment_failure_mean", "prior_mean"]
        metrics += [m for m in ("stress_per_run_phase_adj", "containment_phase_adj",
                                "deep_phase_share") if m in tm.columns]
        res.stability = pd.DataFrame([V.split_half(tm, m) for m in metrics])
        log("\nSplit-half stability (squadre con >=3 match)")
        log(f"  {'metrica':28s} {'mediana':>8s} {'IQR':>17s} {'pos':>5s} {'Sp.-Brown':>10s}")
        for _, r in res.stability.iterrows():
            log(f"  {r.metric:28s} {r['median']:+8.3f} "
                f"[{r.iqr_lo:+.3f},{r.iqr_hi:+.3f}] {r.frac_positive:5.0%} "
                f"{r.spearman_brown:+10.3f}")
        log("  Spearman-Brown = affidabilita' sui 20 match interi; la mediana e'")
        log("  misurata su meta' campione, quindi sottostima quella che interessa.")

    return res
