"""Validazione del DRS(t).

Due test, con scopi diversi.

1. DISSOCIAZIONE (validate.dissociation)
   Il DRS(t) misurato PRIMA di una corsa predice che quella corsa porti a un
   tiro, ma non predice che venga mirata o ricevuta.
   E' il test che dimostra che il DRS(t) misura lo STATO DIFENSIVO e non la
   SCELTA DELL'ATTACCANTE — il difetto che aveva affondato il modello di
   stadio 1 (p(targeted|run)).

   Risultato di riferimento (8877 run, 20 match, controllando per subtype):
       targeted      OR = 0.99  p = 0.59   nessun effetto
       received      OR = 1.03  p = 0.29   nessun effetto
       lead_to_shot  OR = 1.26  p = 1e-16  effetto forte
   Out-of-sample (prior e normalizzazione stimati sul solo train fold):
       lead_to_shot  OR = 1.25  p = 9e-16

2. SPLIT-HALF STABILITY (validate.split_half)
   Una metrica aggregata per squadra e' riproducibile su due meta' casuali
   dei match?
   Riferimento: stress_per_run +0.44 (80% degli split positivi), contro
   -0.13 dei residui del modello di stadio 1.
   ATTENZIONE: drs_mean grezzo da +0.53 ma correla 0.83 con il numero di
   corse subite — quella stabilita' e' quasi tutta volume, non qualita'.
   Residualizzando sul volume crolla a -0.15. Usare sempre metriche
   normalizzate per volume.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DRSConfig, DEFAULT


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

    from .priors import (outcome_priors, spatial_priors, map_spatial_prior,
                         geometric_priors)
    from .stress import (separation_gain_containment,
                         separation_curve_containment)
    from .stress import compute_stress, structural_vulnerability
    from .curve import build_all_curves

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
