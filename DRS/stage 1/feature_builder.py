"""
DRS — Stadio 1: p(targeted | run)
Baseline runnable per capire se lo stadio 1 ha segnale.

Modifiche rispetto alla v0:
  1. Feature set rivisto:
     - RIMOSSO separation_start (feature individuale, va allo stadio 2)
     - AGGIUNTO pp_n_passing_options (portante, verifiche 07 ott)
     - AGGIUNTO pp_pressure, pp_reception_difficulty come ordinali 0-4
     - AGGIUNTO line_height derivata (last_defensive_line_x_start
       normalizzata per attacking_side)
  2. Training:
     - RIMOSSO class_weight="balanced" (sabotava la calibrazione)
     - AGGIUNTA calibrazione isotonica dentro ogni fold su subset di train
  3. Diagnostics:
     - AGGIUNTO split-half stability check (test decisivo del DRS)
     - AGGIUNTO warning su subtype non mappati scartati silenziosamente
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb
import scipy.stats as stats

# ---- paths -----------------------------------------------------------------
DATA = Path("data\matches\layer3_enriched.csv")

# ---- domain maps -----------------------------------------------------------
FAMILY_MAP = {
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

PRESSURE_MAP = {
    "no_pressure": 0, "low_pressure": 1, "medium_pressure": 2,
    "high_pressure": 3, "very_high_pressure": 4,
}

RECEPTION_DIFF_MAP = {
    "very_easy": 0, "easy": 1, "medium": 2, "hard": 3, "very_hard": 4,
}

PITCH_LENGTH = 105.0  # metri, scala tracking SkillCorner


def collapse_channel(s: pd.Series) -> pd.Series:
    """5 livelli -> 3: center / half_space / wide."""
    mapping = {
        "left_wide": "wide", "right_wide": "wide",
        "left_half_space": "half_space", "right_half_space": "half_space",
        "center": "center",
    }
    return s.map(mapping).fillna("unknown")


def compute_line_height(df: pd.DataFrame) -> pd.Series:
    """Altezza linea normalizzata: 0 = sulla propria porta, 105 = porta
    avversaria. Serve per confrontare squadre indipendentemente dal lato.

    attacking_side == 'right' -> porta avversaria a x=105 -> line_height = x
    attacking_side == 'left'  -> porta avversaria a x=0   -> line_height = 105-x
    """
    x = df["last_defensive_line_x_start"]
    side = df["attacking_side"]
    return np.where(side == "right", x, PITCH_LENGTH - x)


# ---- feature builder -------------------------------------------------------
def build_stage1(df: pd.DataFrame, families=None):
    df = df.copy()

    # PATCH: warning esplicito sui subtype non mappati
    all_subtypes = set(df["event_subtype"].dropna().unique())
    mapped = set(FAMILY_MAP.keys())
    unmapped = all_subtypes - mapped
    if unmapped:
        n_dropped = df["event_subtype"].isin(unmapped).sum()
        print(f"WARNING: {n_dropped} run scartate con subtype non mappati: {unmapped}")

    df = df[df["event_subtype"].isin(FAMILY_MAP.keys())].copy()
    df["subtype_family"] = df["event_subtype"].map(FAMILY_MAP)
    if families is not None:
        df = df[df["subtype_family"].isin(families)]

    # channel
    df["channel_3"] = (
        collapse_channel(df["channel_start"])
        if "channel_start" in df.columns else "unknown"
    )

    # PATCH: feature derivata line_height
    if "last_defensive_line_x_start" in df.columns and "attacking_side" in df.columns:
        df["line_height"] = compute_line_height(df)
    else:
        print("WARNING: line_height non calcolabile (colonne mancanti)")
        df["line_height"] = np.nan

    # PATCH: pp_pressure e pp_reception_difficulty come ordinali
    if "pp_pressure" in df.columns:
        df["pp_pressure_ord"] = df["pp_pressure"].map(PRESSURE_MAP)
    if "pp_reception_difficulty" in df.columns:
        df["pp_reception_difficulty_ord"] = df["pp_reception_difficulty"].map(RECEPTION_DIFF_MAP)

    numeric = [
        # PATCH: separation_start RIMOSSA (va allo stadio 2)
        "delta_to_last_defensive_line_start",
        "n_opponents_ahead_start",
        "n_simultaneous_runs",
        # PATCH: feature portanti deterrenza collettiva
        "pp_n_passing_options",
        "pp_pressure_ord",
        "pp_reception_difficulty_ord",
        "line_height",
    ]
    binary = ["inside_defensive_shape_start", "penalty_area_start"]
    categ  = ["channel_3", "team_out_of_possession_phase_type", "subtype_family"]

    numeric = [c for c in numeric if c in df.columns]
    binary  = [c for c in binary  if c in df.columns]

    X = pd.concat([
        df[numeric].astype(float),
        df[binary].astype("Int64").astype(float),
        pd.get_dummies(df[categ], drop_first=False, dummy_na=True),
    ], axis=1)

    y      = df["targeted"].astype(int).reset_index(drop=True)
    groups = df["match_id"].reset_index(drop=True)
    fam    = df["subtype_family"].reset_index(drop=True)

    # PATCH: team difendente per split-half check
    # team_id del CSV e' quello dell'attaccante -> defending_team_id e' il complementare
    # se la colonna defending_team_id esiste (aggiunta da build_full_dataset.py) usa quella
    if "defending_team_id" in df.columns:
        teams = df["defending_team_id"].reset_index(drop=True)
    else:
        print("WARNING: defending_team_id assente, uso team_id (attaccante) come fallback")
        teams = df["team_id"].reset_index(drop=True)

    X = X.reset_index(drop=True)
    print(f"\nfeature set finale ({X.shape[1]} colonne):")
    print(f"  numeric: {numeric}")
    print(f"  binary:  {binary}")
    print(f"  categ:   {categ}")
    return X, y, groups, fam, teams


# ---- training --------------------------------------------------------------
def oof_predict_calibrated(X, y, groups, n_splits=5, seed=42, calib_frac=0.2):
    """OOF predictions con calibrazione isotonica dentro ogni fold.

    Per ogni fold GKF: splitta il train in sub-train (80%) + calib (20%)
    rispettando i gruppi match_id, fitta LightGBM sul sub-train, fitta
    IsotonicRegression sul calib, applica al validation fold.
    """
    oof = np.zeros(len(y))
    oof_raw = np.zeros(len(y))  # per confronto raw vs calibrato
    gkf = GroupKFold(n_splits=n_splits)

    for fold, (tr, va) in enumerate(gkf.split(X, y, groups)):
        # sub-split train -> sub_train + calib rispettando i gruppi
        gss = GroupShuffleSplit(n_splits=1, test_size=calib_frac,
                                random_state=seed + fold)
        sub_tr_rel, calib_rel = next(
            gss.split(X.iloc[tr], y.iloc[tr], groups.iloc[tr])
        )
        tr_sub_idx = tr[sub_tr_rel]
        calib_idx  = tr[calib_rel]

        model = lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=40,
            subsample=0.9, colsample_bytree=0.9,
            # PATCH: class_weight="balanced" RIMOSSO
            random_state=seed, verbosity=-1,
        )
        model.fit(
            X.iloc[tr_sub_idx], y.iloc[tr_sub_idx],
            eval_set=[(X.iloc[calib_idx], y.iloc[calib_idx])],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )

        # calibrazione isotonica sui predict del calib
        p_calib = model.predict_proba(X.iloc[calib_idx])[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_calib, y.iloc[calib_idx].values)

        # predict val, poi applica calibrazione
        p_val_raw = model.predict_proba(X.iloc[va])[:, 1]
        p_val_cal = iso.transform(p_val_raw)

        oof_raw[va] = p_val_raw
        oof[va]     = p_val_cal

        auc_raw = roc_auc_score(y.iloc[va], p_val_raw)
        brier_raw = brier_score_loss(y.iloc[va], p_val_raw)
        brier_cal = brier_score_loss(y.iloc[va], p_val_cal)
        print(f"  fold {fold}: n_tr={len(tr_sub_idx)}, n_calib={len(calib_idx)}, "
              f"n_val={len(va)}, AUC={auc_raw:.3f}, "
              f"Brier raw={brier_raw:.4f} -> cal={brier_cal:.4f}")

    return oof, oof_raw


# ---- diagnostics -----------------------------------------------------------
def report(y, p, fam, label=""):
    auc  = roc_auc_score(y, p)
    base = y.mean()
    brier = brier_score_loss(y, p)
    brier_base = base * (1 - base)
    bss = 1 - brier / brier_base
    print(f"\n== {label} == n={len(y)} base_rate={base:.3f}")
    print(f"  AUC         {auc:.4f}")
    print(f"  Brier       {brier:.4f} (base {brier_base:.4f})")
    print(f"  Brier Skill {bss:+.4f}")
    print(f"  LogLoss     {log_loss(y, np.clip(p,1e-6,1-1e-6)):.4f}")

    # per-family
    print("  per family:")
    for f, idx in fam.groupby(fam).groups.items():
        idx = np.array(idx)
        if len(idx) < 50 or y.iloc[idx].nunique() < 2:
            continue
        print(f"    {f:15s} n={len(idx):5d} "
              f"base={y.iloc[idx].mean():.3f} "
              f"AUC={roc_auc_score(y.iloc[idx], p[idx]):.3f}")

    # calibration (10 bin)
    print("  calibration (decili di p):")
    bins = pd.qcut(p, 10, duplicates="drop")
    cal  = pd.DataFrame({"p": p, "y": y.values, "bin": bins})
    for b, g in cal.groupby("bin", observed=True):
        print(f"    pred {g['p'].mean():.3f}  obs {g['y'].mean():.3f}  n={len(g)}")


# PATCH: split-half stability check
def split_half_stability(y, p, groups, teams, n_iter=200, seed=42, min_matches=3):
    """Test decisivo del DRS stadio 1.

    Dividi i 20 match in 2 metà random, per ogni squadra calcola il residuo
    (osservato - predetto) in ciascuna metà, correla i residui tra le due metà.

    PATCH: filtra squadre con < min_matches match totali.
    Con 13 squadre a mediana 3 match (range 1-7), le squadre a 1-2 match non
    possono finire con >=1 match in entrambe le metà di uno split in 2x10,
    inquinando lo stability check con rumore strutturale non eliminabile.
    """
    # PATCH: lista squadre eligibili prima del loop
    team_match_count = (
        pd.DataFrame({"team": teams.values, "match": groups.values})
        .drop_duplicates()
        .groupby("team")
        .size()
    )
    teams_ok = team_match_count[team_match_count >= min_matches].index
    teams_dropped = team_match_count[team_match_count < min_matches]

    print(f"\n[filtro squadre per stability check]")
    print(f"  squadre totali: {len(team_match_count)}")
    print(f"  squadre con >= {min_matches} match (incluse): {len(teams_ok)}")
    print(f"  squadre con < {min_matches} match (escluse): {len(teams_dropped)}")
    if len(teams_dropped) > 0:
        print(f"    escluse: {dict(teams_dropped)}")

    mask = teams.isin(teams_ok).values
    y_f      = y[mask].reset_index(drop=True)
    p_f      = p[mask]
    groups_f = groups[mask].reset_index(drop=True)
    teams_f  = teams[mask].reset_index(drop=True)
    print(f"  run filtrate: {len(y_f)} su {len(y)} ({len(y_f)/len(y):.1%})")

    rng = np.random.default_rng(seed)
    match_ids = groups_f.unique()
    n_matches = len(match_ids)
    corrs = []
    n_teams_common = []

    for _ in range(n_iter):
        perm = rng.permutation(match_ids)
        h1 = set(perm[:n_matches // 2])
        h2 = set(perm[n_matches // 2:])

        in_h1 = groups_f.isin(h1).values
        in_h2 = groups_f.isin(h2).values

        resid = y_f.values - p_f
        df_h1 = pd.DataFrame({"team": teams_f.values[in_h1], "r": resid[in_h1]})
        df_h2 = pd.DataFrame({"team": teams_f.values[in_h2], "r": resid[in_h2]})
        r_h1 = df_h1.groupby("team")["r"].mean()
        r_h2 = df_h2.groupby("team")["r"].mean()

        common = r_h1.index.intersection(r_h2.index)
        n_teams_common.append(len(common))
        if len(common) < 3:
            continue
        c = r_h1.loc[common].corr(r_h2.loc[common])
        if not np.isnan(c):
            corrs.append(c)

    corrs = np.array(corrs)
    print(f"\n== Split-half stability check (squadre con >= {min_matches} match, "
          f"{len(corrs)} iter validi) ==")
    print(f"  squadre comuni per split: median={int(np.median(n_teams_common))}, "
          f"range=[{min(n_teams_common)}, {max(n_teams_common)}]")
    print(f"  correlation median:  {np.median(corrs):+.3f}")
    print(f"  correlation IQR:    [{np.percentile(corrs, 25):+.3f}, "
          f"{np.percentile(corrs, 75):+.3f}]")
    print(f"  correlation 10-90p: [{np.percentile(corrs, 10):+.3f}, "
          f"{np.percentile(corrs, 90):+.3f}]")
    print(f"  frazione iter con corr > 0: {(corrs > 0).mean():.2%}")
    print(f"  Lettura:")
    print(f"    median > 0.5   -> DRS misura un tratto di squadra stabile")
    print(f"    median 0.3-0.5 -> segnale debole ma presente")
    print(f"    median < 0.3   -> rumore di campione")
    return corrs


# ---- main ------------------------------------------------------------------
if __name__ == "__main__":
    df = pd.read_csv(DATA)
    print(f"loaded {len(df)} rows from {DATA.name}")

    X, y, groups, fam, teams = build_stage1(df)
    print(f"\nstage1 matrix: X={X.shape}, y mean={y.mean():.3f}, "
          f"n_matches={groups.nunique()}, n_teams={teams.nunique()}")

    print("\n== Training (GroupKFold su match_id + calibrazione isotonica) ==")
    oof_cal, oof_raw = oof_predict_calibrated(X, y, groups)

    report(y, oof_raw, fam, label="Stadio 1 pooled -- RAW (pre-calibrazione)")
    report(y, oof_cal, fam, label="Stadio 1 pooled -- CALIBRATO")

    # Il test decisivo
    # Il test decisivo - due soglie per vedere l'effetto del filtro
    print("\n" + "="*72)
    print("STABILITY — tutte le squadre (incluse quelle con 1-2 match)")
    print("="*72)
    _ = split_half_stability(y, oof_cal, groups, teams, min_matches=1)

    print("\n" + "="*72)
    print("STABILITY — solo squadre con >=3 match (robusto)")
    print("="*72)
    _ = split_half_stability(y, oof_cal, groups, teams, min_matches=3)

    def drop_categorical_features(X, drop_phase=True, drop_family=True, drop_channel=False):
        """Rimuove le colonne one-hot generate da pd.get_dummies per le categorie
        indicate. Usata nell'ablation 'geometry-only'."""
        cols_to_drop = []
        if drop_phase:
            cols_to_drop += [c for c in X.columns
                            if c.startswith("team_out_of_possession_phase_type_")]
        if drop_family:
            cols_to_drop += [c for c in X.columns if c.startswith("subtype_family_")]
        if drop_channel:
            cols_to_drop += [c for c in X.columns if c.startswith("channel_3_")]
        dropped = [c for c in cols_to_drop if c in X.columns]
        X_out = X.drop(columns=dropped)
        print(f"  droppate {len(dropped)} colonne one-hot")
        print(f"  colonne residue: {X_out.shape[1]}")
        return X_out
 
 
    # ============================================================================
    # Blocco da appendere al main di drs_stage1.py, DOPO split_half_stability
    # ============================================================================
    
    print("\n" + "=" * 72)
    print("ABLATION — training senza fase difensiva e famiglia subtype")
    print("=" * 72)
    print("Rimuove tutte le colonne one-hot generate da:")
    print("  - team_out_of_possession_phase_type (fase difensiva)")
    print("  - subtype_family (costruzione/progressione/laterali/finalizzazione)")
    print("Tiene: geometria numerica (delta_to_line, line_height, n_opp_ahead,")
    print("       n_simultaneous_runs), pp_* (n_options, pressure_ord,")
    print("       reception_difficulty_ord), binarie (inside_shape, penalty_area),")
    print("       channel_3 (center/half_space/wide, geometria pura).")
    print()
    print("Domanda: quanto del segnale dello stadio 1 sta nel tagging strutturale")
    print("vs nelle feature geometriche + scelta?")
    print()
    
    X_geo = drop_categorical_features(X, drop_phase=True, drop_family=True)
    
    print("\n== Training ablation (geometry-only) ==")
    oof_geo_cal, oof_geo_raw = oof_predict_calibrated(X_geo, y, groups)
    report(y, oof_geo_cal, fam, label="Stadio 1 ABLATION geometry-only -- CALIBRATO")
    
    # confronto riassuntivo
    auc_full   = roc_auc_score(y, oof_cal)
    auc_geo    = roc_auc_score(y, oof_geo_cal)
    brier_full = brier_score_loss(y, oof_cal)
    brier_geo  = brier_score_loss(y, oof_geo_cal)
    
    print("\n" + "=" * 72)
    print("Confronto full vs geometry-only (entrambi calibrati)")
    print("=" * 72)
    print(f"  AUC:   full {auc_full:.4f}  ->  geo {auc_geo:.4f}  "
        f"(delta {auc_geo - auc_full:+.4f})")
    print(f"  Brier: full {brier_full:.4f}  ->  geo {brier_geo:.4f}  "
        f"(delta {brier_geo - brier_full:+.4f})")
    print()
    print("Lettura dei delta:")
    print("  AUC geo scende a 0.55-0.58  -> fase+famiglia catturano il grosso del")
    print("                                 segnale, geometria+scelta sono secondarie")
    print("                                 (fingerprint situazionale regge SOLO su")
    print("                                 fase x famiglia)")
    print()
    print("  AUC geo resta >= 0.60       -> geometria+scelta hanno segnale proprio,")
    print("                                 oltre al tagging strutturale")
    print("                                 (fingerprint situazionale regge su piu'")
    print("                                 assi, inclusi line_height, pp_n_options,")
    print("                                 delta_to_line)")

