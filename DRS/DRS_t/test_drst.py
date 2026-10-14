"""Test del file unico drst.py.  pytest test_drst.py"""

import numpy as np
import pandas as pd
import pytest

from drst import (DRSConfig, DEFAULT, outcome_priors, compute_stress,
                  run_kernel, build_curve, add_defending_team)
from dataclasses import replace

from drst import EventAnchoredIndex
from drst import separation_gain_containment
from drst import geometric_priors
from drst import Tracking, run_geometry


@pytest.fixture
def toy():
    """4 run su 1 match, 2 squadre."""
    return pd.DataFrame({
        "match_id": [1, 1, 1, 1],
        "event_id": ["1_1", "1_2", "1_3", "1_4"],
        "event_subtype": ["cross_receiver", "cross_receiver", "dropping_off", "behind"],
        "frame_start": [100, 300, 500, 700],
        "frame_end":   [130, 340, 530, 760],
        "separation_gain": [2.0, -2.0, 0.5, 1.0],
        "team_id": [10, 10, 20, 10],
        "defender_id": [1.0, 2.0, 3.0, 1.0],
        "targeted": [True, False, True, True],
        "received": [True, False, True, False],
        "lead_to_shot": [True, False, False, True],
    })


# --- kernel

def test_kernel_peaks_at_run_end():
    k = run_kernel(500, 100, 200, 1.0, DEFAULT)
    assert k[:100].sum() == 0, "niente stress prima dell'inizio"
    assert k[199] == pytest.approx(1.0, abs=0.02), "rampa arriva a stress"
    assert k[200] > k[250] > k[300], "decadimento monotono dopo la fine"


def test_kernel_decay_follows_tau():
    cfg = DRSConfig(tau_decay_frames=10)
    k = run_kernel(500, 100, 200, 1.0, cfg)
    assert k[210] == pytest.approx(np.exp(-1.0), rel=0.01), "a t=tau vale 1/e"


def test_kernel_respects_bounds():
    k = run_kernel(150, 100, 200, 1.0, DEFAULT)
    assert len(k) == 150 and np.isfinite(k).all()


def test_kernel_no_ramp_is_step():
    cfg = DRSConfig(use_ramp=False)
    k = run_kernel(500, 100, 200, 1.0, cfg)
    assert k[100] == pytest.approx(1.0) and k[150] == pytest.approx(1.0)


# --- containment

def test_containment_normalized_per_subtype(toy):
    c = separation_gain_containment(toy, DEFAULT)
    assert ((c >= 0) & (c <= 1)).all()
    # le due cross_receiver hanno gain opposto -> containment opposto rispetto a 0.5
    assert c.iloc[0] > 0.5 > c.iloc[1]
    # subtype con una sola osservazione (std NaN) -> 0.5, non NaN
    assert c.iloc[2] == pytest.approx(0.5)


def test_containment_is_relative_not_absolute():
    """Un gain negativo in un subtype dove TUTTI i gain sono negativi non
    deve contare come successo difensivo."""
    df = pd.DataFrame({
        "event_subtype": ["cross_receiver"] * 3,
        "separation_gain": [-3.0, -2.0, -1.0],
    })
    c = separation_gain_containment(df, DEFAULT)
    assert c.iloc[2] > 0.5, "il meno negativo del gruppo e' comunque peggio della media"


# --- prior e stress

def test_priors_shrink_small_subtypes(toy):
    priors, tab = outcome_priors(toy, DEFAULT)
    assert all(0 < v <= 1 for v in priors.values())
    assert tab["shrunk"].between(0, 1).all()


def test_stress_is_product(toy):
    priors, _ = outcome_priors(toy, DEFAULT)
    out = compute_stress(toy, priors, cfg=DEFAULT)
    assert np.allclose(out.stress, out.danger_prior * out.containment_failure)
    assert (out.stress >= 0).all()


def test_stress_raises_on_unknown_subtype(toy):
    priors, _ = outcome_priors(toy, DEFAULT)
    bad = toy.copy()
    bad.loc[0, "event_subtype"] = "subtype_inesistente"
    with pytest.raises(ValueError, match="privo di prior"):
        compute_stress(bad, priors, cfg=DEFAULT)


def test_stress_ignores_outcomes(toy):
    """Lo stress non deve cambiare se cambio gli esiti a parita' di prior.
    E' cio' che rende la validazione non circolare."""
    priors, _ = outcome_priors(toy, DEFAULT)
    a = compute_stress(toy, priors, cfg=DEFAULT).stress.values
    flipped = toy.copy()
    flipped["lead_to_shot"] = ~flipped["lead_to_shot"]
    b = compute_stress(flipped, priors, cfg=DEFAULT).stress.values
    assert np.allclose(a, b)


# --- curva e indicizzazione

def test_curve_requires_stress(toy):
    with pytest.raises(ValueError, match="stress"):
        build_curve(toy, DEFAULT)


def test_curve_support_matches_runs(toy):
    priors, _ = outcome_priors(toy, DEFAULT)
    d = compute_stress(toy, priors, cfg=DEFAULT)
    curve, idx = build_curve(d, DEFAULT)
    assert len(curve) == idx.n_frames
    assert not idx.support[:100].any(), "nessun supporto prima della prima run"
    assert idx.support[110], "supporto dentro la prima run"
    assert 0 < idx.coverage() < 1


def test_curve_aggregation_modes(toy):
    priors, _ = outcome_priors(toy, DEFAULT)
    d = compute_stress(toy, priors, cfg=DEFAULT)
    c_sum, _ = build_curve(d, DRSConfig(aggregation="sum"))
    c_max, _ = build_curve(d, DRSConfig(aggregation="max"))
    c_nor, _ = build_curve(d, DRSConfig(aggregation="noisy_or"))
    assert c_sum.max() >= c_max.max()
    assert c_nor.max() <= 1.0, "noisy_or satura a 1"


def test_masked_ignores_unsupported(toy):
    priors, _ = outcome_priors(toy, DEFAULT)
    d = compute_stress(toy, priors, cfg=DEFAULT)
    curve, idx = build_curve(d, DEFAULT)
    assert idx.masked(curve).mean() > curve.mean(), \
        "la media sui soli frame supportati e' piu' alta di quella con gli zeri"


def test_dense_tracking_is_stub(toy):
    from drst import DenseTrackingIndex
    with pytest.raises(NotImplementedError):
        DenseTrackingIndex().build(toy, DEFAULT)


# --- aggregazione

def test_defending_team_is_complement(toy):
    out = add_defending_team(toy, DEFAULT)
    assert (out.defending_team_id != out.team_id).all()
    assert set(out.defending_team_id) == {10, 20}


def test_defending_team_raises_on_single_team():
    df = pd.DataFrame({"match_id": [1, 1], "team_id": [10, 10]})
    with pytest.raises(ValueError, match="attese 2"):
        add_defending_team(df, DEFAULT)


# --- v1 enriched: prior spaziale e vulnerabilita' strutturale

@pytest.fixture
def toy_enriched(toy):
    """toy + le colonne che servono alla v1."""
    d = toy.copy()
    d["third_start"] = ["attacking_third", "middle_third", "defensive_third", "attacking_third"]
    d["delta_to_last_defensive_line_start"] = [2.0, 20.0, 35.0, 5.0]
    d["n_simultaneous_runs"] = [3.0, 0.0, 0.0, 2.0]
    d["penalty_area_start"] = [True, False, False, False]
    d["player_id"] = [10, 20, 10, 20]
    d["defender_id"] = [11.0, 21.0, 11.0, 21.0]
    return d


def test_has_structural_columns(toy, toy_enriched):
    from drst import has_structural_columns
    assert not has_structural_columns(toy)
    assert has_structural_columns(toy_enriched)


def test_structural_raises_without_enriched(toy):
    from drst import structural_vulnerability
    with pytest.raises(ValueError, match="colonne strutturali mancanti"):
        structural_vulnerability(toy, DEFAULT)


def test_structural_in_unit_interval(toy_enriched):
    from drst import structural_vulnerability
    v = structural_vulnerability(toy_enriched, DEFAULT)
    assert ((v > 0) & (v <= 1)).all()


def test_structural_ranks_danger_correctly(toy_enriched):
    """La run 0 (delta basso, 3 corse simultanee, in area) deve essere piu'
    vulnerabile della run 2 (delta alto, nessuna corsa simultanea)."""
    from drst import structural_vulnerability
    v = structural_vulnerability(toy_enriched, DEFAULT)
    assert v.iloc[0] > v.iloc[2]


def test_structural_reference_is_out_of_sample(toy_enriched):
    """Con `reference` i ranghi vengono dalla distribuzione di riferimento,
    non da se stessi: e' cio' che rende l'OOS onesto."""
    from drst import structural_vulnerability
    sub = toy_enriched.iloc[:2]
    a = structural_vulnerability(sub, DEFAULT)
    b = structural_vulnerability(sub, DEFAULT, reference=toy_enriched)
    assert not np.allclose(a.values, b.values)


def test_spatial_prior_is_finer_than_subtype(toy_enriched):
    from drst import spatial_priors, map_spatial_prior, outcome_priors
    cells, fb = spatial_priors(toy_enriched, DEFAULT)
    sub, _ = outcome_priors(toy_enriched, DEFAULT)
    # due cross_receiver in zone diverse -> due celle distinte
    keys = [k for k in cells if k[0] == "cross_receiver"]
    assert len(keys) == 2
    assert len(cells) >= len(sub)


def test_spatial_prior_falls_back_on_unseen_cell(toy_enriched):
    from drst import spatial_priors, map_spatial_prior
    cells, fb = spatial_priors(toy_enriched, DEFAULT)
    unseen = toy_enriched.iloc[[0]].copy()
    unseen["third_start"] = "zona_mai_vista"
    v = map_spatial_prior(unseen, cells, fb, DEFAULT)
    assert v.notna().all() and 0 < v.iloc[0] <= 1


def test_structural_multiplies_stress(toy_enriched):
    from drst import outcome_priors, compute_stress
    pri, _ = outcome_priors(toy_enriched, DEFAULT)
    a = compute_stress(toy_enriched, pri, cfg=DEFAULT, structural=False)
    b = compute_stress(toy_enriched, pri, cfg=DEFAULT, structural=True)
    assert "structural_vulnerability" in b.columns
    assert (b.stress <= a.stress + 1e-12).all(), "il fattore e' in [0,1], puo' solo ridurre"
    assert np.allclose(b.stress, a.stress * b.structural_vulnerability)


def test_structural_ignores_outcomes(toy_enriched):
    """Come lo stress base: invertire gli esiti non deve cambiare la
    vulnerabilita' strutturale."""
    from drst import structural_vulnerability
    a = structural_vulnerability(toy_enriched, DEFAULT)
    f = toy_enriched.copy()
    f["lead_to_shot"] = ~f["lead_to_shot"]
    b = structural_vulnerability(f, DEFAULT)
    assert np.allclose(a.values, b.values)


def test_dissociation_rejects_missing_control():
    from drst import dissociation
    pre = pd.DataFrame({
        "drs_pre_mean": np.random.rand(50), "subtype": ["behind"] * 50,
        "targeted": [True] * 25 + [False] * 25,
        "received": [True] * 25 + [False] * 25,
        "lead_to_shot": [True] * 25 + [False] * 25,
    })
    with pytest.raises(ValueError, match="assente"):
        dissociation(pre, DEFAULT, extra_controls=["third_start"])


# --- animazione 2D

@pytest.fixture
def toy_tracking():
    from drst import Tracking
    players = {}
    for f in range(100, 800):
        players[f] = {
            10: (0.0 + f * 0.01, 5.0),    # runner 1, si muove verso +x
            11: (-5.0, 5.0),              # difensore 1, fermo
            20: (10.0, -10.0),            # runner 2
            21: (12.0, -10.0),            # difensore 2
            99: (-50.0, 0.0),             # portiere, mai nel layer3
        }
    ball = {f: (0.0, 0.0, 0.1) for f in players}
    return Tracking(players, ball, np.array(sorted(players)))


def test_tracking_nearest_frame(toy_tracking):
    assert toy_tracking.nearest_frame(150) == 150
    assert toy_tracking.nearest_frame(95, max_gap=10) == 100
    assert toy_tracking.nearest_frame(50, max_gap=10) is None, "gap troppo grande"


def test_active_weights_matches_kernel(toy_enriched):
    """La somma dei pesi al frame f deve coincidere con la curva DRS(t)."""
    from drst import outcome_priors, compute_stress, build_curve, active_weights
    pri, _ = outcome_priors(toy_enriched, DEFAULT)
    d = compute_stress(toy_enriched, pri, cfg=DEFAULT)
    curve, _ = build_curve(d, DEFAULT)
    for f in (110, 125, 320, 520, 750):
        if f < len(curve):
            w = active_weights(d, f, DEFAULT)["w"].sum()
            assert abs(w - curve[f]) < 1e-9, f"frame {f}: {w} != {curve[f]}"


def test_active_weights_empty_outside_runs(toy_enriched):
    from drst import outcome_priors, compute_stress, active_weights
    pri, _ = outcome_priors(toy_enriched, DEFAULT)
    d = compute_stress(toy_enriched, pri, cfg=DEFAULT)
    assert len(active_weights(d, 50, DEFAULT)) == 0, "nessuna corsa prima della prima"


def test_stress_field_modes(toy_enriched, toy_tracking):
    from drst import outcome_priors, compute_stress, active_weights, stress_field
    pri, _ = outcome_priors(toy_enriched, DEFAULT)
    d = compute_stress(toy_enriched, pri, cfg=DEFAULT)
    xs = np.arange(-52.5, 53.5, 4.0); ys = np.arange(-34, 35, 4.0)
    grid = np.meshgrid(xs, ys)
    act = active_weights(d, 120, DEFAULT)
    for mode in ("runner", "projected", "gap"):
        fl = stress_field(120, act, toy_tracking, grid, DEFAULT, mode)
        assert fl.shape == grid[0].shape
        assert np.isfinite(fl).all() and (fl >= 0).all()
        assert fl.max() > 0, f"mode {mode} non deposita nulla"


def test_stress_field_rejects_unknown_mode(toy_enriched, toy_tracking):
    from drst import outcome_priors, compute_stress, active_weights, stress_field
    pri, _ = outcome_priors(toy_enriched, DEFAULT)
    d = compute_stress(toy_enriched, pri, cfg=DEFAULT)
    grid = np.meshgrid(np.arange(-52.5, 53.5, 8.0), np.arange(-34, 35, 8.0))
    with pytest.raises(ValueError, match="spatial_mode"):
        stress_field(120, active_weights(d, 120, DEFAULT), toy_tracking, grid,
                     DEFAULT, "modo_inesistente")


def test_peak_windows_are_disjoint():
    from drst import find_peak_windows
    c = np.zeros(10000)
    for pos, amp in [(1000, 1.0), (5000, 0.8), (1050, 0.9)]:
        c[pos:pos + 50] = amp
    w = find_peak_windows(c, n=3, pad_s=5.0, cfg=DEFAULT)
    spans = [(a, b) for _, a, b in w]
    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            assert spans[i][1] <= spans[j][0] or spans[j][1] <= spans[i][0], \
                "finestre sovrapposte"


def test_infer_player_teams_assigns_keeper(toy_enriched, toy_tracking):
    from drst import add_defending_team, infer_player_teams
    d = add_defending_team(toy_enriched, DEFAULT)
    teams = infer_player_teams(d, toy_tracking, DEFAULT, n_sample=20)
    assert 99 in teams, "il portiere non mappato deve essere assegnato"
    assert teams[10] != teams[11], "runner e suo difensore in squadre diverse"


# --- feature dal tracking XY

@pytest.fixture
def toy_curve_df(toy_enriched):
    """toy_enriched + le colonne prodotte da tracking_features."""
    d = toy_enriched.copy()
    d["sep_min"] = [1.0, 5.0, 8.0, 2.0]
    d["sep_late"] = [6.0, 4.0, 8.5, 2.5]        # run 0: fuga tardiva forte
    d["sep_min_frac"] = [0.2, 0.9, 0.5, 0.3]
    d["sep_time_close"] = [0.6, 0.1, 0.0, 0.5]
    d["str_stretch_rate"] = [2.0, -1.0, 0.0, 1.5]
    d["str_recovery_rate"] = [1.5, -2.0, -0.5, 0.8]
    return d


def test_has_curve_columns(toy_enriched, toy_curve_df):
    from drst import has_curve_columns
    assert not has_curve_columns(toy_enriched)
    assert has_curve_columns(toy_curve_df)


def test_curve_containment_requires_columns(toy_enriched):
    from drst import separation_curve_containment
    with pytest.raises(ValueError, match="colonne di curva mancanti"):
        separation_curve_containment(toy_enriched, DEFAULT)


def test_curve_containment_in_unit_interval(toy_curve_df):
    from drst import separation_curve_containment
    c = separation_curve_containment(toy_curve_df, DEFAULT)
    assert ((c >= 0) & (c <= 1)).all() and c.notna().all()


def test_curve_containment_falls_back_without_curve(toy_curve_df):
    """Una corsa senza curva calcolabile deve ricadere sul gain, non su NaN."""
    from drst import separation_curve_containment
    d = toy_curve_df.copy()
    d.loc[0, ["sep_min", "sep_late"]] = np.nan
    c = separation_curve_containment(d, DEFAULT)
    assert c.notna().all()


def test_curve_containment_ignores_outcomes(toy_curve_df):
    from drst import separation_curve_containment
    a = separation_curve_containment(toy_curve_df, DEFAULT)
    f = toy_curve_df.copy()
    f["lead_to_shot"] = ~f["lead_to_shot"]
    assert np.allclose(a.values, separation_curve_containment(f, DEFAULT).values)


def test_tracking_structure_changes_vulnerability(toy_curve_df):
    from dataclasses import replace
    from drst import structural_vulnerability
    a = structural_vulnerability(toy_curve_df, replace(DEFAULT, use_tracking_structure=False))
    b = structural_vulnerability(toy_curve_df, DEFAULT)
    assert not np.allclose(a.values, b.values), "le colonne di tracking devono pesare"
    assert ((b > 0) & (b <= 1)).all()


def test_tracking_structure_ignored_when_columns_absent(toy_enriched):
    from dataclasses import replace
    from drst import structural_vulnerability
    a = structural_vulnerability(toy_enriched, replace(DEFAULT, use_tracking_structure=False))
    b = structural_vulnerability(toy_enriched, DEFAULT)
    assert np.allclose(a.values, b.values), "senza colonne non deve cambiare nulla"


def test_separation_curve_features_shape():
    from drst import separation_curve_features
    sep = np.array([8.0, 6.0, 4.0, 2.0, 1.0, 3.0, 5.0, 7.0])   # V: perde e recupera
    f = separation_curve_features(sep, DEFAULT)
    assert f["sep_min"] == 1.0
    assert 0.4 < f["sep_min_frac"] < 0.7, "il minimo e' a meta' corsa"
    assert f["sep_inversions"] == 1, "una sola inversione"
    assert f["sep_late"] > f["sep_min"], "il runner si libera dopo il minimo"


def test_separation_curve_features_monotone_escape():
    from drst import separation_curve_features
    sep = np.linspace(2.0, 10.0, 10)        # il runner si stacca sempre
    f = separation_curve_features(sep, DEFAULT)
    assert f["sep_min_frac"] == 0.0, "minimo all'inizio"
    assert f["sep_inversions"] == 0
    assert f["sep_time_close"] == pytest.approx(0.2), "solo i primi 2 punti sotto 3 m"
    assert f["sep_slope_min"] > 0, "al minimo il runner si sta gia' staccando"


def test_merge_into_layer3_preserves_rows(toy_enriched):
    from drst import merge_into_layer3
    feats = pd.DataFrame({
        "match_id": [1, 1], "event_id": ["1_1", "1_3"],
        "sep_min": [1.0, 2.0],
    })
    out = merge_into_layer3(toy_enriched, feats, DEFAULT)
    assert len(out) == len(toy_enriched), "il merge non deve duplicare ne' perdere righe"
    assert out["sep_min"].notna().sum() == 2


# --- prior geometrico

def test_geometric_prior_requires_coordinates(toy):
    from drst import geometric_priors
    with pytest.raises(ValueError, match="colonne geometriche mancanti"):
        geometric_priors(toy, DEFAULT)


def test_geometric_prior_decays_with_distance():
    from drst import geometric_priors
    d = pd.DataFrame({"x_start": [52.0, 30.0, 0.0, -40.0], "y_start": [0.0] * 4})
    g = geometric_priors(d, DEFAULT)
    assert (np.diff(g.values) < 0).all(), "il pericolo deve calare allontanandosi"
    assert g.iloc[0] == pytest.approx(1.0), "normalizzato a 1 sul piu' vicino"
    assert (g > 0).all()


def test_geometric_prior_ignores_outcomes():
    """E' il punto: il prior geometrico non tocca gli esiti, quindi la
    validazione non ha bisogno di controlli di zona."""
    from drst import geometric_priors
    d = pd.DataFrame({"x_start": [40.0, 10.0], "y_start": [5.0, -5.0],
                      "lead_to_shot": [True, False]})
    a = geometric_priors(d, DEFAULT)
    f = d.copy(); f["lead_to_shot"] = ~f["lead_to_shot"]
    assert np.allclose(a.values, geometric_priors(f, DEFAULT).values)


def test_geometric_prior_scale_changes_spread():
    from dataclasses import replace
    from drst import geometric_priors
    d = pd.DataFrame({"x_start": [52.0, 20.0, -20.0], "y_start": [0.0] * 3})
    tight = geometric_priors(d, replace(DEFAULT, geo_prior_scale=12.0))
    wide = geometric_priors(d, replace(DEFAULT, geo_prior_scale=45.0))
    assert tight.iloc[-1] < wide.iloc[-1], "scala stretta penalizza di piu' il lontano"


def test_pipeline_picks_geometric_by_default(toy_enriched):
    """Senza coordinate deve ricadere sul prior spaziale, non crashare."""
    from drst import DEFAULT as D
    assert D.prior_kind == "geometric"
    assert "x_start" not in toy_enriched.columns


# ---------------------------------------------------------------------------
# Prior geometrico col punto di arrivo
# ---------------------------------------------------------------------------

def _geo_df(n=40, seed=3):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "x_start": rng.uniform(-40, 45, n),
        "y_start": rng.uniform(-30, 30, n),
        "geo_dist_min": rng.uniform(8, 90, n),
        "ball_dist_goal_start": rng.uniform(12, 100, n),
        "lead_to_shot": rng.random(n) < 0.15,
    })


def test_geometric_prior_endpoint_ignores_outcomes():
    """INVARIANTE. Il prior col punto di arrivo resta indipendente dagli esiti."""
    cfg = replace(DEFAULT, geo_use_endpoint=True)
    df = _geo_df()
    a = geometric_priors(df, cfg)
    flipped = df.assign(lead_to_shot=~df["lead_to_shot"].astype(bool))
    b = geometric_priors(flipped, cfg)
    assert np.allclose(a, b)


def test_geometric_prior_endpoint_takes_minimum():
    """Il prior usa il punto piu' vicino alla porta, non la partenza."""
    df = pd.DataFrame({"x_start": [0.0], "y_start": [0.0], "geo_dist_min": [10.0]})
    near = geometric_priors(df, replace(DEFAULT, prior_normalize=False, geo_use_endpoint=True))
    far = geometric_priors(df.drop(columns="geo_dist_min"),
                           replace(DEFAULT, prior_normalize=False, geo_use_endpoint=True))
    assert near.iloc[0] > far.iloc[0]
    assert np.isclose(near.iloc[0], np.exp(-10.0 / DEFAULT.geo_prior_scale))


def test_geometric_prior_endpoint_never_increases_distance():
    """geo_dist_min piu' GRANDE della partenza non deve abbassare il prior:
    il minimo e' preso su entrambi, quindi la partenza fa da tetto."""
    df = pd.DataFrame({"x_start": [50.0], "y_start": [0.0], "geo_dist_min": [80.0]})
    cfg = replace(DEFAULT, prior_normalize=False, geo_use_endpoint=True)
    assert np.isclose(geometric_priors(df, cfg).iloc[0],
                      geometric_priors(df.drop(columns="geo_dist_min"), cfg).iloc[0])


def test_geometric_prior_endpoint_fallback_on_nan():
    """Copertura parziale del tracking: le corse senza geo_dist_min ricadono
    sulla partenza invece di diventare NaN."""
    df = pd.DataFrame({"x_start": [0.0, 0.0], "y_start": [0.0, 0.0],
                       "geo_dist_min": [10.0, np.nan]})
    g = geometric_priors(df, replace(DEFAULT, prior_normalize=False, geo_use_endpoint=True))
    assert g.notna().all()
    assert np.isclose(g.iloc[1], np.exp(-52.5 / DEFAULT.geo_prior_scale))


def test_geometric_prior_endpoint_off_by_default():
    """DEFAULT OFF: il punto di arrivo peggiora la split-half per squadra."""
    assert DEFAULT.geo_use_endpoint is False
    df = _geo_df()
    off = geometric_priors(df, replace(DEFAULT, geo_use_endpoint=False))
    assert np.allclose(off, geometric_priors(df.drop(columns="geo_dist_min"),
                                             replace(DEFAULT, geo_use_endpoint=False)))


def test_ball_gate_off_by_default_and_attenuates_when_on():
    df = _geo_df()
    base = geometric_priors(df, replace(DEFAULT, prior_normalize=False, geo_use_endpoint=True))
    gated = geometric_priors(df, replace(DEFAULT, prior_normalize=False,
                                         geo_ball_gate=True))
    assert np.allclose(base, geometric_priors(
        df.drop(columns="ball_dist_goal_start"),
        replace(DEFAULT, prior_normalize=False, geo_use_endpoint=True)))
    assert (gated <= base + 1e-12).all()


def test_run_geometry_orientation_flips_with_attacking_side():
    """ORIENTAMENTO. Il tracking e' assoluto: la stessa posizione con
    attacking_side opposto deve dare x di segno opposto."""
    trk = Tracking(
        players={100: {7: (30.0, 5.0)}, 110: {7: (40.0, 2.0)}},
        ball={100: (28.0, 4.0, 0.3), 110: (38.0, 1.0, 0.5)},
        frames=np.array([100, 110]),
    )
    run = pd.Series({"player_id": 7, "frame_start": 100, "frame_end": 110,
                     "attacking_side": "left_to_right"})
    ltr = run_geometry(run, trk, DEFAULT)
    rtl = run_geometry(run.replace({"left_to_right": "right_to_left"}), trk, DEFAULT)
    assert np.isclose(ltr["geo_x_end"], -rtl["geo_x_end"])
    assert ltr["geo_dist_end"] < ltr["geo_dist_start"]      # la corsa avvicina
    # progress = dist_start - dist_end, e la y cambia insieme alla x:
    # hypot(22.5, 5) - hypot(12.5, 2) = 10.39, non i 10 m di solo avanzamento
    assert np.isclose(ltr["geo_progress"],
                      ltr["geo_dist_start"] - ltr["geo_dist_end"])
    assert ltr["geo_progress"] > 10.0
    assert np.isclose(ltr["geo_dist_min"], ltr["geo_dist_end"])


def test_run_geometry_missing_player_returns_empty():
    trk = Tracking(players={100: {}, 110: {}}, ball={},
                   frames=np.array([100, 110]))
    run = pd.Series({"player_id": 7, "frame_start": 100, "frame_end": 110})
    assert run_geometry(run, trk, DEFAULT) == {}


# --- affidabilita': Spearman-Brown e confronto appaiato

def test_spearman_brown_corregge_verso_lalto():
    """La split-half misura meta' campione: la correzione deve salire."""
    from drst import spearman_brown
    assert spearman_brown(0.724) == pytest.approx(0.840, abs=0.001)
    assert spearman_brown(0.534) == pytest.approx(0.696, abs=0.001)
    assert spearman_brown(0.026) == pytest.approx(0.051, abs=0.001)


def test_spearman_brown_punti_fissi():
    """0 e 1 restano se stessi: nessuna affidabilita' non si corregge, e
    un'affidabilita' perfetta non puo' migliorare."""
    from drst import spearman_brown
    assert spearman_brown(0.0) == pytest.approx(0.0)
    assert spearman_brown(1.0) == pytest.approx(1.0)
    assert np.isnan(spearman_brown(float("nan")))


@pytest.fixture
def toy_team_match():
    """6 squadre x 10 match, con un segnale di squadra deliberato.

    10 match e non 4: con 4 match esistono solo 3 partizioni distinte in due
    meta', quindi le 400 iterazioni ricadono sempre sulle stesse tre e la
    mediana non e' una media su nulla. Con 10 le partizioni sono 252 e il
    rumore si concentra intorno a zero come deve.
    """
    rng = np.random.default_rng(0)
    rows = []
    for team in range(6):
        for mid in range(10):
            rows.append(dict(match_id=mid, defending_team_id=team,
                             metric=team + rng.normal(0, 0.3),
                             rumore=rng.normal(0, 1)))
    return pd.DataFrame(rows)


def test_split_half_separa_segnale_da_rumore(toy_team_match):
    from drst import split_half
    segnale = split_half(toy_team_match, "metric", n_iter=100)
    rumore = split_half(toy_team_match, "rumore", n_iter=100)
    assert segnale["median"] > 0.8, "un tratto di squadra netto va riconosciuto"
    assert rumore["median"] < 0.5, "il rumore non deve risultare stabile"
    assert segnale["spearman_brown"] > segnale["median"]


def test_split_half_curves_conserva_la_lunghezza(toy_team_match):
    """I NaN restano al loro posto: il confronto appaiato richiede che le due
    configurazioni siano allineate posizione per posizione."""
    from drst import split_half_curves
    curves, _, n_teams = split_half_curves(toy_team_match, "metric", n_iter=50)
    assert len(curves) == 50
    assert n_teams == 6


def test_paired_split_half_vede_una_differenza_vera(toy_team_match):
    from drst import paired_split_half
    peggiore = toy_team_match.copy()
    rng = np.random.default_rng(1)
    peggiore["metric"] = peggiore["metric"] + rng.normal(0, 3, len(peggiore))
    r = paired_split_half(toy_team_match, peggiore, "metric", n_iter=200)
    assert r["verdict"] == "peggiora"
    assert r["delta_mean"] < 0
    assert r["frac_b_wins"] < 0.3


def test_paired_split_half_identiche_sono_trascurabili(toy_team_match):
    """Due config identiche: delta esattamente zero, non 'migliora'."""
    from drst import paired_split_half
    r = paired_split_half(toy_team_match, toy_team_match.copy(), "metric", n_iter=100)
    assert r["delta_mean"] == pytest.approx(0.0, abs=1e-12)
    assert r["verdict"] == "trascurabile"


def test_paired_split_half_soglia_di_rilevanza(toy_team_match):
    """Un delta piccolo ma statisticamente non nullo resta 'trascurabile'."""
    from drst import paired_split_half
    cfg = replace(DEFAULT, relevance_threshold=0.5)
    quasi = toy_team_match.copy()
    rng = np.random.default_rng(2)
    quasi["metric"] = quasi["metric"] + rng.normal(0, 0.5, len(quasi))
    r = paired_split_half(toy_team_match, quasi, "metric", cfg=cfg, n_iter=200)
    assert r["verdict"] == "trascurabile", "sotto soglia non si chiama miglioramento"


# --- standardizzazione per fase

@pytest.fixture
def toy_phase():
    """2 squadre, 2 match. La squadra 20 difende basso, la 10 alto, e hanno
    lo STESSO stress dentro ciascuna fase: standardizzando devono pareggiare.
    """
    rows = []
    for mid in (1, 2):
        for ph, s in [("low_block", 0.30), ("high_block", 0.05)]:
            n_deep = 8 if ph == "low_block" else 2
            for team, n in [(20, n_deep), (10, 10 - n_deep)]:
                for _ in range(n):
                    rows.append(dict(match_id=mid, defending_team_id=team,
                                     out_of_possession_phase=ph, stress=s,
                                     containment_failure=0.5))
    return pd.DataFrame(rows)


def test_phase_standardized_pareggia_a_parita_di_gestione(toy_phase):
    """Due squadre con lo stesso stress dentro ogni fase ma mix diverso:
    la metrica grezza le separa, quella standardizzata no. E' esattamente la
    differenza fra "difendi basso" e "gestisci male"."""
    from drst import phase_weights, phase_standardized
    w = phase_weights(toy_phase, DEFAULT)
    grezze, adj = {}, {}
    for team, g in toy_phase.groupby("defending_team_id"):
        grezze[team] = g["stress"].mean()
        adj[team] = phase_standardized(g, "stress", w, DEFAULT)
    assert abs(grezze[20] - grezze[10]) > 0.05, "la metrica grezza le separa"
    assert adj[20] == pytest.approx(adj[10], abs=1e-9), \
        "standardizzata per fase devono coincidere"


def test_team_two_axis_due_assi_e_percentili(toy_phase):
    from drst import team_two_axis
    toy_phase = toy_phase.assign(danger_prior=0.2)
    prof = team_two_axis(toy_phase, None, DEFAULT)
    assert set(prof.defending_team_id) == {10, 20}
    assert {"esposizione", "cedimento", "esposizione_pct",
            "cedimento_pct", "deep_phase_share"} <= set(prof.columns)
    deep = prof.set_index("defending_team_id")["deep_phase_share"]
    assert deep[20] > deep[10], "la squadra che difende basso ha quota maggiore"


# --- containment appreso

@pytest.fixture
def toy_learned():
    """Dataset minimo con tutte le colonne che il containment appreso chiede."""
    from drst import LEARNED_NUM, LEARNED_CAT
    rng = np.random.default_rng(0)
    n = 240
    d = pd.DataFrame({
        "match_id": np.repeat(np.arange(6), n // 6),
        "event_id": [f"e{i}" for i in range(n)],
        "event_subtype": rng.choice(["behind", "cross_receiver", "support"], n),
        "frame_start": np.arange(n) * 50,
        "frame_end": np.arange(n) * 50 + 30,
        "team_id": rng.choice([10, 20], n),
        "defender_id": rng.choice([1.0, 2.0, 3.0], n),
        "targeted": rng.random(n) > 0.6,
        "received": rng.random(n) > 0.7,
        "lead_to_shot": rng.random(n) > 0.88,
        "out_of_possession_phase": rng.choice(["low_block", "medium_block"], n),
        "third_start": rng.choice(["attacking_third", "middle_third"], n),
        "channel_start": rng.choice(["left", "centre", "right"], n),
        "pp_pressure": rng.choice(["low", "high"], n),
    })
    for c in LEARNED_NUM:
        d[c] = rng.normal(0, 1, n)
    # il gain dipende dalla geometria piu' rumore: il modello deve trovarlo
    d["separation_gain"] = 2.0 * d["geo_dist_start"] + rng.normal(0, 1, n)
    assert all(c in d.columns for c in LEARNED_NUM + LEARNED_CAT)
    return d


def test_has_learned_columns(toy, toy_learned):
    from drst import has_learned_columns
    assert not has_learned_columns(toy)
    assert has_learned_columns(toy_learned)


def test_learned_containment_raises_without_tracking(toy):
    from drst import learned_containment
    with pytest.raises(ValueError, match="colonne mancanti"):
        learned_containment(toy, DEFAULT)


def test_learned_containment_in_unit_interval(toy_learned):
    from drst import learned_containment
    c = learned_containment(toy_learned, DEFAULT)
    assert ((c > 0) & (c < 1)).all()
    assert len(c) == len(toy_learned)


def test_learned_containment_ignores_outcomes(toy_learned):
    """L'INVARIANTE che blocca la circolarita': il containment appreso non
    deve cambiare se si invertono gli esiti. Il suo target e'
    separation_gain, non lead_to_shot."""
    from drst import learned_containment
    a = learned_containment(toy_learned, DEFAULT)
    flipped = toy_learned.copy()
    for col in DEFAULT.outcomes:
        flipped[col] = ~flipped[col].astype(bool)
    b = learned_containment(flipped, DEFAULT)
    assert np.allclose(a.values, b.values)


def test_learned_containment_segue_il_residuo(toy_learned):
    """Chi si libera piu' del previsto dalla geometria deve avere containment
    piu' alto: il segno del residuo e l'ordine devono corrispondere."""
    from drst import learned_containment
    c = learned_containment(toy_learned, DEFAULT)
    # il gain e' 2*geo_dist_start + rumore: a parita' di geometria, il
    # containment deve ordinare come il rumore, non come il gain grezzo
    resid_proxy = toy_learned["separation_gain"] - 2.0 * toy_learned["geo_dist_start"]
    # 0.7 e non 0.9: su 240 righe e 6 fold il boosting non ricostruisce
    # esattamente una relazione lineare, e non deve farlo per passare
    assert c.corr(resid_proxy) > 0.7
    assert c.corr(resid_proxy) > c.corr(toy_learned["separation_gain"])


def test_fixed_containment_non_rifitta(toy_learned):
    """fixed_containment deve restituire gli stessi valori su un subset,
    allineati per indice: e' cio' che rende l'out-of-fold riusabile in OOS."""
    from drst import learned_containment, fixed_containment
    c = learned_containment(toy_learned, DEFAULT)
    wrapped = fixed_containment(c)
    sub = toy_learned.iloc[10:40]
    assert np.allclose(wrapped(sub, DEFAULT).values, c.iloc[10:40].values)


def test_learned_n_splits_allineato_alla_validazione():
    """learned_n_splits deve coincidere col default di dissociation_oos,
    altrimenti le partizioni GroupKFold non sono le stesse e la previsione
    out-of-fold non e' piu' fuori dal train fold della validazione."""
    import inspect
    from drst import dissociation_oos
    sig = inspect.signature(dissociation_oos)
    assert sig.parameters["n_splits"].default == DEFAULT.learned_n_splits
