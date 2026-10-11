"""Test del package drst. pytest tests/"""

import numpy as np
import pandas as pd
import pytest

from drst import (DRSConfig, DEFAULT, outcome_priors, compute_stress,
                  run_kernel, build_curve, add_defending_team,
                  EventAnchoredIndex, separation_gain_containment, DenseTrackingIndex)




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
