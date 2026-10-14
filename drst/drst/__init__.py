"""DRS(t) — Defensive Response Signature come curva temporale.

Ogni off-ball run e' uno stress test che l'attacco applica alla difesa.
Lo stress si accumula e decade: DRS(t) e' lo stress difensivo istantaneo.

    stress = danger_prior x containment_failure x structural_vulnerability

Lo stress NON usa mai gli esiti (targeted/received/lead_to_shot): restano
fuori per la validazione.

Uso minimo:

    from drst import run_pipeline
    res = run_pipeline("data/layer3_enriched.csv")
    res.save("out/drst_v1")
"""

from .config import DRSConfig, DEFAULT, FAMILY_MAP
from .pipeline import run_pipeline, load_layer3, DRSResult
from .priors import (outcome_priors, spatial_priors, map_spatial_prior,
                     geometric_priors)
from .stress import (compute_stress, separation_gain_containment,
                     structural_vulnerability, has_structural_columns,
                     separation_curve_containment, has_curve_columns,
                     learned_containment, has_learned_columns,
                     fixed_containment)
from .curve import build_curve, build_all_curves, run_kernel
from .indexing import (FrameIndex, IndexingStrategy, EventAnchoredIndex,
                       DenseTrackingIndex, PhaseWindowIndex, get_strategy)
from .aggregate import (add_defending_team, team_match_metrics, team_profile,
                        defender_profile, team_two_axis, phase_weights,
                        phase_standardized)
from .tracking_features import (extract_match, extract_all,
                                merge_into_layer3, separation_curve,
                                separation_curve_features, structure_features,
                                identify_goalkeepers)
from .animate import (load_tracking, infer_player_teams, stress_field,
                      active_weights, find_peak_windows, animate_window,
                      animate_match, draw_pitch, Tracking)
from .validate import (dissociation, dissociation_oos, split_half,
                       split_half_curves, spearman_brown, paired_split_half,
                       pre_run_drs, kernel_sensitivity)

__version__ = "0.1.0"

__all__ = [
    "DRSConfig", "DEFAULT", "FAMILY_MAP",
    "run_pipeline", "load_layer3", "DRSResult",
    "outcome_priors", "spatial_priors", "map_spatial_prior", "geometric_priors",
    "compute_stress", "separation_gain_containment",
    "structural_vulnerability", "has_structural_columns",
    "separation_curve_containment", "has_curve_columns",
    "learned_containment", "has_learned_columns", "fixed_containment",
    "extract_match", "extract_all", "merge_into_layer3", "separation_curve",
    "separation_curve_features", "structure_features", "identify_goalkeepers",
    "build_curve", "build_all_curves", "run_kernel",
    "FrameIndex", "IndexingStrategy", "EventAnchoredIndex",
    "DenseTrackingIndex", "PhaseWindowIndex", "get_strategy",
    "add_defending_team", "team_match_metrics", "team_profile", "defender_profile",
    "team_two_axis", "phase_weights", "phase_standardized",
    "dissociation", "dissociation_oos", "split_half", "split_half_curves",
    "spearman_brown", "paired_split_half", "pre_run_drs", "kernel_sensitivity",
    "load_tracking", "infer_player_teams", "stress_field", "active_weights",
    "find_peak_windows", "animate_window", "animate_match", "draw_pitch", "Tracking",
]
