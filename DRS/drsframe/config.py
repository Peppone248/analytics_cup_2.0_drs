"""Configurazione del modello. Ogni parametro e' una scelta di modellazione
documentata: cambiarlo cambia la metrica, quindi va dichiarato nel paper."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class Config:
    # --- tracking ---------------------------------------------------------
    fps: float = 10.0                 # SkillCorner broadcast tracking
    smooth_frames: int = 5            # media mobile sulle posizioni prima delle velocita'
    max_speed_cap: float = 12.0       # m/s, oltre e' rumore di tracking

    # --- time-to-intercept (Spearman 2018; Bekkers 2025, Pressing Intensity) --
    reaction_time: float = 0.7        # s, il difensore mantiene la velocita' attuale
    v_max: float = 7.0                # m/s, velocita' massima di corsa
    sigma: float = 0.45               # s, incertezza logistica sul tempo di arrivo
    ball_speed: float = 15.0          # m/s, velocita' media del passaggio a terra
    kick_time: float = 0.3            # s, tempo di esecuzione del passaggio
    reception_slack: float = 0.5      # s, il difensore contende se arriva entro questo margine
    lane_points: int = 8              # punti campionati sulla linea di passaggio
    carrier_pressure_T: float = 1.5   # s, soglia Pressing Intensity per il portatore

    # --- marcatura ---------------------------------------------------------
    mark_goal_side_offset: float = 1.5  # m, punto di marcatura ideale verso la propria porta
    mark_max_distance: float = 12.0     # m, oltre il difensore e' "a zona" (non assegnato)
    mark_hysteresis: float = 2.0        # m, bonus per mantenere l'assegnazione precedente

    # --- valore della posizione --------------------------------------------
    value_bandwidth: float = 4.0      # m, kernel Nadaraya-Watson sulla superficie EPV
    value_grid_step: float = 1.0      # m

    # --- esposizione -------------------------------------------------------
    include_carrier: bool = True      # esposizione del portatore (valore x (1 - pressione))
    project_attackers: bool = True    # proietta l'attaccante dove sara' all'arrivo della palla

    # --- modello expected --------------------------------------------------
    n_folds: int = 5                  # GroupKFold su match_id
    train_stride: int = 5             # 1 frame ogni 5 (2 Hz) per ridurre l'autocorrelazione
    random_state: int = 42
    context_phase: bool = False       # fase di possesso SkillCorner nel contesto: OFF, e' etichettata
                                      # a posteriori (es. "finish") e assorbe parte dell'effetto difensivo

    # --- eventi e ranking --------------------------------------------------
    shot_horizon_s: float = 5.0       # finestra per "tiro entro K secondi" (validazione)
    event_pre_s: float = 1.0          # finestra di risposta difensiva: inizio evento - pre
    event_post_s: float = 0.0         #                                  fine evento + post
    z_weight_floor: float = 0.0       # peso minimo di un evento nella media pesata
    min_frames_defender: int = 3000   # 5 min fuori possesso per entrare nel ranking difensori

    # --- video -------------------------------------------------------------
    video_fps: int = 10
    video_dpi: int = 110

    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
