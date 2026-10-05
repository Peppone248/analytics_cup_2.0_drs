"""Batched adaptive-R constant-velocity Kalman filter — copied from the thesis
(master_thesis/src/awareness/kalman/filter.py, run_filter_batched + CWNA Q with
the 2026-08-02 Kronecker-order fix). Changes: (1) it also returns the velocity
estimate (to draw the ghost's heading); (2) optional alternative motion prior
`motion` (see ghost.motion; default None = identical to the thesis).

  State   : [px, py, vx, vy]   Observe: [px, py]
  R_ij(t) = sigma_obs^2 / w_ij(t) * I2 ; update skipped when w_ij <= w_min.
"""

import numpy as np

def _build_cwna_Q(dt: float, sigma_acc: float) -> np.ndarray:
    """
    Build process noise matrix Q under CWNA (Constant White Noise Acceleration).

    Assumes acceleration is zero-mean white noise with std sigma_acc (m/s²).
    Derived by integrating acceleration noise through the CV kinematic model.
    x and y axes are independent; each couples position and velocity.
    Ref: Bar-Shalom 2001, eq. 6.3.3

    Args:
        dt        : time step in seconds (e.g. 0.04 for 25 Hz)
        sigma_acc : acceleration noise std in m/s²
                    NOT literal target kinematics (that range is ~0.3-1.0
                    m/s² for a sprinting player) -- it also encodes how fast
                    the DEFENDER's belief should degrade while unmarked, a
                    cognitive/attentional quantity, not a physical one. The
                    project default (2.0, see run_awareness.py) sits above
                    the pure-kinematic range deliberately: validated
                    2026-07-28 on match 2036211 that values inside the
                    kinematic range make belief uncertainty grow too slowly
                    to be useful (see src/awareness/score.py, thesis
                    decision #6).

    Returns:
        Q : (4, 4) process noise matrix, state order [px, py, vx, vy]

    Note on the Kronecker order (bug fixed 2026-08-02): Bar-Shalom's Q_s is the
    per-axis block for a state ordered [p, v], so the whole-state Q is
    blkdiag(Q_s, Q_s) = I_2 (x) Q_s ONLY when the state is INTERLEAVED
    [px, vx, py, vy]. Our state (and F, H, P0) is GROUPED [px, py, vx, vy], for
    which the correct matrix is the permuted Q_s (x) I_2. The pre-fix code used
    the interleaved form against the grouped state: it put dt^2 on P_yy (2500x
    too large at dt=0.04) and dt^4/4 on vx (2500x too small), dropped the
    position-velocity coupling entirely, and invented spurious x-y correlations
    -- which is why production runs showed a systematic P_yy >> P_xx.
    """
    dt2 = dt ** 2
    dt3 = dt ** 3
    dt4 = dt ** 4

    Q_scalar = np.array([
        [dt4 / 4, dt3 / 2],
        [dt3 / 2, dt2    ],
    ])

    return sigma_acc ** 2 * np.kron(Q_scalar, np.eye(2))




def run_filter_batched(
    positions: np.ndarray,
    w: np.ndarray,
    valid: np.ndarray,
    dt: float = 0.04,
    sigma_obs: float = 1.0,
    sigma_acc: float = 2.0,
    w_min: float = 0.01,
    p0: float = 100.0,
    motion: dict | None = None,
    init_on: str = "valid",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised adaptive-R Kalman over many (observer, target) pairs.

    Same semantics as the thesis `run_filter_batched` (init at first valid
    frame with the observed position, zero velocity, P = p0*I; predict through
    interior gaps; hard-skip updates when w <= w_min).

    Args:
        positions : (P, N, 2) target positions per pair per frame (NaN if invalid).
        w         : (P, N) FoV weights in [0, 1] (0 at invalid frames).
        valid     : (P, N) bool.
        motion    : None = thesis model (CV + DWNA Q with sigma_acc). Otherwise a
                    dict for ghost.motion.make_FQ, e.g.
                    {'model': 'ou', 'tau_v': 1.0, 'sigma_v': 2.0}.
        init_on   : 'valid' (thesis) — belief created at the first frame the pair
                    exists, from the TRUE position even if never seen;
                    'seen' — belief created at the first frame with w > w_min.
                    Before that the pair has no belief (outputs NaN).

    Returns:
        x_hat  : (P, N, 2) belief mean (the "ghost" position). NaN if invalid.
        v_hat  : (P, N, 2) belief velocity. NaN if invalid.
        P_hist : (P, N, 2, 2) positional covariance. NaN if invalid.
    """
    P, N, _ = positions.shape
    if motion is None:
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        Q = _build_cwna_Q(dt=dt, sigma_acc=sigma_acc)
    else:
        from .motion import make_FQ
        F, Q = make_FQ(dt=dt, **{"sigma_acc": sigma_acc, **motion})
    P0 = np.eye(4) * p0
    sig2 = sigma_obs ** 2

    X = np.zeros((P, 4))
    Pc = np.tile(P0, (P, 1, 1))
    initialized = np.zeros(P, dtype=bool)

    x_hat = np.full((P, N, 2), np.nan)
    v_hat = np.full((P, N, 2), np.nan)
    P_hist = np.full((P, N, 2, 2), np.nan)

    for t in range(N):
        vt = valid[:, t]
        can_init = vt if init_on == "valid" else (vt & (w[:, t] > w_min))
        newinit = can_init & ~initialized
        if newinit.any():
            X[newinit, :2] = positions[newinit, t, :]
            X[newinit, 2:] = 0.0
            Pc[newinit] = P0
            initialized[newinit] = True

        init = initialized
        if init.any():
            X[init] = X[init] @ F.T
            Pc[init] = F @ Pc[init] @ F.T + Q

        upd = vt & (w[:, t] > w_min)
        if upd.any():
            z = positions[upd, t, :]
            Pm = Pc[upd]
            R = (sig2 / w[upd, t])[:, None, None] * np.eye(2)
            S = Pm[:, :2, :2] + R
            K = Pm[:, :, :2] @ np.linalg.inv(S)
            y = z - X[upd, :2]
            X[upd] = X[upd] + np.einsum("mij,mj->mi", K, y)
            KH = np.zeros((int(upd.sum()), 4, 4))
            KH[:, :, :2] = K
            Pc[upd] = Pm - KH @ Pm

        rec = vt & initialized
        if rec.any():
            x_hat[rec, t] = X[rec, :2]
            v_hat[rec, t] = X[rec, 2:]
            P_hist[rec, t] = Pc[rec, :2, :2]

    return x_hat, v_hat, P_hist
