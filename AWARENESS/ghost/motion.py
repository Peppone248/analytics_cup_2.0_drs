"""Motion priors for the belief state: transition F and process noise Q.

Two families, per axis (x and y independent), state [p, v]:

* 'cv_thesis' : constant velocity with the thesis' Q
                sigma_acc^2 * [[dt^4/4, dt^3/2], [dt^3/2, dt^2]].
                NB: this is Bar-Shalom's *discrete* white-noise acceleration
                (DWNA, piecewise-constant acceleration, eq. 6.3.2-x) form, not the
                continuous one (CWNA has dt^3/3, dt^2/2, dt). Both are legit;
                only the naming in the thesis docstrings is off.
* 'ou'        : Ornstein-Uhlenbeck (Singer-style) velocity,
                dp = v dt,   dv = -v/tau dt + sqrt(2 sigma_v^2 / tau) dW
                -> stationary speed std sigma_v per axis, velocity memory tau.
                An unseen attacker's ghost extrapolates his last run for ~tau
                seconds then settles, and the uncertainty grows like a random
                walk with diffusion 2 sigma_v^2 tau (bounded velocity, unbounded
                position). Discretised exactly with Van Loan's method.
* 'cv_cont'   : continuous white-noise acceleration (q = sigma_acc^2), exact.

The full 4x4 matrices use the GROUPED state [px, py, vx, vy] like the thesis.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm


def _van_loan(A: np.ndarray, L: np.ndarray, qc: float, dt: float):
    n = A.shape[0]
    M = np.zeros((2 * n, 2 * n))
    M[:n, :n] = -A
    M[:n, n:] = L @ L.T * qc
    M[n:, n:] = A.T
    E = expm(M * dt)
    F = E[n:, n:].T
    Q = F @ E[:n, n:]
    return F, (Q + Q.T) / 2


def _axis_to_grouped(F1: np.ndarray, Q1: np.ndarray):
    # per-axis [p, v] block -> grouped [px, py, vx, vy]
    return np.kron(F1, np.eye(2)), np.kron(Q1, np.eye(2))


def make_FQ(dt: float = 0.04, model: str = "cv_thesis", sigma_acc: float = 2.0,
            tau_v: float = 1.0, sigma_v: float = 2.0):
    if model == "cv_thesis":
        F1 = np.array([[1.0, dt], [0.0, 1.0]])
        Q1 = sigma_acc ** 2 * np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]])
    elif model == "cv_cont":
        F1, Q1 = _van_loan(np.array([[0.0, 1.0], [0.0, 0.0]]), np.array([[0.0], [1.0]]),
                           sigma_acc ** 2, dt)
    elif model == "ou":
        F1, Q1 = _van_loan(np.array([[0.0, 1.0], [0.0, -1.0 / tau_v]]), np.array([[0.0], [1.0]]),
                           2 * sigma_v ** 2 / tau_v, dt)
    else:
        raise ValueError(model)
    return _axis_to_grouped(F1, Q1)
