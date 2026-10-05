"""FoV + occlusion observation weights w_ij — copied from the thesis
(master_thesis/src/awareness/fov/compute_wij.py, itself adapted from Bekkers,
"Wide Open Gazes", SSAC 2026). Kept byte-for-byte in the maths so results are
comparable with the Euro 2024 numbers; only the frame-level wrapper differs
(`frame_wij` works on numpy arrays of one frame instead of a pandas slice).

All positions in pitch-centred metres. head_angle in radians (atan2 convention).
"""

import numpy as np
def _apparent_widths_batch(
    obs_xs: np.ndarray,
    obs_ys: np.ndarray,
    occ_ls_xs: np.ndarray,
    occ_ls_ys: np.ndarray,
    occ_rs_xs: np.ndarray,
    occ_rs_ys: np.ndarray,
) -> np.ndarray:
    """Angular diameter of each occluder as seen from each observer.

    Uses the actual left/right shoulder positions from skeletal data.
    Returns shape (n_obs, n_occ) in radians.
    """
    ls = np.stack([occ_ls_xs, occ_ls_ys], axis=-1)  # (n_occ, 2)
    rs = np.stack([occ_rs_xs, occ_rs_ys], axis=-1)  # (n_occ, 2)

    shoulder_vec = rs - ls                                          # (n_occ, 2)
    shoulder_len = np.linalg.norm(shoulder_vec, axis=-1)           # (n_occ,)

    perp = np.stack(
        [-shoulder_vec[:, 1], shoulder_vec[:, 0]], axis=-1
    ) / (shoulder_len[:, None] + 1e-10)                            # (n_occ, 2)

    half_d = shoulder_len / 4.0                                    # (n_occ,)

    # (n_occ, 4, 2)
    global_corners = np.stack([
        rs + perp * half_d[:, None],
        ls + perp * half_d[:, None],
        ls - perp * half_d[:, None],
        rs - perp * half_d[:, None],
    ], axis=1)

    observers = np.stack([obs_xs, obs_ys], axis=-1)                # (n_obs, 2)
    vectors   = global_corners[None, :, :, :] - observers[:, None, None, :]
    # (n_obs, n_occ, 4, 2)

    v1   = vectors[:, :, [0, 1], :]                                # (n_obs, n_occ, 2, 2)
    v2   = vectors[:, :, [2, 3], :]
    dots = np.sum(v1 * v2, axis=-1)                                # (n_obs, n_occ, 2)
    mags = np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1)
    spans = np.arccos(np.clip(dots / (mags + 1e-10), -1.0, 1.0))

    return np.max(spans, axis=-1)                                  # (n_obs, n_occ)


def compute_wij_batch(
    obs_xs: np.ndarray,
    obs_ys: np.ndarray,
    obs_head_angles: np.ndarray,
    obs_speeds: np.ndarray,
    obs_ids: np.ndarray,
    tgt_xs: np.ndarray,
    tgt_ys: np.ndarray,
    tgt_ids: np.ndarray,
    occ_ls_xs: np.ndarray,
    occ_ls_ys: np.ndarray,
    occ_rs_xs: np.ndarray,
    occ_rs_ys: np.ndarray,
    occ_ids: np.ndarray,
    *,
    wedge_radius: float = 150.0,
    vision_angle: int = 120,
    sigma_r: float = 75.0,
    sigma_a_fov: float = 0.4,
    sigma_a_occ: float = 0.2,
    max_occlusion_alpha: float = 0.7,
    proximity_radius: float = 0.5,
    proximity_alpha: float = 0.7,
) -> np.ndarray:
    """Vectorised compute_wij over all observers simultaneously.

    Parameters
    ----------
    obs_xs, obs_ys : (n_obs,) array
    obs_head_angles : (n_obs,) array
    obs_speeds : (n_obs,) array
    obs_ids : (n_obs,) array
    tgt_xs, tgt_ys : (n_tgt,) array
    tgt_ids : (n_tgt,) array
    occ_ls_xs, occ_ls_ys : (n_occ,) array
        Left shoulder positions of the full occluder set, pitch-centred metres.
    occ_rs_xs, occ_rs_ys : (n_occ,) array
        Right shoulder positions of the full occluder set, pitch-centred metres.
        obs==occ and tgt==occ shadow entries are zeroed internally.
    occ_ids : (n_occ,) array

    Returns
    -------
    w : (n_obs, n_tgt) array
    """
    obs_xs          = np.asarray(obs_xs, dtype=float)
    obs_ys          = np.asarray(obs_ys, dtype=float)
    obs_head_angles = np.asarray(obs_head_angles, dtype=float)
    obs_speeds      = np.asarray(obs_speeds, dtype=float)
    obs_ids         = np.asarray(obs_ids)
    tgt_xs          = np.asarray(tgt_xs, dtype=float)
    tgt_ys          = np.asarray(tgt_ys, dtype=float)
    tgt_ids         = np.asarray(tgt_ids)
    occ_ls_xs       = np.asarray(occ_ls_xs, dtype=float)
    occ_ls_ys       = np.asarray(occ_ls_ys, dtype=float)
    occ_rs_xs       = np.asarray(occ_rs_xs, dtype=float)
    occ_rs_ys       = np.asarray(occ_rs_ys, dtype=float)
    occ_ids         = np.asarray(occ_ids)

    # ------------------------------------------------------------------ FoV --
    dx = tgt_xs[None, :] - obs_xs[:, None]          # (n_obs, n_tgt)
    dy = tgt_ys[None, :] - obs_ys[:, None]
    d  = np.sqrt(dx**2 + dy**2)

    tgt_angles     = np.arctan2(dy, dx)
    vision_radians = vision_angle * (np.pi / 360.0)

    theta_head = np.arctan2(
        np.sin(tgt_angles - obs_head_angles[:, None]),
        np.cos(tgt_angles - obs_head_angles[:, None]),
    )
    in_cone = (d <= wedge_radius) & (np.abs(theta_head) <= vision_radians)

    c_r = np.minimum(0.25 * obs_speeds + 0.1, 2.6)[:, None]  # (n_obs, 1)
    c_a = np.minimum(0.03 * obs_speeds + 0.2, 0.5)[:, None]
    fov_vals = (
        np.exp(-c_r * (d / sigma_r) ** 2)
        * np.exp(-c_a * (theta_head / sigma_a_fov) ** 2)
        * in_cone
    )  # (n_obs, n_tgt)

    if occ_ls_xs.size == 0:
        return np.where(d < proximity_radius, np.maximum(proximity_alpha, fov_vals), fov_vals)

    # -------------------------------------------------------------- Occlusion --
    # Derive occluder centres from shoulder midpoints
    occ_xs = (occ_ls_xs + occ_rs_xs) / 2.0          # (n_occ,)
    occ_ys = (occ_ls_ys + occ_rs_ys) / 2.0

    occ_dx   = occ_xs[None, :] - obs_xs[:, None]    # (n_obs, n_occ)
    occ_dy   = occ_ys[None, :] - obs_ys[:, None]
    dist_occ = np.sqrt(occ_dx**2 + occ_dy**2)
    occ_ang  = np.arctan2(occ_dy, occ_dx)

    # Minimum distance from observer to the nearer shoulder endpoint
    dist_sl = np.sqrt(
        (occ_ls_xs[None, :] - obs_xs[:, None]) ** 2
        + (occ_ls_ys[None, :] - obs_ys[:, None]) ** 2
    )
    dist_sr = np.sqrt(
        (occ_rs_xs[None, :] - obs_xs[:, None]) ** 2
        + (occ_rs_ys[None, :] - obs_ys[:, None]) ** 2
    )
    min_dist_occ = np.minimum(dist_occ, np.minimum(dist_sl, dist_sr))  # (n_obs, n_occ)

    theta_diff = np.arctan2(
        np.sin(tgt_angles[:, :, None] - occ_ang[:, None, :]),
        np.cos(tgt_angles[:, :, None] - occ_ang[:, None, :]),
    )  # (n_obs, n_tgt, n_occ)

    omega       = _apparent_widths_batch(obs_xs, obs_ys, occ_ls_xs, occ_ls_ys, occ_rs_xs, occ_rs_ys)
    c_s         = dist_occ / (omega + 1e-10)         # (n_obs, n_occ)
    occ_profile = np.exp(-c_s[:, None, :] * (theta_diff / sigma_a_occ) ** 2)

    shadow = max_occlusion_alpha * (d[:, :, None] > min_dist_occ[:, None, :]).astype(float)
    # (n_obs, n_tgt, n_occ)

    # Zero out: target cannot occlude itself
    shadow *= ~(tgt_ids[:, None] == occ_ids[None, :])[None, :, :]   # (1, n_tgt, n_occ)

    # Zero out: observer cannot occlude its own targets
    shadow *= ~(obs_ids[:, None] == occ_ids[None, :])[:, None, :]   # (n_obs, 1, n_occ)

    occ_vals = np.prod(1.0 - occ_profile * shadow, axis=2)          # (n_obs, n_tgt)

    w = fov_vals * occ_vals
    return np.where(d < proximity_radius, np.maximum(proximity_alpha, w), w)


def frame_wij(obs: dict, tgt: dict, occ: dict, **kw) -> np.ndarray:
    """w (n_obs, n_tgt) for one frame. Each arg is a dict of 1-D numpy arrays.

    obs: x, y, head, speed, id      tgt: x, y, id
    occ: ls_x, ls_y, rs_x, rs_y, id  (normally every player on the pitch)
    """
    return compute_wij_batch(
        obs["x"], obs["y"], obs["head"], obs["speed"], obs["id"],
        tgt["x"], tgt["y"], tgt["id"],
        occ["ls_x"], occ["ls_y"], occ["rs_x"], occ["rs_y"], occ["id"], **kw)
