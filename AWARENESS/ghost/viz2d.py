"""2D pitch renders: real players + the focal defender's ghosts.

Visual grammar (kept identical in the 3D viewer):
  * filled dots           = real players (team colour); focal defender has a ring
  * hollow dashed circle  = ghost: where the focal defender believes an attacker is
  * thin line ghost->real = ghost error, coloured by size (green < 1.5 m < amber < 4 m < red)
  * shaded ellipse        = 95% belief ellipse around the ghost (from P)
  * wedge                 = focal defender's +-60 deg field of view (head direction)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation
from matplotlib.patches import Circle, Ellipse, Wedge
from mplsoccer import Pitch

from .scores import ellipse_params

C_DEF = "#2f6db5"      # defending team (observer)
C_ATT = "#d1495b"      # attacking team (targets)
C_BALL = "#111111"
ERR_COLORS = [(1.5, "#2a9d8f"), (4.0, "#e9a23b"), (np.inf, "#d1495b")]


def err_color(e: float) -> str:
    for thr, c in ERR_COLORS:
        if e < thr:
            return c
    return ERR_COLORS[-1][1]


def _pitch(ax):
    p = Pitch(pitch_type="skillcorner", pitch_length=105, pitch_width=68,
              pitch_color="#f4f1ea", line_color="#9a9a9a", linewidth=1)
    p.draw(ax=ax)
    return p


def draw_frame(ax, res: pd.DataFrame, pl: pd.DataFrame, ball: pd.DataFrame, frame: int,
               focal: int, highlight_target: int | None = None, fov_len: float = 30.0,
               show_all_ghosts: bool = True, xlim=None, ylim=None):
    """Draw one frame on ax (pitch must already be drawn)."""
    r = res[(res.frame == frame) & (res.observer_id == focal)]
    g = pl[pl.frame == frame]
    artists = []

    # FoV wedge
    if len(r):
        ox, oy, oh = r.obs_x.iloc[0], r.obs_y.iloc[0], np.degrees(r.obs_head.iloc[0])
        src = r.obs_head_source.iloc[0]
        wedge = Wedge((ox, oy), fov_len, oh - 60, oh + 60, color=C_DEF,
                      alpha=0.10 if src != "ball_prior" else 0.04, lw=0, zorder=1)
        ax.add_patch(wedge); artists.append(wedge)

    # ghosts (belief of the focal defender)
    for row in r.itertuples():
        if not show_all_ghosts and row.target_id != highlight_target:
            continue
        if np.isnan(row.x_hat_x):            # never seen -> no ghost
            continue
        strong = highlight_target is None or row.target_id == highlight_target
        if strong:                            # belief ellipse only for the focal pair
            a, b, ang = ellipse_params(row.P_xx, row.P_yy, row.P_xy)
            el = Ellipse((row.x_hat_x, row.x_hat_y), 2 * a, 2 * b, angle=np.degrees(ang),
                         color=C_ATT, alpha=0.12, lw=0, zorder=2)
            ax.add_patch(el); artists.append(el)
        col = err_color(row.ghost_err)
        ln, = ax.plot([row.x_hat_x, row.target_x], [row.x_hat_y, row.target_y],
                      color=col, lw=2.2 if strong else 0.8, alpha=0.95 if strong else 0.35, zorder=3)
        gh = Circle((row.x_hat_x, row.x_hat_y), 0.9, fill=False, ec=C_ATT, ls="--",
                    lw=1.6 if strong else 0.7, alpha=0.95 if strong else 0.35, zorder=4)
        ax.add_patch(gh); artists += [ln, gh]

    # real players
    for p in g.itertuples():
        is_def = p.team_id == g.loc[g.player_id == focal, "team_id"].iloc[0]
        c = C_DEF if is_def else C_ATT
        dot = Circle((p.x, p.y), 0.9, color=c, ec="white", lw=0.8,
                     alpha=1.0 if p.is_detected else 0.45, zorder=5)
        ax.add_patch(dot); artists.append(dot)
        if p.player_id == focal:
            ring = Circle((p.x, p.y), 1.6, fill=False, ec=C_DEF, lw=2, zorder=5)
            ax.add_patch(ring); artists.append(ring)
        if p.player_id == highlight_target:
            ring = Circle((p.x, p.y), 1.6, fill=False, ec=C_ATT, lw=2, zorder=5)
            ax.add_patch(ring); artists.append(ring)
        t = ax.text(p.x, p.y, str(int(p.number)), color="white", fontsize=6,
                    ha="center", va="center", zorder=6, fontweight="bold")
        artists.append(t)

    b = ball[ball.frame == frame]
    if len(b) and not np.isnan(b.ball_x.iloc[0]):
        bd = Circle((b.ball_x.iloc[0], b.ball_y.iloc[0]), 0.5, color="white", ec=C_BALL, lw=1.2, zorder=7)
        ax.add_patch(bd); artists.append(bd)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    return artists


def _auto_zoom(res, focal, pad=12):
    r = res[res.observer_id == focal]
    xs = np.r_[r.obs_x, r.target_x, r.x_hat_x]
    ys = np.r_[r.obs_y, r.target_y, r.x_hat_y]
    xl = (max(np.nanpercentile(xs, 2) - pad, -55), min(np.nanpercentile(xs, 98) + pad, 55))
    yl = (max(np.nanpercentile(ys, 2) - pad, -36), min(np.nanpercentile(ys, 98) + pad, 36))
    return xl, yl


def animate_window(res, pl, ball, focal, target=None, out_path="ghost.gif",
                   step=2, fps=12.5, roi_only=True, title=""):
    """GIF of the window: pitch (left) + focal pair time series (right)."""
    r = res[res.in_roi] if roi_only else res
    frames = np.sort(r.frame.unique())[::step]
    xl, yl = _auto_zoom(r, focal)
    pair = r[(r.observer_id == focal) & (r.target_id == target)].sort_values("frame") if target else None

    fig = plt.figure(figsize=(12, 5.2))
    axp = fig.add_axes([0.01, 0.05, 0.60, 0.88])
    axs = [fig.add_axes([0.67, 0.68 - i * 0.29, 0.31, 0.23]) for i in range(3)]

    if pair is not None and len(pair):
        t = pair.t_s.values
        axs[0].plot(t, pair.w_ij, color=C_DEF); axs[0].set_ylabel("FoV w_ij"); axs[0].set_ylim(-0.05, 1.05)
        axs[1].plot(t, pair.ghost_err, color=C_ATT); axs[1].set_ylabel("ghost err (m)")
        axs[1].axhline(1.5, color="#999", lw=0.8, ls=":")
        axs[2].plot(t, pair.A_cov, color="#555", label="A_cov (thesis)")
        axs[2].plot(t, pair.A_ghost, color=C_ATT, label="A_ghost (new)")
        axs[2].set_ylim(-0.05, 1.05); axs[2].legend(fontsize=7, loc="lower left"); axs[2].set_xlabel("t (s)")
        cursors = [a.axvline(t[0], color="k", lw=0.8) for a in axs]
        for a in axs:
            a.tick_params(labelsize=7); a.yaxis.label.set_size(8)
    else:
        cursors = []

    def update(f):
        axp.clear(); _pitch(axp)
        draw_frame(axp, res, pl, ball, f, focal, highlight_target=target, xlim=xl, ylim=yl)
        ts = (f - r.frame.min()) / 25
        axp.set_title(f"{title}   t = {ts:4.1f} s", fontsize=9)
        for c in cursors:
            c.set_xdata([ (f - r[r.in_roi].frame.min()) / 25 ] * 2)
        return []

    ani = animation.FuncAnimation(fig, update, frames=frames, blit=False)
    ani.save(out_path, writer=animation.PillowWriter(fps=fps), dpi=80)
    plt.close(fig)
    return out_path


def snapshot(res, pl, ball, focal, frame, target=None, out_path=None, title=""):
    fig, ax = plt.subplots(figsize=(8, 5.4))
    _pitch(ax)
    xl, yl = _auto_zoom(res[res.in_roi], focal)
    draw_frame(ax, res, pl, ball, frame, focal, highlight_target=target, xlim=xl, ylim=yl)
    ax.set_title(title, fontsize=9)
    if out_path:
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
    return fig
