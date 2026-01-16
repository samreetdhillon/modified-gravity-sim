import math

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from .diagnostics import two_point_correlation

# ============================================================
# 1. Animation
# ============================================================
def animate_trajectory(trajectory, *, tail_length=None, title=None, frame_skip=1):
    steps, N, _ = trajectory.shape
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = plt.cm.tab10(np.arange(N) % 10)

    scatter = ax.scatter(trajectory[0, :, 0], trajectory[0, :, 1], c=colors, s=36, zorder=3)

    # Pre-create line artists so we can update the paths each frame
    lines = [
        ax.plot([], [], linestyle="--", linewidth=1, alpha=0.7, color=colors[idx], zorder=2)[0]
        for idx in range(N)
    ]

    x_vals = trajectory[:, :, 0]
    y_vals = trajectory[:, :, 1]
    x_min, x_max = x_vals.min(), x_vals.max()
    y_min, y_max = y_vals.min(), y_vals.max()

    # Force a fixed zoom for N-body clarity
    if N > 3:
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
    else:
        x_pad = max((x_max - x_min) * 0.1, 0.1)
        y_pad = max((y_max - y_min) * 0.1, 0.1)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

    '''
    x_pad = max((x_max - x_min) * 0.1, 0.1)
    y_pad = max((y_max - y_min) * 0.1, 0.1)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    '''
    
    ax.set_aspect("equal")
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect(1)
    ax.set_xlabel("x (length units)")
    ax.set_ylabel("y (length units)")
    ax.set_title(title or "Trajectory (x-y plane)")

    frame_skip = max(1, frame_skip)
    frame_indices = range(0, steps, frame_skip)
    def update(frame):
        scatter.set_offsets(trajectory[frame, :, :2])
        start = 0 if tail_length is None else max(0, frame - tail_length + 1)
        for idx, line in enumerate(lines):
            segment = trajectory[start : frame + 1, idx]
            line.set_data(segment[:, 0], segment[:, 1])
        return (scatter, *lines)

    ani = animation.FuncAnimation(fig, update, frames=frame_indices, interval=30, blit=True)
    return ani


# ============================================================
# 2. Trajectory plots
# ============================================================
def plot_trajectories(records, *, figsize=None, xlim=None, ylim=None):
    axis_size = 4
    nplots = max(1, len(records))
    ncols = min(2, nplots)
    nrows = math.ceil(nplots / ncols)
    if figsize is None:
        figsize = (ncols * axis_size, nrows * axis_size)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = np.array(axes).reshape(-1) if isinstance(axes, np.ndarray) else np.array([axes])

    for idx, rec in enumerate(records):
        ax = axes_flat[idx]
        traj = rec["positions"]
        force_type = rec.get("force_type", "unknown")

        # Force fixed zoom for comparison plots
        if traj.shape[1] > 3:
             ax.set_xlim(-15, 15)
             ax.set_ylim(-15, 15)

        # Build dynamic title
        if force_type == "mond":
            mu_type = rec.get("mond_mu", "simple")
            a0 = rec.get("a0", 1e-2)
            title = f"MOND (μ={mu_type}, a0={a0})"
        elif force_type == "yukawa":
            lam = rec.get("lambda", None)
            alpha = rec.get("yukawa_alpha", rec.get("alpha", 1.0))
            title = f"Yukawa (λ={lam}, α={alpha})"
        elif force_type == "dark_photon":
            alpha = rec.get("alpha", rec.get("dp_alpha", 0))
            lam_dp = rec.get("lam", rec.get("dp_lambda", None))
            title = f"Dark Photon (α={alpha}, λ={lam_dp})"
        else:
            title = "Newtonian"

        for body_idx in range(traj.shape[1]):
            ax.plot(traj[:, body_idx, 0], traj[:, body_idx, 1])

        ax.set_title(title)
        ax.set_xlabel("x (length units)")
        ax.set_ylabel("y (length units)")
        ax.set_aspect("equal", adjustable="box")
        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect(1)

        if xlim:
            ax.set_xlim(*xlim)
        if ylim:
            ax.set_ylim(*ylim)

    for ax in axes_flat[len(records):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


# ============================================================
# 3. Snapshot plots
# ============================================================
def plot_snapshots(records, times, *, figsize=None, xlim=None, ylim=None):
    axis_size = 3
    rows = len(records)
    cols = len(times)
    if figsize is None:
        figsize = (max(1, cols) * axis_size, max(1, rows) * axis_size)

    fig, axs = plt.subplots(rows, cols, figsize=figsize, sharex=True, sharey=True)
    if isinstance(axs, plt.Axes):
        axs = np.array([[axs]])
    else:
        axs = np.array(axs)
        if axs.ndim == 1:
            if rows == 1:
                axs = np.expand_dims(axs, 0)
            elif cols == 1:
                axs = np.expand_dims(axs, 1)

    for row_idx, rec in enumerate(records):
        force_type = rec.get("force_type", "unknown")
        if force_type == "mond":
            title_base = f"MOND (μ={rec.get('mond_mu', 'simple')}, a0={rec.get('a0', 1e-2)})"
        elif force_type == "yukawa":
            lam = rec.get("lambda", None)
            alpha = rec.get("yukawa_alpha", rec.get("alpha", 1.0))
            title_base = f"Yukawa (λ={lam}, α={alpha})"
        elif force_type == "dark_photon":
            alpha = rec.get("alpha", rec.get("dp_alpha", 0))
            lam_dp = rec.get("lam", rec.get("dp_lambda", None))
            title_base = f"Dark Photon (α={alpha}, λ={lam_dp})"
        else:
            title_base = "Newtonian"

        for col_idx, t_idx in enumerate(times):
            ax = axs[row_idx, col_idx]
            positions = rec["positions"][t_idx]
            ax.scatter(positions[:, 0], positions[:, 1], c=rec["masses"], cmap="plasma", s=60)

            ax.set_title(f"{title_base}\nt={rec['time'][t_idx]:.2f}")
            if xlim:
                ax.set_xlim(*xlim)
            if ylim:
                ax.set_ylim(*ylim)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.tight_layout()
    return fig


# ============================================================
# 4. Pair separation histogram
# ============================================================
def plot_pair_separation_histogram(records, *, bins=20, density=True, figsize=(6, 4)):
    fig, ax = plt.subplots(figsize=figsize)

    for rec in records:
        positions = rec["positions"][-1]
        diff = positions[:, None, :] - positions[None, :, :]
        i, j = np.triu_indices(positions.shape[0], k=1)
        distances = np.linalg.norm(diff, axis=-1)[i, j]

        force_type = rec.get("force_type", "unknown")

        if force_type == "mond":
            label = f"MOND (μ={rec.get('mond_mu','simple')})"
        elif force_type == "yukawa":
            alpha = rec.get("yukawa_alpha", rec.get("alpha", 1.0))
            label = f"Yukawa (λ={rec.get('lambda')}, α={alpha})"
        elif force_type == "dark_photon":
            alpha = rec.get("alpha", rec.get("dp_alpha", 0))
            lam_dp = rec.get("lam", rec.get("dp_lambda", None))
            label = f"Dark Photon (α={alpha}, λ={lam_dp})"
        else:
            label = "Newtonian"

        ax.hist(
            distances,
            bins=bins,
            density=density,
            histtype="step",
            label=label,
            alpha=0.9,
        )

    ax.set_xlabel("Pair separation")
    ax.set_ylabel("Density")
    ax.set_title("Final pair-separation histograms")
    ax.legend()
    fig.tight_layout()
    return fig


# ============================================================
# 5. Clustering overlay
# ============================================================
def plot_clustering_overlay(records, bin_edges, *, figsize=(6, 4)):
    fig, ax = plt.subplots(figsize=figsize)

    for rec in records:
        centers, xi = two_point_correlation(rec["positions"][-1], bin_edges)
        force_type = rec.get("force_type", "unknown")

        if force_type == "mond":
            label = f"MOND (μ={rec.get('mond_mu', 'simple')})"
        elif force_type == "yukawa":
            alpha = rec.get("yukawa_alpha", rec.get("alpha", 1.0))
            label = f"Yukawa (λ={rec.get('lambda')}, α={alpha})"
        elif force_type == "dark_photon":
            alpha = rec.get("alpha", rec.get("dp_alpha", 0))
            lam_dp = rec.get("lam", rec.get("dp_lambda", None))
            label = f"Dark Photon (α={alpha}, λ={lam_dp})"
        else:
            label = "Newtonian"

        ax.plot(centers, xi, marker="o", label=label)

    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Separation r")
    ax.set_ylabel("ξ(r)")
    ax.set_title("Clustering Comparison")
    ax.legend()
    fig.tight_layout()
    return fig