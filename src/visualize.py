import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from .diagnostics import two_point_correlation


def animate_trajectory(trajectory):
    steps, N, _ = trajectory.shape
    fig = plt.figure()
    ax = plt.axes()
    scatter = ax.scatter(trajectory[0, :, 0], trajectory[0, :, 1])
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    def update(frame):
        scatter.set_offsets(trajectory[frame, :, :2])
        return scatter,

    ani = animation.FuncAnimation(fig, update, frames=steps, interval=30, blit=True)
    return ani


def plot_trajectories(records, *, figsize=(10, 4), xlim=None, ylim=None):
    fig, axes = plt.subplots(1, len(records), figsize=figsize, sharex=True, sharey=True)
    if len(records) == 1:
        axes = [axes]
    for rec, ax in zip(records, axes):
        traj = rec["positions"]
        for idx in range(traj.shape[1]):
            ax.plot(traj[:, idx, 0], traj[:, idx, 1], label=f"body {idx}")
        ax.set_title(f"{rec['force_type'].capitalize()} trajectories")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        if xlim:
            ax.set_xlim(*xlim)
        if ylim:
            ax.set_ylim(*ylim)
    axes[-1].legend(loc="best")
    fig.tight_layout()
    return fig


def plot_snapshots(records, times, *, figsize=(12, 6), xlim=(-2, 2), ylim=(-2, 2)):
    rows = len(records)
    cols = len(times)
    fig, axs = plt.subplots(rows, cols, figsize=figsize, sharex=True, sharey=True)
    if rows == 1:
        axs = np.expand_dims(axs, 0)
    for row_idx, rec in enumerate(records):
        for col_idx, t_idx in enumerate(times):
            ax = axs[row_idx, col_idx]
            pos = rec["positions"][t_idx]
            ax.scatter(pos[:, 0], pos[:, 1], c=rec["masses"], cmap="plasma", s=60)
            ax.set_title(f"{rec['force_type'].capitalize()} t={rec['time'][t_idx]:.2f}")
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    return fig


def plot_pair_separation_histogram(records, *, bins=20, density=True, figsize=(6, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    for rec in records:
        positions = rec["positions"][-1]
        diff = positions[:, None, :] - positions[None, :, :]
        i, j = np.triu_indices(positions.shape[0], k=1)
        distances = np.linalg.norm(diff, axis=-1)[i, j]
        ax.hist(
            distances,
            bins=bins,
            density=density,
            histtype="step",
            label=f"{rec['force_type']} (lam={rec['lambda']})",
            alpha=0.9,
        )
    ax.set_xlabel("Pair separation")
    ax.set_ylabel("Density")
    ax.set_title("Final pair-separation histograms")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_clustering_overlay(records, bin_edges, *, figsize=(6, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    for rec in records:
        centers, xi = two_point_correlation(rec["positions"][-1], bin_edges)
        ax.plot(centers, xi, marker="o", label=f"{rec['force_type']} (lam={rec['lambda']})")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Separation r")
    ax.set_ylabel("ξ(r)")
    ax.set_title("Newtonian vs Yukawa clustering")
    ax.legend()
    fig.tight_layout()
    return fig
