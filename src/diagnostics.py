from __future__ import annotations
from typing import Sequence
import numpy as np
from .energy import compute_total_energy


def mean_interparticle_separation(positions: np.ndarray) -> float:
    """Return the average separation between unique particle pairs."""
    pos = np.asarray(positions, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("positions must be an (N, 3) array")

    N = pos.shape[0]
    if N < 2:
        return 0.0

    diff = pos[:, None, :] - pos[None, :, :]
    distances = np.linalg.norm(diff, axis=-1)
    i, j = np.triu_indices(N, k=1)
    return float(np.mean(distances[i, j]))


def two_point_correlation(
    positions: np.ndarray, 
    bin_edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a simple two-point correlation (ξ) for the provided positions."""
    pos = np.asarray(positions, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("positions must be an (N, 3) array")

    N = pos.shape[0]
    if N < 2:
        return np.zeros(bin_edges.size - 1), np.zeros(bin_edges.size - 1)

    diff = pos[:, None, :] - pos[None, :, :]
    distances = np.linalg.norm(diff, axis=-1)
    i, j = np.triu_indices(N, k=1)
    pair_dists = distances[i, j]

    counts, _ = np.histogram(pair_dists, bins=bin_edges)

    shell_volumes = (
        4.0 / 3.0 * np.pi * (bin_edges[1:]**3 - bin_edges[:-1]**3)
    )
    total_vol = 4.0 / 3.0 * np.pi * (np.max(bin_edges)**3)
    total_pairs = N * (N - 1) / 2

    expected = np.zeros_like(shell_volumes)
    if total_vol > 0:
        expected = total_pairs * (shell_volumes / total_vol)

    xi = np.zeros_like(counts, dtype=float)
    nonzero = expected > 0
    xi[nonzero] = counts[nonzero] / expected[nonzero] - 1.0

    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return centers, xi


def virial_ratio(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    force_type: str = "newtonian",
    lam: float | None = None,
    yukawa_alpha: float | None = None,
    softening: float = 0.0,
    mond_params: dict | None = None,
    dp_params: dict | None = None,
) -> float:
    """
    Virial ratio 2K / |U|.
    """

    pos = np.asarray(positions, dtype=float)
    vel = np.asarray(velocities, dtype=float)
    m = np.asarray(masses, dtype=float)

    kinetic = 0.5 * np.sum(m[:, None] * vel**2)

    if force_type == "dark_photon":
        return float("nan")

    total = compute_total_energy(
        pos,
        vel,
        m,
        force_type=force_type,
        lam=lam,
        yukawa_alpha=yukawa_alpha,
        softening=softening,
        mond_params=mond_params,
        dp_params=dp_params,
    )

    potential = total - kinetic

    if potential == 0:
        return float("nan")

    return float(2 * kinetic / abs(potential))


def energy_drift(energies: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray(energies, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0

    initial = arr[0]
    drift = arr[-1] - initial
    max_dev = float(np.max(np.abs(arr - initial)))

    return float(drift), max_dev