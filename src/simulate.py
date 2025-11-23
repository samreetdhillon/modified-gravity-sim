import numpy as np
from .integrator import leapfrog_step
from tqdm import tqdm

def run_simulation(positions, velocities, masses, dt, steps, force_type="newtonian", lam=None, softening=0.0):
    """
    Runs the N-body simulation and returns arrays of positions and velocities over time.

    Returns
    -------
    positions_history : ndarray
        Array of shape (steps, N, 3) containing positions at each timestep.
    velocities_history : ndarray
        Array of shape (steps, N, 3) containing velocities at each timestep.
    """
    N = len(positions)
    positions_history = np.zeros((steps, N, 3))
    velocities_history = np.zeros((steps, N, 3))
    positions_history[0] = positions
    velocities_history[0] = velocities

    for t in tqdm(range(1, steps)):
        positions, velocities = leapfrog_step(
            positions, velocities, masses, dt, force_type, lam, softening
        )
        positions_history[t] = positions.copy()
        velocities_history[t] = velocities.copy()

    return positions_history, velocities_history