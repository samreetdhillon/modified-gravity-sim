import numpy as np
from .integrator import leapfrog_step
from tqdm import tqdm

def run_simulation(positions, velocities, masses, dt, steps, force_type="newtonian", lam=None):
    """
    Runs the N-body simulation and returns arrays of positions over time.
    """
    N = len(positions)
    trajectory = np.zeros((steps, N, 3))
    trajectory[0] = positions

    for t in tqdm(range(1, steps)):
        positions, velocities = leapfrog_step(
            positions, velocities, masses, dt, force_type, lam
        )
        trajectory[t] = positions.copy()
    return trajectory