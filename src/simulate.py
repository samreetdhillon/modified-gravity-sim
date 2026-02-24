'''
 Run an N-body simulation and return trajectories.
'''

import numpy as np
from tqdm import tqdm
from .integrator import leapfrog_step

def run_simulation(
    positions,
    velocities,
    masses,
    dt,
    steps,
    force_type="newtonian",
    lam=None,
    yukawa_alpha=None,
    softening=0.0,
    mond_params=None,
    dp_params=None,
):

    N = len(positions)
    positions_history = np.zeros((steps, N, 3))
    velocities_history = np.zeros((steps, N, 3))

    positions_history[0] = positions
    velocities_history[0] = velocities

    for t in tqdm(range(1, steps)):
        positions, velocities = leapfrog_step(
            positions=positions,
            velocities=velocities,
            masses=masses,
            dt=dt,
            force_type=force_type,
            lam=lam,
            yukawa_alpha=yukawa_alpha,
            softening=softening,
            mond_params=mond_params,
            dp_params=dp_params,    
        )
        positions_history[t] = positions.copy()
        velocities_history[t] = velocities.copy()

    return positions_history, velocities_history