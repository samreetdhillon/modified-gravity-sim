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
    softening=0.0,
    mond_params=None,
    dp_params=None,
):
    """
    Run an N-body simulation and return trajectories.

    Parameters
    ----------
    positions : ndarray, shape (N, 3)
    velocities : ndarray, shape (N, 3)
    masses : ndarray, shape (N,)
    dt : float
        Timestep.
    steps : int
        Number of simulation steps.
    force_type : str
        "newtonian", "yukawa", "mond", or "dp".
    lam : float or None
        Screening length for Yukawa force (ignored for Newtonian/MOND).
    softening : float
        Softening length.
    mond_params : dict or None
        Required only if force_type="mond".
        Example: {"a0": 1e-2, "mu": "simple"}.
    dp_params : dict or None
        Required only if force_type="dark_photon".
        Example: {"alpha": 1e-2, "lam": 1.0, "charges": np.ones(N)}.

    Returns
    -------
    positions_history : ndarray, shape (steps, N, 3)
    velocities_history : ndarray, shape (steps, N, 3)
    """

    # Default MOND parameters if not supplied
    if force_type == "mond":
        if mond_params is None:
            mond_params = {}
        mond_params.setdefault("a0", 1e-2)
        mond_params.setdefault("mu", "simple")

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
            softening=softening,
            mond_params=mond_params,
            dp_params=dp_params,    
        )
        positions_history[t] = positions.copy()
        velocities_history[t] = velocities.copy()

    return positions_history, velocities_history
