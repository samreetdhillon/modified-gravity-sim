import numpy as np
from .forces import newtonian_force, yukawa_force

def compute_total_energy(positions, velocities, masses, force_type="newtonian", lam=None):
    N = len(positions)
    kinetic = 0.5 * np.sum(masses[:, None] * velocities**2)
    potential = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            r = np.linalg.norm(positions[i] - positions[j])
            if force_type == "newtonian":
                potential += -1.0 * masses[i] * masses[j] / r
            else:
                potential += -1.0 * masses[i] * masses[j] * np.exp(-r/lam) / r
    return kinetic + potential
