import numpy as np
from .forces import newtonian_force, yukawa_force

def compute_total_energy(positions, velocities, masses, force_type="newtonian", lam=None, softening=0.0):
    N = len(positions)
    kinetic = 0.5 * np.sum(masses[:, None] * velocities**2)
    potential = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            r = np.linalg.norm(positions[i] - positions[j])
            softened_r = np.sqrt(r**2 + softening**2)
            if force_type == "newtonian":
                potential += -1.0 * masses[i] * masses[j] / softened_r
            else:
                potential += -1.0 * masses[i] * masses[j] * np.exp(-softened_r/lam) / softened_r
    return kinetic + potential
