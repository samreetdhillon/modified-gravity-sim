import numpy as np
from .forces import newtonian_force, yukawa_force

def compute_accelerations(positions, masses, force_type="newtonian", lam=None, softening=0.0):
    N = len(positions)
    acc = np.zeros_like(positions)
    for i in range(N):
        for j in range(N):
            if i == j: 
                continue
            r_vec = positions[i] - positions[j]
            if force_type == "newtonian":
                F = newtonian_force(r_vec, masses[i], masses[j], softening)
            else:
                F = yukawa_force(r_vec, masses[i], masses[j], lam, softening)
            acc[i] += F / masses[i]
    return acc

def leapfrog_step(positions, velocities, masses, dt, force_type="newtonian", lam=None, softening=0.0):
    acc = compute_accelerations(positions, masses, force_type, lam, softening)
    velocities_half = velocities + 0.5 * dt * acc
    positions_new = positions + dt * velocities_half
    acc_new = compute_accelerations(positions_new, masses, force_type, lam, softening)
    velocities_new = velocities_half + 0.5 * dt * acc_new
    return positions_new, velocities_new

