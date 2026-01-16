'''
Compute total energy (kinetic + potential).
'''

import numpy as np
from .forces import (
    newtonian_force,
    yukawa_force,
    mond_mu_simple,
    mond_mu_standard,
    G
)

def compute_total_energy(
    positions,
    velocities,
    masses,
    force_type="newtonian",
    lam=None,
    yukawa_alpha=None,
    softening=0.0,
    mond_params=None,
    dp_params=None,
):

    if force_type == "dark_photon":
        # Dark Photon is non-conservative (velocity dependent), return Kinetic only
        kinetic = 0.5 * np.sum(masses[:, None] * velocities**2)
        return kinetic

    N = len(positions)

    kinetic = 0.5 * np.sum(masses[:, None] * velocities**2)
    potential = 0.0

    softening = 0.0 if softening is None else softening
    
    if force_type == "mond":
        if mond_params is None:
            raise ValueError("You must supply mond_params when force_type='mond'.")
        a0 = mond_params.get("a0", 1e-2)
        mu_type = mond_params.get("mu", "simple")

    for i in range(N):
        for j in range(i + 1, N):

            # compute softened distance
            r_vec = positions[i] - positions[j]
            r = np.linalg.norm(r_vec)
            softened_r = np.sqrt(r**2 + softening**2)

            if force_type == "newtonian":
                # U = -G * m1 * m2 / r
                potential += -G * masses[i] * masses[j] / softened_r

            elif force_type == "yukawa":
                if lam is None:
                    raise ValueError("You must supply lam when force_type='yukawa'.")
                alpha = yukawa_alpha if yukawa_alpha is not None else 1.0
                pair_mass = masses[i] * masses[j]
                U_newton = -G * pair_mass / softened_r
                yukawa_prefactor = alpha * np.exp(-softened_r / lam)
                U_yukawa = -G * pair_mass / softened_r * yukawa_prefactor
                potential += U_newton + U_yukawa

            elif force_type == "mond":
                # Newtonian potential (pairwise)
                U_N = -G * masses[i] * masses[j] / softened_r
                # Newtonian pairwise acceleration magnitude
                aN_vec = newtonian_force(r_vec, masses[i], masses[j], softening) / masses[i]
                aN = np.linalg.norm(aN_vec) + 1e-12
                x = aN / a0

                if mu_type == "simple":
                    mu = mond_mu_simple(x)
                else:
                    mu = mond_mu_standard(x)
                # MOND pseudo-potential
                potential += U_N

            else:
                raise ValueError(f"Unknown force_type '{force_type}'.")
    return kinetic + potential