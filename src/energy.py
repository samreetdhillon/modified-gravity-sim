import numpy as np
from .forces import (
    newtonian_force,
    yukawa_force,
    mond_mu_simple,
    mond_mu_standard,
)


def compute_total_energy(
    positions,
    velocities,
    masses,
    force_type="newtonian",
    lam=None,
    softening=0.0,
    mond_params=None,
    dp_params=None,
):
    """
    Compute total energy (kinetic + potential).

    For MOND we use a pseudo-potential:
        U_mond = U_newtonian / μ(|a_N|/a0)

    This is not exact MOND physics, but is the standard
    approximation used in simple MOND N-body toy models.
    """

    if force_type == "dark_photon":
        kinetic = 0.5 * np.sum(masses[:, None] * velocities**2)
        return kinetic

    N = len(positions)

    # --- kinetic energy ---
    kinetic = 0.5 * np.sum(masses[:, None] * velocities**2)

    # --- potential energy ---
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
                potential += -masses[i] * masses[j] / softened_r

            elif force_type == "yukawa":
                potential += -masses[i] * masses[j] * np.exp(-softened_r / lam) / softened_r

            elif force_type == "mond":
                # Newtonian potential (pairwise)
                U_N = -masses[i] * masses[j] / softened_r

                # Newtonian pairwise acceleration magnitude
                # (force on i due to j divided by m_i)
                aN_vec = newtonian_force(r_vec, masses[i], masses[j], softening) / masses[i]
                aN = np.linalg.norm(aN_vec) + 1e-12

                x = aN / a0

                if mu_type == "simple":
                    mu = mond_mu_simple(x)
                else:
                    mu = mond_mu_standard(x)

                # MOND pseudo-potential
                potential += U_N / mu

            else:
                raise ValueError(f"Unknown force_type '{force_type}'.")

    return kinetic + potential
