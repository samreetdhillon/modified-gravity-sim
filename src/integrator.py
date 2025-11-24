import numpy as np
from .forces import (
    newtonian_force,
    yukawa_force,
    mond_mu_simple,
    mond_mu_standard,
    dark_photon_force,
)

def compute_accelerations(
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
    Compute accelerations for all particles under the selected force model.
    """
    N = len(positions)
    acc = np.zeros_like(positions)

    # Default MOND parameters
    if mond_params is None:
        mond_params = {"a0": 1e-2, "mu": "simple"}

    a0 = mond_params.get("a0", 1e-2)
    mu_type = mond_params.get("mu", "simple")

    for i in range(N):
        for j in range(N):
            if i == j:
                continue

            r_vec = positions[i] - positions[j]

            # -------------------------------------
            # Newtonian Gravity
            # -------------------------------------
            if force_type == "newtonian":
                F = newtonian_force(r_vec, masses[i], masses[j], softening)

            # -------------------------------------
            # Yukawa Gravity
            # -------------------------------------
            elif force_type == "yukawa":
                F = yukawa_force(r_vec, masses[i], masses[j], lam, softening)

            # -------------------------------------
            # MOND Gravity
            # -------------------------------------
            elif force_type == "mond":
                # Newtonian force first
                F_N = newtonian_force(r_vec, masses[i], masses[j], softening)

                # Newtonian acceleration on particle i
                aN = F_N / masses[i]
                aN_mag = np.linalg.norm(aN) + 1e-12

                # Pick μ function
                x = aN_mag / a0
                if mu_type == "simple":
                    mu = mond_mu_simple(x)
                else:
                    mu = mond_mu_standard(x)

                # Apply MOND correction
                acc[i] += mu * aN
                continue
            
            # -------------------------------------
            # Dark Photon Gravity
            # -------------------------------------            
            elif force_type == "dark_photon":
                if dp_params is None:
                    raise ValueError("dp_params required for dark_photon force")

                alpha = dp_params.get("alpha", 1e-2)
                lam_dp = dp_params.get("lam", 1.0)
                q = dp_params.get("charges")

                v_rel = velocities[i] - velocities[j]

                F = dark_photon_force(
                    r_vec,
                    v_rel,
                    q[i],
                    q[j],
                    alpha=alpha,
                    lam=lam_dp,
                    softening=softening,
                )
                acc[i] += F / masses[i]
                continue

            else:
                raise ValueError(f"Invalid force_type: {force_type}")

            # Add the Newtonian/Yukawa contribution
            acc[i] += F / masses[i]

    return acc

def leapfrog_step(
    positions,
    velocities,
    masses,
    dt,
    force_type="newtonian",
    lam=None,
    softening=0.0,
    mond_params=None,
    dp_params=None,
):
    """
    One full leapfrog step: v_{n+1/2}, x_{n+1}, v_{n+1}.
    """

    # --- 1: compute acceleration at time n ---
    acc = compute_accelerations(
        positions=positions,
        velocities=velocities,
        masses=masses,
        force_type=force_type,
        lam=lam,
        softening=softening,
        mond_params=mond_params,
        dp_params=dp_params,
    )

    # --- 2: half-step velocity ---
    v_half = velocities + 0.5 * dt * acc

    # --- 3: update positions ---
    x_new = positions + dt * v_half

    # --- 4: compute acceleration at x_{n+1} ---
    acc_new = compute_accelerations(
        positions=x_new,
        velocities=v_half,     # <-- IMPORTANT: leapfrog uses v_half here
        masses=masses,
        force_type=force_type,
        lam=lam,
        softening=softening,
        mond_params=mond_params,
        dp_params=dp_params,
    )

    # --- 5: complete velocity step ---
    v_new = v_half + 0.5 * dt * acc_new

    return x_new, v_new
