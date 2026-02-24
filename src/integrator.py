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
    yukawa_alpha=None,
    softening=0.0,
    mond_params=None,
    dp_params=None,
):

    N = len(positions)
    acc = np.zeros_like(positions)

    if force_type == "mond":
        
        if mond_params is None:
             mond_params = {"a0": 1e-2, "mu": "simple"}
        a0 = mond_params.get("a0", 1e-2)
        mu_type = mond_params.get("mu", "simple")
        mu_func = mond_mu_simple if mu_type == "simple" else mond_mu_standard

        aN_total = np.zeros_like(positions)
        
        # Pass 1: Compute total Newtonian acceleration (a_N) for all particles
        for i in range(N):
            for j in range(N):
                if i == j: continue
                r_vec = positions[i] - positions[j]
                F_N = newtonian_force(r_vec, masses[i], masses[j], softening)
                aN_total[i] += F_N / masses[i]
        
        # Pass 2: Apply MOND correction
        for i in range(N):
            aN_mag = np.linalg.norm(aN_total[i])
            aN_mag_safe = aN_mag + 1e-12 
            x = aN_mag_safe / a0
            mu = mu_func(x)
            acc[i] = mu * aN_total[i]            
        return acc

    # -------------------------------------
    # Standard Forces (Newtonian, Yukawa, Dark Photon)
    # -------------------------------------
    
    # Handle Dark Photon parameters outside the loop if needed later
    dp_alpha, lam_dp, q = None, None, None
    if force_type == "dark_photon":
        if dp_params is None:
            raise ValueError("dp_params required for dark_photon force")
        dp_alpha = dp_params.get("alpha", 1e-2)
        lam_dp = dp_params.get("lam", 1.0)
        q = dp_params.get("charges")

    # Single pass loop for remaining forces
    for i in range(N):
        for j in range(N):
            if i == j: continue

            r_vec = positions[i] - positions[j]
            F = np.zeros(3)

            if force_type == "newtonian":
                F = newtonian_force(r_vec, masses[i], masses[j], softening)

            elif force_type == "yukawa":
                # Check for alpha and use 1.0 if not provided
                alpha = yukawa_alpha if yukawa_alpha is not None else 1.0 
                F = yukawa_force(r_vec, masses[i], masses[j], lam, alpha, softening)

            elif force_type == "dark_photon":
                v_rel = velocities[i] - velocities[j]
                F = dark_photon_force(
                    r_vec, v_rel, masses[i], masses[j], q[i], q[j], alpha=dp_alpha, lam=lam_dp, softening=softening
                )
         
            else:
                raise ValueError(f"Invalid force_type: {force_type}")

            # Add the force/acceleration contribution
            acc[i] += F / masses[i]

    return acc

def leapfrog_step(
    positions,
    velocities,
    masses,
    dt,
    force_type="newtonian",
    lam=None,
    yukawa_alpha=None,
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
        yukawa_alpha=yukawa_alpha,
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
        velocities=v_half,
        masses=masses,
        force_type=force_type,
        lam=lam,
        yukawa_alpha=yukawa_alpha,
        softening=softening,
        mond_params=mond_params,
        dp_params=dp_params,
    )

    # --- 5: complete velocity step ---
    v_new = v_half + 0.5 * dt * acc_new

    return x_new, v_new