import numpy as np

G = 1.0   # Gravitational constant (set to 1 for simulation units)

def newtonian_force(r_vec, m1, m2, softening=0.0):
    r_sq = np.dot(r_vec, r_vec)
    if r_sq == 0:
        return np.zeros(3)
    softened_r = np.sqrt(r_sq + softening**2)
    return -G * m1 * m2 * r_vec / softened_r**3

def yukawa_force(r_vec, m1, m2, lam, alpha, softening=0.0): # <-- FIXED: Added 'alpha'
    r_sq = np.dot(r_vec, r_vec)
    if r_sq == 0:
        return np.zeros(3)
    softened_r = np.sqrt(r_sq + softening**2)
    # The 'factor' now includes alpha
    factor = alpha * np.exp(-softened_r / lam) * (1 + softened_r / lam) # <-- FIXED: Used 'alpha'
    return -G * m1 * m2 * factor * r_vec / softened_r**3

def mond_mu_simple(x):
    return x / (1.0 + x)

def mond_mu_standard(x):
    return x / np.sqrt(1.0 + x**2)

def dark_photon_force(r_vec, v_rel, q1, q2, alpha=1e-2, lam=1.0, softening=0.0):
    """
    Pairwise velocity-dependent dark-photon force on particle i due to j.
    r_vec = r_i - r_j
    v_rel = v_i - v_j
    Returns force vector acting on i (note F_ji = -F_ij because v_rel_ji = -v_rel).
    """
    r2 = np.dot(r_vec, r_vec) + softening**2
    r = np.sqrt(r2)
    # Yukawa-like spatial falloff times velocity-dependent coupling
    factor = alpha * q1 * q2 * np.exp(-r / lam) / r2
    return factor * v_rel
