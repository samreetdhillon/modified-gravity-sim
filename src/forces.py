import numpy as np

G = 1.0   # Gravitational constant (set to 1 for simulation units)

def newtonian_force(r_vec, m1, m2, softening=0.0):
    r_sq = np.dot(r_vec, r_vec)
    if r_sq == 0:
        return np.zeros(3)
    softened_r = np.sqrt(r_sq + softening**2)
    return -G * m1 * m2 * r_vec / softened_r**3


def yukawa_force(r_vec, m1, m2, lam, softening=0.0):
    r_sq = np.dot(r_vec, r_vec)
    if r_sq == 0:
        return np.zeros(3)
    softened_r = np.sqrt(r_sq + softening**2)
    factor = np.exp(-softened_r / lam) * (1 + softened_r / lam)
    return -G * m1 * m2 * factor * r_vec / softened_r**3
