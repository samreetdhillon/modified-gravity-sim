import numpy as np

G = 1.0   # Gravitational constant (set to 1 for simulation units)
def newtonian_force(r_vec, m1, m2):
    r = np.linalg.norm(r_vec)
    if r == 0: 
        return np.zeros(3)
    return -G * m1 * m2 * r_vec / r**3
def yukawa_force(r_vec, m1, m2, lam):
    r = np.linalg.norm(r_vec)
    if r == 0:
        return np.zeros(3)
    factor = np.exp(-r/lam) * (1 + r/lam)
    return -G * m1 * m2 * factor * r_vec / r**3
