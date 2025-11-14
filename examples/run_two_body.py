import sys
import pathlib
import numpy as np

# Ensure project root is on sys.path so `src` can be imported as a package.
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.simulate import run_simulation
from src.visualize import animate_trajectory
from src.energy import compute_total_energy

import matplotlib.pyplot as plt

positions = np.array([[ -0.5, 0.0, 0.0 ],
                      [  0.5, 0.0, 0.0 ]], dtype=float)

velocities = np.array([[0.0,  0.5, 0.0],
                       [0.0, -0.5, 0.0]], dtype=float)

masses = np.array([1.0, 1.0])
dt = 0.01
steps = 1000

trajectory = run_simulation(positions, velocities, masses, dt, steps)

ani = animate_trajectory(trajectory)
plt.show()

energies = []
for t in range(len(trajectory)):
    E = compute_total_energy(trajectory[t], velocities, masses)
    energies.append(E)

plt.figure()
plt.plot(energies)
plt.title("Total Energy vs Time")
plt.xlabel("Timestep")
plt.ylabel("Energy")
plt.show()
