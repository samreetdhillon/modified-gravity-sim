import numpy as np

from src.simulate import run_simulation
from src.visualize import animate_trajectory
import matplotlib.pyplot as plt
from src.energy import compute_total_energy

N = 20
np.random.seed(42)  # reproducibility

positions = np.random.uniform(-1, 1, size=(N, 3))
velocities = np.random.uniform(-0.1, 0.1, size=(N, 3))
masses = np.ones(N)

dt = 0.01
steps = 1000

positions_hist, velocities_hist = run_simulation(positions, velocities, masses, dt, steps)
ani = animate_trajectory(positions_hist)
plt.show()

energies = []
for t in range(len(positions_hist)):
    E = compute_total_energy(positions_hist[t], velocities_hist[t], masses)
    energies.append(E)

plt.figure()
plt.plot(energies)
plt.title("Total Energy vs Time (N-body System)")
plt.xlabel("Timestep")
plt.ylabel("Energy")
plt.show()

