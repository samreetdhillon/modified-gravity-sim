import sys
import pathlib
import numpy as np

# Ensure project root is on sys.path so `src` can be imported as a package.
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from src.simulate import run_simulation
from src.visualize import animate_trajectory
import matplotlib.pyplot as plt
from src.energy import compute_total_energy

positions = np.array([[ -1.0,  0.0, 0.0 ],
                      [  0.5,  0.866, 0.0 ],
                      [  0.5, -0.866, 0.0 ]], dtype=float)

velocities = np.array([[ 0.0,  0.3, 0.0 ],
                       [-0.3, -0.15, 0.0],
                       [ 0.3, -0.15, 0.0]], dtype=float)

masses = np.array([1.0, 1.0, 1.0])

dt = 0.01
steps = 1500

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

lam = 1.0  # Yukawa length scale
trajectory_yukawa = run_simulation(positions, velocities, masses, dt, steps, force_type="yukawa", lam=lam)
ani_yukawa = animate_trajectory(trajectory_yukawa)
plt.show()

energies_yukawa = []
for t in range(len(trajectory_yukawa)):
    E = compute_total_energy(trajectory_yukawa[t], velocities, masses, force_type="yukawa", lam=lam)
    energies_yukawa.append(E)

plt.figure()
plt.plot(energies, label="Newtonian")
plt.plot(energies_yukawa, label="Yukawa")
plt.title("Total Energy vs Time (Newtonian vs Yukawa)")
plt.xlabel("Timestep")
plt.ylabel("Energy")
plt.legend()
plt.show()

lam_values = [0.1, 0.5, 1.0, 2.0, 5.0]  # Different Yukawa length scales
results = {}

for lam in lam_values:
    traj = run_simulation(positions, velocities, masses, dt, steps, force_type="yukawa", lam=lam)
    results[lam] = traj

deviations = {}
for lam, traj in results.items():
    com = np.mean(traj[-1], axis=0)  # center of mass at final timestep
    deviations[lam] = np.mean(np.linalg.norm(traj[-1] - com, axis=1))

plt.figure()
plt.plot(list(deviations.keys()), list(deviations.values()), marker='o')
plt.title("Orbit deviation vs Yukawa λ")
plt.xlabel("λ (Yukawa length scale)")
plt.ylabel("Average distance from COM")
plt.show()
