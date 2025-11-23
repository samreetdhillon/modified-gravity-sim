import numpy as np
import matplotlib.pyplot as plt
from src.diagnostics import (
    energy_drift,
    mean_interparticle_separation,
    two_point_correlation,
    virial_ratio,
)
from src.simulate import run_simulation
from src.visualize import (
    animate_trajectory,
    plot_clustering_overlay,
    plot_pair_separation_histogram,
    plot_snapshots,
    plot_trajectories,
)
from src.energy import compute_total_energy

positions = np.array([[ -0.5, 0.0, 0.0 ],
                      [  0.5, 0.0, 0.0 ]], dtype=float)

velocities = np.array([[0.0,  0.5, 0.0],
                       [0.0, -0.5, 0.0]], dtype=float)

masses = np.array([1.0, 1.0])
dt = 0.01
steps = 1000

positions_hist, velocities_hist = run_simulation(positions, velocities, masses, dt, steps)
ani = animate_trajectory(positions_hist)
plt.show()

final_sep = mean_interparticle_separation(positions_hist[-1])
print(f"Two-body Newtonian mean separation: {final_sep:.4f}")

energies = []
for t in range(len(positions_hist)):
    E = compute_total_energy(positions_hist[t], velocities_hist[t], masses)
    energies.append(E)

def log_diagnostics(name, final_positions, final_velocities, energy_trace, force_type, lam=None):
    drift, max_dev = energy_drift(energy_trace)
    ratio = virial_ratio(final_positions, final_velocities, masses, force_type, lam)
    max_dist = np.max(np.linalg.norm(final_positions - final_positions.mean(axis=0), axis=1))
    max_dist = max(max_dist, 1e-3)
    bin_edges = np.linspace(0, max_dist * 2 + 1e-3, 6)
    centers, xi = two_point_correlation(final_positions, bin_edges)
    print(f"{name}: ΔE {drift:.3e}, max dev {max_dev:.3e}, virial ratio {ratio:.3f}")
    print(f"    two-point ξ: {np.round(xi, 3)} @ r={np.round(centers, 2)}")

log_diagnostics("Two-body Newtonian", positions_hist[-1], velocities_hist[-1], energies, "newtonian")

com_positions = np.mean(positions_hist, axis=1)
plt.plot(com_positions)

lam = 100  # Yukawa length scale
positions_yukawa, velocities_yukawa = run_simulation(positions, velocities, masses, dt, steps, force_type="yukawa", lam=lam)
ani_yukawa = animate_trajectory(positions_yukawa)
plt.show()

final_sep_yukawa = mean_interparticle_separation(positions_yukawa[-1])
print(f"Two-body Yukawa mean separation (lam={lam}): {final_sep_yukawa:.4f}")

energies_yukawa = []
for t in range(len(positions_yukawa)):
    E = compute_total_energy(positions_yukawa[t], velocities_yukawa[t], masses, force_type="yukawa", lam=lam)
    energies_yukawa.append(E)

log_diagnostics(
    "Two-body Yukawa",
    positions_yukawa[-1],
    velocities_yukawa[-1],
    energies_yukawa,
    "yukawa",
    lam=lam,
)

time = np.arange(steps) * dt
records = [
    {
        "positions": positions_hist,
        "velocities": velocities_hist,
        "masses": masses,
        "time": time,
        "force_type": "newtonian",
        "lambda": None,
    },
    {
        "positions": positions_yukawa,
        "velocities": velocities_yukawa,
        "masses": masses,
        "time": time,
        "force_type": "yukawa",
        "lambda": lam,
    },
]

plot_trajectories(records)
plot_snapshots(records, [0, steps // 2, -1])
plot_pair_separation_histogram(records)
plot_clustering_overlay(records, np.linspace(0, 4, 16))
plt.show()

plt.figure()
plt.plot(energies, label="Newtonian")
plt.plot(energies_yukawa, label="Yukawa")
plt.title("Total Energy vs Time (Two Body, Newtonian vs Yukawa)")
plt.xlabel("Timestep")
plt.ylabel("Energy")
plt.legend()
plt.show()


lam_values = [0.1, 0.5, 1.0, 2.0, 5.0]  # Different Yukawa length scales
results = {}

for lam in lam_values:
    pos_traj, vel_traj = run_simulation(positions, velocities, masses, dt, steps, force_type="yukawa", lam=lam)
    results[lam] = pos_traj

deviations = {}
for lam, traj in results.items():
    com = np.mean(traj[-1], axis=0)  # center of mass at final timestep
    deviations[lam] = np.mean(np.linalg.norm(traj[-1] - com, axis=1))

plt.figure()
plt.plot(list(deviations.keys()), list(deviations.values()), marker='o')
plt.title("Two-body Orbit deviation vs Yukawa λ")
plt.xlabel("λ (Yukawa length scale)")
plt.ylabel("Average distance from COM")
plt.show()
