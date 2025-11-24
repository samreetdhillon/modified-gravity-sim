import numpy as np

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
import matplotlib.pyplot as plt
from src.energy import compute_total_energy

N = 20
np.random.seed(42)  # reproducibility

positions = np.random.uniform(-1, 1, size=(N, 3))
velocities = np.random.uniform(-0.1, 0.1, size=(N, 3))
masses = np.ones(N)

dt = 0.01
steps = 1000
mond_params = {"a0": 1e-2, "mu": "simple"}

positions_hist, velocities_hist = run_simulation(positions, velocities, masses, dt, steps)
ani = animate_trajectory(positions_hist)
plt.show()

final_sep = mean_interparticle_separation(positions_hist[-1])
print(f"N-body Newtonian mean separation: {final_sep:.4f}")

energies = []
for t in range(len(positions_hist)):
    E = compute_total_energy(positions_hist[t], velocities_hist[t], masses)
    energies.append(E)

def log_diagnostics(name, final_positions, final_velocities, energy_trace, force_type, lam=None, mond_params=None, dp_params=None, softening=0.0):
    drift, max_dev = energy_drift(energy_trace)
    ratio = virial_ratio(
        final_positions,
        final_velocities,
        masses,
        force_type,
        lam=lam,
        softening=softening,
        mond_params=mond_params,
        dp_params=dp_params,
    )
    max_dist = np.max(np.linalg.norm(final_positions - final_positions.mean(axis=0), axis=1))
    max_dist = max(max_dist, 1e-3)
    bin_edges = np.linspace(0, max_dist * 2 + 1e-3, 8)
    centers, xi = two_point_correlation(final_positions, bin_edges)
    print(f"{name}: ΔE {drift:.3e}, max dev {max_dev:.3e}, virial ratio {ratio:.3f}")
    print(f"    two-point ξ: {np.round(xi, 3)} @ r={np.round(centers, 2)}")

log_diagnostics("N-body Newtonian", positions_hist[-1], velocities_hist[-1], energies, "newtonian")

lam = 1  # Yukawa length scale
positions_yukawa, velocities_yukawa = run_simulation(positions, velocities, masses, dt, steps, force_type="yukawa", lam=lam)
ani_yukawa = animate_trajectory(positions_yukawa)
plt.show()

final_sep_yukawa = mean_interparticle_separation(positions_yukawa[-1])
print(f"N-body Yukawa mean separation (lam={lam}): {final_sep_yukawa:.4f}")

energies_yukawa = []
for t in range(len(positions_yukawa)):
    E = compute_total_energy(positions_yukawa[t], velocities_yukawa[t], masses, force_type="yukawa", lam=lam)
    energies_yukawa.append(E)

log_diagnostics("N-body Yukawa", positions_yukawa[-1], velocities_yukawa[-1], energies_yukawa, "yukawa", lam=lam)

positions_mond, velocities_mond = run_simulation(
    positions,
    velocities,
    masses,
    dt,
    steps,
    force_type="mond",
    mond_params=mond_params,
)
ani_mond = animate_trajectory(positions_mond)
plt.show()

final_sep_mond = mean_interparticle_separation(positions_mond[-1])
print(f"N-body MOND mean separation (a0={mond_params['a0']}): {final_sep_mond:.4f}")

energies_mond = []
for t in range(len(positions_mond)):
    E = compute_total_energy(
        positions_mond[t],
        velocities_mond[t],
        masses,
        force_type="mond",
        mond_params=mond_params,
    )
    energies_mond.append(E)

log_diagnostics(
    "N-body MOND",
    positions_mond[-1],
    velocities_mond[-1],
    energies_mond,
    "mond",
    mond_params=mond_params,
)
dp_params = {
    "alpha": 0.02,
    "lam": 2.0,
    "charges": np.random.uniform(-1.0, 1.0, size=N),
}

positions_dp, velocities_dp = run_simulation(
    positions,
    velocities,
    masses,
    dt,
    steps,
    force_type="dark_photon",
    dp_params=dp_params,
)
ani_dp = animate_trajectory(positions_dp)
plt.show()

final_sep_dp = mean_interparticle_separation(positions_dp[-1])
print(f"N-body Dark Photon mean separation (α={dp_params['alpha']}): {final_sep_dp:.4f}")

energies_dp = []
for t in range(len(positions_dp)):
    E = compute_total_energy(
        positions_dp[t],
        velocities_dp[t],
        masses,
        force_type="dark_photon",
        dp_params=dp_params,
    )
    energies_dp.append(E)

log_diagnostics(
    "N-body Dark Photon",
    positions_dp[-1],
    velocities_dp[-1],
    energies_dp,
    "dark_photon",
    dp_params=dp_params,
)

plt.figure()
plt.plot(energies, label="Newtonian")
plt.plot(energies_yukawa, label="Yukawa")
plt.plot(energies_mond, label="MOND")
plt.plot(energies_dp, label="Dark Photon")
plt.title("Total Energy vs Time (N-body, Newtonian vs Yukawa vs MOND vs Dark Photon)")
plt.xlabel("Timestep")
plt.ylabel("Energy")
plt.legend()
plt.show()

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
    {
        "positions": positions_mond,
        "velocities": velocities_mond,
        "masses": masses,
        "time": time,
        "force_type": "mond",
        "mond_mu": mond_params["mu"],
        "a0": mond_params["a0"],
    },
    {
        "positions": positions_dp,
        "velocities": velocities_dp,
        "masses": masses,
        "time": time,
        "force_type": "dark_photon",
        "dp_alpha": dp_params["alpha"],
        "dp_lambda": dp_params["lam"],
        "dp_charges": dp_params["charges"],
    },
]

plot_trajectories(records)
plot_snapshots(records, [0, steps // 2, -1])
plot_pair_separation_histogram(records)
plot_clustering_overlay(records, np.linspace(0, 4, 16))
plt.show()

lam_values = [0.1, 0.5, 1.0, 2.0, 5.0]  # Yukawa length scales
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
plt.title("N-body Orbit Deviation vs Yukawa λ")
plt.xlabel("λ (Yukawa length scale)")
plt.ylabel("Average distance from COM")
plt.show()

