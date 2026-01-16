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

# ==========================================
# 1. Setup Initial Conditions
# ==========================================
positions = np.array([[ -0.5, 0.0, 0.0 ],
                      [  0.5, 0.0, 0.0 ]], dtype=float)

velocities = np.array([[0.0,  0.2, 0.0],
                       [0.0, -0.2, 0.0]], dtype=float)

masses = np.array([1.0, 1.0])
dt = 0.01
steps = 5000

# ==========================================
# 2. Helper: Log Diagnostics
# ==========================================
def log_diagnostics(name, final_positions, final_velocities, energy_trace, force_type, 
                    lam=None, yukawa_alpha=None, mond_params=None, dp_params=None, softening=0.0):
    
    drift, max_dev = energy_drift(energy_trace)
    
    # Pass yukawa_alpha to virial_ratio
    ratio = virial_ratio(
        final_positions,
        final_velocities,
        masses,
        force_type,
        lam=lam,
        yukawa_alpha=yukawa_alpha, # <--- UPDATED
        softening=softening,
        mond_params=mond_params,
        dp_params=dp_params,
    )
    
    max_dist = np.max(np.linalg.norm(final_positions - final_positions.mean(axis=0), axis=1))
    max_dist = max(max_dist, 1e-3)
    bin_edges = np.linspace(0, max_dist * 2 + 1e-3, 6)
    centers, xi = two_point_correlation(final_positions, bin_edges)
    
    print(f"--- {name} ---")
    print(f"ΔE drift: {drift:.3e}, Max dev: {max_dev:.3e}")
    print(f"Virial ratio: {ratio:.3f} (Ideal ~ 1.0 for stable orbit)")
    print(f"Two-point ξ: {np.round(xi, 3)} @ r={np.round(centers, 2)}")
    print("")

# ==========================================
# 3. Newtonian Run
# ==========================================
positions_hist, velocities_hist = run_simulation(
    positions, 
    velocities, 
    masses, 
    dt, 
    steps, 
    force_type="newtonian",
)
ani = animate_trajectory(
    positions_hist,
    title="Two-body Newtonian Trajectories",
    frame_skip=4,
)
plt.show()

energies = []
for t in range(len(positions_hist)):
    E = compute_total_energy(positions_hist[t], velocities_hist[t], masses)
    energies.append(E)

log_diagnostics("Newtonian", positions_hist[-1], velocities_hist[-1], energies, "newtonian")

# ==========================================
# 4. Yukawa Run
# ==========================================
lam = 1.0   # Range
alpha_y = 1.0 # Yukawa coupling strength; adds a short-range, screened correction to Newtonian gravity

positions_yukawa, velocities_yukawa = run_simulation(
    positions, 
    velocities, 
    masses, 
    dt, 
    steps, 
    force_type="yukawa", 
    lam=lam,
    yukawa_alpha=alpha_y,
)
ani_yukawa = animate_trajectory(
    positions_yukawa,
    title=f"Two-body Yukawa Trajectories (λ={lam}, α={alpha_y})",
    frame_skip=4,
)
plt.show()

energies_yukawa = []
for t in range(len(positions_yukawa)):
    E = compute_total_energy(
        positions_yukawa[t], 
        velocities_yukawa[t], 
        masses, 
        force_type="yukawa", 
        lam=lam,
        yukawa_alpha=alpha_y
    )
    energies_yukawa.append(E)

log_diagnostics(
    "Yukawa",
    positions_yukawa[-1],
    velocities_yukawa[-1],
    energies_yukawa,
    "yukawa",
    lam=lam,
    yukawa_alpha=alpha_y
)

# ==========================================
# 5. MOND Run
# ==========================================
mond_params = {"a0": 1e-2, "mu": "simple"}

positions_mond, velocities_mond = run_simulation(
    positions,
    velocities,
    masses,
    dt,
    steps,
    force_type="mond",
    mond_params=mond_params,
)
ani_mond = animate_trajectory(
    positions_mond,
    title=f"Two-body MOND Trajectories (a0={mond_params['a0']})",
    frame_skip=4,
)
plt.show()

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
    "MOND",
    positions_mond[-1],
    velocities_mond[-1],
    energies_mond,
    "mond",
    mond_params=mond_params,
)

# ==========================================
# 6. Dark Photon Run
# ==========================================
dp_params = {
    "alpha": 0.03,
    "lam": 1.5,
    "charges": np.array([1.0, -1.0]),
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
ani_dp = animate_trajectory(
    positions_dp,
    title=f"Two-body Dark Photon Trajectories (α={dp_params['alpha']}, λ={dp_params['lam']})",
    frame_skip=4,
)
plt.show()

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
    "Dark Photon",
    positions_dp[-1],
    velocities_dp[-1],
    energies_dp,
    "dark_photon",
    dp_params=dp_params,
)

# ==========================================
# 7. Visualization & Comparison
# ==========================================
time = np.arange(steps) * dt
records = [
    {
        "positions": positions_hist,
        "velocities": velocities_hist,
        "masses": masses,
        "time": time,
        "force_type": "newtonian",
    },
    {
        "positions": positions_yukawa,
        "velocities": velocities_yukawa,
        "masses": masses,
        "time": time,
        "force_type": "yukawa",
        "lambda": lam,
        "yukawa_alpha": alpha_y,
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

print("Generating Comparison Plots...")
plot_trajectories(records)
plot_snapshots(records, [0, steps // 2, -1])
plot_pair_separation_histogram(records)
plot_clustering_overlay(records, np.linspace(0, 4, 16))
plt.show()

# Energy Plot
plt.figure(figsize=(10, 6))
plt.plot(energies, label="Newtonian")
plt.plot(energies_yukawa, label=f"Yukawa (λ={lam}, α={alpha_y})")
plt.plot(energies_mond, label="MOND")
plt.plot(energies_dp, label="Dark Photon (Kinetic Only)")
plt.title("Total Energy vs Time")
plt.xlabel("Timestep")
plt.ylabel("Energy")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ==========================================
# 8. Yukawa Parameter Sweep
# ==========================================
lam_values = [0.1, 0.5, 1.0, 2.0, 5.0]  
results = {}

print("Running Yukawa Parameter Sweep...")
for l_val in lam_values:
    pos_traj, _ = run_simulation(
        positions, 
        velocities, 
        masses, 
        dt, 
        steps, 
        force_type="yukawa", 
        lam=l_val, 
        yukawa_alpha=alpha_y
    )
    results[l_val] = pos_traj

deviations = {}
for l_val, traj in results.items():
    com = np.mean(traj[-1], axis=0)
    deviations[l_val] = np.mean(np.linalg.norm(traj[-1] - com, axis=1))

plt.figure()
plt.plot(list(deviations.keys()), list(deviations.values()), marker='o')
plt.title(f"Two-body Orbit deviation vs Yukawa λ (fixed α={alpha_y})")
plt.xlabel("λ (Yukawa length scale)")
plt.ylabel("Average distance from COM")
plt.grid(True)
plt.show()