import numpy as np
import matplotlib.pyplot as plt

from src.diagnostics import (
    energy_drift,
    mean_interparticle_separation,
    two_point_correlation,
    virial_ratio,
)
from src.energy import compute_total_energy
from src.simulate import run_simulation
from src.visualize import (
    animate_trajectory,
    plot_clustering_overlay,
    plot_pair_separation_histogram,
    plot_snapshots,
    plot_trajectories,
)


def prompt_with_default(message: str, default: str, display: str | None = None) -> str:
    prompt_value = display if display is not None else default
    response = input(f"{message} [{prompt_value}]: ").strip()
    return response or default


def parse_vector_list(value: str, expected: int | None = None) -> np.ndarray | None:
    if not value or value.strip().lower() == "random":
        return None
    segments = [seg.strip() for seg in value.split(";") if seg.strip()]
    array = np.array([[float(coord.strip()) for coord in seg.split(",")] for seg in segments])
    if expected is not None and array.shape[0] != expected:
        raise ValueError(f"Expected {expected} vectors but got {array.shape[0]}")
    if array.shape[1] != 3:
        raise ValueError("Each vector must have three components")
    return array


def parse_float_list(value: str, expected: int | None = None) -> list[float] | None:
    if not value or value.strip().lower() == "random":
        return None
    items = [float(x.strip()) for x in value.split(",") if x.strip()]
    if expected is not None and len(items) != expected:
        raise ValueError(f"Expected {expected} values but got {len(items)}")
    return items


def log_diagnostics(name, final_positions, final_velocities, energy_trace, force_type, masses, **kwargs):
    drift, max_dev = energy_drift(energy_trace)
    ratio = virial_ratio(
        final_positions,
        final_velocities,
        masses,
        force_type,
        **kwargs,
    )
    max_dist = np.max(np.linalg.norm(final_positions - final_positions.mean(axis=0), axis=1))
    max_dist = max(max_dist, 1e-3)
    bin_edges = np.linspace(0, max_dist * 2 + 1e-3, 6)
    centers, xi = two_point_correlation(final_positions, bin_edges)

    print(f"--- {name} ---")
    print(f"ΔE drift: {drift:.3e}, Max dev: {max_dev:.3e}")
    print(f"Virial ratio: {ratio:.3f}")
    print(f"Two-point ξ: {np.round(xi, 3)} @ r={np.round(centers, 2)}")
    print("")


def format_vectors_for_prompt(array: np.ndarray) -> str:
    rows = []
    for row in array:
        rows.append(",".join(f"{val:.3f}" for val in row))
    return ";".join(rows)


def default_nbody_vectors(N: int) -> tuple[np.ndarray, np.ndarray]:
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    radius = 0.8
    speed = 0.2
    positions = np.vstack([radius * np.cos(angles), radius * np.sin(angles), np.zeros(N)]).T
    velocities = np.vstack([-speed * np.sin(angles), speed * np.cos(angles), np.zeros(N)]).T
    return positions, velocities


def default_vectors(example: int, N: int) -> tuple[np.ndarray, np.ndarray]:
    if example == 2:
        positions = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=float)
        velocities = np.array([[0.0, 0.5, 0.0], [0.0, -0.5, 0.0]], dtype=float)
    elif example == 3:
        positions = np.array([
            [-1.0, 0.0, 0.0],
            [0.5, 0.866, 0.0],
            [0.5, -0.866, 0.0],
        ], dtype=float)
        velocities = np.array([
            [0.0, 0.3, 0.0],
            [-0.3, -0.15, 0.0],
            [0.3, -0.15, 0.0],
        ], dtype=float)
    else:
        positions, velocities = default_nbody_vectors(N)
    return positions, velocities


def random_initial_state(N: int) -> tuple[np.ndarray, np.ndarray]:
    positions = np.random.uniform(-1, 1, size=(N, 3))
    velocities = np.random.uniform(-0.1, 0.1, size=(N, 3))
    return positions, velocities


def build_states(
    example: int,
    masses: np.ndarray,
    positions_text: str,
    velocities_text: str,
) -> tuple[np.ndarray, np.ndarray]:
    N = len(masses)
    custom_positions = parse_vector_list(positions_text, expected=N)
    custom_velocities = parse_vector_list(velocities_text, expected=N)
    if custom_positions is not None and custom_velocities is None:
        raise ValueError("Provide velocities whenever you supply positions")
    if custom_positions is None or custom_velocities is None:
        if custom_positions is None and custom_velocities is None:
            return default_vectors(example, N)
        if custom_positions is not None and custom_velocities is None:
            raise ValueError("Velocities missing for provided positions")
        if custom_velocities is not None and custom_positions is None:
            raise ValueError("Positions missing for provided velocities")
        return random_initial_state(N)
    return custom_positions, custom_velocities


def build_masses(N: int, masses_text: str) -> np.ndarray:
    items = parse_float_list(masses_text, expected=N)
    return np.array(items, dtype=float)


def build_dp_charges(charges_text: str, N: int) -> np.ndarray:
    parsed = parse_float_list(charges_text, expected=N)
    if parsed is None:
        return np.random.uniform(-1.0, 1.0, size=N)
    return np.array(parsed, dtype=float)


def run_all_examples():
    print("Welcome to yukawa-nbody-sim! This tool lets you explore Newtonian, Yukawa, MOND, and Dark Photon forces interactively.")
    while True:
        example_choice = int(prompt_with_default(
            "Choose an example (2=two-body, 3=three-body, any other integer >3 for N-body where that number becomes N)",
            "2",
        ))
        if example_choice >= 2:
            break
        print("Please enter 2 or higher.")
    is_nbody = example_choice > 2

    if is_nbody:
        N = example_choice
        example_tag = f"N-body (N={N})"
        default_pos_array, default_vel_array = default_vectors(example_choice, N)
        pos_example = format_vectors_for_prompt(default_pos_array)
        vel_example = format_vectors_for_prompt(default_vel_array)
    else:
        N = 2 if example_choice == 2 else 3
        example_tag = "Two-body" if N == 2 else "Three-body"
        pos_example = "-0.5,0,0;0.5,0,0" if N == 2 else "-1,0,0;0.5,0.866,0;0.5,-0.866,0"
        vel_example = "0,0.5,0;0,-0.5,0" if N == 2 else "0,0.3,0;-0.3,-0.15,0;0.3,-0.15,0"

    mass_default = ",".join(["1"] * N)
    masses = build_masses(N, prompt_with_default(f"Enter {N} particle masses (mass units, comma separated)", mass_default))

    pos_display = (
        pos_example if N <= 6 else f"{N}-particle circular default; leave blank for ring"
    )
    vel_display = (
        vel_example if N <= 6 else f"tangential velocities for {N}-particle ring; leave blank for default"
    )
    positions_text = prompt_with_default(
        f"Enter {N} initial positions as 'x,y,z' entries separated by ';' (length units)",
        pos_example,
        display=pos_display,
    )
    velocities_text = prompt_with_default(
        f"Enter {N} initial velocities as 'vx,vy,vz' entries separated by ';' (velocity units)",
        vel_example,
        display=vel_display,
    )

    lambda_yukawa = float(prompt_with_default("Yukawa length scale λ (length units)", "1.0"))
    alpha_yukawa = float(prompt_with_default("Yukawa strength α (dimensionless)", "1.0"))
    mond_a0 = float(prompt_with_default("MOND acceleration scale a0 (length/time^2 units)", "1e-2"))
    mond_mu = prompt_with_default("MOND μ interpolation (simple or other)", "simple")
    dp_alpha = float(prompt_with_default("Dark Photon coupling α (dimensionless)", "0.03"))
    dp_lambda = float(prompt_with_default("Dark Photon length scale λ (length units)", "1.5"))
    dp_charges_text = prompt_with_default(
        "Dark Photon charges for each particle (comma separated, dimensionless)",
        "random",
    )

    positions, velocities = build_states(example_choice, masses, positions_text, velocities_text)
    dp_charges = build_dp_charges(dp_charges_text, len(masses))

    dt = 0.02 if N == 2 else 0.001 if N == 3 else 0.01
    steps = 3000 if N == 2 else 4500 if N == 3 else 2500
    frame_skip = 4 if N == 2 else 6 if N == 3 else 5

    records = []
    animations: list = []

    force_runs = [
        ("Newtonian", "newtonian", {}, f"{example_tag} Newtonian Trajectories"),
        ("Yukawa", "yukawa", {"lam": lambda_yukawa, "yukawa_alpha": alpha_yukawa}, f"{example_tag} Yukawa Trajectories (λ={lambda_yukawa}, α={alpha_yukawa})"),
        ("MOND", "mond", {"mond_params": {"a0": mond_a0, "mu": mond_mu}}, f"{example_tag} MOND Trajectories (a0={mond_a0})"),
        ("Dark Photon", "dark_photon", {"dp_params": {"alpha": dp_alpha, "lam": dp_lambda, "charges": dp_charges}}, f"{example_tag} Dark Photon Trajectories (α={dp_alpha}, λ={dp_lambda})"),
    ]

    for label, force_type, extra_args, title in force_runs:
        print(f"Running {label} {example_tag}...")
        positions_hist, velocities_hist = run_simulation(
            positions,
            velocities,
            masses,
            dt,
            steps,
            force_type=force_type,
            **(extra_args or {}),
        )
        energies = [
            compute_total_energy(
                positions_hist[t], velocities_hist[t], masses, force_type=force_type, **(extra_args or {}),
            )
            for t in range(len(positions_hist))
        ]
        log_diagnostics(label, positions_hist[-1], velocities_hist[-1], energies, force_type, masses, **(extra_args or {}))
        ani = animate_trajectory(positions_hist, title=title, frame_skip=frame_skip)
        animations.append(ani)
        plt.show()
        records.append(
            {
                "positions": positions_hist,
                "velocities": velocities_hist,
                "masses": masses,
                "time": np.arange(steps) * dt,
                "force_type": force_type,
                "lambda": lambda_yukawa,
                "yukawa_alpha": alpha_yukawa,
                "mond_mu": mond_mu,
                "a0": mond_a0,
                "dp_alpha": dp_alpha,
                "dp_lambda": dp_lambda,
                "dp_charges": dp_charges,
                "energies": energies,
            }
        )

    print("Generating comparison plots...")
    plot_trajectories(records)
    plot_snapshots(records, [0, steps // 2, -1])
    plot_pair_separation_histogram(records)
    plot_clustering_overlay(records, np.linspace(0, 4, 16))
    plt.show()

    plt.figure(figsize=(10, 6))
    for rec in records:
        label = rec["force_type"].replace("_", " ").title()
        plt.plot(rec["energies"], label=label)
    plt.title(f"Total Energy vs Time ({example_tag})")
    plt.xlabel("Timestep")
    plt.ylabel("Energy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    try:
        run_all_examples()
    except Exception as exc:
        print(f"Error: {exc}", flush=True)
        raise
