# N-Body Dynamics Under Newtonian, MOND, Yukawa & Dark Photon Forces

Small educational N-body simulation exploring Newtonian gravity, Yukawa-modified interactions, MOND (Modified Newtonian Dynamics) via $\mu$-interpolation, and a Dissipative Dark Photon model with charge-based interactions with simple visualization and energy diagnostics.

Full project report is available [here](https://drive.google.com/file/d/1Djdih_a9zyFKvfVbWtiAlEvVw4DohCfi/view?usp=sharing).

---

**Quick summary**

- Simulates N particles under Newtonian gravity, Yukawa-modified interactions, MOND (Modified Newtonian Dynamics) via $\mu$-interpolation, and a Dissipative Dark Photon model with charge-based interactions.
- Employs a Second-Order Leapfrog Integrator for symplectic energy conservation, with a softening parameter to handle high-density close encounters.
- Returns both positions and velocities histories from `run_simulation` so
  per-timestep diagnostics (like kinetic energy, virial ratio, and energy drift
  trace) are computed correctly.
- Provides diagnostics helpers (`mean_interparticle_separation`, `two_point_correlation`, etc.) and visualization utilities to compare behavior under different forces via trajectories, snapshot grids, histograms, and clustering overlays.

### Project layout & helper scripts

- `src/` — core simulation logic (forces, integrators, diagnostics, visualization) that power the CLI and example scripts.
- `examples/` — runnable demos highlighting two-body, three-body, and small N-body chains with diagnostics, plots, and parameter sweeps.
- `results/` — storage for generated animations, diagnostics, stability scans, and statistical snapshots; structure mirrors the example fleets and sweeps so outputs stay organized.

---

### Physics Implementation

- **Yukawa Force:** $F(r) = G \frac{m_1 m_2}{r^2} e^{-r/\lambda} (1 + \frac{r}{\lambda})$
- **MOND:** Implements the "Simple" $\mu(x) = x/(1+x)$ interpolation to simulate galactic-scale acceleration boosts below $a_0$.
- **Dark Photon:** A non-conservative model including velocity-dependent drag and $q_i q_j$ charge interactions, useful for simulating dissipative dark matter.
- **Diagnostics:** Real-time computation of Virial Ratios ($2K/|U|$), Two-Point Correlation Functions $\xi(r)$, and Energy Drift $\Delta E$.

---

### Scientific Results

- **Two-Body:** Observe perihelion precession in Yukawa gravity and orbital decay (inspirals) in the Dark Photon model.
- **Three-Body:** Test the stability of chaotic systems; observe how MONDian "gravity floors" prevent early dispersion compared to Newtonian baselines.
- **N-Body (Structure Formation):** Compare the "Cuspy" halo formation of MOND vs. the smoother "Cored" profiles of Yukawa gravity.

---

**Contents**

- **`src/`**: core library
  - `forces.py` — Newtonian, Yukawa, MOND, and dark-photon helper functions (including μ interpolation curves).
  - `integrator.py` — pairwise acceleration computation and `leapfrog_step`.
  - `simulate.py` — `run_simulation` (returns positions and velocities histories).
  - `energy.py` — compute total energy (kinetic + potential) for diagnostics.
  - `diagnostics.py` — helper metrics such as energy drift, virial ratio, mean separation, and two-point correlation histograms for quantitative checks.
  - `visualize.py` — reusable Matplotlib helpers for trajectories, triggering interpolated snapshots, pair-separation histograms, and simple clustering overlays (trail colors, snapshot grids, etc.).
- **`examples/`**: runnable examples and demos.
  - `run_two_body.py`, `run_three_body.py`, `run_nbody.py`.
- **`main.py`**: interactive CLI that asks for example selection, initial conditions, and force parameters before running the simulations with the same diagnostics/visuals.
- `readme.md` — this document.

---

**Requirements**

- Python 3.8+
- numpy
- matplotlib (for examples / visualization)
- tqdm (for progress bars in `run_simulation`)

You can install the core dependencies with pip, e.g.:

```powershell
python -m pip install numpy matplotlib tqdm
```

---

How to run

Two recommended ways to run examples (from the project root):

- As a module (preferred, keeps package context clean):

```powershell
python -m examples.run_two_body
python -m examples.run_three_body
python -m examples.run_nbody
```

- As a script (this repository supports a fallback so either method works):

```powershell
python .\examples\run_two_body.py
python .\examples\run_three_body.py
python .\examples\run_nbody.py
```

If you see `ModuleNotFoundError: No module named 'src'` when running a
script, use the `-m examples.<name>` invocation from the project root. The
examples include a small fallback to add the project root to `sys.path` so
direct script execution also works.

Or launch the interactive CLI:

```powershell
python main.py
```

### Interactive CLI & diagnostics

- `main.py` drives a conversational prompt flow that asks which demo to run (choose `2` for two-body, `3` for three-body, or any other integer >3 to treat that number as N for the `N`-body case). It accepts custom comma-/semicolon-separated masses, positions, and velocities (or falls back on seeded defaults), then prompts which of the Yukawa, MOND, and Dark Photon variants to include.
- Each non-Newtonian force has configurable parameters (`λ` + `α` for Yukawa, acceleration scale and interpolation for MOND, and per-particle charges plus `α`/`λ` for Dark Photon). The CLI adapts `dt`, step count, softening, and frame-skip heuristics to the selected system size so Newtonian and alternatives share comparable diagnostics.
- Every run stores both position and velocity histories, computes per-step total energy, logs `ΔE` drift/virial ratio/two-point statistics, animates trajectories via `src.visualize.animate_trajectory`, and finally assembles comparison plots (trajectories, snapshots, pair-separation histograms, clustering overlays, and energy traces) to help distinguish the different forces.

---

**API Notes**

- **`src/forces.py`**

  - `newtonian_force(r_vec, m1, m2)` — Standard $1/r^2$ interaction.
  - `yukawa_force(r_vec, m1, m2, lam)` — Screened potential with scale $\lambda$.
  - `mond_acceleration(r_vec, m_source, a0)` — Implements the MOND "simple" $\mu$-interpolation function to boost acceleration in low-$g$ regimes.
  - `dark_photon_force(r_vec, v_vec, m1, m2, q1, q2, alpha, lam)` — Includes both a Yukawa-style $q_i q_j$ interaction and a velocity-dependent dissipative term.

- **`src/integrator.py`**

  - `compute_accelerations(...)` — Vectorized N-body force summation. Supports `force_type` toggles.
  - `leapfrog_step(...)` — Second-order symplectic step ($v_{1/2}, x_1, v_1$). Now includes a **softening parameter** to prevent numerical singularities during close encounters.

- **`src/diagnostics.py`**

  - `compute_energy_drift(history)` — Measures $\Delta E = (E_{final} - E_{initial})/E_{initial}$.
  - `compute_virial_ratio(kinetic, potential)` — Returns $\eta = 2K/|U|$; used to check for gravitational equilibrium.
  - `two_point_correlation(positions, box_size)` — Computes the $\xi(r)$ histogram to quantify matter clumping and spatial structure.

- **`src/visualize.py`**
  - `plot_comparison_grid(...)` — Generates a 4-way comparison (Newton, Yukawa, MOND, Dark Photon) of trajectories and energy traces.
  - `animate_trajectory(...)` — 3D/2D animation with trailing lines to visualize orbital precession or "warm start" cluster evolution.

---

**Examples explained**

- `examples/run_two_body.py`

  - Simple two-body demonstration that now also logs energy drift, virial
    ratio, and two-point statistics while plotting paired energies and
    trajectory snapshots.

- `examples/run_three_body.py`

  - Three-body demo that compares Newtonian and Yukawa interactions, logs the
    diagnostics, and runs a `λ` sweep to show orbit deviations—each run
    visualizes trajectories, snapshots, histograms, and clustering overlays.

- `examples/run_nbody.py`
  - A small random N-body demonstration (default N=20) to show how the
    code behaves for a larger number of bodies.
- Each example also runs the MOND configuration, logging diagnostics and
  plotting the matching trajectory, histogram, clustering, and energy traces.
  Yukawa length-scale space.
