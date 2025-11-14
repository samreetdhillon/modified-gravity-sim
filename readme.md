# Yukawa N-body Simulation

Small educational N-body simulation exploring Newtonian and Yukawa-like
pairwise forces with simple visualization and energy diagnostics.

This repository is intended as a compact, readable codebase for experimenting
with pairwise force laws, symplectic integration (leapfrog), and visualizing
trajectories for small systems (N &lt;~ 100).

---

**Contents**

- **`src/`**: core library
  - `forces.py` — Newtonian and Yukawa force implementations.
  - `integrator.py` — pairwise acceleration computation and `leapfrog_step`.
  - `simulate.py` — `run_simulation` (returns positions and velocities histories).
  - `energy.py` — compute total energy (kinetic + potential) for diagnostics.
  - `visualize.py` — simple Matplotlib animation helper.
- **`examples/`**: runnable examples and demos
  - `run_two_body.py`, `run_three_body.py`, `run_nbody.py`
- `readme.md` — this document.

---

**Quick summary**

- Simulates N particles under either Newtonian gravity or a Yukawa-modified
  interaction: potential ~ exp(-r/λ)/r (and force implemented accordingly).
- Uses a leapfrog (velocity Verlet-like) integrator for time-stepping.
- Returns both positions and velocities histories from `run_simulation` so
  per-timestep diagnostics (e.g., kinetic energy) are computed correctly.

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

---

API Notes (quick)

- `src/forces.py`

  - `newtonian_force(r_vec, m1, m2)`
  - `yukawa_force(r_vec, m1, m2, lam)`
  - Both return a 3-vector force acting on the first body from the second.

- `src/integrator.py`

  - `compute_accelerations(positions, masses, force_type='newtonian', lam=None)`
  - `leapfrog_step(positions, velocities, masses, dt, force_type='newtonian', lam=None)`
  - `leapfrog_step` advances the state by one timestep and returns
    `(positions_new, velocities_new)`.

- `src/simulate.py`

  - `run_simulation(positions, velocities, masses, dt, steps, force_type='newtonian', lam=None)`
  - Returns `(positions_history, velocities_history)` where each array has
    shape `(steps, N, 3)`.
  - Note: velocities history was added so energy/kinetic diagnostics are
    computed from the correct per-timestep velocities.

- `src/energy.py`

  - `compute_total_energy(positions, velocities, masses, force_type='newtonian', lam=None)`
  - Compute kinetic + potential energy (pairwise potential)

- `src/visualize.py`
  - `animate_trajectory(trajectory)` — expects `trajectory` shape `(steps, N, 3)`
  - Uses Matplotlib `FuncAnimation` to animate x-y projections.

---

Examples explained

- `examples/run_two_body.py`

  - Simple two-body demonstration and energy-over-time plot.

- `examples/run_three_body.py`

  - Three-body demo that compares Newtonian and Yukawa interactions, and
    runs a small parameter sweep over `lam` to show orbit deviations.

- `examples/run_nbody.py`
  - A small random N-body demonstration (default N=20) to show how the
    code behaves for a larger number of bodies.

---

Recent important changes

- `run_simulation` now returns both positions and velocities histories. This
  fixes a prior issue where energy was computed using the initial velocities
  for every timestep. The examples were updated to use the per-timestep
  velocities when computing kinetic energy.
- `src` and `examples` are packaged (contain `__init__.py`) and examples are
  runnable either via `python -m examples.<name>` or directly as scripts
  (examples include a `ModuleNotFoundError` fallback that inserts the
  project root into `sys.path` for backward compatibility).

---

Potential next steps (ideas you may want to pick from)

- Add `requirements.txt` and/or `pyproject.toml` for easier installs.
- Add unit tests (e.g., check force symmetry, energy conservation for short
  runs, shape/return-value tests for `run_simulation`).
- Add softening parameter to avoid singular accelerations at very small r.
- Vectorize the O(N^2) acceleration calculation or add `numba` to speed up
  loops for moderate N.
- Implement Barnes–Hut / octree for large-N scaling.
- Improve visualization (trails, color by mass/energy, interactive plots
  with Plotly/Bokeh).

---

Troubleshooting

- If you get `ModuleNotFoundError: No module named 'src'`:
  - Make sure you are running from the project root and use `python -m examples.<name>`.
  - Or run the script directly; the examples include a fallback that adds the
    project root to `sys.path`.
