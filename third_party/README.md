# third_party — external comparison planners

Upstream implementations of the planners we compare Frontier Voltage Boost
against, wrapped as normal algorithm classes in
[`Algorithms/External Algorithms/`](../Algorithms/External%20Algorithms) and
registered in the benchmark app.

```bash
pip install pybind11 pyyaml                     # build/runtime deps (once)
python third_party/setup_external.py            # fetch pinned sources + build (~30 s)
python third_party/setup_external.py --selftest # solve a small map with every planner
```

The benchmark app runs the setup automatically before the first run that
uses an external planner (and again if the app's Python version changes,
since compiled modules are per-interpreter). If it cannot build — no git, no
C++ compiler, or pybind11/pyyaml missing — the run stops with the reason.
Unsolved maps report why (e.g. `no path: local minimum near (41.0, 17.3)` or
`unsupported map: 2 obstacle(s) touch the map border …`).

`setup_external.py` checks each upstream repo out at a **pinned commit** into
`third_party/src/` and compiles the native parts into `third_party/build/`.
Both directories are git-ignored: compiled modules are platform-specific (a
macOS arm64 build will not load on a Linux VM), so each machine builds its
own from the same pinned sources. Upstream code is compiled **unmodified** —
only the small pybind11 bindings in `bindings/` are ours. Needs git and a
C++17 compiler; no system libraries (Eigen and tinyxml2 are fetched too).

| Component | Upstream (pinned) | License | How it runs |
|---|---|---|---|
| `fmm` | [thinks/fast-marching-method](https://github.com/thinks/fast-marching-method) @ `b20f0b2` | MIT | C++ eikonal solver via `bindings/fmm_module.cpp` (2–6D) |
| `hntf2d` | [maxchaos/hntf2d](https://github.com/maxchaos/hntf2d) @ `583527e` (+ Eigen 3.4.0, tinyxml2 10.0.0) | GPL-3.0 | C++ harmonic map via `bindings/hntf2d_module.cpp`; upstream `poly2circle` used as-is |
| `cust_fields` | [Shuaikang-Wang/cust_fields](https://github.com/Shuaikang-Wang/cust_fields) @ `5ca178e` | none stated | Pure Python, imported directly (`NF/`, `TOPO/`) |
| — | [johnyf/nfsim](https://github.com/johnyf/nfsim) @ `21bf45d` | BSD-2 | MATLAB — not fetched; the KR navigation function is ported (see below) |

**Licensing note.** Nothing from hntf2d (GPL-3.0) or cust_fields (no license
file, so all rights reserved by default) is committed to this repository —
they are fetched on the user's machine at build time. Keep it that way unless
the repo's own license is chosen with that in mind.

## The four wrappers

Principle: **run each algorithm as published.** The wrappers call the upstream
code directly wherever it exists and add only what is unavoidable to run it
on our occupancy-grid maps (listed below). Nothing is tuned to make a
baseline succeed more often; problems found upstream are listed at the end
rather than patched.

All return continuous `(x, y, ...)` paths like the other planners, run in the
app's watchdog subprocess, and report why a map was not solved: either
`unsupported map: …` (outside the method's world class) or `no path: …`.

| Registry id | Upstream code called | Written here (and why) |
|---|---|---|
| `fast_marching` | the C++ eikonal solver (`HighAccuracyVaryingSpeedEikonalSolver` + `SignedArrivalTime`) | path extraction — the library computes the arrival-time field only. Uses the repo's standard descent (same as the potential-field baselines). |
| `navigation_function` | — (nfsim is MATLAB source; no runtime here) | port of `krf.m` / `bi2b.m`, evaluated on the grid, then the repo's standard descent. |
| `harmonic_map` | the C++ `HarmonicMap2D` (panel-method map, `map`, `jacob`, `puncts`) and upstream's `poly2circle` | the controller (paper eqs. 6–16) — it is not in the upstream repo. |
| `custom_fields` | `NF.geometry.World`, `ForestToStar`, `StarToSphere`, `SphereToPoint`, `TOPO.integrate_forest` (upstream defaults) | grid → forest-world conversion, and the 3-line map composition from upstream's `test_cust_nf.PointMap` (that script imports matplotlib at module level). |

### Choices needed to run on grid maps

- **Geometry.** Cell (x, y) is the unit square around (x, y). The continuous
  methods use the start's face-connected free region; obstacle cells touching
  at a corner count as one obstacle (a zero-width gap is not traversable).
  For hntf2d's Jordan-curve boundaries, a boundary passing twice through such
  a corner is separated by 0.05 cells.
- **Units.** Maps are scaled into each method's native world size where its
  behaviour depends on units: nfsim demo worlds (radius 5, so the map's
  longer side = 10 units) for the NF, `world_size` (default 5, upstream's
  example world) for cust_fields. hntf2d's paths do not depend on scale.
- **Obstacles for the NF.** nfsim's implicit obstacle functions β_i become
  the distance to each obstacle component; the grid border is obstacle 0.
- **FMM speed.** The solver requires speed > 0, so obstacle cells get
  `obstacle_speed` (1e-6). The goal's free neighbours are seeded with exact
  distances, as the upstream README recommends for the high-accuracy solver.
  `clearance_saturation > 0` switches to the Fast Marching Square speed
  field — a separate, published variant; off by default.
- **hntf2d integration.** The paper simulates with explicit Euler at 10 ms
  (`dt`) and §6.1 gains. At grid scale a 10 ms step can travel several
  cells, so such a step is split into shorter Euler steps of at most
  `step_size` cells — same scheme, finer step, no other change.
- **Termination.** Goal reached within 1 cell; failure on leaving free
  space, a stalled field (no decrease for `stall_window` steps) or the
  iteration limit — the same rules for every wrapper.
- **cust_fields workspace.** Upstream's `Workspace` adds +0.1 to its size,
  which would let paths leave the map; that padding is cancelled so the
  workspace equals the map. Obstacle padding is left as upstream has it.
- **Not run:** cust_fields' homotopy *customization* — it needs a target
  homotopy class, which a start→goal benchmark does not define.

### Reading the results

- **Timing is not like-for-like across languages:** the FMM solve and the
  hntf2d map are compiled C++; FVB, the NF port and the hntf2d controller
  are NumPy/Python; cust_fields is pure Python.
- **With the repo's standard descent, plain FMM and the NF zig-zag along
  walls** (the ±1-cell stencil samples obstacle values) and plain FMM can
  bounce at the entrance of 1-cell corridors, which shortest paths use.
- **KR navigation functions** only guarantee a unique minimum on sphere-like
  worlds with large enough k; on cluttered maps expect local minima and flat
  regions (φ ≈ 1), reported as unsolved. `k = 0` uses #obstacles + 1.
- **cust_fields** handles only forest worlds: free-standing obstacles that are
  chains of overlapping rectangles, off the border. Its star→sphere λ is
  map-dependent (upstream tunes it per example); most default-generator maps
  are rejected.

### Problems found upstream (not patched)

- **hntf2d README example** builds its "inner box" from `linspace(-1, 1)`
  instead of `linspace(-0.1, 0.1)`, so the example boundary self-intersects
  and its printed output is meaningless. `HarmonicMap2D` does not validate
  input: a repeated vertex (zero-length segment) silently yields NaNs.
- **hntf2d computes in single precision** (`Eigen::MatrixXf`), so the map and
  Jacobian lose accuracy within about a panel of the boundary.
- **cust_fields `ForestToStar.compute_virtual_ws`** calls
  `Rectangular(_type, center, w, h)` without the required `theta` and `s`
  arguments, so any obstacle tree attached to the workspace boundary raises
  `TypeError`.
- **cust_fields purges each obstacle tree as a parent→child chain**
  (`obs[depth]` into `obs[depth-1]`), but its own `example_world.yaml` has a
  three-star tree whose second and third stars do not overlap.
- **cust_fields `SphereToPoint.compute_T_q`** starts with `T = q` and then
  `T += …`, mutating its input array in place, so later obstacles in the
  loop see the modified point.
- **cust_fields uses two different squircle formulas** for the same obstacle:
  `Rectangular.potential` (the transformation; mixes length² and length⁴
  terms, so its shape depends on the world's scale) and `check_point_inside`
  (the integrator's collision check). `NF/utils.py`'s ray length divides by
  |q − centre| and emits 0/0 warnings.
