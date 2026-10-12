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

All return continuous `(x, y, ...)` paths like the other planners, run in the
app's watchdog subprocess, and report maps outside their world class as
unsolved with the reason (`UnsupportedMapError`) rather than crashing.

| Registry id | Class | Dims | Field / method | Path extraction |
|---|---|---|---|---|
| `fast_marching` | `FastMarchingAlgorithm` | 2–6 | Arrival time T from the goal; speed 1 (plain FMM) or min(1, d/sat) (Fast Marching Square) | Shared N-linear descent on T |
| `navigation_function` | `KoditschekRimonNF` | any | KR NF φ = γ/(γᵏ+β)^(1/k) with β = Π (distance to obstacle i) | Shared descent on log(γᵏ/β) (monotone in φ) |
| `harmonic_map` | `HarmonicMapNavigation2D` | 2 | Harmonic map to the punctured disk + the paper's adaptive harmonic potential | Paper's control law u = −K_u s J⁻¹∇ψ, integrated |
| `custom_fields` | `CustomHarmonicFields2D` | 2 | Forest→point diffeomorphism + harmonic potential (Lemma 5 weights) | Upstream `TOPO.integrate_forest` |

### Things to know when reading the results

- **Timing is not apples to apples across languages.** The FMM solve and the
  hntf2d map construction are compiled C++; FVB, the NF port and the hntf2d
  controller are NumPy/Python; cust_fields is pure Python. Compare solve
  times with that in mind (or compare like with like).
- **Field descent is shared.** FMM and the NF use the same normalised ±1-cell
  N-linear descent as the repo's potential-field planners, so they differ only
  in the field. Large obstacle values would swamp that stencil in narrow
  corridors, so samples are capped: at T's free-space Lipschitz bound for FMM
  (|∇T| = 1/F), and by the upwind (downhill-only) stencil for the NF, whose
  log field has no useful global bound. Both are an `upwind_gradient`
  parameter, so you can sweep them.
- **Plain FMM descends into 1-cell corridors** (shortest paths use them) and
  can then bounce at the entrance with a 1-cell step; FM² avoids narrow
  passages. A smaller `step_size` helps.
- **KR navigation functions** only guarantee a unique minimum on sphere-like
  worlds with large enough `k`; cluttered grid maps produce local minima,
  reported as unsolved. `k = 0` picks #obstacles + 1.
- **hntf2d** ships only the transformation; the controller (adaptive gains
  k_d, k_i, gate s(q), Sensors 2023 eqs. 6–16, §6.1 defaults) is implemented
  in the wrapper. The domain is the start's face-connected free region; two
  obstacle cells touching at a corner count as one obstacle. Integration
  steps shrink near walls (as the continuous controller would slow), and the
  adaptive law can make the robot approach the outer wall and reverse — a
  sharp turn the steering metric will see.
- **cust_fields** only handles forest worlds: free-standing obstacles that are
  chains of overlapping rectangles, not touching the border (upstream's
  workspace-attached trees crash: `compute_virtual_ws` builds `Rectangular`
  with missing arguments). Its constants (λ, μ, a +0.1 squircle padding)
  are tuned for ~5-unit worlds with a few obstacles, so maps are scaled to
  `world_size` and λ is exposed — the right λ is map-dependent. The paper's
  homotopy *customization* needs a target homotopy class, which a plain
  start→goal benchmark doesn't define, so only the underlying harmonic field
  runs. On the default map generator most maps are rejected; use maps with a
  few obstacles kept off the border to evaluate it.
- **nfsim** is MATLAB and no MATLAB/Octave runtime is assumed, and calling it
  per gradient step would make its timing meaningless anyway. The KR NF is a
  ~10-line formula (`krf.m`, `grad_krf.m`, `bi2b.m`); the port follows it
  exactly and represents arbitrary grid obstacles by distance fields, the
  grid analogue of nfsim's implicit obstacle functions.
