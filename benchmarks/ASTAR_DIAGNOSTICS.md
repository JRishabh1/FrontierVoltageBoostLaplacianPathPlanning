# A* performance and planner failure diagnostics

## Findings

The app times `algo.solve()` plus final device synchronization. Process startup,
imports, backend warm-up, smoothness metrics and image rendering are outside that
timer. The tiles average **successful maps only**. Thus A*'s 1000/1000 mean and a
planner's 408/1000 mean cover different map subsets.

The previous A* at commit `dfe4714` was slow in its search itself:

* Every candidate neighbor constructed lists, tuples and generator expressions,
  performed generic dimension/bounds checks, and indexed NumPy scalars.
* Strict diagonal collision checking rebuilt `itertools.product` for every move,
  including axial moves, and rechecked cells already known to be free.
* Euclidean distance is admissible but weaker than octile distance for the
  8-neighbor graph (and the analogous exact obstacle-free 26-neighbor metric).
* Heap ties preferred smaller g, expanding more of equal-f plateaus.

A profile of one seeded 512x512 map recorded 7.27 million Python calls, 111,041
heap pops, and about 1.89 of 4.25 profiled seconds in the generic corner-check
`any(...)` expression. The heap itself accounted for only about 0.11 seconds.
Profiler timings have overhead; the unprofiled measurements below are separate.
The unpatched upstream A* permits corner cutting, so its timings are not directly
interchangeable with the strict-corner baseline used here.

## Changes and measured results

The optimized Python search uses flat occupancy bytes with a blocked halo,
indexed cost/parent/closed arrays, precomputed moves and only the necessary corner
checks. It uses exact obstacle-free grid-distance heuristics and larger-g tie
breaking. The optional C++17 implementation uses the same graph, costs and
heuristic. Both remain exact A*; neither uses weighted heuristics or path smoothing.
Tie breaking can change which equal-cost path is selected and its steering score.

Local serial measurements, 12 seeded synthetic 512x512 maps, 3 repeats per map,
20 square obstacles/map, strict corners, Python 3.12.14 on AMD EPYC 9V74:

| Implementation | Successful trials | Mean solve time | Median solve time |
| --- | ---: | ---: | ---: |
| Previous Python (`dfe4714`) | 36/36 | 1.5041 s | 1.5012 s |
| Optimized Python | 36/36 | 0.1253 s | 0.1276 s |
| Compiled C++ CPU | 36/36 | 0.0110 s | 0.0106 s |

Shortest-path costs agreed to 1e-8 on every trial. These are **not the user's
original 1000 maps and not the spark machine**; their exact 1.156 s mean has not
been reproduced. Raw measurements are in
[validation/astar_profile_20261102.json](validation/astar_profile_20261102.json).
Source and generation parameters are in `profile_astar.py`. Backend construction
is outside the timer, matching the app; array conversion and search allocation
are inside. Timed implementations rotate order between repeats.

## Install and rerun

After applying this change, run in the same Python environment as the app:

```bash
python3 -m pip install pybind11
python3 benchmarks/build_astar.py
python3 -m pytest -q tests/test_astar_diagnostics.py tests/test_gpu_benchmarks.py
```

Restart the app. A* defaults to the compiled backend if available; a new result
must say `cpp-astar`. Without a built extension it reports `python-astar`. No build
or compilation is hidden in the solve timer. Set `use_native=false` to compare the
Python fallback. The overnight tuning launcher also builds the extension.

To run both new implementations serially on the user's saved maps:

```bash
python3 benchmarks/profile_astar.py \
  --map-set ms_20261026-200410_fbe6 --limit 20 --repeats 3
```

It reads `Benchmark Data/map_sets/<id>/manifest.json` and the map files. A folder
path may also be supplied. Add `--baseline-ref <old-commit>` to compare the old
implementation from your local Git history with identical corner settings; a
baseline that only supports corner cutting requires `--allow-corner-cutting`.
Do not compare a strict-corner new search to a corner-cutting old search silently.

Use one app worker for latency comparisons. A* now records `cpu_time_s` as well
as wall-clock `time_s`: for this single-threaded search, a much larger wall time
than CPU time is evidence of scheduling waits/contention. For threaded algorithms
process CPU time can exceed wall time. `search_stats` records expanded nodes,
popped/stale heap entries, pushes, peak queue size and path cost. The app/CSV now
preserve these fields and actual backend/device, which the runner previously dropped.

## Failure classification

`failure_code` is a stable category; `error` retains the human explanation.
`failure_details` includes available stage, position, iteration or tree counts.
A budget exhaustion is not proof of unreachability. A zero gradient is not proof
of a local minimum (it may also be a saddle or a flat/degenerate field).

| Code | Observed condition / applicable planners |
| --- | --- |
| `invalid_endpoint` | Invalid/blocked start or goal detected by A*/RRT/RRT-Laplace |
| `unreachable` | A* exhausted its reachable search component |
| `search_timeout` | A*'s internal search budget expired |
| `zero_gradient` | FVB/APF/FMM/Eikonal/KR descent reached zero gradient before the goal |
| `nonfinite_gradient` | Descent gradient contained NaN/Inf |
| `stalled` | APF/external descent or navigation hit its configured progress/stall rule |
| `out_of_bounds` | Descent left the grid or the interior required by its stencil |
| `collision` | A planner's collision check found an occupied cell |
| `collision_or_bounds` | Harmonic navigation left its free-space domain |
| `step_limit` | Path extraction/integration exhausted its step cap |
| `iteration_limit` | RRT/RRT-Laplace exhausted samples or outer iterations; connectivity is unknown |
| `nonconvergence` | Eikonal field failed its convergence test within the sweep cap |
| `singularity`, `singular_jacobian`, `nonfinite_control` | Harmonic-map numerical/control termination |
| `unsupported_map` | A method rejected geometry outside its supported world class |
| `dependency_missing` | Import/native dependency unavailable |
| `invalid_parameters` | Value/type error during planner construction |
| `out_of_memory`, `numerical_error`, `exception` | Execution error; original message/traceback retained |
| `setup_timeout` | Watchdog expired during child startup, setup or backend warm-up |
| `solver_timeout` | Watchdog expired after entry to solve; finer internal stage is unknown |
| `process_crash`, `cancelled` | Child died without a result, or the user cancelled |
| `invalid_path` | Standalone tuning suite's existing strict path validator rejected a returned path |
| `no_path_unclassified` | A return without a specific reason; do not infer a cause |

RRT-Laplace additionally counts failed branch descents by zero/nonfinite gradient,
bounds exit and step limit. Batched RRT reports samples, batches and tree nodes,
which helps distinguish inadequate tree growth from setup or numerical errors.
FVB continues to use its existing 10000-step cap: `step_limit` is not automatically
labelled oscillation. This change does not add collision validation to FVB extraction
or retrofit the standalone tuner's strict path validation into the web app.

For the earlier `n_l=1, epsilon=0` case, a controlled 24x24 reproduction terminates
with `zero_gradient` at the start. Epsilon zero admits untouched cells into the
solved set on the first wave, consistent with that failure. This does not determine
the cause of the ten interpolated failures from the separate screenshot.

Failure counts are visible on new dashboard tiles, per-map rows expand to the full
reason/details, and CSV exports retain the code/details. Existing results are not
rewritten or assigned guessed causes; they require rerunning for instrumentation.

## Validation

`python3 -m pytest -q tests` exercises independent Dijkstra comparisons on 320
random 2D/3D cases across both backends and both corner policies, including path
legality and cost, endpoints, timeouts, diagnostic reset, field failures, RRT budget
exhaustion, branch failure counts, worker transport and CSV export. Existing GPU
benchmark tests still run; CUDA hardware checks skip on this CPU-only test host.
