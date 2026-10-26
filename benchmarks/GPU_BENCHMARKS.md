# Best-effort GPU comparison and parameter search

Run each planner with a suitable implementation, tune only on training maps,
and compare frozen settings on independent test maps. This is an **empirical
search**, not a claim that these settings are globally optimal or that a GPU
is faster for every method. CUDA performance must be measured on your machine.

## Overnight command (Linux, NVIDIA GPU)

From the repository root:

```bash
tmux new -s gpu-tuning
bash benchmarks/start_tuning.sh --suite gpu-night-1 --hours 8
```

Detach with Ctrl-B, D. Reattach with `tmux attach -t gpu-tuning`. To resume after
the wall budget is exhausted, run the **same command** again. `--hours` is the
budget for each invocation, so increasing it does not invalidate the run.
The default search may need multiple nights; it stops at the budget and keeps
completed work. Setup/compilation and launch-time tests precede this budget.

The launcher creates `.venv` only if absent, installs the benchmark extra,
installs the CuPy wheel for the CUDA major version used by PyTorch, builds the
pinned CPU reference implementations, and runs numerical tests **on your GPU**
before starting. It requires `python3-venv`, a C++ compiler, git, and a working
NVIDIA driver. If your Python was built with a compiler absent on the machine,
set `CC=gcc CXX=g++` when launching. It never replaces/deletes an existing env.

If your installed PyTorch is CPU-only, choose the CUDA install command from
[PyTorch's installer](https://pytorch.org/get-started/locally/). You can pass its
official wheel index as `TORCH_INDEX_URL` to the launcher. Check `nvidia-smi`.
Never install multiple conflicting CuPy distributions into one environment.
`BENCH_VENV=/path/to/env` selects another environment; `SKIP_SETUP=1` skips
installation/build but still checks CUDA and runs the tests.

For a shorter first run on the GPU:

```bash
bash benchmarks/start_tuning.sh --suite gpu-pilot --sizes 100 \
  --train-maps 4 --test-maps 8 --trials 4 --repeats 3 --hours 2
```

The raw CLI can inspect the plan without running or installing anything:

```bash
.venv/bin/python benchmarks/tune_gpu.py --check
.venv/bin/python benchmarks/tune_gpu.py --dry-run
```

## Implementations

| Method | Accelerated work | Qualifications |
|---|---|---|
| FVB 2D | One fused CUDA kernel per Laplace sweep, two persistent field buffers | Float64 and the existing boost/mask semantics. PyTorch convolution fallback when CuPy is absent. CPU descent. |
| Eikonal | Fused 2D Godunov updates, stable difference-form quadratic, convergence checks every 32 sweeps | First-order **iterative Eikonal**, not FMM. Float64 default; explicitly labelled float32 candidates. Exceeding the sweep cap is failure. ND retains the torch solver. |
| KR navigation function | Exact CUDA EDT per obstacle component, float64 field assembly | Replaces quadratic pairwise surface distances. SciPy EDT fallback for absent CuPy or >3D. CPU component labelling and descent. Same grid adaptation as before, not an analytic sphere-world solver. |
| Potential field 2D | Exact CUDA EDT and elementwise field assembly | Same potential; CPU descent. Local minima remain real failures. |
| Harmonic map | Chunked panel matrix assembly and GPU dense linear solve | Float64, unlike upstream hntf2d float32. Memory guard reports oversized systems. Exact single-point evaluation/controller run on CPU after one coefficient transfer, avoiding a GPU round trip each step. |
| Batched RRT | Batched nearest neighbours, steering, and collision proposals | **Separate variant**, not sequential RRT or RRT*. A batch expands against a frozen tree snapshot. No path shortcutting. |
| RRT–Laplace predecessors | Existing torch convolution implementation | Included and tuned; no new fused implementation claimed. |
| CPU controls | FVB, compiled FMM, KR, compiled hntf2d, PF, sequential RRT, A*, CustFields | Keep strong CPU implementations in the comparison. CustFields remains upstream CPU code with its existing domain restrictions. |

The app exposes the implementation options and the new `rrt_batch_gpu` entry.
The old `run_suite.py` remains available for its original fixed-configuration
CPU/torch-CPU/CUDA comparison. This new runner uses its own SQLite/CSV exports;
map sets are visible in the web app, but tuning trial records are not app runs.

## Search and fairness rules

Defaults in [`best_effort_config.json`](best_effort_config.json):

- Sizes 100, 512, 1000. Eight training maps and 24 independently seeded test maps
  per size; identical maps for every planner at that size.
- At most eight parameter settings per family, including its default. Deterministic
  random search over the listed grid, then successive halving on 2, 4, 8 training
  maps. Identical map/timeout caps apply to candidates in each completed round;
  these are evaluation-count caps, **not equal GPU-hour budgets**. A* has only one
  setting. Related CPU/CUDA implementations sample the same settings where their
  parameter spaces match.
- Pruning prioritises success rate, then `log1p(median_time) +
  log1p(mean_steering30_per_length)`, then path length. This explicit heuristic is
  used only for search. It is not an overall ranking on the test set.
- Among the final candidates, select the fastest and smoothest at the highest
  observed training success rate (often the same setting). Save these choices
  **before generating test maps**. No per-test-map tuning, oracle, or failed-map
  substitution. This search cannot guarantee that every Pareto-optimal setting
  survives early pruning.
- Three held-out repetitions. Randomised methods receive three reproducible seeds
  per map; deterministic methods repeat the same solve for timing. Reported
  success rates count attempts, not independent maps. `maps` reports map count.
- One solve at a time, shuffled candidate order, one CPU math-library thread by
  default (`--cpu-threads` changes it). Do not run a competing benchmark on the
  same GPU. FMM/A* remain CPU implementations, with explicit device labels.

The parameter grid uses FVB `n_l=16..400`, positive epsilon values, Eikonal
clearance/precision settings, KR exponents, harmonic panel sizes/gains, PF
repulsion scales, and RRT batch/step/bias settings. Spatial influence radii and
panel lengths scale with map width. PF repulsive gain scales as width^4 because
attraction has distance-squared units and the repulsion has inverse-distance-
squared units. Defaults are included unscaled as an anchor. All are candidates,
not settings represented as already tuned. FVB epsilon zero is excluded from
the grid: equality can mark undiffused cells solved.

All maps have a blocked outer ring and face-connected, separated endpoints,
matching the continuous planners and FVB's boundary conditions. Narrow corridors
and concave obstacles are still possible. CustFields rejects maps outside its
world class; `unsupported` is reported separately and still counts as failure
in the all-map success rate. For claims about that method's valid domain, use a
separate explicitly constructed supported-domain experiment.

## Correctness and metrics

Success requires correct endpoints, finite coordinates, and a collision-free
**whole polyline**. Cells are closed unit squares centred on integer coordinates;
touching a blocked edge/corner is rejected. The validator traverses all grid
crossings, including the final goal segment; waypoint-only checks and approximate
collision sampling can miss collisions. Raw paths are preserved even on rejection.
A* now rejects diagonal corner cutting by default (a legacy toggle is available),
and RRT/PF checks use the same cell-centre convention.

Smoothness is measured on raw points **and** on samples every one unit of arc
length along the original polyline. This prevents small descent steps or sparse
RRT nodes from changing the metric's physical scale. Resampling is for evaluation,
not smoothing or changing the returned path. Scores include excess steering
angles at 10°, 20°, 30°, 45°, the raw 30° score, path length, and 30° score per unit
length. The five-offset definition is the existing repository metric.

Timing uses wall time around `solve()` with CUDA synchronisation before/after.
Field construction, transfers, and CPU descent are included. Python imports,
construction, generic device warm-up, and fused-kernel compilation are outside
`time_s` and reported as `setup_s`; `wall_s` includes the whole subprocess and
watchdog wait. First-use work not covered by generic warm-up remains in solve
time. Fresh subprocesses isolate every solve. No pure-kernel or throughput timing
is presented as end-to-end single-query latency. The process timeout includes
startup; all methods receive the same per-size watchdog cap.

Do not compare solved-only means without success rates: one algorithm may solve
only easy maps. `paired.json` additionally compares time and smoothness on the
same map/repetition cases successfully solved by both configurations. Those
paired numbers describe only the common successful subset. No bootstrap intervals
or statistical-significance claims are generated.

## Outputs and restart behaviour

Under `benchmarks/results/<suite>/`:

| File | Contents |
|---|---|
| `status.json` | `complete`, `budget_exhausted`, `interrupted`, or `error`; trial count |
| `plan.json` | Exact config, source fingerprint, commit, package/device/driver/thread metadata |
| `maps-*.json` | Train/test map manifests, endpoints, content hashes |
| `trials.sqlite3` | Transactional per-trial records, errors, seeds, raw paths and metrics |
| `results.csv` | Every attempt, including failures, device/backend, actual precision and params |
| `summary.csv`, `summary.json` | Success, timing, smoothness, unsupported/timeout/invalid-path counts |
| `selected-*.json` | Frozen fast/smooth settings per family and size |
| `paired.json` | Comparisons on common successful held-out instances |
| `sweep.log` | Progress |

Use `tail -f benchmarks/results/<suite>/sweep.log`. A result saved in SQLite is
not repeated on restart. An in-flight solve interrupted before commit is rerun.
Partially generated map sets are regenerated with the same seed; partial orphan
sets may remain visible in the app. An OS-owned lock prevents concurrent writes
to one suite. A config/source/hardware change requires a new suite name so data
from different experiments are not silently mixed. Source snapshots and Python
environment versions should be kept with any report.

The runner processes sizes in order; within a training round it interleaves
families. An expired budget may leave larger sizes unfinished. Consult `status`
and attempt counts rather than treating partial tables as a finished comparison.

## Validation without a GPU

```bash
python -m pip install -e '.[benchmark]'
python -m pytest -q tests/test_gpu_benchmarks.py
python benchmarks/tune_gpu.py --suite cpu-smoke --allow-cpu --sizes 24 \
  --families astar_cpu fvb_cpu fvb_cuda eikonal_cuda nf_cuda harmonic_cuda rrt_batch_cuda \
  --train-maps 2 --test-maps 2 --trials 2 --repeats 1 --hours 1
```

`--allow-cpu` is only a correctness smoke mode. Its rows say `device=cpu`; family
names ending in `_cuda` identify the requested implementation, **not evidence of
CUDA execution**. The normal runner fails if CUDA is unavailable, and refuses a
successful CPU fallback during a requested GPU trial. Four hardware tests cover
fused FVB, float32/float64 Eikonal, and CUDA EDT/NF parity; these are skipped on a
CPU machine and run in the overnight launcher.
