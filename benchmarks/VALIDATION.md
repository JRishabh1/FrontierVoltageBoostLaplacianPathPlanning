# Implementation validation

Validated in the development workspace on Python 3.12.14, PyTorch 2.14.1+cpu,
NumPy 2.3.5 and SciPy 1.17.0. No NVIDIA GPU was exposed. These checks establish
CPU correctness/integration and CUDA compilation, **not CUDA execution or speed**.

- `python -m pytest -q tests/test_gpu_benchmarks.py`: 22 passed, 4 skipped
  (the skipped tests require CUDA). Includes FVB/PF/NF CPU equivalence, an
  independent heap-based Eikonal reference, float32 accuracy, harmonic assembly
  and evaluator parity, seeded batched RRT, segment collision checks, metric
  invariance, watchdog termination and device/timing metadata.
- `python third_party/setup_external.py --selftest`: passed for the compiled
  FMM, KR port, compiled hntf2d and upstream CustFields, including the available
  2D/3D checks. Native extensions were built with GCC/G++.
- `python third_party/setup_external.py --gputest`: passed with **torch on CPU**.
  NF field difference was about 1.1e-16; Eikonal relative differences from the
  first-order upstream FMM were below 1e-13; harmonic-map values differed from
  upstream float32 hntf2d by about 1.5e-6 on its small check map.
- All three fused-kernel specialisations compiled to PTX with NVIDIA NVRTC
  12.9.86 targeting compute_80: float64 Laplace, float64 Eikonal, float32 Eikonal.
  This caught and removed an unnecessary dependency on toolkit math headers.
- A 66-trial smoke experiment exercised candidate selection, independent
  train/test maps, repeated trials, path storage/validation and CSV summaries.
- A further 32-trial smoke experiment exercised all 16 configured families.
  Unsupported-domain and no-path outcomes were retained as failures.
- A final 18-trial checkpoint test reran the same command and executed **zero**
  new trials; the database stayed at 18 rows. SQLite uses a single-file rollback
  journal with FULL synchronisation and a per-suite process lock.
- Normal GPU mode correctly refused to start without CUDA. `git diff --check`,
  Python compilation and shell syntax checks passed.

The launcher reruns the numerical tests on the target GPU before the overnight
search. Inspect those results before using the accelerated implementations in a
research comparison. The supplied search settings are candidates, not measured
optimal parameters. The overnight sweep itself has not run on CUDA here.
