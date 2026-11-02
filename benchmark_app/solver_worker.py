"""
Run a single (algorithm, map) solve in a child process with a hard timeout.

Rationale: some algorithms have no internal timeout (FrontierVoltageBoostLaplace
loops until the start cell is solved, which never happens if the start is
unreachable under its rules). Python threads cannot be killed, but a child
process can be terminated — so each solve runs in its own process. This also
isolates the app from crashes inside algorithm code.

The child measures wall-clock time around ``solve()`` itself, so process
startup overhead is not counted in the reported timing.
"""

import multiprocessing as mp
import time
import traceback

import numpy as np


def describe_error(exc: BaseException, limit: int = 5) -> str:
    """Error text with the reason first (the UI shows only its start).

    Expected outcomes — a map outside an algorithm's world class, or an
    external planner that is not set up — are reported by message alone;
    anything else keeps the traceback after the summary for debugging.
    """
    summary = f'{type(exc).__name__}: {exc}'
    if type(exc).__name__ == 'UnsupportedMapError':
        return f'unsupported map: {exc}'
    if isinstance(exc, ImportError):
        return summary
    return f'{summary}\n\n{traceback.format_exc(limit=limit)}'


def _worker(conn, algo_id: str, params: dict, grid: np.ndarray, start: tuple, end: tuple):
    """Child-process entry: build map + algorithm, solve, send result back."""
    stage = 'setup'
    try:
        conn.send({'event': 'stage', 'stage': stage})
        setup_start = time.perf_counter()
        from benchmark_app.registry import get_spec
        from map_generation import Map, Map2D, Map3D

        if grid.ndim == 2:
            map_cls = Map2D
        elif grid.ndim == 3:
            map_cls = Map3D
        else:
            map_cls = Map   # N-dimensional grids use the base Map
        map_obj = map_cls(grid, tuple(start), tuple(end))

        spec = get_spec(algo_id)
        algo = spec.build(map_obj, params)
        device = 'cpu'
        stage = 'warmup'
        conn.send({'event': 'stage', 'stage': stage})

        # Importing PyTorch in this fresh subprocess costs ~0.5s+ and the first
        # tensor/conv op pays one-time dispatcher + algorithm-selection setup —
        # that's environment overhead, not algorithm work. Do it all here so
        # the timed region below measures the solve itself.
        if spec.gpu or (params or {}).get('use_gpu'):
            import torch
            import torch.nn.functional as tF
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if device == 'cuda':
                torch.cuda.init()
                if algo_id in ('fvb_laplace', 'eikonal_gpu') and (params or {}).get('fused_cuda', True):
                    from Algorithms.gpu_ops import warmup_cuda_kernels
                    warmup_cuda_kernels()
            warm = torch.zeros((1, 1, 4, 4), dtype=torch.double, device=device)
            tF.conv2d(warm, torch.zeros((1, 1, 3, 3), dtype=torch.double, device=device))
            (warm + 1).sum().item()   # forces execution (and a CUDA sync when on GPU)
            if device == 'cuda':
                torch.cuda.synchronize()

        setup_s = time.perf_counter() - setup_start
        stage = 'solve'
        conn.send({'event': 'stage', 'stage': stage})
        t0 = time.perf_counter()
        cpu0 = time.process_time()
        path = algo.solve()
        if device == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        cpu_s = time.process_time() - cpu0

        if path is not None:
            # Preserve continuous coordinates — algorithms like FVB return
            # decimal positions and metrics must be computed on the raw path.
            path = [tuple(float(c) for c in p) for p in path]

        # Planners may say why they found no path (external wrappers do).
        reason = getattr(algo, 'failure_reason', None) if path is None else None
        conn.send({'ok': True, 'path': path, 'time_s': elapsed, 'reason': reason,
                   'failure_code': getattr(algo, 'failure_code', None),
                   'failure_details': getattr(algo, 'failure_details', {}),
                   'search_stats': getattr(algo, 'search_stats', {}), 'cpu_time_s': cpu_s,
                   'device': device, 'backend': getattr(algo, 'backend_used', None),
                   'setup_s': setup_s, 'converged': getattr(algo, 'converged', None),
                   'sweeps_used': getattr(algo, 'sweeps_used', None)})
    except Exception as exc:
        code = 'exception'
        if type(exc).__name__ == 'UnsupportedMapError':
            code = 'unsupported_map'
        elif isinstance(exc, ImportError):
            code = 'dependency_missing'
        elif isinstance(exc, MemoryError) or 'OutOfMemory' in type(exc).__name__:
            code = 'out_of_memory'
        elif isinstance(exc, (FloatingPointError, np.linalg.LinAlgError)):
            code = 'numerical_error'
        elif stage == 'setup' and isinstance(exc, (ValueError, TypeError)):
            code = 'invalid_parameters'
        conn.send({'ok': False, 'error': describe_error(exc, limit=5),
                   'failure_code': code, 'failure_details': {'stage': stage}})
    finally:
        conn.close()


def solve_with_timeout(algo_id: str, params: dict, grid: np.ndarray,
                       start: tuple, end: tuple, timeout_s: float = 120.0,
                       cancel_event=None) -> dict:
    """Solve in a subprocess. Returns {solved, path, time_s, error, cancelled?}.

    ``cancel_event`` (a ``threading.Event``) is polled while waiting; setting
    it terminates the child immediately and returns a cancelled outcome.
    """
    ctx = mp.get_context('spawn')
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_worker,
                       args=(child_conn, algo_id, params, grid, start, end))
    proc.start()
    child_conn.close()  # parent keeps only the read end

    # Wait in short slices so cancellation is responsive.
    result = None
    stage = 'startup'
    cancelled = False
    deadline = time.monotonic() + timeout_s
    def stop_child():
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=2)
        if proc.is_alive():
            proc.kill()
        proc.join()

    try:
        while time.monotonic() < deadline:
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                break
            if parent_conn.poll(min(0.2, max(0.0, deadline - time.monotonic()))):
                try:
                    result = parent_conn.recv()
                except EOFError:
                    result = None
                if result and result.get('event') == 'stage':
                    stage = result['stage']
                    result = None
                    continue
                break
            if not proc.is_alive():
                break
    except BaseException:
        stop_child()
        parent_conn.close()
        raise

    if cancelled:
        stop_child()
        parent_conn.close()
        return {'solved': False, 'path': None, 'time_s': None,
                'error': 'cancelled', 'cancelled': True, 'failure_code': 'cancelled',
                'failure_details': {'stage': stage}}

    # Once the watchdog deadline is reached, do not give a stuck solver an
    # additional five seconds to run before terminating it.
    proc.join(timeout=0 if result is None else 5)
    timed_out = proc.is_alive()
    if timed_out:
        stop_child()

    parent_conn.close()

    if result is None:
        if timed_out:
            return {'solved': False, 'path': None, 'time_s': timeout_s,
                    'error': f'timed out after {timeout_s:.0f}s during {stage} (process terminated)',
                    'failure_code': 'solver_timeout' if stage == 'solve' else 'setup_timeout',
                    'failure_details': {'stage': stage, 'budget_s': timeout_s}}
        return {'solved': False, 'path': None, 'time_s': None,
                'error': f'solver process died (exit code {proc.exitcode})',
                'failure_code': 'process_crash',
                'failure_details': {'stage': stage, 'exit_code': proc.exitcode}}

    if not result.get('ok'):
        return {'solved': False, 'path': None, 'time_s': None,
                'error': result.get('error', 'unknown error'),
                'failure_code': result.get('failure_code', 'exception'),
                'failure_details': result.get('failure_details', {})}

    path = result['path']
    return {
        'solved': path is not None and len(path) > 0,
        'path': path,
        'time_s': result['time_s'],
        'cpu_time_s': result.get('cpu_time_s'),
        'search_stats': result.get('search_stats', {}),
        'failure_code': None if path else (result.get('failure_code') or 'no_path_unclassified'),
        'failure_details': {} if path else result.get('failure_details', {}),
        'device': result.get('device'),
        'backend': result.get('backend'),
        'setup_s': result.get('setup_s'),
        'converged': result.get('converged'),
        'sweeps_used': result.get('sweeps_used'),
        'error': None if path else (f"no path: {result['reason']}" if result.get('reason')
                                    else 'algorithm returned no path'),
    }
