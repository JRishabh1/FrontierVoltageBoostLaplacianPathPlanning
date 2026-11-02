import importlib.util
import itertools
import math
import os
import sys

import numpy as np

from Algorithms.algorithm import BaseAlgorithm


def _load_common():
    name = 'external_algorithms_common'
    if name not in sys.modules:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'externalcommon.py')
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


_common = _load_common()


class EikonalIterativeGPU(BaseAlgorithm):
    """Eikonal planner with a parallel iterative solver (PyTorch; CUDA when
    available) — the GPU counterpart of :class:`FastMarchingAlgorithm`.

    **Not the Fast Marching Method.** FMM finalises cells one at a time from a
    priority queue, which does not parallelise. This solves the same eikonal
    equation |∇T| = 1/F with the data-parallel scheme used by GPU solvers such
    as the Fast Iterative Method (Jeong & Whitaker, 2008): every cell is
    repeatedly given the first-order Godunov upwind update

        Σ_d max(T − a_d, 0)² = (1/F)²,   a_d = min(T at the two axis-d neighbours)

    in parallel (Jacobi iterations) until no value changes by more than
    ``tolerance``. Its fixed point is the first-order upwind solution: it
    matches the upstream library run with ``high_accuracy=False`` to ~1e-13
    (checked on 2D/3D/4D maps), but not the default second-order
    (high-accuracy) FMM, which is about 3× more accurate against exact
    distances. Our implementation, not upstream code.

    Everything else matches :class:`FastMarchingAlgorithm` so the two can be
    compared directly: the same speed field (plain, or Fast Marching Square
    when ``clearance_saturation > 0``), obstacle cells at ``obstacle_speed``,
    the goal and its free neighbours seeded with exact distances, and the
    repo's standard descent on T (on the CPU).
    """

    def __init__(self, map, clearance_saturation: float = 0.0, obstacle_speed: float = 1e-6,
                 tolerance: float = 1e-9, max_sweeps: int = 0, step_size: float = 1.0,
                 bilinear_interpolation: bool = True, max_iters: int = 20000,
                 stall_window: int = 200, check_every: int = 32,
                 single_precision: bool = False, fused_cuda: bool = True):
        super().__init__(map)
        if check_every < 1 or max_sweeps < 0 or tolerance < 0 or obstacle_speed <= 0:
            raise ValueError('invalid iteration, tolerance or speed parameter')
        self.clearance_saturation = clearance_saturation
        self.obstacle_speed = obstacle_speed
        self.tolerance = tolerance
        self.max_sweeps = max_sweeps
        self.step_size = step_size
        self.bilinear_interpolation = bilinear_interpolation
        self.max_iters = max_iters
        self.stall_window = stall_window
        self.check_every = check_every
        self.single_precision = single_precision
        self.fused_cuda = fused_cuda
        self.converged = False
        self._path = None
        self._arrival = None
        self.sweeps_used = None
        self.failure_reason = None   # why the last solve found no path

    def solve(self, visualize=False):
        self.reset_diagnostics()
        self._path = None
        grid = self.map.grid
        self._arrival = self._arrival_times(grid, self.map.end)
        if not self.converged:
            self._path = None
            return self.fail('nonconvergence', f'eikonal field did not converge in {self.sweeps_used} sweeps', stage='field', sweeps=self.sweeps_used)
        diag = {}
        self._path = _common.descend_field(
            self._arrival, grid, self.map.start, self.map.end,
            step_size=self.step_size, max_iters=self.max_iters,
            stall_window=self.stall_window, interpolate=self.bilinear_interpolation,
            diag=diag)
        self.failure_reason = diag.get('reason')
        self.failure_code = diag.get('code')
        self.failure_details = diag.get('details', {})
        return self._path

    def result(self, visualize=False):
        return self._path

    def _arrival_times(self, grid, end):
        import torch
        from Algorithms.gpu_ops import cupy_for, distance_edt, eikonal_cuda
        device = _common.torch_device()
        ndim = grid.ndim
        free = grid == 0

        # Speed field and boundary values: identical to FastMarchingAlgorithm.
        if self.clearance_saturation > 0:
            dist, self.distance_backend = distance_edt(free, device)
            speed = np.minimum(1.0, dist.cpu().numpy() / self.clearance_saturation)
        else:
            speed = np.ones(grid.shape)
        speed[~free] = self.obstacle_speed
        speed = np.maximum(speed, self.obstacle_speed)

        initial = np.full(grid.shape, math.inf)
        fixed_np = np.zeros(grid.shape, dtype=bool)
        end = tuple(int(c) for c in end)
        for delta in itertools.product((-1, 0, 1), repeat=ndim):
            g_idx = tuple(reversed(tuple(c + d for c, d in zip(end, delta))))
            if all(0 <= g_idx[i] < grid.shape[i] for i in range(ndim)) and free[g_idx]:
                initial[g_idx] = math.sqrt(sum(d * d for d in delta)) / speed[g_idx]
                fixed_np[g_idx] = True

        # Each sweep advances the front by at least one cell, so the number of
        # cells bounds the sweeps needed; convergence normally stops far sooner.
        limit = self.max_sweeps or int(np.prod(grid.shape))
        if self.fused_cuda and ndim == 2 and cupy_for(device) is not None:
            self.backend_used = 'cupy-fused-' + ('float32' if self.single_precision else 'float64')
            field, self.sweeps_used, self.converged = eikonal_cuda(
                initial, fixed_np, 1.0 / speed, limit, self.tolerance,
                self.check_every, self.single_precision)
            return field
        dtype = torch.float32 if self.single_precision else torch.float64
        self.backend_used = f'torch-{device}-{dtype}'
        T = torch.as_tensor(initial, dtype=dtype, device=device)
        fixed = torch.as_tensor(fixed_np, device=device)
        inv_speed = torch.as_tensor(1.0 / speed, dtype=dtype, device=device)
        self.converged = False
        for sweep in range(1, limit + 1):
            T_new = torch.where(fixed, T, torch.minimum(T, self._godunov(T, inv_speed)))
            if sweep % self.check_every == 0 or sweep == limit:
                changed = (T_new < T - self.tolerance) | (torch.isfinite(T_new) & ~torch.isfinite(T))
                T = T_new
                if not bool(changed.any()):
                    self.converged = True
                    break
            else:
                T = T_new
        self.sweeps_used = sweep
        return T.cpu().numpy()

    @staticmethod
    def _godunov(T, inv_speed):
        """First-order Godunov upwind update of every cell (N-D, unit spacing)."""
        import torch
        ndim = T.ndim
        mins = []
        for d in range(ndim):
            P = _pad_inf(T, d)
            mins.append(torch.minimum(P.narrow(d, 0, T.shape[d]), P.narrow(d, 2, T.shape[d])))
        if ndim == 2:
            lo, hi = torch.minimum(*mins), torch.maximum(*mins)
            delta = hi - lo
            two_axis = lo + 0.5 * (delta + torch.sqrt(
                torch.clamp(2 * inv_speed.square() - delta.square(), min=0)))
            return torch.where(torch.isfinite(hi) & (delta < inv_speed), two_axis, lo + inv_speed)
        a, _ = torch.sort(torch.stack(mins), dim=0)          # a_0 <= a_1 <= ...
        t = a[0] + inv_speed
        s1 = a[0].clone()
        s2 = a[0] * a[0]
        for m in range(2, ndim + 1):
            am = a[m - 1]
            use = t > am                                       # include the m-th direction
            s1 = s1 + torch.where(use, am, torch.zeros_like(am))
            s2 = s2 + torch.where(use, am * am, torch.zeros_like(am))
            disc = s1 * s1 - m * (s2 - inv_speed * inv_speed)
            cand = (s1 + torch.sqrt(torch.clamp(disc, min=0.0))) / m
            t = torch.where(use & (disc >= 0), cand, t)
            # a is sorted, so once a direction is excluded the larger ones are too
        return t


def _pad_inf(T, d):
    """Pad axis ``d`` of ``T`` by one cell of +inf on each side (any ndim)."""
    import torch
    shape = list(T.shape)
    shape[d] = 1
    edge = torch.full(shape, math.inf, dtype=T.dtype, device=T.device)
    return torch.cat([edge, T, edge], dim=d)
