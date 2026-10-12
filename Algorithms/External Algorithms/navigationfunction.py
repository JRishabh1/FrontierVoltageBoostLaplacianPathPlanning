import importlib.util
import os
import sys

import numpy as np
from scipy import ndimage

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


class KoditschekRimonNF(BaseAlgorithm):
    """Koditschek–Rimon navigation function planner (any dimensionality).

    Port of the KRF of johnyf/nfsim (BSD-2; MATLAB) — ``krf.m``::

        phi = gd / (gd^k + beta)^(1/k),   gd = ||q - q_d||^2,  beta = prod_i beta_i

    with ``bi2b.m``'s product obstacle function. nfsim represents arbitrary
    obstacles by implicit functions beta_i (zero on the boundary, positive in
    free space); on an occupancy grid the natural implicit function for each
    obstacle is its Euclidean distance field, so beta_i is the distance (in
    cells, measured to the obstacle's cell faces) to obstacle component i. The
    workspace boundary is one more obstacle: the region outside the grid plus
    every obstacle touching the border. Components merge obstacles that touch,
    including diagonally. Like nfsim (``b(b < 0) = 0``), phi = 1 on obstacles.

    The descent runs on ``log(gd^k / beta) = k·log gd − Σ log beta_i``,
    a strictly increasing function of phi (phi^-k = 1 + beta/gd^k), so it has
    the same critical points and gradient direction as phi — exactly the
    direction of nfsim's analytic gradient ``grad_krf.m``,
    ``(Dgd·b − Db·gd/k)·c`` with c > 0. Working in log space avoids the
    flat-plateau underflow of phi itself (phi ≈ 1 away from the goal for large
    k) without changing the planner.

    ``k`` is the KR tuning parameter (0 = automatic: number of obstacles + 1,
    nfsim's "> #obstacles" rule of thumb). KR guarantees a unique minimum only
    on sphere(-like) worlds with k large enough; on cluttered grid maps local
    minima are expected and are reported as unsolved.

    Descent: the repo's normalised N-linear gradient descent (shared
    ``descend_field``). By default it uses the one-sided (upwind) stencil —
    only the downhill neighbour of each axis — because the log field has no
    useful global Lipschitz bound (it is singular at the goal) and with a
    central stencil the obstacle wall makes the path zig-zag along obstacle
    faces. ``upwind_gradient=False`` caps the central stencil at the largest
    free-to-free neighbour change instead. Returned positions are continuous
    (x, y, ...).
    """

    def __init__(self, map, k: float = 0.0, step_size: float = 1.0,
                 bilinear_interpolation: bool = True, upwind_gradient: bool = True,
                 max_iters: int = 20000, stall_window: int = 200):
        super().__init__(map)
        self.k = k
        self.step_size = step_size
        self.bilinear_interpolation = bilinear_interpolation
        self.upwind_gradient = upwind_gradient
        self.max_iters = max_iters
        self.stall_window = stall_window
        self._path = None
        self._field = None
        self.k_used = None

    def solve(self, visualize=False):
        grid = self.map.grid
        self._field = self._log_field(grid, self.map.end)
        self._path = _common.descend_field(
            self._field, grid, self.map.start, self.map.end,
            step_size=self.step_size, max_iters=self.max_iters,
            stall_window=self.stall_window, interpolate=self.bilinear_interpolation,
            max_rise=0.0 if self.upwind_gradient else _common.free_lipschitz(
                self._field, grid, exclude=tuple(reversed(self.map.end))))
        if visualize:
            self._visualize()
        return self._path

    def result(self, visualize=False):
        if visualize:
            self._visualize()
        return self._path

    def _log_field(self, grid, end):
        ndim = grid.ndim
        free = grid == 0

        # Obstacle components, with the grid surroundings as obstacle 0.
        padded = np.pad(~free, 1, constant_values=True)
        labels, n_obs = ndimage.label(padded, structure=ndimage.generate_binary_structure(ndim, ndim))
        inner = tuple(slice(1, -1) for _ in range(ndim))

        k = self.k if self.k > 0 else n_obs + 1
        self.k_used = k

        log_beta = np.zeros(grid.shape)
        for lab in range(1, n_obs + 1):
            dist = ndimage.distance_transform_edt(labels != lab)[inner]
            # beta_i = distance to the obstacle's faces (0.5 from its cell
            # centres); free cells are always >= 0.5 away.
            with np.errstate(divide='ignore'):
                log_beta += np.log(np.maximum(dist - 0.5, 0.0))

        coords = np.indices(grid.shape, dtype=float)
        end_g = tuple(reversed(end))
        gd = sum((coords[d] - end_g[d]) ** 2 for d in range(ndim))
        gd[end_g] = 1e-12   # goal: log gd -> -inf, keep it finite

        field = k * np.log(gd) - log_beta
        # phi = 1 on obstacles (nfsim clamps beta < 0 to 0): a wall well above
        # every free-space value.
        field[~free] = field[free].max() + 1e3
        return field

    def _visualize(self):
        if self.map.grid.ndim != 2:
            return
        from PIL import Image
        grey = self.map.to_image()
        rgb = np.stack([grey, grey, grey], axis=2).copy()
        for x, y in self._path or []:
            xi, yi = int(round(x)), int(round(y))
            if 0 <= yi < rgb.shape[0] and 0 <= xi < rgb.shape[1]:
                rgb[yi, xi] = [128, 128, 255]
        sx, sy = self.map.start
        ex, ey = self.map.end
        rgb[sy, sx] = [0, 255, 0]
        rgb[ey, ex] = [255, 0, 0]
        Image.fromarray(rgb, mode='RGB').save('navigation_function_result.png')
