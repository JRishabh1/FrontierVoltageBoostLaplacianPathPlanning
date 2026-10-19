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

        b   = prod_i beta_i          (bi2b.m)
        b(b < 0) = 0
        phi = gd ./ (gd.^k + b).^(1/k),   gd = ||q - q_d||^2

    evaluated at every grid cell. nfsim represents arbitrary obstacles by
    implicit functions beta_i (zero on the boundary, positive in free space);
    on an occupancy grid each obstacle's implicit function is its Euclidean
    distance field (to the obstacle's cell faces). The workspace boundary is
    one more obstacle: the region outside the grid plus every obstacle
    touching the border; obstacles that touch (also diagonally) merge.
    Coordinates are scaled to nfsim's demo worlds (radius 5): the map's
    longer side spans 10 units — phi depends on the length unit through
    gd^k versus b. The arithmetic is done in log space only to avoid
    overflowing gd^k; the values are krf.m's.

    ``k`` is the KR tuning parameter (0 = automatic: number of obstacles + 1,
    from krf.m's note "k > #obstacles"). KR guarantees a unique minimum only
    on sphere(-like) worlds with k large enough; on cluttered grid maps local
    minima and flat plateaus (phi ≈ 1) are expected and are reported as
    unsolved.

    Descent: the repo's standard normalised N-linear gradient descent (the
    same as the potential-field baselines; shared ``descend_field``) on phi.
    Returned positions are continuous (x, y, ...).
    """

    WORLD_SIZE = 10.0   # nfsim demo worlds: workspace radius r0 = 5

    def __init__(self, map, k: float = 0.0, step_size: float = 1.0,
                 bilinear_interpolation: bool = True, max_iters: int = 20000,
                 stall_window: int = 200):
        super().__init__(map)
        self.k = k
        self.step_size = step_size
        self.bilinear_interpolation = bilinear_interpolation
        self.max_iters = max_iters
        self.stall_window = stall_window
        self._path = None
        self.failure_reason = None   # why the last solve found no path
        self._phi = None
        self.k_used = None

    def solve(self, visualize=False):
        grid = self.map.grid
        self._phi = self._krf(grid, self.map.end)
        diag = {}
        self._path = _common.descend_field(
            self._phi, grid, self.map.start, self.map.end,
            step_size=self.step_size, max_iters=self.max_iters,
            stall_window=self.stall_window, interpolate=self.bilinear_interpolation,
            diag=diag)
        self.failure_reason = diag.get('reason')
        if visualize:
            self._visualize()
        return self._path

    def result(self, visualize=False):
        if visualize:
            self._visualize()
        return self._path

    def _krf(self, grid, end):
        ndim = grid.ndim
        free = grid == 0
        unit = self.WORLD_SIZE / max(grid.shape)   # world units per cell

        # Obstacle components, with the grid surroundings as obstacle 0.
        padded = np.pad(~free, 1, constant_values=True)
        labels, n_obs = ndimage.label(padded, structure=ndimage.generate_binary_structure(ndim, ndim))
        inner = tuple(slice(1, -1) for _ in range(ndim))

        k = self.k if self.k > 0 else n_obs + 1
        self.k_used = k

        # log b = sum_i log beta_i, beta_i = distance to obstacle i's faces
        log_b = np.zeros(grid.shape)
        with np.errstate(divide='ignore'):
            for lab in range(1, n_obs + 1):
                dist = ndimage.distance_transform_edt(labels != lab)[inner]
                log_b += np.log(np.maximum(dist - 0.5, 0.0) * unit)

        coords = np.indices(grid.shape, dtype=float)
        end_g = tuple(reversed(end))
        gd = sum(((coords[d] - end_g[d]) * unit) ** 2 for d in range(ndim))

        # phi = gd / (gd^k + b)^(1/k)  ==  exp(log gd - logaddexp(k log gd, log b) / k)
        with np.errstate(divide='ignore'):
            log_gd = np.log(gd)
        log_phi = log_gd - np.logaddexp(k * log_gd, log_b) / k
        phi = np.exp(log_phi)
        phi[end_g] = 0.0
        phi[~free] = 1.0   # b(b < 0) = 0 -> phi = 1 on obstacles
        return phi

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


class KoditschekRimonNFGPU(KoditschekRimonNF):
    """GPU variant of :class:`KoditschekRimonNF` (PyTorch; CUDA when available).

    Same phi (krf.m) and the same descent — only where the field is computed
    changes. The per-obstacle Euclidean distances, which dominate the CPU
    version's cost (one SciPy distance transform per obstacle), are computed
    on the device as exact nearest-cell searches against each obstacle's
    surface cells, and phi is assembled there in float64. Component labelling
    and the descent stay on the CPU, as in the repo's other GPU variants.
    Not upstream code; results match the CPU version to rounding.
    """

    def _krf(self, grid, end):
        import torch
        device = _common.torch_device()
        ndim = grid.ndim
        free = grid == 0
        unit = self.WORLD_SIZE / max(grid.shape)

        padded = np.pad(~free, 1, constant_values=True)
        labels, n_obs = ndimage.label(padded, structure=ndimage.generate_binary_structure(ndim, ndim))
        k = self.k if self.k > 0 else n_obs + 1
        self.k_used = k

        # Query points: every grid cell, in padded index coordinates.
        cells = torch.from_numpy(np.indices(grid.shape).reshape(ndim, -1).T + 1.0).to(device)
        cross = ndimage.generate_binary_structure(ndim, 1)
        log_b = torch.zeros(cells.shape[0], dtype=torch.float64, device=device)
        for lab in range(1, n_obs + 1):
            comp = labels == lab
            surface = comp & ~ndimage.binary_erosion(comp, structure=cross)
            pts = torch.from_numpy(np.argwhere(surface).astype(np.float64)).to(device)
            dist = self._nearest(cells, pts)
            log_b += torch.log(torch.clamp(dist - 0.5, min=0.0) * unit)

        end_g = torch.tensor([float(c) + 1.0 for c in reversed(end)], dtype=torch.float64, device=device)
        gd = (((cells - end_g) * unit) ** 2).sum(dim=1)
        log_gd = torch.log(gd)
        phi = torch.exp(log_gd - torch.logaddexp(k * log_gd, log_b) / k)
        phi = phi.reshape(grid.shape).cpu().numpy()
        phi[tuple(reversed(end))] = 0.0
        phi[~free] = 1.0
        return phi

    @staticmethod
    def _nearest(cells, pts, budget=2 ** 24):
        """Exact distance from each cell to its nearest point in ``pts``, in
        chunks of at most ``budget`` pairs."""
        import torch
        out = torch.empty(cells.shape[0], dtype=torch.float64, device=cells.device)
        step = max(1, budget // max(1, pts.shape[0]))
        for i in range(0, cells.shape[0], step):
            diff = cells[i:i + step, None, :] - pts[None, :, :]
            out[i:i + step] = torch.sqrt((diff * diff).sum(dim=2).min(dim=1).values)
        return out

