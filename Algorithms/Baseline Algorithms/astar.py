import heapq
import itertools
import math
import time
import numpy as np

from Algorithms.algorithm import BaseAlgorithm


class AStarAlgorithm(BaseAlgorithm):
    # Move tables in internal (reversed) coordinate order.
    # 2D entries: (dr, dc, cost)  —  internal coords are (row, col)
    # 3D entries: (dz, dy, dx, cost)  —  internal coords are (z, y, x)
    _MOVES = {
        2: [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
            (1, -1, math.sqrt(2)),  (1,  1, math.sqrt(2)),
        ],
        3: [
            (dz, dy, dx, math.sqrt(dx * dx + dy * dy + dz * dz))
            for dz in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if not (dz == 0 and dy == 0 and dx == 0)
        ],
    }

    def __init__(self, map, timeout: float = 30.0, allow_corner_cutting: bool = False,
                 use_native: bool = True):
        super().__init__(map)
        self.timeout = timeout
        self.allow_corner_cutting = allow_corner_cutting
        self._path = None
        self._native = self._load_native() if use_native else None
        self.backend_used = 'cpp-astar' if self._native else 'python-astar'
        self.search_stats = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self, visualize=False):
        self.reset_diagnostics()
        self._path, code, stats = self._search(
            self.map.grid, self.map.start, self.map.end, self.timeout,
            self.allow_corner_cutting, self._native)
        self.search_stats = stats
        if code:
            self.fail(code, {'invalid_endpoint': 'start or goal is outside the grid or blocked',
                             'search_timeout': 'A* search exceeded its time budget',
                             'unreachable': 'A* exhausted the reachable component'}[code],
                      stage='search', **stats)
        if visualize:
            self._visualize()
        return self._path

    def result(self, visualize=False):
        if visualize:
            self._visualize()
        return self._path

    @staticmethod
    def is_solvable(grid: np.ndarray, start: tuple, end: tuple, timeout: float = 30.0) -> bool:
        """Standalone solvability check that does not require a Map object."""
        return AStarAlgorithm._astar(grid, start, end, timeout) is not None

    # ------------------------------------------------------------------
    # Core search (classmethod so is_solvable can call it without an instance)
    # ------------------------------------------------------------------

    @staticmethod
    def _load_native():
        try:
            from Algorithms import _astar_native
            return _astar_native
        except ImportError:
            return None

    @classmethod
    def _astar(cls, grid, start, end, timeout, allow_corner_cutting=False):
        return cls._search(grid, start, end, timeout, allow_corner_cutting,
                           cls._load_native())[0]

    @classmethod
    def _search(cls, grid, start, end, timeout, allow_corner_cutting, native):
        if grid.ndim not in cls._MOVES:
            raise ValueError(f'Unsupported grid dimensionality: {grid.ndim}')
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError('timeout must be finite and nonnegative')
        s, e = cls._to_internal(start), cls._to_internal(end)
        stats = dict(expanded=0, popped=0, stale_pops=0, pushes=0, peak_open=0,
                     path_cost=None)
        for pt in (s, e):
            if (len(pt) != grid.ndim or any(not isinstance(c, (int, np.integer)) or
                    not 0 <= c < size for c, size in zip(pt, grid.shape))):
                return None, 'invalid_endpoint', stats
            if grid[pt] != 0:
                return None, 'invalid_endpoint', stats
        if native is not None:
            return native.search(np.ascontiguousarray(grid != 0, dtype=np.uint8),
                                 s, e, timeout, allow_corner_cutting)
        return cls._python_search(grid, s, e, timeout, allow_corner_cutting, stats)

    @classmethod
    def _python_search(cls, grid, start, end, timeout, corners, stats):
        # A blocked halo removes per-neighbor bounds checks. Flat byte occupancy
        # avoids NumPy scalar indexing, tuples and generators in the hot loop.
        deadline = time.perf_counter() + timeout
        shape = tuple(n + 2 for n in grid.shape)
        strides = tuple(math.prod(shape[i+1:]) for i in range(grid.ndim))
        occupied = np.pad(grid != 0, 1, constant_values=True).tobytes()
        s = sum((c+1)*w for c,w in zip(start,strides))
        e = sum((c+1)*w for c,w in zip(end,strides))
        moves = []
        for *delta, cost in cls._MOVES[grid.ndim]:
            offset = sum(d*w for d,w in zip(delta,strides))
            checks = () if corners else tuple(
                sum(d*w for d,w in zip(part,strides))
                for part in itertools.product(*[(0,d) if d else (0,) for d in delta])
                if any(part) and tuple(part) != tuple(delta))
            moves.append((offset, cost, checks))
        size = len(occupied)
        costs, prev, closed = [math.inf]*size, [-1]*size, bytearray(size)
        costs[s] = 0.
        if grid.ndim == 2:
            width = shape[1]
            ey, ex = end[0]+1, end[1]+1
            def h(idx):
                y,x = divmod(idx,width)
                dy,dx = abs(y-ey), abs(x-ex)
                return max(dx,dy) + (math.sqrt(2)-1)*min(dx,dy)
        else:
            plane, width = strides[0], shape[2]
            ez,ey,ex = (c+1 for c in end)
            def h(idx):
                z,rem = divmod(idx,plane); y,x = divmod(rem,width)
                a,b,c = sorted((abs(z-ez),abs(y-ey),abs(x-ex)))
                return c + (math.sqrt(2)-1)*b + (math.sqrt(3)-math.sqrt(2))*a
        queue = [(h(s), 0., s)]
        stats.update(pushes=1, peak_open=1)
        while queue:
            if time.perf_counter() >= deadline:
                return None, 'search_timeout', stats
            _, neg_cost, cur = heapq.heappop(queue)
            stats['popped'] += 1
            cost = -neg_cost
            if closed[cur] or cost > costs[cur]:
                stats['stale_pops'] += 1
                continue
            if cur == e:
                path = []
                stats['path_cost'] = cost
                while cur != -1:
                    rem, point = cur, []
                    for stride in strides:
                        c, rem = divmod(rem, stride); point.append(c-1)
                    path.append(tuple(reversed(point)))
                    cur = prev[cur]
                return path[::-1], None, stats
            closed[cur] = 1
            stats['expanded'] += 1
            for offset, weight, checks in moves:
                nb = cur + offset
                if occupied[nb] or closed[nb]:
                    continue
                ng = cost + weight
                if ng >= costs[nb]:
                    continue
                blocked = False
                for check in checks:
                    if occupied[cur + check]:
                        blocked = True
                        break
                if blocked:
                    continue
                costs[nb], prev[nb] = ng, cur
                # Larger g wins ties on f; avoid flooding an optimal-cost plateau.
                heapq.heappush(queue, (ng + h(nb), -ng, nb))
                stats['pushes'] += 1
            stats['peak_open'] = max(stats['peak_open'], len(queue))
        return None, 'unreachable', stats

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_internal(point: tuple) -> tuple:
        """(x, y, ...) → (..., y, x) to match numpy grid indexing."""
        return tuple(reversed(point))

    @staticmethod
    def _from_internal(point: tuple) -> tuple:
        """(..., y, x) → (x, y, ...) back to user-facing convention."""
        return tuple(reversed(point))

    @staticmethod
    def _h(a: tuple, b: tuple) -> float:
        return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))

    @classmethod
    def _reconstruct(cls, prev: dict, start: tuple, end: tuple) -> list:
        path, cur = [], end
        while cur in prev:
            path.append(cls._from_internal(cur))
            cur = prev[cur]
        path.append(cls._from_internal(start))
        return path[::-1]

    def _visualize(self):
        ndim = self.map.grid.ndim
        if ndim == 2:
            self._visualize_2d()
        elif ndim == 3:
            self._visualize_3d()

    def _visualize_2d(self):
        from PIL import Image
        grey = self.map.to_image()
        rgb = np.stack([grey, grey, grey], axis=2).copy()

        if self._path:
            for x, y in self._path:
                if 0 <= y < rgb.shape[0] and 0 <= x < rgb.shape[1]:
                    rgb[y, x] = [128, 128, 255]

        sx, sy = self.map.start
        ex, ey = self.map.end
        rgb[sy, sx] = [0, 255, 0]
        rgb[ey, ex] = [255, 0, 0]

        Image.fromarray(rgb, mode='RGB').save('astar_result.png')

    def _visualize_3d(self):
        self.map.save_image_slices('astar_result_3d')
