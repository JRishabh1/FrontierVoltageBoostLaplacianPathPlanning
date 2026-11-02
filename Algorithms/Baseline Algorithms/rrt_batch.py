"""A labelled batched RRT variant; not sequential RRT and not RRT*."""
import math

import numpy as np

from Algorithms.algorithm import BaseAlgorithm


class BatchedRRTGPU(BaseAlgorithm):
    """Expand many samples against one frozen tree snapshot per batch.

    Nearest-neighbour searches, steering and segment checks run on the device.
    All accepted proposals attach only to the old tree, so parent chains are
    acyclic. Batching changes the sampling process and is reported explicitly.
    No shortcutting or smoothing is applied. Final benchmark validation checks
    exact cell intersections in addition to these 0.2-cell proposal checks.
    """
    def __init__(self, map, max_iterations=20000, batch_size=128, step_size=5.,
                 goal_sample_rate=0.1, goal_tolerance=1.5, seed=0):
        super().__init__(map)
        if max_iterations < 1 or batch_size < 1 or step_size <= 0 or goal_tolerance <= 0:
            raise ValueError('iteration counts and lengths must be positive')
        if not 0 <= goal_sample_rate <= 1:
            raise ValueError('goal_sample_rate must be in [0, 1]')
        self.max_iterations, self.batch_size = max_iterations, batch_size
        self.step_size, self.goal_sample_rate = step_size, goal_sample_rate
        self.goal_tolerance, self.seed = goal_tolerance, seed
        self._path = None

    def solve(self, visualize=False):
        self.reset_diagnostics()
        self._path = None
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.backend_used = f'torch-{device}-batched-rrt-float64'
        self._path = None
        grid = self.map.grid
        ndim = grid.ndim
        shape = torch.tensor(grid.shape, device=device)
        occupancy = torch.as_tensor(np.ascontiguousarray(grid), device=device)
        nodes = torch.empty((self.max_iterations + 1, ndim), dtype=torch.float64, device=device)
        parents = torch.full((self.max_iterations + 1,), -1, dtype=torch.long, device=device)
        nodes[0] = torch.tensor(self.map.start, device=device)
        goal = torch.tensor(self.map.end, dtype=torch.float64, device=device)
        sizes = torch.tensor(grid.shape[::-1], dtype=torch.float64, device=device) - 1
        rng = torch.Generator(device=device)
        rng.manual_seed(self.seed) if self.seed >= 0 else rng.seed()
        count = 1

        def segments_free(a, b, max_length):
            n = max(1, math.ceil(max_length / 0.2))
            t = torch.linspace(0, 1, n + 1, dtype=torch.float64, device=device)
            points = a[:, None] + (b - a)[:, None] * t[None, :, None]
            idx = (points.flip(-1) + 0.5).floor().long()
            inside = ((idx >= 0) & (idx < shape)).all(dim=-1)
            safe = torch.minimum(torch.maximum(idx, torch.zeros_like(idx)), shape - 1)
            blocked = occupancy[tuple(safe[..., d] for d in range(ndim))] != 0
            return (inside & ~blocked).all(dim=1)

        endpoints = torch.stack((nodes[0], goal))
        if not bool(segments_free(endpoints, endpoints, 0).all()):
            return self.fail('invalid_endpoint', 'start or goal is blocked or outside the grid', stage='search')
        if tuple(self.map.start) == tuple(self.map.end):
            self._path = [tuple(self.map.start)]
            return self._path
        for offset in range(0, self.max_iterations, self.batch_size):
            batch = min(self.batch_size, self.max_iterations - offset)
            samples = torch.rand((batch, ndim), generator=rng, dtype=torch.float64, device=device) * sizes
            bias = torch.rand(batch, generator=rng, device=device) < self.goal_sample_rate
            samples = torch.where(bias[:, None], goal, samples)
            # Bound distance-matrix scratch to roughly 32 MiB.
            chunk = max(1, (2**22) // count)
            nearest_ids = torch.cat([
                torch.cdist(samples[i:i+chunk], nodes[:count]).argmin(dim=1)
                for i in range(0, batch, chunk)])
            nearest = nodes[nearest_ids]
            delta = samples - nearest
            distances = torch.linalg.vector_norm(delta, dim=1)
            proposals = nearest + delta * (self.step_size / distances.clamp_min(1e-12)).clamp_max(1)[:, None]
            accepted = segments_free(nearest, proposals, self.step_size) & (distances > 1e-9)
            ids = accepted.nonzero().flatten()
            n = len(ids)
            if n == 0:
                continue
            fresh = proposals[ids]
            nodes[count:count+n] = fresh
            parents[count:count+n] = nearest_ids[ids]
            close = (torch.linalg.vector_norm(fresh - goal, dim=1) <= self.goal_tolerance).nonzero().flatten()
            if len(close):
                connects = segments_free(fresh[close], goal.expand(len(close), -1), self.goal_tolerance)
                wins = connects.nonzero().flatten()
                if len(wins):
                    idx = count + int(close[wins[0]])
                    # Transfer the tree once, then backtrack on CPU.
                    points = nodes[:count+n].cpu().numpy()
                    links = parents[:count+n].cpu().numpy()
                    path = [tuple(float(c) for c in self.map.end)]
                    while idx >= 0:
                        path.append(tuple(float(c) for c in points[idx]))
                        idx = int(links[idx])
                    self._path = path[::-1]
                    return self._path
            count += n
        return self.fail('iteration_limit', 'batched RRT exhausted its sample budget before connecting to the goal', stage='search', samples=self.max_iterations, tree_nodes=count, batches=math.ceil(self.max_iterations/self.batch_size))

    def result(self, visualize=False):
        return self._path
