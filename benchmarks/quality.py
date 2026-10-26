"""Shared geometric acceptance and sampling-independent benchmark scores.

Cells are closed unit squares centred at integer coordinates, matching the
external continuous planners. Touching an obstacle (including at a corner)
is a collision. Validation covers every segment, including the final goal
connection; dense sampling alone can miss short corner intersections.
"""
import itertools

import numpy as np


def validate_path(grid, path, start, goal):
    if path is None or len(path) == 0:
        return False, 'no path'
    p = np.asarray(path, dtype=float)
    if p.ndim != 2 or p.shape[1] != grid.ndim or not np.isfinite(p).all():
        return False, 'invalid or non-finite coordinates'
    if not np.allclose(p[0], start, atol=1e-6, rtol=0):
        return False, 'path does not start at start'
    if not np.allclose(p[-1], goal, atol=1e-6, rtol=0):
        return False, 'path does not end at goal'
    p = p[:, ::-1] + 0.5
    for a, b in zip(p, p[1:] if len(p) > 1 else p):
        delta = b - a
        ts = [0., 1.]
        for d in range(grid.ndim):
            if abs(delta[d]) > 1e-15:
                lo, hi = sorted((a[d], b[d]))
                edges = np.arange(np.floor(lo) + 1, np.ceil(hi))
                ts.extend(((edges - a[d]) / delta[d]).tolist())
        ts = np.unique(np.clip(ts, 0, 1))
        ts = np.concatenate([ts, (ts[:-1] + ts[1:]) / 2])
        points = a + ts[:, None] * delta
        base = np.floor(points).astype(int)
        # At boundaries check both adjacent cells, not just floor's choice.
        on_edge = np.isclose(points, np.rint(points), atol=1e-10, rtol=0)
        base = np.where(on_edge, np.rint(points).astype(int), base)
        for side in itertools.product((0, 1), repeat=grid.ndim):
            indices = base - on_edge * np.array(side)
            if (indices < 0).any() or (indices >= np.array(grid.shape)).any():
                return False, 'segment leaves map'
            if np.any(grid[tuple(indices.T)] != 0):
                return False, 'segment intersects an obstacle'
    return True, None


def resample_path(path, spacing=1.0):
    if spacing <= 0:
        raise ValueError('spacing must be positive')
    p = np.asarray(path, dtype=float)
    if len(p) < 2:
        return p
    lengths = np.linalg.norm(np.diff(p, axis=0), axis=1)
    p = p[np.r_[True, lengths > 1e-12]]
    if len(p) < 2:
        return p
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    samples = np.r_[np.arange(0, arc[-1], spacing), arc[-1]]
    return np.column_stack([np.interp(samples, arc, p[:, d]) for d in range(p.shape[1])])


def steering(path, theta=30., sweep=5):
    """Vectorised equivalent of the repository's excess-angle metric."""
    p = np.asarray(path, dtype=float)
    total = 0.
    for k in range(1, min(sweep, (len(p) - 1) // 2) + 1):
        left, right = p[k:-k] - p[:-2*k], p[2*k:] - p[k:-k]
        denom = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
        valid = denom > 0
        cos = np.sum(left[valid] * right[valid], axis=1) / denom[valid]
        angles = np.degrees(np.arccos(np.clip(cos, -1., 1.)))
        total += np.maximum(0., angles - theta).sum()
    return float(total)


def path_metrics(path, spacing=1.0):
    p = np.asarray(path, dtype=float)
    length = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
    uniform = resample_path(p, spacing)
    result = {'path_length': length, 'path_steps': len(p),
              'steering30_raw': steering(p)}
    for theta in (10, 20, 30, 45):
        result[f'steering{theta}'] = steering(uniform, theta)
    result['steering30_per_length'] = result['steering30'] / max(length, 1e-12)
    return result
