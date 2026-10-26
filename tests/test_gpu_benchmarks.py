import heapq
import math

import numpy as np
import pytest
import torch

from benchmark_app.registry import get_spec, _load_class
from benchmarks.quality import path_metrics, resample_path, steering, validate_path
from benchmarks.tune_gpu import aggregate, candidates, rank
from map_generation import Map2D, Map

torch.set_num_threads(1)


def map2d(size=24):
    grid = np.zeros((size, size), dtype=np.uint8)
    grid[[0, -1], :] = 1
    grid[:, [0, -1]] = 1
    grid[6:15, 10:12] = 1
    return Map2D(grid, (3, 3), (size-4, size-4))


def test_fvb_torch_matches_numpy(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    m = map2d()
    a = get_spec('fvb_laplace').build(m, {'n_l': 32, 'epsilon': 1e-8})
    expected = a._solve_phi(m.grid, m.start, m.end)
    actual = a._solve_phi_torch(m.grid, m.start, m.end)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('shape', [(17, 21), (9, 11, 13)])
def test_nf_exact_edt_matches_cpu(shape, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    grid = np.zeros(shape, dtype=np.uint8)
    grid[tuple(slice(3, 5) for _ in shape)] = 1
    m = Map(grid, (1,)*len(shape), tuple(s-2 for s in shape[::-1]))
    a = get_spec('navigation_function').build(m, {'k': 5})
    b = get_spec('navigation_function_gpu').build(m, {'k': 5})
    np.testing.assert_allclose(a._krf(grid, m.end), b._krf(grid, m.end), atol=1e-14, rtol=1e-13)


def test_potential_matches_numpy(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    m = map2d()
    a = get_spec('potential_field').build(m, {})
    np.testing.assert_allclose(a._compute_potential(m.grid, m.end),
                               a._compute_potential_torch(m.grid, m.end), rtol=1e-12)


def reference_eikonal(grid, end, obstacle_speed=1e-6):
    """Independent accepted-front heap solver for first-order 2D Eikonal."""
    h, w = grid.shape
    T = np.full(grid.shape, np.inf)
    accepted = np.zeros(grid.shape, dtype=bool)
    fixed = np.zeros(grid.shape, dtype=bool)
    queue = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            y, x = end[1]+dy, end[0]+dx
            if 0 <= y < h and 0 <= x < w and grid[y, x] == 0:
                T[y, x] = math.hypot(dx, dy)
                fixed[y, x] = True
                heapq.heappush(queue, (T[y, x], y, x))
    def value(y, x):
        return T[y, x] if 0 <= y < h and 0 <= x < w and accepted[y, x] else math.inf
    while queue:
        distance, y, x = heapq.heappop(queue)
        if accepted[y, x] or distance != T[y, x]:
            continue
        accepted[y, x] = True
        for yy, xx in ((y-1,x), (y+1,x), (y,x-1), (y,x+1)):
            if not (0 <= yy < h and 0 <= xx < w) or fixed[yy, xx] or accepted[yy, xx]:
                continue
            a, b = sorted((min(value(yy-1,xx), value(yy+1,xx)),
                           min(value(yy,xx-1), value(yy,xx+1))))
            s = 1. if grid[yy, xx] == 0 else 1 / obstacle_speed
            candidate = a+s if b-a >= s else a + (b-a + math.sqrt(2*s*s-(b-a)**2))/2
            if candidate < T[yy, xx]:
                T[yy, xx] = candidate
                heapq.heappush(queue, (candidate, yy, xx))
    return T


@pytest.mark.parametrize('single', [False, True])
def test_eikonal_matches_independent_heap(single, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    m = map2d(20)
    a = get_spec('eikonal_gpu').build(m, {'single_precision': single, 'tolerance': 0.})
    actual = a._arrival_times(m.grid, m.end)
    expected = reference_eikonal(m.grid, m.end)
    assert a.converged
    np.testing.assert_allclose(actual, expected, rtol=2e-6 if single else 1e-12,
                               atol=1e-4 if single else 1e-10)


def test_eikonal_exhaustion_is_failure():
    a = get_spec('eikonal_gpu').build(map2d(), {'max_sweeps': 1})
    assert a.solve() is None
    assert not a.converged
    assert 'did not converge' in a.failure_reason


def test_harmonic_chunking_and_cpu_eval():
    cls = _load_class('Algorithms/External Algorithms/harmonicmap2d.py', 'PanelHarmonicMapTorch')
    # Outer CCW, inner CW, each side divided into panels.
    common = _load_class('Algorithms/External Algorithms/harmonicmap2d.py', '_common')
    outer = np.array([[0., 0.], [12, 0], [12, 12], [0, 12], [0, 0]])
    hole = np.array([[5., 5.], [5, 7], [7, 7], [7, 5], [5, 5]])
    loops = [common.resample_polygon(p, 1.) for p in (outer, hole)]
    a = cls(loops, 'cpu', chunk_size=7, cpu_evaluation=True)
    b = cls(loops, 'cpu', chunk_size=1000, cpu_evaluation=False)
    np.testing.assert_allclose(a._puncts, b._puncts, atol=1e-11)
    for x, y in [(2.2, 3.1), (9., 8.5), (3., 9.)]:
        np.testing.assert_allclose(a.map(x,y), b.map(x,y), atol=1e-11)
        np.testing.assert_allclose(a.jacob(x,y), b.jacob(x,y), atol=1e-11)


@pytest.mark.parametrize('batch', [1, 32, 128])
def test_batched_rrt_valid_and_seeded(batch):
    m = Map2D(np.zeros((24, 24), dtype=np.uint8), (2, 2), (20, 20))
    params = {'seed': 123, 'batch_size': batch, 'step_size': 3, 'max_iterations': 5000}
    a = get_spec('rrt_batch_gpu').build(m, params).solve()
    b = get_spec('rrt_batch_gpu').build(m, params).solve()
    assert a == b
    assert validate_path(m.grid, a, m.start, m.end)[0]


@pytest.mark.parametrize('path', [[(1,1), (7,7)], [(1,1), (4,3.5), (7,7)], [(1,1), (math.nan,2), (7,7)]])
def test_validator_rejects_collision_or_nan(path):
    grid = np.zeros((9,9), dtype=np.uint8)
    grid[4,4] = 1
    assert not validate_path(grid, path, (1,1), (7,7))[0]


def test_validator_catches_corner_and_final_jump():
    grid = np.zeros((5,5), dtype=np.uint8)
    grid[1,2] = 1
    assert not validate_path(grid, [(1,1),(2,2)], (1,1), (2,2))[0]
    assert not validate_path(grid, [(0,1),(1,1),(3,1)], (0,1), (3,1))[0]
    assert validate_path(grid, [(0,2),(1,2),(2,2),(3,2)], (0,2), (3,2))[0]


def test_astar_does_not_cut_corners():
    grid = np.array([[0,1], [1,0]], dtype=np.uint8)
    m = Map2D(grid, (0,0), (1,1))
    assert get_spec('astar').build(m, {}).solve() is None
    assert get_spec('astar').build(m, {'allow_corner_cutting': True}).solve() is not None


def test_uniform_metric_independent_of_waypoint_density():
    path = [(0,0), (10,0), (10,10)]
    dense = resample_path(path, 0.25)
    a, b = path_metrics(path), path_metrics(dense)
    assert a['steering30'] == pytest.approx(b['steering30'])
    assert a['path_length'] == pytest.approx(b['path_length'])


def test_vectorized_metric_matches_repository():
    from smoothness_metrics.steering_penalty import SteeringPenaltySmoothness
    p = np.random.default_rng(17).normal(size=(40,2)).cumsum(axis=0)
    assert steering(p) == pytest.approx(SteeringPenaltySmoothness().compute(p))


def test_sampling_is_reproducible_and_strict():
    d = {'id':'fvb_laplace', 'base': {'use_gpu':True}, 'grid': {'n_l':[16,32,64,100]}}
    assert candidates('fvb_cuda', d, 100, 3, 17) == candidates('fvb_cuda', d, 100, 3, 17)
    assert len(candidates('fvb_cuda', d, 100, 100, 17)) == 4
    with pytest.raises(ValueError, match='unknown'):
        candidates('fvb_cuda', {**d, 'base': {'typo':True}}, 100, 3, 17)


def test_failures_never_win_ranking():
    failed = aggregate([{'valid':False, 'wall_s':1, 'error':'timeout'}])
    good = {'valid':1, 'success_rate':0.5, 'median_time_s':1000.,
            'mean_steering30_per_length':1000., 'mean_path_length':1000.}
    assert rank(good) < rank(failed)


def test_watchdog_terminates_unreachable_solver():
    from benchmark_app.solver_worker import solve_with_timeout
    grid = np.ones((9,9), dtype=np.uint8)
    grid[2,2] = grid[6,6] = 0
    result = solve_with_timeout('fvb_laplace', {'n_l':16, 'epsilon':1e-8},
                                grid, (2,2), (6,6), timeout_s=0.2)
    assert not result['solved']
    assert 'timed out' in result['error']


def test_watchdog_records_device_and_setup():
    from benchmark_app.solver_worker import solve_with_timeout
    m = Map2D(np.zeros((9,9), dtype=np.uint8), (1,1), (7,7))
    result = solve_with_timeout('astar', {}, m.grid, m.start, m.end, timeout_s=20)
    assert result['solved']
    assert result['device'] == 'cpu'
    assert result['setup_s'] >= 0 and result['time_s'] >= 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA hardware required')
@pytest.mark.parametrize('single', [False, True])
def test_cuda_eikonal_fused_parity(single):
    pytest.importorskip('cupy')
    m = map2d(20)
    a = get_spec('eikonal_gpu').build(m, {'single_precision':single, 'tolerance':0.})
    actual = a._arrival_times(m.grid, m.end)
    assert a.converged and a.backend_used.startswith('cupy-fused')
    np.testing.assert_allclose(actual, reference_eikonal(m.grid, m.end),
                               rtol=2e-6 if single else 1e-12, atol=1e-4 if single else 1e-10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA hardware required')
def test_cuda_fvb_fused_parity():
    pytest.importorskip('cupy')
    m = map2d()
    a = get_spec('fvb_laplace').build(m, {'n_l':32, 'epsilon':1e-8})
    expected = a._solve_phi(m.grid, m.start, m.end)
    actual = a._solve_phi_torch(m.grid, m.start, m.end)
    assert a.backend_used.startswith('cupy-fused')
    np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-11)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA hardware required')
def test_cuda_edt_and_nf_parity():
    pytest.importorskip('cupy')
    from Algorithms.gpu_ops import distance_edt
    from scipy.ndimage import distance_transform_edt
    m = map2d()
    actual, backend = distance_edt(m.grid == 0, 'cuda')
    assert backend == 'cupy-exact-edt'
    np.testing.assert_allclose(actual.cpu().numpy(), distance_transform_edt(m.grid == 0), atol=1e-12)
    a = get_spec('navigation_function').build(m, {})
    b = get_spec('navigation_function_gpu').build(m, {})
    np.testing.assert_allclose(a._krf(m.grid, m.end), b._krf(m.grid, m.end), atol=1e-12)
