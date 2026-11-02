"""Correctness against independent Dijkstra, plus failure propagation contracts."""
import csv
import heapq
import io
import itertools
import math

import numpy as np
import pytest

from benchmark_app.registry import get_spec, _load_class
from benchmark_app.solver_worker import solve_with_timeout
from benchmark_app.storage import results_csv_text
from map_generation import Map, Map2D


def dijkstra(grid, start, end, corners):
    start, end = tuple(reversed(start)), tuple(reversed(end))
    if grid[start] or grid[end]:
        return math.inf
    distances = {start: 0.}
    queue = [(0., start)]
    while queue:
        distance, cell = heapq.heappop(queue)
        if distance != distances[cell]:
            continue
        if cell == end:
            return distance
        for step in itertools.product((-1, 0, 1), repeat=grid.ndim):
            if not any(step):
                continue
            other = tuple(c+d for c,d in zip(cell,step))
            if any(c < 0 or c >= n for c,n in zip(other,grid.shape)) or grid[other]:
                continue
            if not corners:
                bounds = [range(min(a,b),max(a,b)+1) for a,b in zip(cell,other)]
                if any(grid[p] for p in itertools.product(*bounds)):
                    continue
            cost = distance + math.sqrt(sum(d*d for d in step))
            if cost < distances.get(other, math.inf):
                distances[other] = cost
                heapq.heappush(queue, (cost, other))
    return math.inf


@pytest.mark.parametrize('native', [False, True])
@pytest.mark.parametrize('corners', [False, True])
@pytest.mark.parametrize('shape', [(7, 9), (5, 6, 4)])
def test_astar_matches_independent_dijkstra(native, corners, shape):
    if native:
        pytest.importorskip('Algorithms._astar_native')
    rng = np.random.default_rng(31)
    for _ in range(40):
        grid = (rng.random(shape) < 0.28).astype(np.uint8)
        start = tuple(int(rng.integers(n)) for n in shape[::-1])
        end = tuple(int(rng.integers(n)) for n in shape[::-1])
        expected = dijkstra(grid, start, end, corners)
        algo = get_spec('astar').build(Map(grid,start,end),
                                      {'use_native':native, 'allow_corner_cutting':corners})
        path = algo.solve()
        assert (path is None) == math.isinf(expected)
        if path is not None:
            assert tuple(path[0]) == start and tuple(path[-1]) == end
            cost = sum(math.dist(a,b) for a,b in zip(path,path[1:]))
            assert cost == pytest.approx(expected, abs=1e-11)
            assert algo.search_stats['path_cost'] == pytest.approx(expected)
            for a,b in zip(path,path[1:]):
                assert max(abs(x-y) for x,y in zip(a,b)) == 1
                assert grid[tuple(reversed(b))] == 0
                if not corners:
                    bounds = [range(min(x,y),max(x,y)+1) for x,y in zip(a,b)]
                    assert all(grid[tuple(reversed(p))] == 0 for p in itertools.product(*bounds))
        else:
            assert algo.failure_code in ('invalid_endpoint','unreachable')


@pytest.mark.parametrize('native', [False, True])
def test_astar_endpoints_timeout_and_reset(native):
    if native:
        pytest.importorskip('Algorithms._astar_native')
    m = Map2D(np.zeros((9,9),np.uint8),(1,1),(7,7))
    a = get_spec('astar').build(m, {'timeout':0, 'use_native':native})
    assert a.solve() is None and a.failure_code == 'search_timeout'
    a.timeout = 5
    assert a.solve() is not None and a.failure_code is None
    m.start = (-1,2)
    assert a.solve() is None and a.failure_code == 'invalid_endpoint'
    m.start = m.end
    assert len(a.solve()) == 1 and a.search_stats['path_cost'] == 0


@pytest.mark.parametrize('algo_id,ndim', [('fvb_laplace',2),('fvb_laplace_3d',3),('fvb_laplace_nd',4),('potential_field',2),('potential_field_nd',3)])
def test_field_failures_are_specific(algo_id, ndim):
    shape=(9,)*ndim
    a=get_spec(algo_id).build(Map(np.zeros(shape,np.uint8),(2,)*ndim,(6,)*ndim),{})
    args = [np.ones(shape), a.map.start, a.map.end]
    if algo_id.startswith('potential'):
        args.insert(1,a.map.grid)
    assert a._gradient_descent(*args) is None
    assert a.failure_code == 'zero_gradient'
    assert a.failure_details['steps'] == 0
    args[-2] = (0,)*ndim
    assert a._gradient_descent(*args) is None
    assert a.failure_code == 'out_of_bounds'


def test_fvb_epsilon_zero_observed_failure():
    grid=np.zeros((24,24),np.uint8)
    a=get_spec('fvb_laplace').build(Map2D(grid,(2,2),(20,20)),{'n_l':1,'epsilon':0.})
    assert a.solve() is None
    assert a.failure_code == 'zero_gradient'
    assert a.failure_details['position'] == [2.,2.]


@pytest.mark.parametrize('algo_id', ['rrt','rrt_batch_gpu'])
def test_rrt_budget_is_not_reported_as_proven_unreachable(algo_id):
    a=get_spec(algo_id).build(Map2D(np.zeros((32,32),np.uint8),(2,2),(28,28)),
                             {'max_iterations':1,'step_size':1.,'seed':42})
    assert a.solve() is None
    assert a.failure_code == 'iteration_limit'
    assert a.failure_details['samples'] == 1


def test_external_nonfinite_distinct_from_zero_gradient():
    common=_load_class('Algorithms/External Algorithms/eikonalgpu.py','_common')
    diag={}
    assert common.descend_field(np.full((9,9),np.nan),np.zeros((9,9)),(2,2),(6,6),diag=diag) is None
    assert diag['code'] == 'nonfinite_gradient'


def test_worker_and_csv_preserve_codes_and_search_statistics():
    grid=np.zeros((9,9),np.uint8)
    outcome=solve_with_timeout('astar',{},grid,(1,1),(7,7),timeout_s=20)
    assert outcome['solved'] and outcome['failure_code'] is None
    assert outcome['search_stats']['expanded'] > 0
    assert outcome['cpu_time_s'] >= 0
    grid[:,4]=1
    outcome=solve_with_timeout('astar',{},grid,(1,1),(7,7),timeout_s=20)
    assert not outcome['solved'] and outcome['failure_code']=='unreachable'
    exported=next(csv.DictReader(io.StringIO(results_csv_text([outcome],[]))))
    assert exported['failure_code']=='unreachable'
    assert 'expanded' in exported['search_stats']
    assert exported['backend'] in ('cpp-astar','python-astar')


@pytest.mark.parametrize('algo_id',['rrt_laplace_frontier','rrt_laplace_random'])
def test_incremental_failure_has_branch_breakdown(algo_id):
    a=get_spec(algo_id).build(Map2D(np.zeros((32,32),np.uint8),(2,2),(28,28)),
                             {'n_l':1,'n_w':0,'max_outer_iters':1,'seed':12})
    assert a.solve() is None
    assert a.failure_code=='iteration_limit'
    assert isinstance(a.failure_details['branch_failures'],dict)


def test_worker_preserves_fvb_failure_position():
    result=solve_with_timeout('fvb_laplace',{'n_l':1,'epsilon':0.},
                              np.zeros((24,24),np.uint8),(2,2),(20,20),timeout_s=20)
    assert result['failure_code']=='zero_gradient'
    assert result['failure_details']['position']==[2.,2.]
