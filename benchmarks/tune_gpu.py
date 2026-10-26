"""Resumable, serial, held-out best-effort comparison. See GPU_BENCHMARKS.md."""
import argparse
import csv
import hashlib
import itertools
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import sqlite3
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:20]


def atomic_json(path, obj):
    temp = Path(str(path) + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temp.replace(path)


def source_hash():
    files = subprocess.check_output(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=ROOT).split(b'\0')
    h = hashlib.sha256()
    for name in sorted(set(files)):
        if not name:
            continue
        path = ROOT / os.fsdecode(name)
        if path.is_file() and path.suffix in ('.py', '.json', '.cpp', '.toml'):
            h.update(name)
            h.update(path.read_bytes())
    return h.hexdigest()


def environment():
    import torch
    info = {'python': sys.version, 'platform': platform.platform(), 'cpu': platform.processor(),
            'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
            'cuda_available': torch.cuda.is_available(), 'source_sha256': source_hash(),
            'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'threads': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}
    info['packages'] = {name: importlib.metadata.version(name) for name in ('numpy', 'scipy')}
    info['cpu_count'] = os.cpu_count()
    if info['cuda_available']:
        p = torch.cuda.get_device_properties(0)
        info.update(gpu=p.name, gpu_memory=p.total_memory, compute_capability=[p.major, p.minor])
        try:
            info['gpu_driver'] = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                text=True, timeout=5).strip()
        except (OSError, subprocess.SubprocessError):
            info['gpu_driver'] = None
    try:
        import cupy
        info['cupy'] = cupy.__version__
    except ImportError:
        info['cupy'] = None
    return info


def candidates(family, definition, size, count, seed):
    from benchmark_app.registry import get_spec
    spec = get_spec(definition['id'])
    declared = {p.name for p in spec.params}
    unknown = (set(definition.get('base', {})) | set(definition.get('grid', {}))) - declared
    if unknown:
        raise ValueError(f'{family}: unknown parameters {sorted(unknown)}')
    base = spec.coerce_params(definition.get('base', {}))
    grid = {}
    for key, values in definition.get('grid', {}).items():
        if isinstance(values, dict):
            values = [v * (size / 100.)**values['power'] for v in values['at_size_100']]
        if not values:
            raise ValueError(f'{family}: empty grid for {key}')
        grid[key] = values
    combos = [dict(zip(grid, v)) for v in itertools.product(*grid.values())]
    # Related CPU/CUDA implementations receive the same sampled knob settings.
    stream = family.removesuffix('_cuda').removesuffix('_cpu')
    random.Random(int(digest([stream, size, seed]), 16)).shuffle(combos)
    seen, out = set(), []
    # Include published/default parameters as a candidate, even outside the grid.
    for knobs in [base, *[{**base, **v} for v in combos]]:
        params = spec.coerce_params(knobs)
        key = digest([spec.id, params])
        if key not in seen:
            seen.add(key)
            out.append({'key': key, 'family': family, 'id': spec.id, 'params': params})
        if len(out) >= count:
            break
    return out


def aggregate(rows):
    good = [r for r in rows if r['valid']]
    def mean(key):
        return statistics.mean(r[key] for r in good) if good else None
    times = [r['time_s'] for r in good]
    return {'attempts': len(rows), 'valid': len(good),
            'maps': len({(r.get('map_set_id'), r.get('map')) for r in rows}),
            'success_rate': len(good) / len(rows) if rows else 0.,
            'median_time_s': statistics.median(times) if times else None,
            'mean_time_s': mean('time_s'), 'mean_path_length': mean('path_length'),
            'mean_steering30_per_length': mean('steering30_per_length'),
            'mean_steering30': mean('steering30'),
            'mean_all_attempt_wall_s': statistics.mean(r['wall_s'] for r in rows) if rows else None,
            'timeouts': sum('timed out' in (r.get('error') or '') for r in rows),
            'unsupported': sum((r.get('error') or '').startswith('unsupported map') for r in rows),
            'invalid_paths': sum(r.get('returned_path', False) and not r['valid'] for r in rows)}


def rank(summary, profile='balanced'):
    if summary['valid'] == 0:
        return (0, math.inf)
    speed = summary['median_time_s']
    smooth = summary['mean_steering30_per_length']
    length = summary['mean_path_length']
    if profile == 'fast':
        return (-summary['success_rate'], speed, smooth, length)
    if profile == 'smooth':
        return (-summary['success_rate'], smooth, length, speed)
    # Explicit heuristic used only to prune TRAINING trials, not to rank test results.
    score = math.log1p(speed) + math.log1p(smooth)
    return (-summary['success_rate'], score, length)


class BudgetExpired(Exception):
    pass


class Sweep:
    def __init__(self, out, cfg, env, deadline, allow_cpu):
        self.out, self.cfg, self.env = out, cfg, env
        self.deadline, self.allow_cpu = deadline, allow_cpu
        self.db = sqlite3.connect(out / 'trials.sqlite3')
        # A single writer holds the suite lock. Use a single-file checkpoint
        # rather than depending on sidecar WAL files surviving a host restart.
        self.db.execute('PRAGMA journal_mode=DELETE')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS trials (key TEXT PRIMARY KEY, record TEXT NOT NULL)')
        self.db.commit()
        self.plan_id = digest({'config': cfg, 'environment': env, 'allow_cpu': allow_cpu})
        self.logfile = out / 'sweep.log'

    def log(self, message):
        line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {message}'
        print(line, flush=True)
        with self.logfile.open('a') as f:
            f.write(line + '\n')

    def check_budget(self):
        if time.monotonic() >= self.deadline:
            raise BudgetExpired()

    def rows(self):
        return [json.loads(r[0]) for r in self.db.execute('SELECT record FROM trials ORDER BY key')]

    def maps(self, size, phase, count):
        """Independent deterministic map seeds; boundary & connectivity shared by all."""
        import numpy as np
        from scipy.ndimage import label
        from benchmark_app import mapgen, storage
        path = self.out / f'maps-{size}-{phase}.json'
        if path.exists():
            manifest = json.loads(path.read_text())
            for entry in manifest['maps']:
                grid = storage.load_map_grid(manifest['id'], entry['name'])
                if hashlib.sha256(grid.tobytes()).hexdigest() != entry['sha256']:
                    raise RuntimeError('map content changed; use a new suite')
            return manifest
        seed = self.cfg['seed'] + size * 1009 + (0 if phase == 'train' else 1_000_000_007)
        rng = random.Random(seed)
        ms_id = storage.new_id('ms')
        folder = Path(storage.map_set_dir(ms_id))
        folder.mkdir(parents=True)
        manifest = {'id': ms_id, 'name': f'{self.out.name}-{size}-{phase}',
                    'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                    'status': 'generating', 'dims': 2, 'shape': [size, size],
                    'width': size, 'height': size, 'num_maps': count, 'maps': [],
                    'gen_params': {'seed': seed, 'obstacles': self.cfg['obstacles'],
                                   'closed_boundary': True, 'face_connected': True}}
        storage.save_map_set_manifest(manifest)
        gen = mapgen.InteriorMapGenerator2D(size, size, tuple(self.cfg['obstacles']))
        self.log(f'generate {phase}: {count} maps at {size}, seed={seed}')
        for i in range(count):
            self.check_budget()
            # Reject unsolvable/trivial maps before any planner is evaluated.
            for attempt in range(1000):
                grid = np.zeros((size, size), dtype=np.uint8)
                random.seed(rng.randrange(2**32))
                for _ in range(random.randint(*self.cfg['obstacles'])):
                    gen._place_obstacle(grid)
                grid[[0, -1], :] = 1
                grid[:, [0, -1]] = 1
                start, end = gen._find_free_point(grid), gen._find_free_point(grid)
                labels, _ = label(grid == 0)
                if (np.linalg.norm(np.array(start) - end) >= max(2, size / 5)
                        and labels[start[1], start[0]] == labels[end[1], end[0]]):
                    break
            else:
                raise RuntimeError('map generation failed after 1000 attempts')
            stem = f'map_{i:03d}'
            np.save(folder / f'{stem}.npy', grid)
            mapgen._save_thumb(str(folder), stem, grid, start, end)
            manifest['maps'].append({'name': f'{stem}.npy', 'stem': stem,
                                     'start': list(start), 'end': list(end),
                                     'sha256': hashlib.sha256(grid.tobytes()).hexdigest()})
        manifest['status'] = 'ready'
        storage.save_map_set_manifest(manifest)
        atomic_json(path, manifest)
        return manifest

    def run_cell(self, candidate, size, phase, manifest, entry, repeat=0):
        from benchmark_app import storage
        from benchmark_app.registry import get_spec
        from benchmark_app.solver_worker import solve_with_timeout
        from benchmarks.quality import validate_path, path_metrics
        key = digest([self.plan_id, candidate['key'], size, phase, entry['sha256'],
                      entry['start'], entry['end'], repeat])
        cached = self.db.execute('SELECT record FROM trials WHERE key=?', (key,)).fetchone()
        if cached:
            return json.loads(cached[0])
        self.check_budget()
        params = dict(candidate['params'])
        if 'seed' in params:
            # Common random seed per map/repeat for every candidate of a method.
            params['seed'] = (int(digest([self.cfg['seed'], size, phase, entry['stem'], repeat]), 16) % (2**31))
        grid = storage.load_map_grid(manifest['id'], entry['name'])
        timeout = self.cfg['timeouts_s'].get(str(size), 120)
        remaining = self.deadline - time.monotonic()
        if remaining < timeout:
            # Do not count a budget-truncated trial as an algorithm timeout.
            raise BudgetExpired()
        self.log(f'{phase} {size} {candidate["family"]} {candidate["key"][:6]} {entry["stem"]} rep={repeat}')
        t0 = time.monotonic()
        outcome = solve_with_timeout(candidate['id'], params, grid,
                                     entry['start'], entry['end'], timeout_s=timeout)
        wall = time.monotonic() - t0
        spec = get_spec(candidate['id'])
        wants_cuda = spec.gpu or params.get('use_gpu', False)
        if wants_cuda and outcome.get('device') == 'cpu' and not self.allow_cpu:
            raise RuntimeError('GPU trial fell back to CPU; refusing to record it as CUDA')
        path = outcome.pop('path', None)
        valid, reason = validate_path(grid, path, entry['start'], entry['end'])
        record = {'trial_key': key, **candidate, 'effective_params': params,
                  'size': size, 'phase': phase, 'map_set_id': manifest['id'],
                  'map': entry['stem'], 'map_sha256': entry['sha256'], 'repeat': repeat,
                  'wall_s': wall, **outcome, 'returned_path': path is not None,
                  'valid': valid, 'validation_error': reason,
                  'timeout_s': timeout, 'path': path}
        if valid:
            record.update(path_metrics(path, self.cfg['spacing']))
        self.db.execute('INSERT INTO trials VALUES (?, ?)',
                        (key, json.dumps(storage.clean_json(record), allow_nan=False)))
        self.db.commit()
        return record

    def export(self):
        rows = self.rows()
        scalars = [{k: (json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v)
                    for k, v in r.items() if k != 'path'} for r in rows]
        if scalars:
            fields = sorted(set().union(*(r.keys() for r in scalars)))
            tmp = self.out / 'results.csv.tmp'
            with tmp.open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(scalars)
            tmp.replace(self.out / 'results.csv')
        groups = {}
        for row in rows:
            groups.setdefault((row['phase'], row['size'], row['family'], row['key']), []).append(row)
        summaries = [{'phase': phase, 'size': size, 'family': family, 'key': key,
                      'algorithm': rs[0]['id'], 'params': rs[0]['params'], **aggregate(rs)}
                     for (phase, size, family, key), rs in sorted(groups.items())]
        atomic_json(self.out / 'summary.json', summaries)
        if summaries:
            with (self.out / 'summary.csv').open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(summaries[0]))
                w.writeheader()
                w.writerows([{**s, 'params': json.dumps(s['params'], sort_keys=True)} for s in summaries])
        # Compare on identical successful map/repeat cases as well as marginal means.
        test = {key: rs for key, rs in groups.items() if key[0] == 'test'}
        paired = []
        for (ka, a), (kb, b) in itertools.combinations(test.items(), 2):
            if ka[1] != kb[1]:
                continue
            b_good = {(r['map_set_id'], r['map'], r['repeat']): r for r in b if r['valid']}
            pairs = [(r, b_good[(r['map_set_id'], r['map'], r['repeat'])]) for r in a
                     if r['valid'] and (r['map_set_id'], r['map'], r['repeat']) in b_good]
            paired.append({'size': ka[1], 'a': ka[2], 'a_key': ka[3], 'b': kb[2], 'b_key': kb[3],
                           'common_valid_attempts': len(pairs),
                           'median_a_over_b_time': statistics.median(x['time_s'] / max(y['time_s'], 1e-12) for x, y in pairs) if pairs else None,
                           'mean_a_minus_b_steering30': statistics.mean(x['steering30'] - y['steering30'] for x, y in pairs) if pairs else None})
        atomic_json(self.out / 'paired.json', paired)

    def execute(self):
        for size in self.cfg['sizes']:
            train = self.maps(size, 'train', self.cfg['train_maps'])
            frozen = self.out / f'selected-{size}.json'
            if frozen.exists():
                winners = json.loads(frozen.read_text())
            else:
                pools = {f: candidates(f, d, size, self.cfg['trials_per_family'], self.cfg['seed'])
                         for f, d in self.cfg['families'].items()}
                # A completed round has identical maps and budgets for all its candidates.
                # Interleave families to avoid giving one family the whole night's budget.
                for round_i, n_maps in enumerate(self.cfg['rounds']):
                    for map_i in range(n_maps):
                        tasks = [c for pool in pools.values() for c in pool]
                        random.Random(self.cfg['seed'] + size + round_i * 100 + map_i).shuffle(tasks)
                        for candidate in tasks:
                            self.run_cell(candidate, size, 'train', train, train['maps'][map_i])
                    self.export()
                    all_rows = self.rows()
                    for family, pool in pools.items():
                        summaries = {c['key']: aggregate([r for r in all_rows
                                     if r['size'] == size and r['phase'] == 'train' and r['key'] == c['key']])
                                     for c in pool}
                        pool.sort(key=lambda c: rank(summaries[c['key']]))
                        if round_i + 1 < len(self.cfg['rounds']):
                            pools[family] = pool[:max(2, math.ceil(len(pool) / 2))]
                    self.log(f'completed training round {round_i+1}, {size}px, {n_maps} maps per candidate')
                winners = []
                for family, pool in pools.items():
                    summaries = {c['key']: aggregate([r for r in all_rows
                                 if r['size'] == size and r['phase'] == 'train' and r['key'] == c['key']]) for c in pool}
                    chosen = {}
                    for profile in ('fast', 'smooth'):
                        c = min(pool, key=lambda c: rank(summaries[c['key']], profile))
                        chosen.setdefault(c['key'], {**c, 'profiles': [], 'training': summaries[c['key']]})['profiles'].append(profile)
                    winners.extend(chosen.values())
                # Freeze before even generating test maps; test results never choose knobs.
                atomic_json(frozen, winners)
            test = self.maps(size, 'test', self.cfg['test_maps'])
            for i, entry in enumerate(test['maps']):
                for repeat in range(self.cfg['test_repeats']):
                    order = list(winners)
                    random.Random(self.cfg['seed'] + size + i*17 + repeat).shuffle(order)
                    for candidate in order:
                        self.run_cell(candidate, size, 'test', test, entry, repeat)
                self.export()
            self.log(f'completed held-out evaluation at {size}px')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, default=ROOT / 'benchmarks/best_effort_config.json')
    ap.add_argument('--suite', default='gpu-best-effort')
    ap.add_argument('--hours', type=float, default=8., help='wall budget per invocation; rerun to resume')
    ap.add_argument('--sizes', type=int, nargs='+')
    ap.add_argument('--families', nargs='+')
    ap.add_argument('--train-maps', type=int)
    ap.add_argument('--test-maps', type=int)
    ap.add_argument('--trials', type=int)
    ap.add_argument('--repeats', type=int)
    ap.add_argument('--cpu-threads', type=int, default=1)
    ap.add_argument('--allow-cpu', action='store_true', help='explicit CPU correctness smoke test, NOT GPU evidence')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()
    if Path(args.suite).name != args.suite or args.suite in ('.', '..'):
        ap.error('--suite must be a simple directory name')
    for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[variable] = str(args.cpu_threads)
    cfg = json.loads(args.config.read_text())
    for arg, key in [('sizes', 'sizes'), ('train_maps', 'train_maps'), ('test_maps', 'test_maps'),
                     ('trials', 'trials_per_family'), ('repeats', 'test_repeats')]:
        if getattr(args, arg) is not None:
            cfg[key] = getattr(args, arg)
    if args.families:
        cfg['families'] = {f: cfg['families'][f] for f in args.families}
    if (args.hours <= 0 or args.cpu_threads < 1 or not cfg['families']
            or min(cfg['sizes']) < 8 or any(cfg[k] < 1 for k in ('train_maps', 'test_maps', 'trials_per_family', 'test_repeats'))):
        ap.error('positive budgets/counts and map sizes >= 8 required')
    cfg['rounds'] = sorted(set([min(n, cfg['train_maps']) for n in cfg['rounds']] + [cfg['train_maps']]))
    env = environment()
    if args.check:
        print(json.dumps(env, indent=2))
        return
    plan = {'config': cfg, 'environment': env, 'allow_cpu': args.allow_cpu}
    for size in cfg['sizes']:
        for family, definition in cfg['families'].items():
            candidates(family, definition, size, cfg['trials_per_family'], cfg['seed'])
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        print('No solves started. All families use equal trial/map caps; timings are serial.')
        return
    if not env['cuda_available'] and not args.allow_cpu:
        ap.error('no CUDA device; install CUDA PyTorch or use --allow-cpu ONLY for a smoke test')
    out = ROOT / 'benchmarks/results' / args.suite
    out.mkdir(parents=True, exist_ok=True)
    # OS-owned advisory lock is released on crash; no stale PID lock to remove.
    import fcntl
    lock = (out / 'run.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        ap.error('another process is already running this suite')
    plan_path = out / 'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        ap.error('config, source or hardware changed; choose a new --suite name')
    atomic_json(plan_path, plan)
    sweep = Sweep(out, cfg, env, time.monotonic() + args.hours * 3600, args.allow_cpu)
    status = 'running'
    atomic_json(out / 'status.json', {'status': status, 'started': time.time()})
    try:
        sweep.execute()
        status = 'complete'
    except (BudgetExpired, KeyboardInterrupt) as exc:
        status = 'budget_exhausted' if isinstance(exc, BudgetExpired) else 'interrupted'
        sweep.log(f'{status}: saved completed trials; rerun the identical command to resume')
    except Exception:
        status = 'error'
        raise
    finally:
        sweep.export()
        atomic_json(out / 'status.json', {'status': status, 'finished': time.time(), 'trials': len(sweep.rows())})
        sweep.db.close()
        lock.close()
    sweep.log(f'{status}; results: {out}')


if __name__ == '__main__':
    main()
