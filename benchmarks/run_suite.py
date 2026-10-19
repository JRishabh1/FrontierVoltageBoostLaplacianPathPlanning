"""
Headless benchmark suite: every algorithm × map sizes × device conditions.

    python benchmarks/run_suite.py --check                 # environment report
    python benchmarks/run_suite.py --dry-run               # what would run
    python benchmarks/run_suite.py                         # full suite (resumable)
    python benchmarks/run_suite.py --sizes 100 --maps 20   # quick trial

Conditions
    cpu-cpu   CPU implementations (suite_config.json "cpu" entries)
    gpu-gpu   GPU implementations ("gpu" entries) with CUDA visible
    gpu-cpu   the same GPU implementations with CUDA hidden
              (CUDA_VISIBLE_DEVICES=""), so PyTorch runs them on the CPU

Everything goes through the benchmark app's own job runner and storage, so
each batch is a normal app run under ``Benchmark Data/`` (open the web app
later to browse/compare), and it needs no browser — run it inside tmux.
Work is split into batches of maps; finished batches are skipped on restart,
so the suite can be stopped and resumed. Each solve keeps the app's
watchdog subprocess and per-size timeout.

Combined results are written to ``benchmarks/results/<suite>/``:
    results.csv   one row per (size, condition, algorithm config, map)
    summary.csv   per (size, condition, algorithm config): solved, times, metrics
    runs.csv      which app run holds which batch
    suite.log     progress log
"""

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, ROOT)

CONDITIONS = ('cpu-cpu', 'gpu-gpu', 'gpu-cpu')
_ORIGINAL_CUDA_ENV = os.environ.get('CUDA_VISIBLE_DEVICES')


def log(msg, logfile=None):
    line = f'[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}'
    print(line, flush=True)
    if logfile:
        with open(logfile, 'a') as f:
            f.write(line + '\n')


def cuda_status():
    """(available, description) checked in a child process so this process
    never initialises CUDA itself."""
    code = ('import torch; c = torch.cuda.is_available(); '
            'print(c, torch.cuda.get_device_name(0) if c else "-", torch.__version__)')
    try:
        out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                             timeout=120).stdout.split()
        return out[0] == 'True', f'torch {out[-1]}, GPU: {" ".join(out[1:-1])}'
    except Exception as exc:   # torch missing or broken
        return False, f'PyTorch unavailable ({type(exc).__name__})'


def set_condition(condition):
    if condition == 'gpu-cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    elif _ORIGINAL_CUDA_ENV is None:
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = _ORIGINAL_CUDA_ENV


def ensure_map_set(name, size, n_maps, obstacles, seed, logfile):
    from benchmark_app import mapgen, storage
    for ms in storage.list_map_sets():
        if ms.get('name') == name:
            if ms.get('status') == 'ready' and ms.get('num_maps') == n_maps:
                return storage.get_map_set(ms['id'])
            log(f'removing incomplete map set {ms["id"]} ({name})', logfile)
            storage.delete_map_set(ms['id'])
    log(f'generating {n_maps} maps of {size}x{size} ({name}); each is A*-validated', logfile)
    t0 = time.time()

    def progress(i, total, msg):
        if i and i % 25 == 0:
            log(f'  {name}: {i}/{total} maps ({time.time() - t0:.0f}s)', logfile)

    return mapgen.generate_map_set(name=name, dims=2, width=size, height=size, num_maps=n_maps,
                                   obstacles_min=obstacles[0], obstacles_max=obstacles[1],
                                   seed=seed, progress_cb=progress)


def wait_for(job_id, logfile, label):
    from benchmark_app import runner
    last = None
    while True:
        job = runner.get_job(job_id)
        status = job['status']
        msg = job['progress'].get('message')
        if msg != last and msg:
            prog = job['progress']
            log(f'  {label}: {prog.get("current", 0)}/{prog.get("total", 0)} {msg}', logfile)
            last = msg
        if status in ('done', 'error', 'cancelled'):
            return job
        time.sleep(5)


def export_run(run_id, size, condition, batch, results_csv):
    from benchmark_app import storage
    from smoothness_metrics import SmoothnessMetrics
    manifest = storage.get_run(run_id)
    labels = {a['key']: (a.get('graph_label') or a['label']) for a in manifest['algorithms']}
    params = {a['key']: a.get('params') for a in manifest['algorithms']}
    keys = list(SmoothnessMetrics.METRIC_KEYS)
    fields = ['size', 'condition', 'batch', 'run_id', 'config', 'algorithm_id', 'params', 'map',
              'solved', 'time_s', 'path_length', 'path_steps', 'error'] + keys
    new = not os.path.isfile(results_csv)
    with open(results_csv, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        for r in storage.get_run_results(run_id):
            k = r.get('algorithm_key')
            err = (r.get('error') or '').strip().splitlines()
            row = {'size': size, 'condition': condition, 'batch': batch, 'run_id': run_id,
                   'config': labels.get(k, k), 'algorithm_id': r.get('algorithm_id'),
                   'params': json.dumps(params.get(k), sort_keys=True), 'map': r.get('map_stem'),
                   'solved': int(bool(r.get('solved'))), 'time_s': r.get('time_s'),
                   'path_length': r.get('path_length_euclidean'), 'path_steps': r.get('path_steps'),
                   'error': err[0][:200] if err else ''}
            for m in keys:
                row[m] = (r.get('metrics') or {}).get(m)
            w.writerow(row)


def write_summary(results_csv, summary_csv):
    if not os.path.isfile(results_csv):
        return
    groups = {}
    with open(results_csv, newline='') as f:
        for r in csv.DictReader(f):
            groups.setdefault((int(r['size']), r['condition'], r['config']), []).append(r)

    def num(rows, key):
        return [float(r[key]) for r in rows if r['solved'] == '1' and r[key] not in ('', 'None')]

    with open(summary_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['size', 'condition', 'config', 'maps', 'solved', 'solved_pct',
                    'mean_time_s', 'median_time_s', 'mean_path_length',
                    'mean_steering_penalty', 'median_steering_penalty', 'timeouts', 'unsupported'])
        for (size, cond, config), rows in sorted(groups.items()):
            t, pl, sp = num(rows, 'time_s'), num(rows, 'path_length'), num(rows, 'steering_penalty')
            solved = sum(r['solved'] == '1' for r in rows)
            fmt = lambda v: f'{v:.6g}' if v is not None else ''
            w.writerow([size, cond, config, len(rows), solved, f'{100 * solved / len(rows):.1f}',
                        fmt(statistics.mean(t) if t else None), fmt(statistics.median(t) if t else None),
                        fmt(statistics.mean(pl) if pl else None), fmt(statistics.mean(sp) if sp else None),
                        fmt(statistics.median(sp) if sp else None),
                        sum('timed out' in r['error'] for r in rows),
                        sum(r['error'].startswith('unsupported map') for r in rows)])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--config', default=os.path.join(ROOT, 'benchmarks', 'suite_config.json'))
    ap.add_argument('--suite', default='suite', help='name for map sets, runs and the results folder')
    ap.add_argument('--sizes', type=int, nargs='+', default=[100, 512, 1000])
    ap.add_argument('--maps', type=int, default=1000, help='maps per size')
    ap.add_argument('--batch', type=int, default=100, help='maps per app run (resume granularity)')
    ap.add_argument('--conditions', nargs='+', default=list(CONDITIONS), choices=CONDITIONS)
    ap.add_argument('--workers', type=int, default=1,
                    help='parallel solves per run; >1 is faster but timings then share the machine '
                         '(GPU conditions always use 1)')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    out_dir = os.path.join(ROOT, 'benchmarks', 'results', args.suite)
    os.makedirs(out_dir, exist_ok=True)
    logfile = os.path.join(out_dir, 'suite.log')
    results_csv = os.path.join(out_dir, 'results.csv')
    runs_csv = os.path.join(out_dir, 'runs.csv')

    cuda_ok, cuda_desc = cuda_status()
    sys.path.insert(0, os.path.join(ROOT, 'third_party'))
    import setup_external
    if args.check:
        print(f'python {sys.version.split()[0]} ({sys.executable}); {os.cpu_count()} CPU cores')
        print(cuda_desc, '| CUDA available:', cuda_ok)
        for comp, (ok, detail) in setup_external.check().items():
            print(f'external {comp:12s} {"OK" if ok else "--"} {detail}')
        print('(missing external planners are built automatically before the first run that needs them)')
        return

    from benchmark_app.registry import get_spec
    plan = []
    for cond in args.conditions:
        entries = cfg['algorithms']['cpu' if cond == 'cpu-cpu' else 'gpu']
        for e in entries:
            get_spec(e['id'])   # fail early on a typo
        plan.append((cond, entries))
    if 'gpu-gpu' in args.conditions and not cuda_ok:
        log(f'WARNING: no CUDA GPU visible ({cuda_desc}) — skipping gpu-gpu', logfile)
        plan = [p for p in plan if p[0] != 'gpu-gpu']

    batches = [(i, min(i + args.batch, args.maps)) for i in range(0, args.maps, args.batch)]
    n_solves = sum(len(e) for _, e in plan) * args.maps * len(args.sizes)
    log(f'suite "{args.suite}": sizes {args.sizes}, {args.maps} maps each, conditions '
        f'{[c for c, _ in plan]}, {len(batches)} batch(es) per size/condition, '
        f'{n_solves} solves total; {cuda_desc}', logfile)
    if args.dry_run:
        for cond, entries in plan:
            print(f'  {cond}: ' + ', '.join(e.get('label', e['id']) for e in entries))
        for s in args.sizes:
            print(f'  size {s}: timeout {cfg["timeouts_s"].get(str(s), 300)}s per solve')
        return

    from benchmark_app import runner, storage
    done_names = {r['name']: r for r in storage.list_runs() if r.get('status') == 'done'}
    exported = set()
    if os.path.isfile(runs_csv):
        with open(runs_csv, newline='') as f:
            exported = {r['run_id'] for r in csv.DictReader(f)}

    for size in args.sizes:
        ms = ensure_map_set(f'{args.suite}-{size}px', size, args.maps, cfg['obstacles'],
                            cfg['seed'] + size, logfile)
        timeout = float(cfg['timeouts_s'].get(str(size), 300))
        names = [m['name'] for m in ms['maps']]
        for cond, entries in plan:
            set_condition(cond)
            workers = 1 if cond.startswith('gpu') else args.workers
            for b0, b1 in batches:
                run_name = f'{args.suite} {size}px {cond} maps {b0:04d}-{b1 - 1:04d}'
                run = done_names.get(run_name)
                if run is None:
                    log(f'start {run_name} ({len(entries)} configs x {b1 - b0} maps, timeout {timeout:.0f}s, '
                        f'{workers} worker(s))', logfile)
                    job = runner.enqueue_run({
                        'name': run_name, 'map_set_id': ms['id'], 'map_names': names[b0:b1],
                        'algorithms': [{'id': e['id'], 'params': e.get('params', {}),
                                        'graph_label': e.get('label')} for e in entries],
                        'solve_timeout_s': timeout, 'parallel_workers': workers,
                    })
                    job = wait_for(job['id'], logfile, run_name)
                    run = storage.get_run(job['result']['run_id'])
                    if job['status'] != 'done':
                        log(f'FAILED {run_name}: {job.get("error") or job["status"]}', logfile)
                        continue
                if run['id'] not in exported:
                    export_run(run['id'], size, cond, f'{b0:04d}-{b1 - 1:04d}', results_csv)
                    new = not os.path.isfile(runs_csv)
                    with open(runs_csv, 'a', newline='') as f:
                        w = csv.writer(f)
                        if new:
                            w.writerow(['run_id', 'name', 'size', 'condition', 'batch', 'map_set_id'])
                        w.writerow([run['id'], run_name, size, cond, f'{b0:04d}-{b1 - 1:04d}', ms['id']])
                    exported.add(run['id'])
                    write_summary(results_csv, os.path.join(out_dir, 'summary.csv'))
                    log(f'done  {run_name} -> {run["id"]}', logfile)
    set_condition('gpu-gpu')
    write_summary(results_csv, os.path.join(out_dir, 'summary.csv'))
    log(f'suite finished; results in {out_dir}', logfile)


if __name__ == '__main__':
    main()
