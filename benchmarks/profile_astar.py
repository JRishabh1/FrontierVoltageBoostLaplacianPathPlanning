"""Serial A* timings on real app maps or seeded synthetic maps.

python benchmarks/profile_astar.py --map-set ms_... --limit 20 --baseline-ref HEAD~1
python benchmarks/profile_astar.py --baseline-ref HEAD~1 --output /tmp/astar.json
No process startup, metric calculation, visualization or map generation is timed.
"""
import argparse
import cProfile
import importlib.util
import inspect
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark_app.registry import get_spec
from map_generation import Map


def synthetic_maps(size, count, seed):
    rng=np.random.default_rng(seed)
    for i in range(count):
        grid=np.zeros((size,size),np.uint8)
        for _ in range(20):
            width=int(rng.integers(max(1,size//20),max(2,size//8)+1))
            x,y=rng.integers(1,size-width,2)
            grid[y:y+width,x:x+width]=1
        start=(max(1,size//25),max(1,size//13))
        end=(size-max(2,size//16),size-max(2,size//8))
        grid[start[::-1]]=grid[end[::-1]]=0
        yield f'synthetic_{i:03}',Map(grid,start,end)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--map-set', help='App map-set ID or folder containing manifest.json')
    ap.add_argument('--baseline-ref', help='Git ref holding the previous A* implementation')
    ap.add_argument('--size',type=int,default=512)
    ap.add_argument('--limit',type=int,default=12)
    ap.add_argument('--repeats',type=int,default=3)
    ap.add_argument('--seed',type=int,default=17)
    ap.add_argument('--timeout',type=float,default=60.)
    ap.add_argument('--allow-corner-cutting',action='store_true')
    ap.add_argument('--output',type=Path,default=Path('benchmarks/results/astar-profile.json'))
    ap.add_argument('--cprofile',type=Path,help='Profile old A* on the first map, outside timed trials')
    args=ap.parse_args()
    if args.limit<1 or args.repeats<1 or args.size<8:
        ap.error('limit/repeats must be positive, size >= 8')
    baseline=None
    with tempfile.TemporaryDirectory(prefix='astar-profile-') as folder:
        if args.baseline_ref:
            source=subprocess.check_output(['git','show',f'{args.baseline_ref}:Algorithms/Baseline Algorithms/astar.py'],cwd=ROOT)
            filename=Path(folder)/'astar_baseline.py';filename.write_bytes(source)
            spec=importlib.util.spec_from_file_location('astar_baseline',filename)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            baseline=module.AStarAlgorithm
            if 'allow_corner_cutting' not in inspect.signature(baseline).parameters and not args.allow_corner_cutting:
                ap.error('This baseline allows corner cutting; use --allow-corner-cutting for a matched comparison')
        if args.map_set:
            base=Path(args.map_set)
            if not base.is_dir():base=ROOT/'Benchmark Data/map_sets'/args.map_set
            manifest=json.loads((base/'manifest.json').read_text())
            maps=[(e['name'],Map(np.load(base/e['name']),tuple(e['start']),tuple(e['end'])))
                  for e in manifest['maps'][:args.limit]]
        else:
            maps=list(synthetic_maps(args.size,args.limit,args.seed))
        if not maps:ap.error('No maps selected')
        variants=['python']
        new_cls=get_spec('astar').build(maps[0][1],{}).__class__
        if new_cls._load_native() is not None:variants.append('native')
        if baseline:variants.insert(0,'baseline')
        def build(variant,m):
            params={'timeout':args.timeout,'allow_corner_cutting':args.allow_corner_cutting}
            if variant=='baseline':
                if 'allow_corner_cutting' not in inspect.signature(baseline).parameters:
                    params.pop('allow_corner_cutting')
                return baseline(m,**params)
            return get_spec('astar').build(m,{**params,'use_native':variant=='native'})
        rows=[]
        for name,m in maps:
            for repeat in range(args.repeats):
                # Rotate order so the old search is not always the cold-cache one.
                order=variants[repeat%len(variants):]+variants[:repeat%len(variants)]
                for variant in order:
                    a=build(variant,m)
                    t=time.perf_counter();c=time.process_time()
                    path=a.solve()
                    cpu=time.process_time()-c;elapsed=time.perf_counter()-t
                    row={'map':name,'variant':variant,'repeat':repeat,'solved':path is not None,
                         'time_s':elapsed,'cpu_time_s':cpu,'shape':list(m.grid.shape),
                         'backend':getattr(a,'backend_used',None),'stats':getattr(a,'search_stats',{}),
                         'path_cost':sum(math.dist(p,q) for p,q in zip(path,path[1:])) if path else None}
                    rows.append(row)
                    print(f'{name} {variant:8} {elapsed:.6f}s cost={row["path_cost"]}',flush=True)
            solved=[r for r in rows if r['map']==name and r['solved']]
            if solved and any(not r['solved'] for r in rows if r['map']==name):
                raise RuntimeError(f'{name}: solve status differs; inspect timeouts before comparing speeds')
            if solved and max(r['path_cost'] for r in solved)-min(r['path_cost'] for r in solved)>1e-8:
                raise RuntimeError(f'{name}: shortest-path costs differ')
        if args.cprofile:
            if not baseline:ap.error('--cprofile requires --baseline-ref')
            args.cprofile.parent.mkdir(parents=True,exist_ok=True)
            pr=cProfile.Profile();pr.runcall(build('baseline',maps[0][1]).solve);pr.dump_stats(str(args.cprofile))
        summary={}
        for variant in variants:
            subset=[r for r in rows if r['variant']==variant]
            good=[r for r in subset if r['solved']]
            summary[variant]={'attempts':len(subset),'solved':len(good),
                              'mean_s':statistics.mean(r['time_s'] for r in good) if good else None,
                              'median_s':statistics.median(r['time_s'] for r in good) if good else None}
        report={'python':sys.version,'platform':platform.platform(),
                'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                'settings':vars(args),'summary':summary,'trials':rows}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2,default=str)+'\n')
        print(json.dumps(summary,indent=2));print(f'Saved {args.output}')


if __name__=='__main__':main()
