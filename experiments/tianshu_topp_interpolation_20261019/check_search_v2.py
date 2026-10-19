# SPDX-License-Identifier: Apache-2.0
"""CPU feasibility study only: probe placement, not a production or GPU kernel."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics


def search(probs, mode, p=.95):
    lo, hi = 0., max(probs)
    gl, gh = 1.-p, -p
    previous = 0
    for it in range(32):
        geometric = math.sqrt(lo)*math.sqrt(hi) if lo > 0 else (lo+hi)*.5
        pivot = geometric
        if mode != 'geo' and lo > 0 and gl > 0 > gh and not (mode == 'guarded' and it % 4 == 3):
            fraction = min(.9, max(.1, gl/(gl-gh)))
            pivot = math.exp(math.log(lo)+(math.log(hi)-math.log(lo))*fraction)
        values = [x for x in probs if x > pivot]
        mass = math.fsum(values)
        smallest = min(values, default=1.)
        duplicates = values.count(smallest)
        if mass >= p and mass-smallest*duplicates < p:
            return dict(iterations=it+1, found=True, boundary=smallest,
                        selected=len(values)-int((mass-p)/smallest))
        if mass > p:
            lo, gl = pivot, mass-p
            if mode in ('illinois','guarded') and previous == 1:gh *= .5
            previous = 1
        elif mass < p:
            hi, gh = pivot, mass-p
            if mode in ('illinois','guarded') and previous == -1:gl *= .5
            previous = -1
        else:
            return dict(iterations=it+1, found=False, reason='mass equality needs production tie handling')
        if hi-lo < 1e-12:
            return dict(iterations=it+1, found=False, reason='interval limit without exact boundary')
    return dict(iterations=32, found=False, reason='iteration limit')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();rows=[]
    for n in (8192,130560):
        for shape in ('normal','peaked','uniform','duplicate'):
            for seed in range(2):
                rng=random.Random(20261019000+seed)
                logits=[rng.gauss(0,3 if shape=='peaked' else 1) for _ in range(n)]
                if shape=='uniform':logits=[0.]*n
                if shape=='duplicate':logits=[round(x*2)/2 for x in logits]
                maximum=max(logits);weights=[math.exp(x-maximum) for x in logits];total=math.fsum(weights)
                probs=[x/total for x in weights]
                ordered=sorted(probs,reverse=True)
                # Balanced FP64 prefix sums avoid a one-item error on large uniform rows.
                left,right=0,len(ordered)
                while left<right:
                    middle=(left+right)//2
                    if math.fsum(ordered[:middle]) >= .95:right=middle
                    else:left=middle+1
                target=left
                results={mode:search(probs,mode) for mode in ('geo','secant','illinois','guarded')}
                for result in results.values():
                    result['exact_prefix_count']=bool(result['found'] and result['selected']==target)
                rows.append(dict(n=n,shape=shape,seed=seed,reference_count=target,results=results))
    summary={mode:dict(mean_iterations=statistics.mean(row['results'][mode]['iterations'] for row in rows),
                       exact_cases=sum(row['results'][mode]['exact_prefix_count'] for row in rows),cases=len(rows))
             for mode in ('geo','secant','illinois','guarded')}
    output=dict(CPU_only=True,all_arithmetic_FP64=True,production_FP32_or_outlier_buffer_not_reproduced=True,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rows=rows,summary=summary,
                GPU_test=False,performance_score=False,adopted=False)
    args.output.write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(summary))


if __name__=='__main__':main()
