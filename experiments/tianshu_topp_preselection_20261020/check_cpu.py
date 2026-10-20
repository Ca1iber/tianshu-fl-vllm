# SPDX-License-Identifier: Apache-2.0
"""Independent sorted-nucleus support retention proof for safe preselection proposals."""
import argparse,hashlib,json,math,random,statistics
from pathlib import Path
import source
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.output.resolve()
assert root.is_relative_to('/workspace/benchmark-results');root.mkdir(parents=True,exist_ok=False)
native=Path('/usr/local/lib/python3.12/site-packages/vllm/v1/sample/ops/topk_topp_triton.py')
proofs={}
for mode in source.MODES:
    text,proof=source.generate(native.read_bytes(),mode);(root/f'generated_{mode}.py').write_text(text);proofs[mode]=proof
rng=random.Random(20261020);rows=[]
for n in (17,129,1025,4097):
    normal=[rng.gauss(0,1) for _ in range(n)]
    corpora={'normal':normal,'wide':[4*x for x in normal],'peaked':[20. if i==n//2 else x for i,x in enumerate(normal)],
             'bimodal':[x+(4 if i%2 else -4) for i,x in enumerate(normal)],'ties':[round(x) for x in normal],
             'flat':[0.]*n,'masked':[x if i%3 else -math.inf for i,x in enumerate(normal)],
             'one_finite':[1. if i==0 else -math.inf for i in range(n)],'all_inf':[-math.inf]*n,
             'heavy_tail':[-math.log(i+1) for i in range(n)]}
    for name,values in corpora.items():
        finite=[v for v in values if math.isfinite(v)]
        if finite:
            maximum=max(finite);weights=[math.exp(v-maximum) if math.isfinite(v) else 0. for v in values];total=math.fsum(weights);probs=[w/total for w in weights]
            sample=[v for v in values[:32768] if math.isfinite(v)]
            avg=math.fsum(sample)/len(sample);std=math.sqrt(max(0.,math.fsum(v*v for v in sample)/len(sample)-avg*avg))
        else:maximum=avg=std=0.;probs=[0.]*n;total=1.
        for pvalue in (.01,.5,.8,.95,.999):
            support=[];mass=0.
            for i in sorted(range(n),key=lambda i:(-probs[i],i)):
                if not probs[i] or mass>=pvalue:break
                support.append(i);mass=math.fsum(probs[j] for j in support)
            sigma=statistics.NormalDist().inv_cdf(1-pvalue);sigma-=.25*abs(sigma)
            for mode,factor in source.MODES.items():
                pivot=avg+std*sigma+factor*std*std
                threshold=math.exp(min(700.,pivot-maximum))/total if finite else 1.
                kept={i for i,v in enumerate(probs) if v>threshold};gathered_mass=math.fsum(probs[i] for i in kept)
                uses_subset=gathered_mass>pvalue
                if uses_subset:assert set(support)<=kept,(n,name,pvalue,mode)
                rows.append(dict(n=n,distribution=name,p=pvalue,mode=mode,outliers=len(kept),uses_subset=uses_subset,
                                 reference_nucleus_size=len(support),all_exact_sorted_nucleus_tokens_retained=True))
sources=[Path(__file__),*source.dependency_sources()]
result=dict(complete=True,CPU_only=True,GPU_executed=False,adopted=False,independent_sorted_support_checks=len(rows),
            all_safe_pruned_sets_retain_exact_sorted_nucleus=True,scope='Mathematical support retention only, no Triton numeric, speed or model quality proof',
            normal_mass_shift_reason='e^x times a normal density with mean mu and variance s^2 has shifted mean mu+s^2; correctness never assumes normality because mass<=p uses the existing full-vocabulary fallback',
            provenance=proofs,cases=rows,sources={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
(root/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n')
for path in sources:
    dst=root/'source_snapshot'/path.resolve().relative_to('/');dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(path.read_bytes())
print(json.dumps(dict(complete=True,CPU_only=True,checks=len(rows),provenance=proofs)))
