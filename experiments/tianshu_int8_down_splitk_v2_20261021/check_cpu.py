# SPDX-License-Identifier: Apache-2.0
"""Check split partitions, launch geometry and exact integer limits without GPU."""
import ast,hashlib,json
from pathlib import Path
from partition import chunk_size
BASE=Path(__file__).resolve().parent
OLD=BASE.parent/'tianshu_int8_down_grouped_20261021'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for path in BASE.glob('*.py'):ast.parse(path.read_text())
for name in ('baseline.py','qualified_down_route.json','weight_signature.json'):assert sha(BASE/name)==sha(OLD/name)
plans=json.loads((BASE/'tiles.json').read_text());assert len(plans)==9
partition_checks=geometry_checks=0
for plan in plans:
    splits=plan['splits'];bk=plan['bk'];assert 1<=splits<=8
    for k in (1,259,6144,6151):
        chunk=chunk_size(k,splits,bk)
        covered=[i for split in range(splits) for block in range(chunk//bk) for lane in range(bk) if (i:=split*chunk+block*bk+lane)<k]
        assert covered==list(range(k)) and k*127*127<2**31;partition_checks+=1
    for m in (1,3,17,129,1024,2048,4096):
      for n in (1,130,2048):
        rows=(m+plan['bm']-1)//plan['bm'];cols=(n+plan['bn']-1)//plan['bn'];group=plan['group_m'];seen=set()
        for pid in range(rows*cols):
            first=pid//(group*cols)*group;size=min(rows-first,group);local=pid%(group*cols)
            location=(first+local%size,local//size)
            assert 0<=location[0]<rows and 0<=location[1]<cols and location not in seen;seen.add(location)
        assert len(seen)==rows*cols;geometry_checks+=1
kernel=(BASE/'kernel.py').read_text();assert 'quantize=parent.quantize' in kernel and 'atomic_' not in kernel
probe=(BASE/'probe_gpu.py').read_text()
assert 'kernel.mm(q,wq.T,asc,ws,out,integer,config)' not in probe
assert 'assert torch.equal(out,unsplit_out)' in probe and 'partial_integer=partial[' in probe
mm_calls=[n for n in ast.walk(ast.parse(probe)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='kernel' and n.func.attr=='mm']
assert len(mm_calls)==3 and all(any(kw.arg=='partial' and isinstance(kw.value,ast.Name) and kw.value.id=='partial' for kw in call.keywords) for call in mm_calls)
assert 'unsplit_int8_ms' in (BASE/'verify_gpu.py').read_text()
route=json.loads((BASE/'qualified_down_route.json').read_text())
paths=[*BASE.glob('*.py'),*BASE.glob('*.sh'),BASE/'tiles.json',BASE/'weight_signature.json',BASE/'qualified_down_route.json',OLD/'kernel.py',Path(route['kernel_source']),Path(route['vendor_source'])]
assert sha(route['kernel_source'])==route['kernel_sha256'] and sha(route['vendor_source'])==route['vendor_sha256']
result=dict(complete=True,CPU_only=True,no_GPU_started=True,INT32_no_overflow=True,worst_absolute_integer_sum=6144*127*127,
    all_K_partitions_complete_unique=True,partition_cases=partition_checks,all_MN_geometry_complete_unique=True,geometry_cases=geometry_checks,
    activation_quantization_original_function=True,original_BF16_baseline_and_checkpoint_signature_identical=True,
    no_atomics_in_source=True,partial_storage_and_reduction_cost_in_timed_Graph=True,original_unsplit_INT8_also_measured=True,
    GPU_and_model_accuracy_not_proven=True,sources={str(p.resolve()):sha(p) for p in paths})
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='sources'}))
