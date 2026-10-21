# SPDX-License-Identifier: Apache-2.0
"""Independently recompute full CPU FP64 nucleus diagnostics from saved GPU tensors."""
import argparse,csv,hashlib,json,math,statistics
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
root=args.root.resolve();out=args.output.resolve()
assert root.is_relative_to('/workspace/benchmark-results') and not out.exists()
out.mkdir(parents=True)
torch.set_num_threads(8);assert not torch.cuda.is_initialized()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
data=json.loads((root/'micro/result.json').read_text());cpu=json.loads(Path(__file__).with_name('CPU_GATE_32g.json').read_text())
assert (root/'complete.txt').is_file() and (root/'exit_code.txt').read_text().strip()=='0'
assert data['complete'] and data['same_Graph_RNG_checks'] and len(data['cases'])==12
assert data['CPU_tolerance']==1e-6 and data['vocab_size']==130560
labels=['geo','online'];assert [x[0] for x in data['plans']]==labels
expected={mode:cpu['provenance'][mode]['generated_sha256'] for mode in labels}
assert expected['geo']=='09156e66b6aa2f4460423ea5b8d70bee043022bd75f0a784417455d0e979479d'
for name,digest in data['frozen_sources'].items():
    assert sha(name)==digest==sha(root/'micro/source_snapshot'/Path(name).relative_to('/'))
rows=[];full_checks=[];file_hashes={};counts=0;total_values=0
for index,case in enumerate(data['cases']):
    assert [x['label'] for x in case['variants']]==labels
    baseline=case['variants'][0].get('median_ms')
    assert baseline is not None
    for phase,field in [('original','input_p_before'),('changed','input_p_after')]:
        input_path=root/f'micro/input_{index}_{phase}.pt';inp=torch.load(input_path,map_location='cpu',weights_only=True)
        file_hashes[str(input_path)]=sha(input_path)
        values=inp['logits'].double();probs_p=inp['p'].double();assert probs_p.tolist()==case[field]
        if inp['k'] is not None:
            assert case['topk'] and bool((inp['k']==100).all())
            threshold=values.topk(100,dim=1).values[:,-1:];values=values.masked_fill(values<threshold,-math.inf)
        probs=values.softmax(-1)
        ordered,indices=values.sort(dim=1,descending=True,stable=True)
        op=probs.gather(1,indices);keep=op.cumsum(1)-op<probs_p[:,None]
        ideal=torch.zeros_like(keep).scatter_(1,indices,keep)&torch.isfinite(values)
        for entry in case['variants']:
            path=root/f'micro/output_{index}_{entry["label"]}_{phase}.pt'
            if not path.exists():
                assert phase=='changed' and not entry.get('original',{}).get('relative_gate',False)
                continue
            actual=torch.load(path,map_location='cpu',weights_only=True);file_hashes[str(path)]=sha(path)
            assert actual.shape==inp['logits'].shape and actual.dtype==torch.float32
            finite=torch.isfinite(actual)
            retained=torch.equal(actual[finite],inp['logits'][finite])
            mass=(probs*finite).sum(1);minimum=torch.where(finite,probs,math.inf).min(1).values
            low=torch.where(finite,values,math.inf).min(1).values;high=torch.where(~finite,values,-math.inf).max(1).values
            strict=bool((finite.any(1)&(mass>=probs_p-1e-6)&(mass-minimum<=probs_p+1e-6)&(high<=low)).all())
            recorded=entry[phase]
            assert strict==recorded['CPU_strict_semantics'] and retained==recorded['retained_values_unchanged']
            assert strict and retained if entry['label']=='geo' else strict==recorded['relative_gate']
            assert mass.tolist()==recorded['kept_mass']
            assert minimum.tolist()==recorded['smallest_kept_probability']
            assert int((finite!=ideal).sum())==recorded['stable_CPU_mask_difference']
            counts+=actual.shape[0];total_values+=actual.numel()
            full_checks.append(dict(case=index+1,label=entry['label'],phase=phase,rows=actual.shape[0],values=actual.numel(),strict=strict,retained=retained,ideal_mask_difference=int((finite!=ideal).sum())))
    for entry in case['variants']:
        if 'compiled_metadata' in entry:
            meta=entry['compiled_metadata'];wanted=data['native_source_sha256'] if case['topk'] else expected[entry['label']]
            assert wanted==meta['source_sha256']==sha(meta['source'])
            assert meta['num_warps']==16 and meta['num_stages']==1
            assert meta['BLOCK_SIZE']==(8192 if case['topk'] else 32768)
            assert meta['BLOCK_SIZE_TRUNC']==(4096 if case['topk'] else 16384)
            assert meta['grid']==[min(case['batch'],16)]
        if 'median_ms' in entry:
            assert len(entry['raw_ms'])==7 and all(math.isfinite(t) and t>0 for t in entry['raw_ms'])
            assert statistics.median(entry['raw_ms'])==entry['median_ms']
            assert entry['original_graph_exact'] and entry['changed_graph_exact']
        rows.append(dict(case=index+1,batch=case['batch'],distribution=case['distribution'],topk=case['topk'],label=entry['label'],median_ms=entry.get('median_ms'),time_reduction_percent=None if 'median_ms' not in entry else 100*(1-entry['median_ms']/baseline),strict_original=entry.get('original',{}).get('relative_gate',False),strict_changed=entry.get('changed',{}).get('relative_gate',False),error=entry.get('error')))
assert not torch.cuda.is_initialized()
passed=[label for label in labels if all(any(e['label']==label and 'median_ms' in e and e.get('original_graph_exact') and e.get('changed_graph_exact') and 'error' not in e for e in c['variants']) for c in data['cases'])]
assert passed==data['relative_passing_plans'] and 'geo' in passed
result=dict(complete=True,independently_CPU_recomputed_full_outputs=True,CPU_recompute_did_not_initialize_GPU=True,adopted=False,official_benchmark=False,model_service_not_run=True,
    GPU_cases=12,plans_per_case=2,phases=2,full_output_row_checks=counts,full_output_float_values=total_values,
    all_candidates_passed=passed==labels,relative_passing_plans=passed,source_sha256=expected,frozen_sources=data['frozen_sources'],
    tensor_file_sha256=file_hashes,raw_result_path=str(root/'micro/result.json'),raw_result_sha256=sha(root/'micro/result.json'),rows=rows,full_output_checks=full_checks)
(out/'GPU_GATE_32g.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
with (out/'GPU_TIMINGS_32g.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(json.dumps(dict(complete=True,passed=passed,full_rows=counts,full_float_values=total_values)))
for row in rows:
    if row['case'] in (3,4,8,9,12):print(json.dumps(row))
