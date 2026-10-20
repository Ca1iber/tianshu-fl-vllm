# SPDX-License-Identifier: Apache-2.0
"""Rebuild expected full-array hashes independently; recheck every saved timing."""
import argparse,csv,hashlib,json,math,statistics
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def hash_array(value):return hashlib.sha256(value.tobytes()).hexdigest()
def expected(capacity,n,phase):
    comp=((np.arange(capacity,dtype=np.int32)*13+phase*4091)%131072).astype(np.int32)
    if phase:comp=comp[::-1].copy()
    arrays={}
    for name,size,dtype,active in [('req_indices',4096,np.int64,np.arange(n,dtype=np.int64)),('query_pos',4096,np.int64,np.zeros(n,dtype=np.int64)),('num_scheduled_tokens',capacity,np.int32,np.ones(n,dtype=np.int32))]:
        cpu=np.arange(size,dtype=dtype)+701;gpu=cpu+1000
        cpu[:n]=active;gpu[:n]=active;arrays[name+'_CPU']=cpu;arrays[name+'_GPU']=gpu
    pos=np.arange(4096,dtype=np.int64)-100;pos[:n]=comp[:n].astype(np.int64)
    seq=np.zeros(capacity,dtype=np.int32);seq[:n]=comp[:n]+1
    arrays.update(positions=pos,seq_lens=seq,num_computed_tokens=comp)
    return {name:hash_array(value) for name,value in arrays.items()}

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args();root=args.root.resolve()
assert root.is_relative_to('/workspace/benchmark-results') and (root/'complete.txt').is_file()
assert (root/'exit_code.txt').read_text().strip()=='0'
path=root/'micro/result.json';data=json.loads(path.read_text());assert data['complete'] and len(data['cases'])==14
for name,digest in data['frozen_sources'].items():
    assert sha(name)==digest and sha(root/'micro/source_snapshot'/Path(name).relative_to('/'))==digest,name
assert data['actual_CpuGpuBuffer'] and data['FlagGems_all_ops_enabled']
assert data['production_FL_block_source']=='/usr/local/lib/python3.12/site-packages/vllm_fl/worker/model_runner.py'
assert data['production_FL_block_sha256']=='b0144bcff96bef15b04eb1ad45385b01c47faea056bc56352a6486b7c497fa2e'
rows=[];checks=0
for row in data['cases']:
    cap,n=row['capacity'],row['batch'];assert cap in (64,256) and n in (1,8,32,40,41,63,64)
    assert row['same_Graph_changed_inputs_verified'] and row['full_buffers_and_tails_bitwise'] and row['RNG_unchanged']
    assert [r['phase'] for r in row['phases']]==[0,1,31]
    for phase in row['phases']:
        hashes=expected(cap,n,phase['phase'])
        assert phase['control']==phase['candidate']==hashes,(cap,n,phase['phase'])
        checks+=len(hashes)*2
    meta=row['compiled_metadata'];assert meta['num_warps']==4 and meta['num_stages']==1
    assert meta['N']==n and meta['CAPACITY']==cap and meta['BLOCK']==256 and meta['grid']==[(cap+255)//256]
    assert meta['source_sha256']==sha(BASE/'kernel.py')
    for measurement in ('graph','wall'):
        for mode in ('control','candidate'):
            values=row[measurement+'_raw_ms'][mode];assert len(values)==7 and all(math.isfinite(x) and x>0 for x in values)
            assert statistics.median(values)==row[measurement+'_median_ms'][mode]
    graph=row['graph_median_ms'];wall=row['wall_median_ms']
    rows.append(dict(capacity=cap,batch=n,graph_control_ms=graph['control'],graph_candidate_ms=graph['candidate'],
        eager_control_ms=wall['control'],eager_candidate_ms=wall['candidate'],
        eager_time_reduction_percent=100*(1-wall['candidate']/wall['control']),n_regs=meta['n_regs'],n_spills=meta['n_spills']))
assert len({(r['capacity'],r['batch']) for r in rows})==14
gate=dict(complete=True,independently_rechecked=True,official_benchmark=False,adopted=False,
    GPU_cases=14,input_phases_per_case=3,whole_buffer_digest_comparisons=checks,
    CPU_hash_formula_independent_of_native_and_candidate=True,actual_CpuGpuBuffer=True,
    same_Graph_changed_computed_tokens=True,RNG_unchanged_recorded=True,offline_audit_did_not_repeat_GPU=True,
    kernel_sha256=sha(BASE/'kernel.py'),generated_method_sha256=json.loads((BASE/'CPU_GATE_32g.json').read_text())['generated_source_sha256'],
    raw_result_path=str(path),raw_result_sha256=sha(path),analyzer_sha256=sha(__file__),frozen_sources=data['frozen_sources'],rows=rows)
(BASE/'GPU_GATE_32g.json').write_text(json.dumps(gate,indent=2,allow_nan=False)+'\n')
with (BASE/'GPU_TIMINGS_32g.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
print(json.dumps(dict(complete=True,GPU_cases=14,whole_buffer_digest_comparisons=checks)))
