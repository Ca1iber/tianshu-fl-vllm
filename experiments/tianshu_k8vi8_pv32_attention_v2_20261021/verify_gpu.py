# SPDX-License-Identifier: Apache-2.0
"""Independent CPU requantization, FP64 saved references and timing arithmetic."""
import argparse,hashlib,json,statistics
from pathlib import Path
import torch
BASE=Path(__file__).resolve().parent

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args();root=args.root.resolve();state=read(root/'result.json')
    torch.set_num_threads(8)
    assert state['phase']=='complete' and state['complete'] and len(state['cases'])==5
    assert state['modes']==['K8V16','K8VI8Heads','PV32']
    for path,digest in state['source_snapshots'].items():assert sha(path)==sha(root/'source_snapshot'/Path(path).relative_to('/'))==digest
    checks=writer_values=cpu_vectors=0;rows=[];fixtures={}
    for case in state['cases']:
        assert case['same_page_table_for_all_modes'] and case['timing_includes_real_writer_quantizer_scales'] and len(case['phases'])==3
        for phase in case['phases']:
            assert phase['changed_K_V_scales_slots_lengths']==bool(phase['phase']) and len(phase['checks'])==3
            for c in phase['checks']:
                assert all(c[k] for k in ('full_physical_storage_checked','output_and_scratch_guards','inputs_unchanged','RNG_unchanged')) and c['GPU_reference']['passed']
                for v in c['CPU_references']:
                    a=torch.tensor(v['actual'],dtype=torch.float64);r=torch.tensor(v['expected'],dtype=torch.float64)
                    assert v['CPU_FP64_reference'] and v['passed'] and torch.isfinite(a).all() and torch.isfinite(r).all() and bool(((a-r).abs()<=.02+.01*r.abs()).all());cpu_vectors+=1
                if c['writer_fixture']:
                    file=root/c['writer_fixture'];data=torch.load(file,map_location='cpu',weights_only=True);x=data['value_input'].float()
                    scale=x.abs().amax(-1).clamp_min(1.e-8)/127.;q=(x/scale[...,None]).round().clamp(-127,127).to(torch.int8)
                    assert torch.equal(data['actual_V_INT8'],q) and torch.equal(data['actual_V_scale'],scale) and bool((data['actual_K_scale']==1.).all())
                    fixtures[file.name]=dict(sha256=sha(file),bytes=file.stat().st_size);writer_values+=q.numel()
                else:assert c['mode']=='K8V16'
                checks+=1
        medians={m:statistics.median(case['raw_ms'][m]) for m in state['modes']}
        assert medians==case['median_ms'] and all(len(v)==7 and all(x>0 for x in v) for v in case['raw_ms'].values())
        layouts={v['mode']:v for v in case['layouts']}
        for m in ('K8VI8Heads','PV32','PV64'):
            assert layouts[m]['page_bytes']==layouts[m]['actual_spec_page_bytes']==8448 and layouts[m]['all_scale_bytes_in_native_page_accounting']
        for m,t,w,pv in [('K8V16',16,1,False),('K8VI8Heads',16,1,False),('PV32',32,2,True)]:
            record=state['producer_records'][str((m,case['batch'],case['context']))]
            assert record['IS_3D'] and record['tile']==t and record['warps']==w and record['PV_INT8']==pv and record['segments']==case['segments']
            assert record['KV_QUANT_MODE']==(0 if m=='K8V16' else 2)
        rows.append(dict(batch=case['batch'],context=case['context'],strided=case['strided'],segments=case['segments'],median_ms=medians,
            PV32_vs_K8V16_time_change_fraction=medians['PV32']/medians['K8V16']-1,
            PV32_vs_native_V8_time_change_fraction=medians['PV32']/medians['K8VI8Heads']-1))
    gate=dict(complete=True,component_only=True,adopted=False,full_engine_not_installed=True,Math_not_run=True,
        includes_different_tile_and_warp_configurations=True,checked_cases=5,checked_phase_paths=checks,
        CPU_FP64_reference_vectors_rechecked=cpu_vectors,actual_writer_V_values_requantized_on_CPU=writer_values,
        timing_includes_real_writer_quantizer_scales_producer_reducer=True,raw_root=str(root),raw_result_sha256=sha(root/'result.json'),
        writer_fixture_hashes=fixtures,source_snapshots=state['source_snapshots'],producer_records=state['producer_records'],rows=rows)
    (BASE/'GPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n')
    (BASE/'ROUNDS_32g.json').write_text(json.dumps({str((c['batch'],c['context'])):c['raw_ms'] for c in state['cases']},indent=2)+'\n')
    print(json.dumps({k:v for k,v in gate.items() if k not in ('writer_fixture_hashes','source_snapshots','producer_records')}))
if __name__=='__main__':main()
