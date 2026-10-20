# SPDX-License-Identifier: Apache-2.0
"""Independent CPU requantization and saved writer/reference/timing verification."""
import argparse,hashlib,json,statistics
from pathlib import Path
import torch
BASE=Path(__file__).resolve().parent

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args();root=args.root.resolve();state=read(root/'result.json')
    assert state['phase']=='complete' and state['complete'] and len(state['cases'])==10
    for path,digest in state['source_snapshots'].items():
        assert sha(path)==digest and sha(root/'source_snapshot'/Path(path).relative_to('/'))==digest
    checks=0;writer_values=0;cpu_vectors=0;rows=[];fixtures={}
    for case in state['cases']:
        assert case['same_page_table_for_all_modes'] and case['timing_includes_real_writer_quantizer_scales']
        assert len(case['phases'])==3
        for phase in case['phases']:
            assert phase['changed_K_V_scales_slots_lengths']==bool(phase['phase']) and len(phase['checks'])==3
            for c in phase['checks']:
                assert all(c[k] for k in ('full_physical_storage_checked','output_and_scratch_guards','inputs_unchanged','RNG_unchanged')) and c['GPU_reference']['passed']
                for v in c['CPU_references']:
                    actual=torch.tensor(v['actual']);reference=torch.tensor(v['expected']);bad=(actual-reference).abs()>.02+.01*reference.abs()
                    assert v['passed'] and torch.isfinite(actual).all() and not bad.any();cpu_vectors+=1
                if c['writer_fixture']:
                    file=root/c['writer_fixture'];data=torch.load(file,map_location='cpu',weights_only=True);x=data['value_input'].float()
                    scale=x.abs().amax(-1).clamp_min(1.e-8)/127.;q=(x/scale[...,None]).round().clamp(-127,127).to(torch.int8)
                    assert torch.equal(data['actual_V_INT8'],q) and torch.equal(data['actual_V_scale'],scale) and bool((data['actual_K_scale']==1.).all())
                    fixtures[file.name]=dict(sha256=sha(file),bytes=file.stat().st_size);writer_values+=q.numel()
                checks+=1
        medians={mode:statistics.median(case['raw_ms'][mode]) for mode in state['modes']}
        assert medians==case['median_ms'] and all(len(values)==7 and all(v>0 for v in values) for values in case['raw_ms'].values())
        layouts={v['mode']:v for v in case['layouts']}
        assert layouts['K8VI8Heads']['page_bytes']==layouts['K8VI8Heads']['actual_spec_page_bytes']==8448
        assert layouts['K8VI8Heads']['all_scale_bytes_in_native_page_accounting']
        rows.append(dict(batch=case['batch'],context=case['context'],strided=case['strided'],segments=case['segments'],median_ms=medians,
            new_vs_K8V16_time_change_fraction=medians['K8VI8Heads']/medians['K8V16']-1))
    result=dict(complete=True,component_only=True,adopted=False,full_engine_not_installed=True,Math_not_run=True,
        checked_cases=10,checked_phase_paths=checks,CPU_reference_vectors_rechecked=cpu_vectors,actual_writer_V_values_requantized_on_CPU=writer_values,
        writer_fixture_hashes=fixtures,raw_root=str(root),raw_result_sha256=sha(root/'result.json'),
        source_snapshots=state['source_snapshots'],producer_records=state['producer_records'],rows=rows)
    (BASE/'GPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(complete=True,checked_phase_paths=checks,CPU_writer_values=writer_values,rows=rows)))
if __name__=='__main__':main()
