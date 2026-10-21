# SPDX-License-Identifier: Apache-2.0
"""Reload original BF16 sources and actual outputs; independently redo CPU math."""
import argparse,csv,hashlib,json,statistics,sys
from pathlib import Path
import numpy as np
import torch
ORIGINAL=Path(__file__).resolve().parent.parent/'tianshu_k8vi8_pv_full_20261021';sys.path.insert(0,str(ORIGINAL))
from plans import PLANS,MODES,CASES,CURRENT_SHA
from reference_numpy import Oracle,quantize,soft8
from verification import compare
BASE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
torch.set_num_threads(4);assert not torch.cuda.is_initialized()
state=read(root/'result.json');assert state['complete'] and state['phase']=='complete'
assert state['modes']==list(MODES) and [(c['batch'],c['context']) for c in state['cases']]==list(CASES)
assert len(state['profile_environment'])==62 and len(state['server_command'])==13
assert state['codec_proof']['current_source_sha256']==CURRENT_SHA
for path,h in state['source_snapshots'].items():assert sha(path)==h==sha(root/'source_snapshot'/Path(path).relative_to('/'))
paths=0;vectors=0;integer_vectors=0;writer_values=0;key_bytes=0;fixtures={};reference_files={};rows=[];ideal_failures=[];bitwise=0
for case_id,case in enumerate(state['cases']):
    n,context=case['batch'],case['context'];segments=case['segments'];previous=None
    assert case['same_page_table_for_all_modes'] and case['timing_includes_real_writer_quantizer_scales'] and len(case['phases'])==3
    for phase in case['phases']:
        path=root/phase['reference_fixture'];assert sha(path)==phase['reference_sha256'];reference_files[path.name]=phase['reference_sha256']
        data=torch.load(path,map_location='cpu',weights_only=True);oracle=Oracle(data)
        assert (data['case'],data['phase'],data['batch'],data['segments'])==(case_id,phase['phase'],n,segments)
        assert data['indices']==sorted(set((0,n//2,n-1)))
        assert data['lengths']==[phase['lengths'][i] for i in data['indices']] and set(data['actual_outputs'])==set(MODES)
        if previous is not None:
            width=data['k'].shape[1]
            assert torch.equal(data['q'],-previous['q']) and torch.equal(data['k'],previous['k'][:,:width]*.5) and torch.equal(data['v'],previous['v'][:,:width]*-.5)
            assert data['lengths']==[max(1,v-7) for v in previous['lengths']]
        previous=data
        assert phase['default_false_bitwise_native_V8'] and torch.equal(data['actual_outputs']['PVFalse'],data['actual_outputs']['V8Native']);bitwise+=1
        if n>64:
            assert set(phase['actual_2D_fallback_bitwise_labels'])=={m for m in MODES if PLANS[m]['pv']}
            for mode in phase['actual_2D_fallback_bitwise_labels']:assert torch.equal(data['actual_outputs'][mode],data['actual_outputs']['V8Native']);bitwise+=1
        assert len(phase['checks'])==7 and {c['mode'] for c in phase['checks']}==set(MODES)
        for check in phase['checks']:
            mode=check['mode'];plan=PLANS[mode];active=plan['pv'] and n<=64
            assert check['PV_INT8_active']==active and check['GPU_reference']['passed']
            assert all(check[k] for k in ('full_physical_storage_checked','output_and_scratch_guards','inputs_unchanged','RNG_unchanged'))
            if not active:assert check['ideal_GPU_reference']['passed']
            elif not check['ideal_GPU_reference']['passed']:ideal_failures.append(dict(case=case_id,phase=phase['phase'],mode=mode,**check['ideal_GPU_reference']))
            actual=data['actual_outputs'][mode];assert actual.dtype==torch.bfloat16 and actual.shape==(len(data['indices']),16,128)
            for local,sequence in enumerate(data['indices']):
                for head in (0,7,8,15):
                    vector=actual[local,head].float();ideal=oracle.ideal(local,head,'current' if mode=='K8V16' else 'v8')
                    if active:
                        expected,bound=oracle.integer_pv(local,head,plan['tile'],segments)
                        proof=compare(vector,torch.from_numpy(expected),True);assert proof['passed'],(case_id,phase['phase'],mode,sequence,head,proof)
                        assert bool(np.all(np.abs(vector.numpy().astype(np.float64)-ideal)<=bound+.004*np.abs(expected.astype(np.float64))+5.e-4));integer_vectors+=1
                    else:assert compare(vector,torch.from_numpy(ideal))['passed']
                    vectors+=1
            if check['writer_fixture'] and check['writer_fixture'] not in fixtures:
                file=root/check['writer_fixture'];saved=torch.load(file,map_location='cpu',weights_only=True)
                vq,vs=quantize(saved['value_input'].float().numpy());_,kc=soft8(saved['key_input'])
                assert torch.equal(saved['actual_V_INT8'],torch.from_numpy(vq)) and torch.equal(saved['actual_V_scale'],torch.from_numpy(vs))
                assert torch.equal(saved['actual_K_bytes'],torch.from_numpy(kc)) and bool((saved['actual_K_scale']==1.).all())
                for local,sequence in enumerate(data['indices']):
                    position=data['lengths'][local]-1
                    assert torch.equal(saved['key_input'][sequence],data['k'][local,position]) and torch.equal(saved['value_input'][sequence],data['v'][local,position])
                fixtures[file.name]=dict(sha256=sha(file),bytes=file.stat().st_size);writer_values+=vq.size;key_bytes+=kc.size
            paths+=1
    medians={mode:statistics.median(case['raw_ms'][mode]) for mode in MODES};assert medians==case['median_ms']
    assert all(len(v)==7 and all(x>0 for x in v) for v in case['raw_ms'].values())
    layout={v['mode']:v for v in case['layouts']}
    assert layout['K8V16']['actual_spec_page_bytes']==12288
    for mode in MODES:
        assert layout[mode]['all_planes_alias_actual_raw']
        if mode!='K8V16':assert layout[mode]['actual_spec_page_bytes']==8448 and layout[mode]['all_scale_bytes_in_native_page_accounting']
        record=state['producer_records'][str((mode,n,context))];assert record['PV_INT8']==(PLANS[mode]['pv'] and n<=64)
        assert record['segments']==(segments if n<=64 else 1) and record['IS_3D']==(n<=64) and not record['USE_TD']
        rows.append(dict(batch=n,context=context,mode=mode,median_ms=medians[mode],current_ms=medians['K8V16'],V8_native_ms=medians['V8Native'],
            time_change_vs_current_percent=100*(medians[mode]/medians['K8V16']-1),time_change_vs_V8_native_percent=100*(medians[mode]/medians['V8Native']-1),segments=segments,tile=record['tile'],warps=record['warps'],n_regs=record['n_regs'],n_spills=record['n_spills']))
assert paths==210 and vectors==2184 and integer_vectors==1104 and writer_values==302592 and key_bytes==302592 and bitwise==42
assert not torch.cuda.is_initialized()
gate=dict(complete=True,CPU_metadata_only_recovery=True,parent_verifier_remains_failed=True,no_GPU_rerun=True,component_only=True,adopted=False,model_Math_not_run=True,full_engine_not_installed=True,
    all_declared_integer_algorithm_checks_passed=True,independent_CPU_original_inputs_reloaded=True,phase_paths=paths,CPU_reference_vectors=vectors,CPU_integer_PV_vectors=integer_vectors,
    exact_actual_writer_V_values=writer_values,exact_actual_writer_K_bytes=key_bytes,default_false_and_2D_full_bitwise_checks=bitwise,
    ideal_attention_tolerance_failures_preserved=ideal_failures,ideal_attention_failure_count=len(ideal_failures),model_quality_requires_fresh_Math_over95=True,
    current_407_profile_and_reader_verified=True,current_reader_sha256=CURRENT_SHA,environment_count=62,rows=rows,producer_records=state['producer_records'],
    source_snapshots=state['source_snapshots'],writer_fixture_hashes=fixtures,reference_file_hashes=reference_files,raw_root=str(root),raw_result_sha256=sha(root/'result.json'),verifier_sha256=sha(__file__))
(BASE/'GPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');(BASE/'ROUNDS_32g.json').write_text(json.dumps([{k:v for k,v in c.items() if k in ('batch','context','raw_ms','median_ms','segments','strided')} for c in state['cases']],indent=2)+'\n')
with (BASE/'KEY_TIMINGS_32g.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
print(json.dumps({k:v for k,v in gate.items() if k not in ('source_snapshots','producer_records','rows','writer_fixture_hashes','reference_file_hashes','ideal_attention_tolerance_failures_preserved')}))
for r in rows:
    if r['batch']==64 and r['context']==4096:print(json.dumps(r))
