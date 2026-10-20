# SPDX-License-Identifier: Apache-2.0
"""Offline recheck full outputs/cache bytes and all alternating timing groups."""
import csv,hashlib,json,statistics,sys
from pathlib import Path
import numpy as np
import torch
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
BASE=Path(__file__).resolve().parent
MODES=('compiled_native','fused_w1','fused_w2','fused_w4')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def quantize(bits):
    bits=bits.astype(np.uint32);mag=bits&32767
    rounded=(mag+7+((mag>>4)&1))>>4
    payload=np.where(rounded>=936,np.clip(rounded.astype(np.int64)-928,0,127),0).astype(np.uint32)
    return (payload|((bits&32768)>>8)).astype(np.uint8)
def bf16(bits):return torch.from_numpy(bits.copy()).view(torch.bfloat16)

def main():
    root=Path(sys.argv[1]);state=read(root/'result.json');torch.set_num_threads(1)
    assert state['complete'] and state['phase']=='complete' and state['modes']==list(MODES)
    assert state['case_count']==14 and state['phase_count']==42 and state['mode_checks']==168
    assert state["components_per_graph"]==16 and state["iterations_per_timing_group"]==100
    assert len(state['cases'])==14 and state['RNG_unchanged'] and state['cos_sin_cache_unchanged']
    for path,digest in state['sources'].items():
        assert sha(path)==digest and sha(root/'source_snapshot'/Path(path).relative_to('/'))==digest
    assert state['compiled_inductor_sources']
    for path,digest in state['compiled_inductor_sources'].items():assert sha(path)==digest
    cachefile=root/'cos_sin_bf16_bits.npy';assert sha(cachefile)==state['cache_artifact_sha256']
    cache=bf16(np.load(cachefile));assert cache.shape==(131072,128)
    rows=[];checks=0;reference_elements=0;max_compiler_rel_RMSE=0.;compiled_exact=True
    assert {(c['batch'],c['input_row_stride']) for c in state['cases']}=={(n,s) for n in (1,8,32,40,41,63,64) for s in (2560,2688)}
    for case in state['cases']:
        n=case['batch'];stride=case['input_row_stride'];case_exact=True
        assert len(case['phases'])==3
        for phase,row in enumerate(case['phases']):
            assert row['phase']==phase and sha(row['artifact'])==row['sha256']
            with np.load(row['artifact']) as data:
                inp=bf16(data['input_bf16']);pos=torch.from_numpy(data['positions'].copy());slots=data['slots']
                assert inp.shape==(n,stride) and pos.shape==(n,) and slots.shape==(n,)
                if phase==2:
                    assert (slots>=0).all() and torch.equal(pos,4096+torch.arange(n,dtype=torch.int64))
                q,k=RotaryEmbedding.forward_static(pos,inp[:,:2048],inp[:,2048:2304],128,128,cache,True)
                qbits=q.contiguous().view(torch.uint16).numpy();kbits=k.contiguous().view(torch.uint16).numpy()
                assert np.array_equal(data['expected_q_bf16'],qbits) and np.array_equal(data['expected_k_bf16'],kbits)
                for mode in MODES:
                    aq=data[mode+'_q'];ak=data[mode+'_k'];actual=data[mode+'_raw']
                    assert aq.shape==(n,2048) and ak.shape==(n,256) and actual.dtype==np.uint8
                    assert torch.isfinite(bf16(aq)).all() and torch.isfinite(bf16(ak)).all()
                    same=np.array_equal(aq,qbits) and np.array_equal(ak,kbits)
                    assert row['modes'][mode]['bitwise_vs_native_eager']==same
                    if mode!='compiled_native':assert same
                    else:
                        case_exact &= same
                        for observed,reference in ((bf16(aq).float(),q.float()),(bf16(ak).float(),k.float())):
                            rms=float(((observed-reference).square().mean()/reference.square().mean().clamp_min(1e-30)).sqrt())
                            # Sanity of the compiled *control* against mathematical
                            # native reference; this is not a model quality gate.
                            assert rms<.02,(case['name'],phase,rms)
                            max_compiler_rel_RMSE=max(max_compiler_rel_RMSE,rms)
                    reference_elements+=aq.size+ak.size
                    expected=np.full(actual.shape,173,dtype=np.uint8)
                    valid=np.flatnonzero(slots>=0);block=slots[valid]//16;token=slots[valid]%16
                    kd=128+block[:,None]*12288+token[:,None]*256+np.arange(256)[None,:]
                    vd=128+block[:,None]*12288+4096+token[:,None]*512+np.arange(512)[None,:]
                    expected[kd]=quantize(ak[valid])
                    expected[vd]=inp[:,2304:2560].contiguous().view(torch.uint8).numpy()[valid]
                    assert np.array_equal(expected,actual),(case['name'],phase,mode,'all physical bytes')
                    checks+=1
        assert case['compiled_baseline_vs_fused_bitwise_all_phases']==case_exact
        compiled_exact &= case_exact
        for mode in MODES:
            values=case['timing_ms'][mode];assert len(values)==7 and all(v>0 for v in values)
            median=statistics.median(values);assert median==case['median_ms'][mode]
            if mode!='compiled_native':
                meta=case['kernel_metadata'][mode]
                assert meta['num_warps']==int(mode[-1]) and meta['num_stages']==1 and meta['grid']==[n,18]
            rows.append(dict(batch=n,input_row_stride=stride,mode=mode,component_median_ms=median,
                speedup_percent=100*(case['median_ms']['compiled_native']/median-1),
                compiled_native_exact_vs_eager=case_exact))
    assert checks==168
    with (BASE/'GPU_TIMING_32g.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    gate=dict(complete=True,independently_rechecked=True,component_only=True,adopted=False,
        case_count=14,changed_data_phases=42,full_output_and_physical_cache_checks=checks,reference_QK_elements=reference_elements,
        timing_groups=392,iterations_per_timing_group=100,components_per_graph=16,all_fused_outputs_bitwise_match_native_eager=True,
        compiled_native_also_bitwise_matches_eager=compiled_exact,compiled_control_max_relative_RMSE=max_compiler_rel_RMSE,
        compiled_native_baseline_not_eager=True,compiled_native_source_files=len(state['compiled_inductor_sources']),
        current_model_compiled_graph_or_service_not_covered=True,new_Math_and_4K_service_required_if_component_positive=True,
        RNG_unchanged=state['RNG_unchanged'],inputs_unchanged=True,cos_sin_cache_unchanged=True,
        raw_root=str(root),raw_result_sha256=sha(root/'result.json'),sources=state['sources'],summary=rows,
        verifier_source_sha256=sha(__file__))
    (BASE/'GPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n')
    (root/'independent_gpu_gate.json').write_text(json.dumps(gate,indent=2)+'\n')
    print(json.dumps({k:v for k,v in gate.items() if k not in ('summary','sources')},ensure_ascii=False))
if __name__=='__main__':main()
