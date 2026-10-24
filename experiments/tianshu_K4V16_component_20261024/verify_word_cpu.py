# SPDX-License-Identifier: Apache-2.0
"""Rebuild every packed cache and independently recompute every output in FP64."""
import argparse,csv,hashlib,importlib.util,json,math,statistics,sys
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent
CONTRACT=BASE.parent/'tianshu_K4V16_contract_20261024'
import fixtures_cpu as fx
s=importlib.util.spec_from_file_location('_K4_verify_independent_codec',CONTRACT/'codec.py')
codec=importlib.util.module_from_spec(s);s.loader.exec_module(codec)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def fp64(qb,k,v,scales,lengths):
    q=fx.values(qb).astype(np.float64);answer=np.empty(q.shape,np.float64)
    for seq,length in enumerate(lengths):
        for h in range(2):
            qq=q[seq,h*8:(h+1)*8]
            kk=k[seq,:length,h].astype(np.float64)
            vv=v[seq,:length,h].astype(np.float64)
            score=qq@kk.T
            if scales is None:score*=float(np.float32(128**-.5))
            else:score*=(scales[seq,:length,h]*np.float32(128**-.5)).astype(np.float64)[None,:]
            score-=np.max(score,axis=1,keepdims=True)
            probability=np.exp(score);probability/=np.sum(probability,axis=1,keepdims=True)
            answer[seq,h*8:(h+1)*8]=probability@vv
    assert np.isfinite(answer).all();return answer

def compare(actual,expected,atol,rtol):
    a=np.asarray(actual,dtype=np.float64);e=np.asarray(expected,dtype=np.float64)
    delta=np.abs(a-e);bad=delta>atol+rtol*np.abs(e)
    return dict(passed=bool(np.isfinite(a).all() and np.isfinite(e).all() and not bad.any()),
        elements=a.size,failed_elements=int(bad.sum()),max_abs=float(delta.max()),atol=atol,rtol=rtol)

def raw_cache(label,k,v,scales,table,strided):
    n,width=table.shape;count=n*width;factor=2 if strided else 1
    page=12288 if label=='current' else 10496
    raw=np.full(((count+2)*factor,page),17,np.uint8)
    physical=table.reshape(-1).astype(np.int64)*factor
    if label=='current':
        raw[physical,:4096]=k.reshape(count,4096)
        raw[physical,4096:12288]=v.reshape(count,-1).view(np.uint8)
    else:
        raw[physical,:2048]=k.reshape(count,2048)
        raw[physical,2048:2176]=scales.reshape(count,32).view(np.uint8)
        raw[physical,2304:10496]=v.reshape(count,-1).view(np.uint8)
    return raw

def main():
    p=argparse.ArgumentParser();p.add_argument('--raw-root',type=Path,required=True);args=p.parse_args();root=args.raw_root.resolve()
    r=json.loads((root/'result.json').read_text());assert r['complete'] and len(r['cases'])==len(fx.CASES)==8
    assert r['FP64_all_output_verification_pending'] and r['numpy_version']==np.__version__
    assert not (BASE/'WORD_GPU_GATE_32g.json').exists()
    primitive=r['decoder_primitive']
    assert primitive['complete'] and len(primitive['cases'])==4
    assert primitive==json.loads((root/'decode_primitive/result.json').read_text())
    assert primitive['generated_source_sha256']==r['reader_proof']['generated_sha256']
    primitive_outputs=0
    for unit in primitive['cases']:
        p=Path(unit['artifact']);assert sha(p)==unit['artifact_sha256']
        a=np.load(p);packed=a['packed_words'];assert set(packed.tolist())==set(range(65536))
        q=np.stack(tuple(((packed>>(4*d))&15).astype(np.int16)-8 for d in range(4)),axis=-1)
        expected=fx.bits(q.astype(np.float32)).reshape(-1)
        assert np.array_equal(a['observed_BF16_bits'],expected) and np.array_equal(a['expected_BF16_bits'],expected)
        binary=root/'decode_primitive'/f"{unit['stride']}_{unit['binary_sha256']}.bin"
        assert sha(binary)==unit['binary_sha256']
        assert unit['bitwise_equal'] and unit['input_and_output_guards'] and unit['RNG_unchanged']
        primitive_outputs+=expected.size
    assert primitive_outputs==primitive['total_output_elements']==1048576
    for path,h in r['frozen_sources'].items():assert sha(path)==h,path
    assert r['reader_proof']['parent_sha256']=='f1df731bae6cf48d5f38eef0a0adb5863ade2145667aab1ac976b9d473bdb002'
    rows=[];audits=[];total_outputs=0;total_raw=0
    for index,(case,definition) in enumerate(zip(r['cases'],fx.CASES)):
        name,base_lengths,strided,transition=definition;n=len(base_lengths);width=(max(base_lengths)+15)//16
        assert case['name']==name and case['batch']==n and case['strided']==strided and case['width']==width
        assert len(case['phases'])==2
        for scope in ('full','reader','writer'):
            trials=case['raw_ms'][scope];assert len(trials)==2 and all(len(t)==9 for t in trials)
            assert all(math.isfinite(v) and v>0 for t in trials for v in t)
            assert list(map(statistics.median,trials))==case['median_ms'][scope]
        slow=case['median_ms']['full'][1]/case['median_ms']['full'][0]
        assert math.isclose(case['full_component_speedup_percent'],100*(1/slow-1),abs_tol=1e-10)
        for phase,record in enumerate(case['phases']):
            kb,vb,qb,table=fx.generate(case['seed'],n,width,phase)
            lengths=([1]*n if transition and phase==0 else list(base_lengths) if transition or not phase
                else [x-min(7,x-1) for x in base_lengths])
            assert record['lengths']==lengths
            assert record['input_hashes']==dict(K_bits=fx.digest(kb),V_bits=fx.digest(vb),Q_bits=fx.digest(qb),table=fx.digest(table))
            path=Path(record['artifact']['path']);assert sha(path)==record['artifact']['sha256']
            saved=np.load(path)
            assert np.array_equal(saved['table'],table) and np.array_equal(saved['lengths'],lengths)
            encoded=fx.k8_encode(kb);packed,scales=codec.encode(fx.values(kb));v=fx.values(vb)
            for label,check in zip(('current','k4'),record['checks']):
                assert check['label']==label and check['numerical']['passed']
                key=fx.values(fx.k8_decode(encoded)) if label=='current' else codec.unpack(packed)
                expected=fp64(qb,key,v,None if label=='current' else scales,lengths)
                output=compare(fx.values(saved[label+'_output_bits']),expected,.002,.01)
                ref=compare(saved[label+'_GPU_reference'],expected,2e-5,2e-4)
                assert output['passed'] and ref['passed'],(name,phase,label,output,ref)
                raw=raw_cache(label,encoded if label=='current' else packed,vb,scales,table,strided)
                assert fx.digest(raw)==check['raw_storage_sha256'],(name,phase,label,'raw cache digest')
                assert raw.nbytes==check['raw_cache_bytes'] and raw.shape[0]==check['physical_pages']
                total_outputs+=output['elements'];total_raw+=raw.nbytes
                observed=case['actual_launch'][label]
                assert observed['IS_3D'] and observed['BLOCK_M']==16 and observed['BLOCK_Q']==2 and observed['TILE_SIZE']==16
                assert observed['num_warps']==observed['num_stages']==1 and observed['n_spills']>=0
                assert observed['K_shape'][-1]==(128 if label=='current' else 64)
                binary=root/f"{label}_{observed['binary_sha256']}.bin"
                assert sha(binary)==observed['binary_sha256']
                audits.append(dict(case=name,phase=phase,label=label,CPU_FP64_all_outputs=output,
                    full_GPU_FP32_reference_vs_FP64=ref,raw_cache_bytes_rebuilt=raw.nbytes,raw_sha256=check['raw_storage_sha256'],
                    all_inputs_regenerated_and_hashed=True,FP64_reference_sha256=fx.digest(expected)))
                del raw,key,expected
            del kb,vb,qb,table,encoded,packed,scales,v
        assert case['actual_launch']['current']['grid']==case['actual_launch']['k4']['grid']
        row=dict(case=name,current_full_ms=case['median_ms']['full'][0],k4_full_ms=case['median_ms']['full'][1],
            k4_slowdown_factor=slow,current_reader_ms=case['median_ms']['reader'][0],k4_reader_ms=case['median_ms']['reader'][1],
            current_writer_ms=case['median_ms']['writer'][0],k4_writer_ms=case['median_ms']['writer'][1])
        rows.append(row);print(json.dumps(row),flush=True)
    assert total_outputs==sum(len(x[1]) for x in fx.CASES)*16*128*2*2
    sources={str(BASE/'verify_word_cpu.py'):sha(BASE/'verify_word_cpu.py'),str(BASE/'fixtures_cpu.py'):sha(BASE/'fixtures_cpu.py'),
        str(CONTRACT/'codec.py'):sha(CONTRACT/'codec.py')}
    gate=dict(complete=True,CPU_only=True,raw_root=str(root),raw_result_sha256=sha(root/'result.json'),
        all_model_sources_preserved=True,independent_full_FP64_output_elements=total_outputs,
        independent_raw_cache_bytes_reconstructed=total_raw,all_input_hashes_verified=True,all32_arm_phases_verified=len(audits)==32,
        all_GPU_FP32_references_independently_checked=True,all_actual_binary_hashes_verified=True,
        all_components_slower=all(x['k4_slowdown_factor']>1 for x in rows),meets_consistent_component_gain=False,
        qualified_for_model_integration=False,adopted=False,rows=rows,checks=audits,sources=sources,
        native_Spec_count_verified=True,actual_component_page_bytes=10496,component_page_reduction_percent=100*(1-10496/12288),
        engine_allocator_connected=False,model_math_not_run=True,unmodified_current_release='r5')
    primary=[x for x in rows if x['case'] in ('c64_4096','c64_5120','c64_ragged_strided')]
    assert len(primary)==3
    eligible=all(100*(1/x['k4_slowdown_factor']-1)>=1 for x in primary)
    gate.update(word_helper_GPU_verified=True,word_helper_output_elements=primitive_outputs,
        meets_consistent_component_gain=eligible,qualified_for_model_integration=eligible,
        declared_gate_scope='C64 4096/5120/ragged all gain >=1%; other cases recorded as diagnostics')
    with (BASE/'WORD_KEY_RUNS_32g.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    (BASE/'WORD_GPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n')
    print(json.dumps({k:v for k,v in gate.items() if k not in ('checks','sources','rows')},indent=2))
if __name__=='__main__':main()
