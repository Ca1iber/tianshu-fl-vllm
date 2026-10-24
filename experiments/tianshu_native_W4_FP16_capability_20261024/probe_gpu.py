# SPDX-License-Identifier: Apache-2.0
"""Try actual native FP16 W4 contract; record rejections, do not time unqualified math."""
import os,signal,subprocess,traceback
from datetime import datetime,timezone
from common import *
def main():
    remaining=int((datetime.fromisoformat(DEADLINE)-datetime.now(timezone.utc)).total_seconds())
    assert remaining>900;signal.alarm(min(900,remaining))
    assert not GPU.exists();GPU.mkdir()
    state=dict(complete=False,phase='initializing',pid=os.getpid(),outputs=[],FP16_call_accepted=None,
        performance_measured=False,model_executed=False,adopted=False)
    def save():
        tmp=GPU/'result.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(GPU/'result.json')
    save()
    try:
        hardware=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hardware
        (GPU/'hardware_before.txt').write_text(hardware)
        gate=json.loads((BASE/'CPU_GATE_32g.json').read_text())
        assert gate['complete'] and all(sha(p)==h for p,h in gate['sources'].items())
        assert all(sha(p)==h for p,h in gate['libraries'].items()) and all(sha(p)==h for p,h in gate['files'].items())
        import torch,vllm_fl
        torch.set_num_threads(4)
        guarded=[];immutable=[]
        def tensor(array,dtype,name):
            array=np.ascontiguousarray(array)
            cpu=torch.from_numpy(array.copy())
            if array.dtype==np.uint16:cpu=cpu.view(dtype)
            raw=torch.full((array.nbytes+8192,),113,dtype=torch.uint8,device='cuda')
            t=raw[4096:-4096].view(cpu.dtype).reshape(cpu.shape);t.copy_(cpu)
            guarded.append(raw);immutable.append(dict(name=name,tensor=t,hash=ahash(array),version=int(t._version)))
            return t
        data=np.load(gate['bank']['file'],allow_pickle=False);ref0=np.load(gate['references'][0]['file'],allow_pickle=False)
        w=tensor(data['W'],None,'packed')
        sb=tensor(data['S'],torch.bfloat16,'S_BF16');zb=tensor(data['Z'],torch.bfloat16,'Z_BF16')
        sh=tensor(ref0['S_half_words'],torch.float16,'S_FP16');zh=tensor(ref0['Z_half_words'],torch.float16,'Z_FP16')
        cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone()
        for kind,dtype,s,z in [('BF16',torch.bfloat16,sb,zb),('FP16',torch.float16,sh,zh)]:
            initial=ref0['H'] if kind=='BF16' else ref0['H_half_words']
            x=tensor(initial,dtype,'H_'+kind)
            def compute():return torch.ops.aten._weight_int4pack_mm_dotsaddz.default(x,w,128,s,z,size_n=2048)
            try:
                warm=compute();torch.cuda.synchronize()
            except RuntimeError as error:
                if kind!='FP16':raise
                state.update(FP16_call_accepted=False,FP16_error=repr(error),
                    FP16_error_scope='Exact layer0 M64/K6144/N2048, Half A/Scale/Zero and unchanged natural packedW')
                save();break
            if kind=='FP16':state['FP16_call_accepted']=True
            assert warm.dtype==dtype and tuple(warm.shape)==(64,2048)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):output=compute()
            for entry in gate['references']:
                phase=entry['phase'];reference=np.load(entry['file'],allow_pickle=False)
                array=reference['H'] if kind=='BF16' else reference['H_half_words']
                x.copy_(torch.from_numpy(array).view(dtype))
                tracked=next(item for item in immutable if item['name']=='H_'+kind)
                tracked.update(hash=ahash(array),version=int(x._version))
                eager=compute();torch.cuda.synchronize()
                expected=eager.contiguous().view(torch.uint16).cpu().numpy().copy()
                repeated=compute();torch.cuda.synchronize()
                assert np.array_equal(expected,repeated.contiguous().view(torch.uint16).cpu().numpy())
                output.fill_(float('nan'));graph.replay();torch.cuda.synchronize()
                actual=output.contiguous().view(torch.uint16).cpu().numpy().copy()
                path=GPU/f'{kind}_phase{phase}.npz'
                np.savez(path,actual_words=actual,eager_words=expected)
                values=f32(actual) if kind=='BF16' else actual.view(np.float16).astype(np.float32)
                check=metrics(values,reference['reference_'+kind],kind)
                row=dict(kind=kind,phase=phase,file=str(path),sha256=sha(path),metrics=check,
                    actual_dtype=str(output.dtype),actual_shape=list(output.shape),
                    poison_Graph_eager_word_exact=bool(np.array_equal(actual,expected)))
                state['outputs'].append(row);save()
                assert row['poison_Graph_eager_word_exact'] and check['passed'],row
            del graph,output,warm
        assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
        for entry in immutable:
            t=entry['tensor'];assert t._version==entry['version'] and ahash(t.contiguous().view(torch.uint8).cpu().numpy())==entry['hash']
        assert all(bool((raw[:4096]==113).all() and (raw[-4096:]==113).all()) for raw in guarded)
        assert all(sha(p)==h for p,h in gate['sources'].items()) and all(sha(p)==h for p,h in gate['libraries'].items())
        state.update(complete=True,phase='complete',full_inputs_versions_guards_unchanged=True,CPU_GPU_RNG_unchanged=True,
            CPU_gate_sha256=sha(BASE/'CPU_GATE_32g.json'));save()
    except BaseException as exc:
        state.update(phase='failed',error=repr(exc),traceback=traceback.format_exc());save();raise
    finally:
        (GPU/'hardware_after.txt').write_text(subprocess.check_output(['ixsmi'],text=True));signal.alarm(0)
if __name__=='__main__':main()
