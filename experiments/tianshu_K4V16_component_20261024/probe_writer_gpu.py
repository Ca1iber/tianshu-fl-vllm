# SPDX-License-Identifier: Apache-2.0
"""Exhaustive actual K4 writer, raw bytes and Graph mutation; no performance claim."""
import argparse,hashlib,importlib.util,json,os,signal,subprocess,sys,traceback
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent
CONTRACT=BASE.parent/'tianshu_K4V16_contract_20261024'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    root=args.output.resolve();assert root.is_relative_to('/dev/shm') and not root.exists();root.mkdir()
    remaining=int((datetime.fromisoformat('2026-10-24T15:58:00+00:00')-datetime.now(timezone.utc)).total_seconds())
    assert remaining>300;signal.alarm(min(900,remaining))
    hardware=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hardware
    (root/'hardware_before.txt').write_text(hardware)
    gate=json.loads((CONTRACT/'CPU_GATE_32g.json').read_text());assert gate['complete'] and gate['CPU_only']
    for path,h in gate['sources'].items():assert sha(path)==h
    fixture=Path(gate['raw_fixture']);assert sha(fixture)==gate['raw_fixture_sha256']
    codec=load('_K4_primitive_CPU_codec',CONTRACT/'codec.py')
    sys.path.insert(0,str(CONTRACT));reference=load('_K4_primitive_scalar_reference',CONTRACT/'check_cpu.py').reference
    sys.path.pop(0)
    import torch
    import vllm_fl
    import writer,pages
    torch.set_num_threads(8)
    sources=list(BASE.glob('*.py'))+[BASE/'PLAN_32g.md',CONTRACT/'codec.py',CONTRACT/'check_cpu.py',CONTRACT/'CPU_GATE_32g.json',
        Path('/usr/local/lib/python3.12/site-packages/triton/language/extra/corex/libdevice.py'),
        Path('/usr/local/lib/python3.12/site-packages/vllm/v1/kv_cache_interface.py')]
    frozen={str(f):sha(f) for f in sources}
    for f in sources:
        copied=root/'source_snapshot'/f.relative_to('/');copied.parent.mkdir(parents=True,exist_ok=True);copied.write_bytes(f.read_bytes())
    result=dict(complete=False,phase='GPU writer primitive',runner_pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),
        frozen_sources=frozen,original_CPU_gate_sha256=sha(CONTRACT/'CPU_GATE_32g.json'),cases=[],
        native_spec=pages.native_spec(),model_executed=False,performance_measured=False,adopted=False)
    def save():
        t=root/'result.tmp';t.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');t.replace(root/'result.json')
    save();raw_fixture=np.load(fixture)
    try:
        n=272;logical_pages=n//16+2
        with torch.inference_mode():
            for strided in (False,True):
                physical_pages=logical_pages*(2 if strided else 1)
                raw=torch.full((physical_pages*pages.PAGE_BYTES,),17,dtype=torch.uint8,device='cuda')
                k,v,s=pages.views(raw,physical_pages,strided)
                ks=torch.full((n+3,3,128),19,dtype=torch.bfloat16,device='cuda');key=ks[:,:2]
                vs=torch.full_like(ks,23);value=vs[:,:2]
                slots=torch.full((n+3,),-1,dtype=torch.int64,device='cuda')
                graph=None
                for phase in range(2):
                    kb=np.zeros((n,2,128),np.uint16)
                    kb[:256].reshape(-1)[:65280]=raw_fixture[f'input_bits_{phase}']
                    rows=kb.reshape(-1,128)
                    ties=np.array([-7,-6.5,-5.5,-4.5,-3.5,-2.5,-1.5,-.5,-0.,0.,.5,1.5,2.5,3.5,4.5,5.5,6.5,7],np.float32)
                    special=np.zeros((32,128),np.float32)
                    for i in range(20):special[i,:ties.size]=ties*np.float32(2.**(i-10))
                    special[20]=-0.;special[21]=np.float32(2.**-110);special[22]=np.float32(2.**-101)
                    special[23,::2]=np.float32(2.**-100);special[23,1::2]=-np.float32(2.**-100)
                    special[24:]=special[:8]
                    rows[512:]=(special.view(np.uint32)>>16).astype(np.uint16)
                    x=(kb.astype(np.uint32)<<16).view(np.float32)
                    packed,scale=codec.encode(x)
                    for row,pb,sc in zip(x.reshape(-1,128),packed.reshape(-1,64),scale.reshape(-1)):
                        expected,scale_bytes=reference(row)
                        assert pb.tobytes()==expected and sc.tobytes()==scale_bytes
                    assert np.array_equal(packed.reshape(-1,64)[:510],raw_fixture[f'packed_{phase}'])
                    assert np.array_equal(scale.reshape(-1)[:510].view(np.uint32),raw_fixture[f'scale_{phase}'].view(np.uint32))
                    vb=np.zeros_like(kb);vb[:256].reshape(-1)[:]=np.arange(65536,dtype=np.uint16)
                    if phase:vb[:256].reshape(-1)[:]=vb[:256].reshape(-1)[::-1].copy()
                    order=np.random.default_rng(2026102400+phase).permutation(n).astype(np.int64)
                    allslots=np.concatenate([order,np.full(3,-1,np.int64)])
                    key[:n].copy_(torch.from_numpy(kb.view(np.int16)).view(torch.bfloat16).to('cuda'))
                    value[:n].copy_(torch.from_numpy(vb.view(np.int16)).view(torch.bfloat16).to('cuda'))
                    slots.copy_(torch.from_numpy(allslots).to('cuda'));raw.fill_(17)
                    expected=np.full(raw.numel(),17,np.uint8)
                    for row,slot in enumerate(order):
                        block,token=divmod(int(slot),16);physical=block*(2 if strided else 1)
                        for head in range(2):
                            base=physical*pages.PAGE_BYTES
                            at=base+(token*2+head)*64;expected[at:at+64]=packed[row,head]
                            at=base+pages.SCALE_OFFSET+(token*2+head)*4
                            expected[at:at+4]=np.asarray([scale[row,head]],dtype=np.float32).view(np.uint8).reshape(-1)
                            at=base+pages.V_OFFSET+(token*2+head)*256
                            expected[at:at+256]=vb[row,head].view(np.uint8)
                    key_before=ks.view(torch.int16).clone();value_before=vs.view(torch.int16).clone();slots_before=slots.clone()
                    if graph is None:
                        compiled=writer.write(key,value,k,v,s,slots);torch.cuda.synchronize()
                        raw.fill_(17);graph=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):writer.write(key,value,k,v,s,slots)
                    rng=torch.cuda.get_rng_state().clone()
                    graph.replay();torch.cuda.synchronize()
                    observed=raw.cpu().numpy();different=np.flatnonzero(observed!=expected)
                    assert different.size==0,dict(first_byte=int(different[0]),actual=int(observed[different[0]]),expected=int(expected[different[0]]))
                    assert torch.equal(torch.cuda.get_rng_state(),rng)
                    assert torch.equal(ks.view(torch.int16),key_before) and torch.equal(vs.view(torch.int16),value_before)
                    assert torch.equal(slots,slots_before)
                    artifact=root/f'writer_strided{int(strided)}_phase{phase}.npz'
                    np.savez_compressed(artifact,observed_raw=observed,expected_raw=expected,input_K_bits=kb,input_V_bits=vb,
                        slots=allslots,expected_packed=packed,expected_scales=scale)
                    case=dict(strided=strided,phase=phase,finite_K_patterns=65280,all_V_BF16_patterns=65536,
                        independent_scalar_rows=int(n*2),raw_storage_bytes=int(raw.numel()),all_raw_bytes_equal=True,
                        negative_slots=3,input_and_padding_unchanged=True,RNG_unchanged=True,same_Graph_changed_inputs=bool(phase),
                        actual_writer=dict(grid=[n+3,2],num_warps=compiled.metadata.num_warps,num_stages=compiled.metadata.num_stages,
                            n_regs=getattr(compiled,'n_regs',None),n_spills=getattr(compiled,'n_spills',None),
                            binary_sha256=hashlib.sha256(compiled.kernel).hexdigest()),
                        artifact=dict(path=str(artifact),sha256=sha(artifact),bytes=artifact.stat().st_size))
                    result['cases'].append(case);save();print(json.dumps(case),flush=True)
        assert len(result['cases'])==4
        for path,h in frozen.items():assert sha(path)==h
        result.update(complete=True,phase='complete',all_writer_bytes_and_CPU_contract_verified=True,
            finite_K_pattern_visits=4*65280,V_BF16_pattern_visits=4*65536,ended_at=datetime.now(timezone.utc).isoformat());save()
    except BaseException as e:
        result.update(phase='failed',error=repr(e),traceback=traceback.format_exc());save();raise
    finally:
        signal.alarm(0)
        (root/'hardware_after.txt').write_text(subprocess.check_output(['ixsmi'],text=True))
if __name__=='__main__':main()
