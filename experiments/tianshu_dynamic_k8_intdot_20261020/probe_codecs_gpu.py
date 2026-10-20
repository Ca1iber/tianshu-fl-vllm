# SPDX-License-Identifier: Apache-2.0
"""Full physical storage and BF16 output-bit tests; no model/performance result."""
import argparse
from datetime import datetime,timezone
import hashlib,json,os
from pathlib import Path
import signal,struct,subprocess,traceback
BASE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--stop-at-utc',required=True);args=parser.parse_args()
    remaining=int((datetime.fromisoformat(args.stop_at_utc)-datetime.now(timezone.utc)).total_seconds())
    assert remaining>0;signal.alarm(min(600,remaining))
    root=args.output.resolve();assert root.is_relative_to('/workspace/benchmark-results')
    root.mkdir(parents=True,exist_ok=False)
    hardware=subprocess.check_output(['ixsmi'],text=True);(root/'hardware_before.txt').write_text(hardware)
    assert 'No running processes found' in hardware
    import torch
    import vllm_fl
    import codec,writer
    from vllm.platforms import current_platform
    assert current_platform.vendor_name=='iluvatar'
    torch.set_num_threads(1)
    gate=json.loads((BASE/'CPU_GATE_32g.json').read_text());assert gate['complete']
    for name,expected in gate['sources'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected
    sources=[Path(__file__),BASE/'codec.py',BASE/'writer.py',BASE/'CPU_GATE_32g.json',Path('/usr/local/lib/python3.12/site-packages/triton/language/extra/corex/libdevice.py')]
    frozen={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    for p in sources:
        dest=root/'source_snapshot'/p.relative_to('/');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
    result=dict(complete=False,component_only=True,official_benchmark=False,adopted=False,model_executed=False,service_executed=False,
                no_performance_claim=True,runner_pid=os.getpid(),frozen_sources=frozen,cases=[])
    def save():(root/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    finite=[bits for bits in range(65536) if (bits&32640)!=32640]
    original=torch.tensor([x if x<32768 else x-65536 for x in finite],dtype=torch.int16).view(torch.bfloat16).reshape(255,2,128)
    try:
        for strided in (False,True):
            generator=torch.Generator(device='cpu').manual_seed(2026102001+strided)
            pages=18;stride=12416*(2 if strided else 1);n=255;total=n+3
            raw=torch.full((pages*stride,),17,dtype=torch.uint8,device='cuda')
            k=torch.as_strided(raw,(pages,16,2,128),(stride,256,128,1))
            v=torch.as_strided(raw.view(torch.bfloat16),(pages,16,2,128),(stride//2,256,128,1),4096//2)
            shape=(total,4,256) if strided else (total,2,128)
            key_storage=torch.full(shape,19.,device='cuda',dtype=torch.bfloat16);value_storage=torch.full_like(key_storage,23.)
            key=key_storage[:,::2,::2] if strided else key_storage
            value=value_storage[:,::2,::2] if strided else value_storage
            output_storage=torch.full((total+13,4,256),17.,device='cuda',dtype=torch.bfloat16)
            out=output_storage[:total,::2,::2]
            slots=torch.full((total,),-1,device='cuda',dtype=torch.int64)
            graph=None;phases=[]
            for phase in range(3):
                host_key=original if phase==0 else original.flip(0) if phase==1 else (original*.5)
                host_value=original.roll(17,0) if phase==0 else -original if phase==1 else original.flip(-1)
                key[:n].copy_(host_key);value[:n].copy_(host_value)
                slot_host=torch.cat((torch.randperm(pages*16,generator=generator)[:n],torch.full((3,),-1,dtype=torch.int64)))
                slots.copy_(slot_host)
                codes,scales=codec.encode_cpu(host_key);decoded=codec.decode_cpu(codes,scales)
                expected=bytearray([17])*raw.numel()
                for row,slot in enumerate(slot_host[:n].tolist()):
                    page,position=divmod(slot,16)
                    for head in range(2):
                        offset=page*stride+position*256+head*128
                        expected[offset:offset+128]=bytes(codes[row,head].tolist())
                        offset=page*stride+4096+position*512+head*256
                        expected[offset:offset+256]=host_value[row,head].contiguous().view(torch.uint8).numpy().tobytes()
                        offset=page*stride+12288+(position*2+head)*4
                        expected[offset:offset+4]=struct.pack('<f',float(scales[row,head]))
                raw.fill_(17);output_storage.fill_(17)
                key_before=key_storage.view(torch.int16).clone();value_before=value_storage.view(torch.int16).clone()
                def invoke():writer.write(key,value,k,v,slots);writer.decode(k,slots,out)
                if graph is None:
                    for _ in range(3):invoke()
                    raw.fill_(17);output_storage.fill_(17)
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):invoke()
                raw.fill_(17);output_storage.fill_(17)
                rng=torch.cuda.get_rng_state().clone();graph.replay();torch.cuda.synchronize()
                assert torch.equal(torch.cuda.get_rng_state(),rng)
                assert bytes(raw.cpu().tolist())==bytes(expected),'physical K/V/FP32 scales/padding mismatch'
                expected_output=torch.full(output_storage.shape,17.,dtype=torch.bfloat16)
                expected_output[:n,::2,::2]=decoded
                assert torch.equal(output_storage.view(torch.int16).cpu(),expected_output.view(torch.int16))
                assert torch.equal(key_storage.view(torch.int16),key_before) and torch.equal(value_storage.view(torch.int16),value_before)
                phases.append(dict(phase=phase,encoded_values=host_key.numel(),scale_values=scales.numel(),
                                   complete_physical_storage_exact=True,complete_output_bits_and_guards_exact=True,
                                   inputs_unchanged=True,RNG_unchanged=True,negative_slots_ignored=True,
                                   all_source_pointers_reused=True))
                save()
            result['cases'].append(dict(strided=strided,page_bytes=12416,actual_page_stride=stride,phases=phases));save()
            del graph,raw,k,v,key_storage,value_storage,output_storage,out,key,value,key_before,value_before
            torch.cuda.empty_cache()
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in frozen.items())
        result.update(complete=True,encoded_value_checks=391680,FP32_scale_checks=3060,ended_utc=datetime.now(timezone.utc).isoformat());save()
    except BaseException as error:
        result.update(error=repr(error),traceback=traceback.format_exc());save();raise
    finally:(root/'hardware_after.txt').write_text(subprocess.check_output(['ixsmi'],text=True))

if __name__=='__main__':main()
