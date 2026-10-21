# SPDX-License-Identifier: Apache-2.0
"""Full active-cell/guard proof for tighter prefill maxK; isolated gather only."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
from types import SimpleNamespace

BASE=Path(__file__).resolve().parent
EXPERIMENTS=BASE.parent
PIPE=EXPERIMENTS/'tianshu_k8v16_pipeline_20261019'
HELPER=EXPERIMENTS/'tianshu_prefill_key_bound_20261021'
CAPACITY=8192
CASES=[
    dict(name='pure_C8',nd=0,keys=[512]*8,queries=[512]*8,maximum=512),
    dict(name='mixed_63decode_short_prefill',nd=63,keys=[65],queries=[33],maximum=32768),
    dict(name='mixed_32decode_32prefill',nd=32,keys=[129+i%15 for i in range(32)],queries=[16]*32,maximum=2048),
    dict(name='mixed_62decode_prefix',nd=62,keys=[4097,1202],queries=[128,17],maximum=32768),
    dict(name='mixed_60decode_4prefill',nd=60,keys=[8192,7169,65,4096],queries=[2048,512,33,128],maximum=32768),
    dict(name='pure_C8_prefix',nd=0,keys=[1025,1202,1297,1536,1665,1792,1921,2048],queries=[128]*8,maximum=2048),
    dict(name='mixed_57decode_unaligned',nd=57,keys=[1201,16,33,65,129,257,513],queries=[17,16,17,33,64,128,128],maximum=16384),
    dict(name='mixed_full_context_bound',nd=63,keys=[2404],queries=[128],maximum=131072),
]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def sources():
    paths=[Path(__file__).resolve(),PIPE/'bounded_prefill.py',PIPE/'dense_prefill.py',PIPE/'value_planes.py',HELPER/'bound.py',HELPER/'CPU_GATE_32g.json']
    return {str(p):sha(p) for p in paths}

def prepare():
    for name in ('gpu_probe.py',):ast.parse((BASE/name).read_text())
    sys.path.insert(0,str(HELPER));from bound import derive
    import torch
    assert not torch.cuda.is_initialized()
    import runpy
    runpy.run_path(str(HELPER/'check_cpu.py'),run_name='__main__')
    assert not torch.cuda.is_initialized()
    gate=json.loads((HELPER/'CPU_GATE_32g.json').read_text());assert gate['complete']
    fixtures=[]
    for case in CASES:
        n=len(case['keys']);nd=case['nd'];queries=[1]*nd+case['queries'];starts=[0]
        for q in queries:starts.append(starts[-1]+q)
        common=SimpleNamespace(num_reqs=nd+n,num_actual_tokens=starts[-1],max_seq_len=case['maximum'],
            seq_lens_cpu_upper_bound=torch.tensor([case['maximum']]*nd+case['keys'],dtype=torch.int32),
            query_start_loc_cpu=torch.tensor(starts,dtype=torch.int32))
        bound=derive(common,SimpleNamespace(num_decodes=nd,max_prefill_query=max(case['queries'])))
        assert bound is not None and bound.max_key_length==max(case['keys'])
        old=n*((case['maximum']+15)//16);new=n*((bound.max_key_length+15)//16)
        assert new<=old<=CAPACITY
        fixtures.append(dict(case=case['name'],prefills=n,decodes=nd,old_pages=old,new_pages=new,
            metadata_bound=bound.max_key_length,full_prefix_key_length_preserved=True))
    assert not torch.cuda.is_initialized()
    result=dict(complete=True,CPU_only=True,capacity_pages=CAPACITY,fixtures=fixtures,
        inherited_helper_complete=True,GPU_not_run=True,model_hook_not_installed=True,sources=sources())
    (BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='sources'},indent=2))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--output',type=Path);parser.add_argument('--stop-at-utc')
    args=parser.parse_args()
    if args.prepare_only:prepare();return
    assert args.output and args.stop_at_utc
    root=args.output.resolve();assert not root.exists()
    deadline=datetime.fromisoformat(args.stop_at_utc)
    seconds=int((deadline-datetime.now(timezone.utc)).total_seconds());assert seconds>0
    signal.alarm(min(seconds,1800))
    hardware=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hardware
    gate=json.loads((BASE/'CPU_GATE_32g.json').read_text());assert gate['complete'] and gate['sources']==sources()
    root.mkdir(parents=True);(root/'hardware_before.txt').write_text(hardware)
    snapshot=root/'source_snapshot'
    for path,expected in gate['sources'].items():
        dest=snapshot/Path(path).relative_to('/');dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(Path(path).read_bytes());assert sha(dest)==expected
    import torch
    torch.set_num_threads(8)
    sys.path.insert(0,str(PIPE));import bounded_prefill as bounded
    from value_planes import value_plane,clear_cache
    bounded.CAPACITY_PAGES=CAPACITY
    data=dict(complete=False,phase='numerical',runner_pid=os.getpid(),capacity_pages=CAPACITY,
        source_manifest=gate['sources'],CPU_gate_sha256=sha(BASE/'CPU_GATE_32g.json'),cases=[],
        model_integration=False,quality_not_claimed=True,official_score_not_claimed=True,
        stored_K_uint8_V_BF16=True,actual_native_BoundedDensePrefill_and_gather_unmodified=True,
        only_isolated_process_capacity_constant_set8192=True,
        started_utc=datetime.now(timezone.utc).isoformat())
    def save():
        (root/'result.tmp').write_text(json.dumps(data,indent=2)+'\n');(root/'result.tmp').replace(root/'result.json')
    save()
    codes=torch.arange(256,dtype=torch.int32)
    magnitude=codes&127;lut=(torch.where(magnitude>=8,(magnitude<<4)+14848,0)|((codes&128)<<8)).to(torch.int16)
    for case in CASES:
      for page_stride,offset in ((12288,0),(12544,128),(24576,256)):
        clear_cache();lengths=case['keys'];n=len(lengths);old_blocks=(case['maximum']+15)//16
        newmax=max(lengths);new_blocks=(newmax+15)//16
        physical=sum((length+15)//16 for length in lengths)+11
        raw=torch.full((offset+physical*page_stride,),37,dtype=torch.uint8,device='cuda')
        k=torch.as_strided(raw,(physical,16,2,128),(page_stride,256,128,1),offset)
        raw_v=torch.as_strided(raw,(physical,16,2,128),(page_stride,256,128,1),offset+4096)
        v=value_plane(raw_v)
        indices=torch.arange(physical*4096,dtype=torch.int64).reshape(physical,16,2,128)
        cpu_k=((indices*17+29)%256).to(torch.uint8)
        vb=((indices*19)%2048+15360)|(((indices%3)==0).to(torch.int64)<<15)
        cpu_v=vb.to(torch.int16).view(torch.bfloat16)
        k.copy_(cpu_k);v.copy_(cpu_v)
        table=torch.full((n,old_blocks+3),2147483647,dtype=torch.int32)
        active_ids=torch.randperm(physical,generator=torch.Generator().manual_seed(3721))
        begin=0
        for row,length in enumerate(lengths):
            count=(length+15)//16;table[row,:count]=active_ids[begin:begin+count];begin+=count
        used=torch.tensor(lengths,device='cuda',dtype=torch.int32);gpu_table=table.to('cuda')
        kw=dict(k=k,v=v,block_table=gpu_table,seqused_k=used,max_seqlen_k=case['maximum'])
        stages=[];records=[]
        for label,maximum in (('control',case['maximum']),('candidate',newmax)):
            stage=bounded.BoundedDensePrefill();stage.reserve('cuda')
            staged=stage(dict(kw,max_seqlen_k=maximum));assert staged is not None
            torch.cuda.synchronize()
            blocks=(maximum+15)//16;pages=n*blocks
            assert staged['max_seqlen_k']==maximum
            assert torch.equal(staged['block_table'].cpu(),torch.arange(pages,dtype=torch.int32).reshape(n,blocks))
            valid=torch.zeros(CAPACITY+1,16,dtype=torch.bool)
            digest=hashlib.sha256();checked=0
            for row,length in enumerate(lengths):
                count=(length+15)//16;ids=table[row,:count].to(torch.int64)
                expected_k=lut[cpu_k[ids].to(torch.int64)].reshape(-1,2,128)[:length]
                expected_v=cpu_v[ids].view(torch.int16).reshape(-1,2,128)[:length]
                start=row*blocks
                actual_k=staged['k'][start:start+count].view(torch.int16).reshape(-1,2,128)[:length].cpu()
                actual_v=staged['v'][start:start+count].view(torch.int16).reshape(-1,2,128)[:length].cpu()
                assert torch.equal(actual_k,expected_k) and torch.equal(actual_v,expected_v),(case['name'],label,row)
                for value in (actual_k,actual_v):digest.update(value.contiguous().numpy().tobytes());checked+=value.numel()
                for token in range(length):valid[start+token//16,token%16]=True
            invalid=(~valid).to('cuda')
            invalid_values=stage.storage.permute(0,2,1,3,4)[invalid]
            assert bool((invalid_values==17).all()),(case['name'],label,'inactive or guard overwritten')
            records.append(dict(arm=label,max_seqlen_k=maximum,blocks_per_sequence=blocks,pages=pages,
                checked_active_elements=checked,active_bits_sha256=digest.hexdigest(),
                all_active_K_V_bits_equal_independent_CPU_expected=True,
                all_inactive_tokens_and_extra_guard_preserved=True,
                poison_table_inactive_pages_masked=True,dense_table_exact=True,
                actual_kernel_records=stage.records))
            stages.append(stage)
        assert records[0]['active_bits_sha256']==records[1]['active_bits_sha256']
        # Isolated component graphs remove Python launch gaps; this is not model Graph evidence.
        graphs=[]
        for stage,maximum in zip(stages,(case['maximum'],newmax)):
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(16):stage(dict(kw,max_seqlen_k=maximum))
            for _ in range(3):graph.replay()
            graphs.append(graph)
        torch.cuda.synchronize();times=[[],[]]
        for group in range(7):
            for arm in ((0,1) if group%2==0 else (1,0)):
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(3):graphs[arm].replay()
                end.record();end.synchronize();times[arm].append(start.elapsed_time(end)/48)
        for record,timing in zip(records,times):record.update(kernel_ms_groups=timing,kernel_ms_median=statistics.median(timing))
        data['cases'].append(dict(name=case['name'],decodes=case['nd'],prefills=n,page_stride_bytes=page_stride,
            storage_offset_bytes=offset,layout_real_typed_V_alias=True,prefix_queries=case['queries'],
            full_key_lengths=lengths,arms=records,
            kernel_speedup_percent=(statistics.median(times[0])/statistics.median(times[1])-1)*100))
        save();print(json.dumps(dict(case=case['name'],stride=page_stride,control_ms=records[0]['kernel_ms_median'],
            candidate_ms=records[1]['kernel_ms_median'],full_active_and_guard_pass=True)),flush=True)
        del graphs,stages,stage,staged,invalid_values,invalid,raw,k,v,raw_v,gpu_table,used
        clear_cache();torch.cuda.empty_cache()
    assert len(data['cases'])==24
    data.update(complete=True,phase='complete',all24cases_every_active_bit_and_inactive_guard_passed=True,
        completed_utc=datetime.now(timezone.utc).isoformat(),model_speedup_not_inferred=True)
    assert gate['sources']==sources();save()

if __name__=='__main__':main()
