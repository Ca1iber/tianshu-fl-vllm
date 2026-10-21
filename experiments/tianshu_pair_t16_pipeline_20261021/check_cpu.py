# SPDX-License-Identifier: Apache-2.0
"""Actual launch-argument path and current407 binding, without running a GPU."""
import ast,hashlib,json
from pathlib import Path
from types import SimpleNamespace
import torch
import source
import launch
BASE=Path(__file__).resolve().parent
PARENT=Path('/workspace/benchmark-results/20261020_k8v16_pair_load_GPU_tianshu/pair_rows_reader/generated_pair_rows.py')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
assert not torch.cuda.is_initialized()
for p in BASE.glob('*.py'):ast.parse(p.read_text())
proofs={mode:source.render(PARENT,mode)[1] for mode in source.MODES}
assert len(proofs)==7 and sha(PARENT)==source.PARENT_SHA
qualified=read(BASE.parent/'tianshu_prefill_int8_official_results_20261021/OFFICIAL_GATE_32g.json')
assert qualified['complete'] and qualified['math_accuracy']>.95
original=Path(qualified['raw_root'])/'comparison.json';current=read(original)['profile']
assert current['active_decode_source_sha256']==source.PARENT_SHA
c64=[r for r in current['pair_load_record']['launches'].values() if r['sequences']==64]
assert c64 and all((r['tile'],r['warps'],r['stages'],r['segments'])==(16,1,1,4) for r in c64)
reference_path=BASE.parent/'tianshu_qkv_prefill_exact_reference_20261021/CPU_GATE_32g.json'
reference=read(reference_path);assert reference['complete'] and reference['runtime_source_count']==407
sources=dict(reference['actual_reference_runtime_sources']);assert len(sources)==407
for p,h in sources.items():assert sha(p)==h
native_proof=launch.native.source_proof();assert native_proof['normalized_launch_AST_unchanged'] and native_proof['reducer_AST_unchanged']
class Spy:
    def __getitem__(self,grid):
        def invoke(*args,**kw):
            assert not args;self.kw=dict(kw);self.grid=grid
            return SimpleNamespace(metadata=SimpleNamespace(num_warps=kw['num_warps'],num_stages=kw['num_stages'],shared=0),kernel=b'CPU launch argument stub only',hash='CPU_stub',n_regs=None,n_spills=None)
        return invoke
cases=transition=stub_checks=0
lengths=(1,16,33,64,128,129,4096,5120,8192,8193,16384)
for n in range(1,65):
 for length in lengths:
    raw=dict(IS_3D=True,BLOCK_SIZE=16,HEAD_SIZE=128,num_query_heads=16,num_queries_per_kv=8,KV_QUANT_MODE=0,
        USE_TD=False,USE_TD_QO=False,query_ptr=SimpleNamespace(dtype=torch.bfloat16),key_cache_ptr=SimpleNamespace(dtype=torch.uint8),
        value_cache_ptr=SimpleNamespace(dtype=torch.bfloat16),BLOCK_M=16,BLOCK_Q=2,num_seqs=n,TILE_SIZE=128,num_warps=8,num_stages=7)
    baseline=None
    for mode in source.MODES:
        config=launch.choose_config(mode,n,length);segments=config['segments'];tile=config['TILE_SIZE']
        active=4 if n>24 and length<=8192 else segments
        tiles=(length+active*tile-1)//(active*tile);counts=[0]*length
        for segment in range(segments):
            start=segment*tiles*tile
            for token in range(start,min(length,start+tiles*tile)):counts[token]+=1
        assert all(c==1 for c in counts) and segments>=((length+tiles*tile-1)//(tiles*tile));cases+=1
        if n>24:
            assert launch.choose_config(mode,n,1)['segments']==segments;transition+=1
        spy=Spy();observed=launch.ObservedKernel(spy,mode);producer=launch.native.Producer(observed)
        producer.decode_config=config;producer.pure_decode=True;kw=dict(raw,NUM_SEGMENTS_PER_SEQ=segments)
        producer[(n//2+n,2,segments)](**kw)
        expected=source.configuration(mode,n)
        assert spy.kw['TILE_SIZE']==16 and spy.kw['num_warps']==expected['num_warps'] and spy.kw['num_stages']==expected['num_stages']
        assert producer.last_record['stages']==expected['num_stages'] and observed.last['stages_applied_after_legacy_default']
        normalized={k:v for k,v in spy.kw.items() if k not in ('num_warps','num_stages')}
        if mode=='current':baseline=normalized
        else:assert normalized==baseline
        stub_checks+=1
assert (cases,transition,stub_checks)==(4928,3080,4928)
paths=[*BASE.glob('*.py'),*BASE.glob('*.sh'),BASE/'PLAN_32g.md',PARENT,reference_path,original,
    BASE.parent/'tianshu_prefill_int8_official_results_20261021/OFFICIAL_GATE_32g.json',launch.NATIVE_PATH,launch.native.LEGACY,launch.native.POLICY,
    launch.NATIVE_PATH.parent/'guard.py',Path('/usr/local/lib/python3.12/site-packages/vllm/v1/attention/ops/triton_attention_helpers.py')]
sources.update({str(p.resolve()):sha(p) for p in paths})
assert not torch.cuda.is_initialized()
gate=dict(complete=True,CPU_only=True,GPU_not_executed=True,CUDA_not_initialized=True,coverage_cases=cases,
    capture_length1_to_runtime_long_length_transition_cases=transition,actual_legacy_Producer_to_JIT_argument_stub_checks=stub_checks,
    CPU_stub_is_not_GPU_compile_proof=True,all_selected_stages_survive_legacy_default=True,only_warp_and_stages_differ_in_actual_launch=True,
    all_valid_tokens_covered_once=True,all_active_segment_scratch_within_reserved_capacity=True,n1to24_configuration_unchanged=True,
    current_407_live_sources_verified=True,current_407_B64_launch_verified=True,native_launch_AST_proof=native_proof,provenance=proofs,sources=sources)
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps({k:v for k,v in gate.items() if k not in ('sources','provenance','native_launch_AST_proof')}))
