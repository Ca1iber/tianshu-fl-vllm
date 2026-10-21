# SPDX-License-Identifier: Apache-2.0
"""Reversible source proof, exact current binding and common producer/reducer fallback."""
import ast,hashlib,json,random
from pathlib import Path
from types import SimpleNamespace
import torch
import source,launch,pair_selector
BASE=Path(__file__).resolve().parent
PARENT=BASE/'qualified_current_reader.py'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not torch.cuda.is_initialized()
for path in BASE.glob('*.py'):ast.parse(path.read_text())
proofs={mode:source.render(PARENT,mode)[1] for mode in source.MODES}
assert len(proofs)==4 and sha(PARENT)==source.PARENT_SHA
for mode,proof in proofs.items():
    if 'late_v' in mode:assert proof['late_V_proof']['LATE_false_AST_identical_to_parent'] and proof['late_V_proof']['exact_inverse_recovers_parent_bytes']
    if 'addr32' in mode:assert proof['only_integer_width_AST_changes_after_scheduling'] and proof['physical_pointer_and_slot_mapping_not_narrowed']
reference_path=BASE.parent/'tianshu_qkv_prefill_exact_reference_20261021/CPU_GATE_32g.json'
reference=read(reference_path);assert reference['complete'] and reference['runtime_source_count']==407
sources=dict(reference['actual_reference_runtime_sources']);assert len(sources)==407
for p,h in sources.items():assert sha(p)==h
qualified_path=BASE.parent/'tianshu_prefill_int8_official_results_20261021/OFFICIAL_GATE_32g.json';qualified=read(qualified_path)
assert qualified['complete'] and qualified['math_accuracy']>.95
comparison_path=Path(qualified['raw_root'])/'comparison.json';current=read(comparison_path)['profile']
assert current['active_decode_source_sha256']==source.PARENT_SHA
c64=[r for r in current['pair_load_record']['launches'].values() if r['sequences']==64]
assert c64 and all((r['tile'],r['warps'],r['stages'],r['segments'])==(16,1,1,4) for r in c64)
native_proof=launch.native.source_proof();assert native_proof['normalized_launch_AST_unchanged']
class TensorMeta:
    def __init__(self,shape,strides,dtype):self.shape=shape;self.strides=strides;self.dtype=dtype
    def stride(self):return self.strides
class Spy:
    def __init__(self,name):self.name=name;self.calls=0
    def __getitem__(self,grid):
        def run(*args,**kw):
            assert not args;self.calls+=1;self.kw=dict(kw);self.grid=grid
            self.result=SimpleNamespace(metadata=SimpleNamespace(num_warps=kw.get('num_warps',1),num_stages=kw.get('num_stages',1),shared=0),kernel=b'CPU argument stub, not GPU proof',hash=self.name,n_regs=None,n_spills=None)
            return self.result
        return run

def arguments(n,length):
    width=(length+15)//16;pages=n*width+2
    return dict(IS_3D=True,BLOCK_SIZE=16,HEAD_SIZE=128,HEAD_SIZE_PADDED=128,BLOCK_M=16,BLOCK_Q=2,num_query_heads=16,num_queries_per_kv=8,
        KV_QUANT_MODE=0,USE_TD=False,USE_TD_QO=False,num_seqs=n,TILE_SIZE=128,num_warps=8,num_stages=4,
        query_ptr=TensorMeta((n,16,128),(2048,128,1),torch.bfloat16),key_cache_ptr=TensorMeta((pages,16,2,128),(12288,256,128,1),torch.uint8),
        value_cache_ptr=TensorMeta((pages,16,2,128),(6144,256,128,1),torch.bfloat16),output_ptr=TensorMeta((n,16,128),(2048,128,1),torch.bfloat16),
        segm_output_ptr=TensorMeta((n,16,64,128),(131072,8192,128,1),torch.float32),segm_max_ptr=TensorMeta((n,16,64),(1024,64,1),torch.float32),
        segm_expsum_ptr=TensorMeta((n,16,64),(1024,64,1),torch.float32),block_tables_ptr=TensorMeta((n,width),(width,1),torch.int32),
        seq_lens_ptr=TensorMeta((n,),(1,),torch.int32),query_start_len_ptr=TensorMeta((n+1,),(1,),torch.int32),output_stride_0=2048,output_stride_1=128)

def stack(mode):
    wide,candidate,wide_reduce,candidate_reduce=[Spy(s) for s in ('wide','candidate','wide_reduce','candidate_reduce')]
    selector=pair_selector.Selector(wide,candidate,mode,source.PARENT_SHA,proofs[mode]['generated_sha256'])
    producer=launch.native.Producer(selector);reducer=pair_selector.Reducer(wide_reduce,candidate_reduce,producer,selector)
    return selector,producer,reducer,wide,candidate,wide_reduce,candidate_reduce
cases=transition=stub_checks=0
lengths=(1,16,33,64,128,129,4096,5120,8192,8193,16384)
for n in range(1,65):
 for length in lengths:
  for mode in source.MODES:
    config=launch.choose_config(mode,n,length);segments=config['segments'];tile=config['TILE_SIZE'];active=4 if n>24 and length<=8192 else segments
    tiles=(length+active*tile-1)//(active*tile);counts=[0]*length
    for segment in range(segments):
        start=segment*tiles*tile
        for token in range(start,min(length,start+tiles*tile)):counts[token]+=1
    assert all(c==1 for c in counts);cases+=1
    if n>24:assert launch.choose_config(mode,n,1)['segments']==segments;transition+=1
    selector,producer,reducer,w,c,wr,cr=stack(mode);producer.decode_config=config;producer.pure_decode=True
    kw=dict(arguments(n,length),NUM_SEGMENTS_PER_SEQ=segments);grid=(n//2+n,2,segments)
    result=producer[grid](**kw);chosen=mode!='current' and n>24
    assert selector.last_selection['candidate_selected']==chosen and selector.last_selection['signed_element_guard_passed']
    assert selector.last_selection['late_V']==bool(chosen and 'late_v' in mode)
    expected_width=32 if chosen and 'addr32' in mode else 64
    assert selector.last_selection['address_bits']==expected_width
    selected=c if chosen else w;assert result is selected.result and selected.grid==grid
    marker=object();rgrid=(n,16);reduced=reducer[rgrid](marker=marker)
    expected_reducer=cr if expected_width==32 else wr
    assert reduced is expected_reducer.result and expected_reducer.grid==rgrid and expected_reducer.kw==dict(marker=marker,TILE_SIZE=16)
    assert reducer.last['address_bits']==expected_width and reducer.last['producer_and_reducer_guard_decision_consistent'];stub_checks+=1
assert (cases,transition,stub_checks)==(2816,1760,2816)
base=dict(arguments(64,4096),NUM_SEGMENTS_PER_SEQ=4,TILE_SIZE=16,num_warps=1,num_stages=1);unsafe=[]
for updates in ({'TILE_SIZE':32},{'num_warps':2},{'num_stages':2},{'output_stride_0':2**31},{'IS_3D':False},{'KV_QUANT_MODE':2},{'USE_TD':True},{'USE_TD_QO':True},{'USE_SINKS':True},{'USE_MM_PREFIX':True},{'USE_QQ_BIAS':True},{'USE_SOFTCAP':True},{'USE_FP8':True},{'SLIDING_WINDOW':1024},{'CHUNK_LOOKBACK':1},{'num_seqs':24},{'num_seqs':65},
    {'query_ptr':TensorMeta((64,16,128),(2**31,128,1),torch.bfloat16)},
    {'key_cache_ptr':TensorMeta((2**31,16,2,128),(12288,256,128,1),torch.uint8)},
    {'value_cache_ptr':TensorMeta((64,16,2,128),(-6144,256,128,1),torch.bfloat16)},
    {'query_ptr':TensorMeta((64,16,128),(2048,128,1),torch.float32)}):unsafe.append(dict(base,**updates))
fallbacks=0
for mode in source.MODES:
 for kw in unsafe:
    selector,producer,reducer,w,c,wr,cr=stack(mode);grid=(96,2,4);original=dict(kw)
    result=selector[grid](**kw);assert not selector.last_selection['candidate_selected'] and result is w.result and w.kw==original and w.grid==grid
    producer.tile=16;marker=object();reduced=reducer[(64,16)](marker=marker)
    assert reduced is wr.result and reducer.last['address_bits']==64 and wr.kw==dict(marker=marker,TILE_SIZE=16);fallbacks+=1
rng=random.Random(202610211405);addresses=0
for pages in (16,32768,65536):
 for gap in (1,2):
  for _ in range(4096):
    p=rng.randrange(pages);head=rng.randrange(2);slot=rng.randrange(16);d=rng.randrange(128)
    expected=p*(12288*gap)+head*128+slot*256+d
    word=p*(12288*gap//2)+head*64+slot*128+d//2
    assert 2*word+d%2==expected and 0<=expected<2**31 and 0<=word<2**31;addresses+=1
paths=[*BASE.glob('*.py'),*BASE.glob('*.sh'),BASE/'PLAN_32g.md',reference_path,qualified_path,comparison_path,
    launch.NATIVE_PATH,launch.native.LEGACY,launch.native.POLICY,source.OLD,pair_selector.GUARD,
    Path('/usr/local/lib/python3.12/site-packages/vllm/v1/attention/ops/triton_attention_helpers.py')]
sources.update({str(p.resolve()):sha(p) for p in paths})
assert not torch.cuda.is_initialized()
gate=dict(complete=True,CPU_only=True,GPU_not_executed=True,CUDA_not_initialized=True,coverage_cases=cases,
    capture_length1_to_runtime_long_length_transition_cases=transition,actual_producer_reducer_selection_stub_checks=stub_checks,
    actual_wide_and_unsupported_common_fallback_checks=fallbacks,independent_pair_word_to_byte_address_samples=addresses,
    CPU_stubs_not_GPU_compile_or_wide_allocation_proof=True,producer_and_reducer_share_guarded_fallback=True,
    current407_and_real_B64_launch_verified=True,LATE_default_false_AST_and_exact_inverse_verified=True,
    integer_only_AST_change_verified=True,all_valid_tokens_covered_once=True,n1to24_configuration_unchanged=True,
    provenance=proofs,native_launch_AST_proof=native_proof,sources=sources)
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps({k:v for k,v in gate.items() if k not in ('sources','provenance','native_launch_AST_proof')}))
