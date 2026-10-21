# SPDX-License-Identifier: Apache-2.0
"""M64-only native FX fusion, unchanged dynamic prefill and M8 decode."""
import atexit,hashlib,importlib.util,json,os,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
COMPONENT=BASE.parent/'tianshu_nn_silu_two_dot_20261021'
RECORD=None

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

def validate_capture(nn_record):
    assert RECORD is not None
    if RECORD['mode']=='control':
        routes=[r for r in nn_record['routes'].values() if r['M']==64 and r['actually_inside_capture']];assert len({r['prefix'] for r in routes})==42
    else:
        routes=[r for r in RECORD['routes'].values() if r['M']==64 and r['actually_inside_capture']];assert len({r['prefix'] for r in routes})==42
        assert all(r['role']=='gateup' and r['implementation']=='NN_two_dot_fused_SiLU' and r['calls']>0 and r['graph_key'] is not None for r in routes)
    RECORD['all_42_M64_real_capture_routes_verified']=True
    nn_record['M64_capture_validation']='original NN' if RECORD['mode']=='control' else 'actual NN_two_dot_fused_SiLU routes; no fictitious old MM launches'
    nn_record['M64_capture_validator_source']=str(BASE/'policy.py');nn_record['M64_capture_validator_sha256']=sha(BASE/'policy.py')
    SAVE()

def install(worker,mode,directory):
    global RECORD
    import torch
    from fx_pass import FusionPass
    assert RECORD is None and mode in ('control','candidate')
    cfg=worker.vllm_config
    assert cfg.speculative_config is None and cfg.model_config.model=='/workspace/MiniCPM5-2B'
    assert cfg.scheduler_config.max_num_seqs==64 and cfg.scheduler_config.max_num_batched_tokens==4096
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    gate=json.loads((COMPONENT/'GPU_GATE_32g.json').read_text());assert gate['complete'] and gate['independently_CPU_rechecked'] and gate['full_actual_timed_outputs_bitwise_verified']
    assert all(sha(p)==h for p,h in gate['frozen_sources'].items())
    selected=next(r for r in gate['best_by_shape'] if r['M']==64 and not r['strided']);config=selected['configuration'];assert config==dict(bm=64,bn=64,bk=128,warps=4,stages=1,mode='two_dot')
    fused_kernel=load('_nn_silu_fx_fused_component',COMPONENT/'kernel.py')
    native=load('_nn_silu_fx_native_quantizer',BASE.parent/'tianshu_int8_weight_nn_20261021/kernel_nn.py')
    record=dict(mode=mode,installed=True,all_42_M64_real_capture_routes_verified=False,component_gate_sha256=sha(COMPONENT/'GPU_GATE_32g.json'),component_expected_binary_sha256=selected['actual_timed_compiled']['binary_sha256'],configuration=config,routes={},policy_source=str(BASE/'policy.py'),policy_sha256=sha(__file__),kernel_source=str(COMPONENT/'kernel.py'),kernel_sha256=sha(COMPONENT/'kernel.py'),quantizer_source=str(BASE.parent/'tianshu_int8_weight_nn_20261021/kernel_nn.py'),quantizer_sha256=sha(BASE.parent/'tianshu_int8_weight_nn_20261021/kernel_nn.py'),no_additional_packed_weight_payload=True,M8_decode_unchanged=True,dynamic_prefill_graph_unchanged=True,model_quality_not_yet_evaluated=True,activation_FP32_operation_order_differs_from_native_sigmoid_multiply=True,Python_launch_counts_are_not_Graph_replays=True)
    RECORD=record;dummy=None
    def save():
        path=directory/f'route_{os.getpid()}.json';temp=path.with_suffix('.tmp');temp.write_text(json.dumps(record,indent=2)+'\n');temp.replace(path)
    globals()['SAVE']=save
    @torch.library.custom_op('tianshu_nn_silu_fx::fused',mutates_args=(),device_types='cuda')
    def fused(x:torch.Tensor,weight:torch.Tensor,packed:torch.Tensor,scales:torch.Tensor,bank_id:int)->torch.Tensor:
        nonlocal dummy
        assert mode=='candidate' and x.shape==(64,2048) and x.dtype==torch.bfloat16 and x.stride(-1)==1
        assert weight.shape==(12288,2048) and packed.shape==(12288,2048) and packed.stride()==(1,12288) and packed.T.is_contiguous()
        assert scales.shape==(12288,) and scales.dtype==torch.float32
        bank=sys.modules['_nn_silu_fx_NN_policy'].POINTERS[packed.data_ptr()];assert bank['bank_id']==bank_id
        capturing=torch.cuda.is_current_stream_capturing()
        if dummy is None:
            assert not capturing;dummy=torch.empty(1,device=x.device,dtype=torch.int32)
        q=torch.empty((64,2048),device=x.device,dtype=torch.int8);s=torch.empty(64,device=x.device,dtype=torch.float32);out=torch.empty((64,6144),device=x.device,dtype=torch.bfloat16)
        native.quantize(x,q,s);compiled=fused_kernel.fused(q,packed.T,s,scales,out,dummy,dummy.view(torch.bfloat16),config)
        trace=sys.modules.get('_tianshu_long_segments_actual_graph_trace');graph_key=getattr(trace,'STATE',{}).get('active_graph');key=str((bank_id,64,capturing,graph_key))
        row=record['routes'].setdefault(key,dict(bank_id=bank_id,prefix=bank['prefix'],role='gateup',M=64,K=2048,N=12288,output_columns=6144,actually_inside_capture=capturing,graph_key=graph_key,implementation='NN_two_dot_fused_SiLU',configuration=config,kernel_source=record['kernel_source'],kernel_sha256=record['kernel_sha256'],calls=0))
        if 'binary_sha256' not in row:
            binary=hashlib.sha256(compiled.kernel).hexdigest();row.update(binary_sha256=binary,component_binary_exact_match=(binary==record['component_expected_binary_sha256']),cache_hash=compiled.hash,n_regs=compiled.n_regs,n_spills=compiled.n_spills,shared=compiled.metadata.shared)
        row['calls']+=1
        if not capturing and bank['prefix']=='model.layers.41.mlp.gate_up_proj' and row['calls']&(row['calls']-1)==0:save()
        return out
    @fused.register_fake
    def fake(x,weight,packed,scales,bank_id):return torch.empty((x.shape[0],6144),device=x.device,dtype=x.dtype)
    cc=cfg.compilation_config;assert 'post_grad_custom_post_pass' not in cc.inductor_compile_config
    old_sizes=list(cc.compile_sizes or []);assert all(isinstance(n,int) for n in old_sizes)
    cc.compile_sizes=sorted(set(old_sizes)|{64});record.update(original_compile_sizes=old_sizes,compile_sizes=cc.compile_sizes)
    cc.inductor_compile_config['post_grad_custom_post_pass']=FusionPass(mode,directory/'fx_pass',COMPONENT/'kernel.py')
    from vllm_fl.worker.model_runner import ModelRunnerFL
    import functools
    original_capture=ModelRunnerFL.capture_model
    @functools.wraps(original_capture)
    def capture(owner,*args,**kwargs):
        nonlocal dummy
        cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone()
        nn=sys.modules['_nn_silu_fx_NN_policy'];assert len(nn.LAYERS)==42
        layer=next(layer for layer in nn.LAYERS.values() if layer.prefix=='model.layers.0.mlp.gate_up_proj')
        with torch.inference_mode():
            dummy=torch.empty(1,device=layer.weight.device,dtype=torch.int32)
            if mode=='candidate':
                x=torch.zeros((64,2048),device=layer.weight.device,dtype=torch.bfloat16)
                out=fused(x,layer.weight,layer._tianshu_dense_w8_packed,layer._tianshu_dense_w8_scale,int(layer._tianshu_dense_w8_bank))
                torch.cuda.synchronize();assert bool((out==0).all());del x,out
        assert torch.equal(torch.get_rng_state(),cpu_rng) and torch.equal(torch.cuda.get_rng_state(),gpu_rng)
        record['warm_before_capture_RNG_unchanged']=True;save()
        value=original_capture(owner,*args,**kwargs)
        assert record['all_42_M64_real_capture_routes_verified']
        record.update(completed=True,after_original_capture_and_prefill_warmup=True);save();return value
    ModelRunnerFL.capture_model=capture
    atexit.register(save);save()
