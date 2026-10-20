# SPDX-License-Identifier: Apache-2.0
"""Opaque GateUp+activation in actual LlamaMLP; native down projection retained."""
import atexit,functools,hashlib,importlib.util,inspect,json,os,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

def install_fusion(banks,record,save,dummy_get):
    import torch
    from flag_gems.fused.silu_and_mul import silu_and_mul_out
    from vllm.model_executor.models.llama import LlamaMLP
    mode=os.environ['TIANSHU_MLP_FUSION_MODE'];assert mode in ('control','candidate')
    component=BASE.parent/'tianshu_int8_gateup_silu_20261021'
    gate=read(component/'GPU_GATE_32g.json');assert gate['complete'] and gate['independently_CPU_rechecked'] and gate['full_actual_timed_outputs_bitwise_verified']
    assert gate['checked_configurations']==24 and gate['changed_input_Graph_phases']==72
    for path in (component/'kernel.py',BASE.parent/'tianshu_dense_tile_padding_20261021/kernel_padded.py'):assert sha(path)==gate['frozen_sources'][str(path)]
    implementation=load('_tianshu_actual_MLP_int8_fusion',component/'kernel.py')
    padding=load('_tianshu_actual_MLP_int8_padding',BASE.parent/'tianshu_dense_tile_padding_20261021/kernel_padded.py')
    choices={row['M']:row for row in read(component/'SELECTED_CONFIGURATIONS_32g.json')}
    configs={m:choices[m]['configuration'] for m in (8,64)}
    assert configs[8]==dict(bm=32,bn=64,bk=128,warps=2,stages=1) and configs[64]==dict(bm=64,bn=64,bk=128,warps=2,stages=1)
    binary_proofs={}
    fallback_file=Path(inspect.getsourcefile(silu_and_mul_out)).resolve()
    native_forward=LlamaMLP.forward
    record.update(fallback_activation_policy='FlagGems_scalar_for_all_nonfused_shapes',fallback_activation_source=str(fallback_file),fallback_activation_sha256=sha(fallback_file),selected_native_activation_overridden_inside_custom_op=True)
    native_file=Path(inspect.getsourcefile(native_forward)).resolve()
    record.update(mlp_fusion_mode=mode,fused_launches={},MLP_bindings=[],MLP_native_fallback_calls={},
        fusion_source=str(Path(__file__).resolve()),fusion_source_sha256=sha(__file__),fusion_kernel_source=str(component/'kernel.py'),fusion_kernel_sha256=sha(component/'kernel.py'),
        padding_kernel_source=str(BASE.parent/'tianshu_dense_tile_padding_20261021/kernel_padded.py'),padding_kernel_sha256=sha(BASE.parent/'tianshu_dense_tile_padding_20261021/kernel_padded.py'),
        original_LlamaMLP_forward_source=str(native_file),original_LlamaMLP_forward_sha256=sha(native_file),fused_supported_M=[8,64],
        unsupported_M_preserves_original_GateUp_precision=True,down_projection_unchanged=True,fusion_preserves_BF16_projection_round=True,
        fresh_ephemeral_buffers=True,configurations={str(m):cfg for m,cfg in configs.items()})
    @torch.library.custom_op('tianshu_int8_mlp_fusion::gate_activation',mutates_args=(),device_types='cuda')
    def gate_activation(x:torch.Tensor,weight:torch.Tensor,packed:torch.Tensor,scales:torch.Tensor,bank_id:int)->torch.Tensor:
        bank=banks[bank_id];m,k=x.shape
        assert bank['role']=='gateup' and weight.shape==(12288,2048) and x.dtype==torch.bfloat16
        capturing=torch.cuda.is_current_stream_capturing()
        if mode!='candidate' or m not in configs:
            projection,bias=bank['layer'](x);assert bias is None
            output=torch.empty((m,6144),device=x.device,dtype=x.dtype);silu_and_mul_out(projection[:,:6144],projection[:,6144:],output)
            key=str((bank_id,m,capturing));row=record['MLP_native_fallback_calls'].setdefault(key,dict(bank_id=bank_id,prefix=bank['prefix'],M=m,actually_inside_capture=capturing,activation_source=str(fallback_file),activation_sha256=sha(fallback_file),activation_function='silu_and_mul_out',calls=0))
            row['calls']+=1
            if bank["prefix"]=="model.layers.41.mlp.gate_up_proj" and not capturing and row["calls"]&(row["calls"]-1)==0:save()
            return output
        target=32 if m==8 else 64
        q=torch.empty((target,k),device=x.device,dtype=torch.int8);ascale=torch.empty(target,device=x.device,dtype=torch.float32)
        output=torch.empty((target,6144),device=x.device,dtype=torch.bfloat16)
        dummy=dummy_get();assert dummy is not None
        quant=padding.quantize(x,q,ascale)
        compiled=implementation.fused(q,packed,ascale,scales,output,dummy,dummy.view(torch.bfloat16),configs[m])
        if compiled.hash not in binary_proofs:binary_proofs[compiled.hash]=hashlib.sha256(compiled.kernel).hexdigest()
        trace=getattr(sys.modules.get('_tianshu_long_segments_actual_graph_trace'),'STATE',{})
        graph_key=trace.get('active_graph');key=str((bank_id,m,capturing,graph_key))
        row=record['fused_launches'].setdefault(key,dict(bank_id=bank_id,role='gateup',prefix=bank['prefix'],M=m,K=k,N=12288,I=6144,
            allocated_M=target,actually_inside_capture=capturing,graph_key=graph_key,configuration=configs[m],
            fused_kernel_sha256=record['fusion_kernel_sha256'],padding_kernel_sha256=record['padding_kernel_sha256'],
            quantizer_cache_hash=quant.hash,compiled_cache_hash=compiled.hash,compiled_binary_sha256=binary_proofs[compiled.hash],n_regs=getattr(compiled,'n_regs',None),n_spills=getattr(compiled,'n_spills',None),
            allocated_q_shape=list(q.shape),allocated_scale_shape=list(ascale.shape),allocated_output_shape=list(output.shape),returned_output_shape=[m,6144],
            actual_ephemeral_buffer_bytes=q.numel()+4*ascale.numel()+2*output.numel(),matrix_pointer_alignment_mod256=[t.data_ptr()%256 for t in (q,packed,ascale,scales,output,dummy)],calls=0))
        row['calls']+=1
        if bank["prefix"]=="model.layers.41.mlp.gate_up_proj" and not capturing and row["calls"]&(row["calls"]-1)==0:save()
        return output[:m]
    @gate_activation.register_fake
    def fake(x,weight,packed,scales,bank_id):return torch.empty((x.shape[0],weight.shape[0]//2),device=x.device,dtype=x.dtype)
    @functools.wraps(native_forward)
    def forward(owner,x):
        if not hasattr(owner,'_tianshu_mlp_fusion_bank_id') or x.ndim!=2 or x.dtype!=torch.bfloat16 or x.stride(-1)!=1:
            return native_forward(owner,x)
        layer=owner.gate_up_proj
        value=gate_activation(x,layer.weight,layer._tianshu_dense_w8_packed,layer._tianshu_dense_w8_scale,owner._tianshu_mlp_fusion_bank_id)
        value,_=owner.down_proj(value)
        return value
    LlamaMLP.forward=forward
    def bind_model(model):
        assert not record['MLP_bindings']
        for name,module in model.named_modules():
            if not isinstance(module,LlamaMLP):continue
            bank_id=module.gate_up_proj._tianshu_dense_w8_bank;bank=banks[bank_id]
            assert bank['role']=='gateup' and bank['prefix']==name+'.gate_up_proj' and module.gate_up_proj.bias is None
            assert not hasattr(module,'_tianshu_mlp_fusion_bank_id');module._tianshu_mlp_fusion_bank_id=bank_id
            bank['activation']=module.act_fn
            selected=getattr(module.act_fn,'_forward_method',module.act_fn.forward);selected=inspect.unwrap(selected)
            activation_file=Path(inspect.getsourcefile(selected)).resolve()
            record['MLP_bindings'].append(dict(bank_id=bank_id,prefix=bank['prefix'],module=name,MLP_class=type(module).__module__+'.'+type(module).__name__,
                activation_class=type(module.act_fn).__module__+'.'+type(module.act_fn).__name__,selected_activation_qualname=getattr(selected,"__qualname__",type(selected).__qualname__),selected_activation_source=str(activation_file),selected_activation_sha256=sha(activation_file),
                installed_forward_source=str(Path(__file__).resolve()),installed_forward_sha256=sha(__file__),down_projection_class=type(module.down_proj).__module__+'.'+type(module.down_proj).__name__))
        assert len(record['MLP_bindings'])==42
        save()
    atexit.register(save);save();return bind_model
