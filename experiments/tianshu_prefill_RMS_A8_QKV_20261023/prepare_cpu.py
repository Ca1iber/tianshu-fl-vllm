# SPDX-License-Identifier: Apache-2.0
"""Read checkpoint and independently reconstruct three original QKV INT8 packs."""
import ast,hashlib,json,struct,sys
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;EXP=BASE.parent;MODEL=Path('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors');ROOT=Path('/workspace/benchmark-results/20261023_prefill_RMS_A8_QKV_CPU_tianshu')
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def h(array):return hashlib.sha256(array.tobytes()).hexdigest()
assert not ROOT.exists() and not (BASE/'CPU_PREPARE_GATE_32g.json').exists();ROOT.mkdir()
selection_path=EXP/'tianshu_native_int8_attention_service_20261022/selected_plan.json';selection=json.loads(selection_path.read_text());assert selection['complete'] and selection['rows_by_role']['QKV']==[128,512,4096]
state=json.loads(Path('/workspace/benchmark-results/20261023_RMS_LM_math64_tianshu/result.json').read_text());assert state['phase']=='complete' and state['correct']==101 and state['frozen_sources'][str(selection_path)]==sha(selection_path)
parent_path=EXP/'tianshu_IR_RMS_A8_20261023/INDEPENDENT_CPU_GATE_32g.json';parent=json.loads(parent_path.read_text());assert parent['complete'] and parent['numeric_cases']==108
banks=[]
with MODEL.open('rb') as checkpoint:
 length=struct.unpack('<Q',checkpoint.read(8))[0];header=json.loads(checkpoint.read(length));offset=8+length
 def read(name,shape):
  meta=header[name];assert meta['dtype']=='BF16' and meta['shape']==list(shape);lo,hi=meta['data_offsets'];assert hi-lo==int(np.prod(shape))*2 and offset+hi<=MODEL.stat().st_size;checkpoint.seek(offset+lo);raw=checkpoint.read(hi-lo);assert len(raw)==hi-lo;return np.frombuffer(raw,dtype='<u2').reshape(shape)
 for layer in (0,20,41):
  prefix=f'model.layers.{layer}.self_attn.qkv_proj';proof=next(x for x in selection['banks'] if x['prefix']==prefix);assert proof['role']=='QKV' and proof['N']==2560 and proof['K']==2048
  parts=[read(f'model.layers.{layer}.self_attn.{role}_proj.weight',(rows,2048)) for role,rows in (('q',2048),('k',256),('v',256))];bits=np.concatenate(parts,axis=0);original=h(bits);assert original==proof['original_BF16_sha256']
  weight=(bits.astype(np.uint32)<<16).view(np.float32);scales=(np.maximum(np.max(np.abs(weight),axis=1),np.float32(1e-8))/np.float32(127)).astype('<f4');packed=np.clip(np.rint(weight/scales[:,None]),-127,127).astype(np.int8);assert h(packed)==proof['packed_sha256'] and h(scales)==proof['scales_sha256']
  norm_name=f'model.layers.{layer}.input_layernorm.weight';norm=read(norm_name,(2048,));norm_path=ROOT/f'input_norm_layer{layer}.u16';norm_path.write_bytes(norm.tobytes());assert sha(norm_path)==h(norm)
  banks.append(dict(layer=layer,has_residual=layer!=0,prefix=prefix,N=2560,K=2048,original_BF16_sha256=original,packed_NT_sha256=h(packed),scale_sha256=h(scales),CPU_exactly_matches_original_live_pack=True,norm_name=norm_name,norm_path=str(norm_path),norm_sha256=h(norm),full_weight_copy_written=False))
  del parts,bits,weight,scales,packed
checkpoint_sha=sha(MODEL);assert checkpoint_sha=='14fb8e7f0a18d53d1f239773758bf581cee7e456a4523a54622c3a245b64402c'
paths=[BASE/'prepare_cpu.py',BASE/'PLAN_32g.md',selection_path,parent_path,EXP/'tianshu_IR_RMS_A8_20261023/kernel.py',EXP/'tianshu_dense_w8a8_20261020/kernel.py',EXP/'tianshu_native_int8_lm64_20261022/kernel.py',EXP/'tianshu_w8_gpu_packing_20261020/packing.py',Path('/usr/local/lib/python3.12/site-packages/vllm/ir/ops/layernorm.py'),Path('/usr/local/lib/python3.12/site-packages/vllm/ir/op.py')]
for path in BASE.glob('*.py'):ast.parse(path.read_text())
result=dict(complete=True,CPU_only=True,no_GPU_imports=True,checkpoint_sha256=checkpoint_sha,selected_banks=banks,three_QKV_weight_packs_independently_recomputed_exact=True,M_values=[512,4096],phases=[0,1,2],expected_numeric_cases=18,expected_full_K_INT64_projection_samples=4608,expected_timing_rows=180,primary_timing_M=4096,diagnostic_timing_M=512,actual_residual_cases_by_layer={'0':False,'20':True,'41':True},fixed_norm_kernel_sha256=sha(EXP/'tianshu_IR_RMS_A8_20261023/kernel.py'),expected_norm_binaries={k:v['binary_sha256'] for k,v in parent['compiled_candidate_kernels'].items()},expected_QKV_dequant_binaries={str(m):selection['expected_dequant_binaries'][f'QKV:{m}'] for m in (512,4096)},block_by_M={str(m):selection['block_by_M'][str(m)] for m in (512,4096)},sources={str(path):sha(path) for path in paths},GPU_probe_not_implemented_yet=True,no_new_M4096_numeric_or_perf_result=True,whole_model_CUBIN_equivalence_not_asserted=True,model_not_started=True,adopted=False)
assert len(banks)==3 and not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules);(BASE/'CPU_PREPARE_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('sources','selected_banks')}))
