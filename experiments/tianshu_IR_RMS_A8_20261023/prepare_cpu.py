# SPDX-License-Identifier: Apache-2.0
import ast,hashlib,json,struct,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent;ROOT=Path('/workspace/benchmark-results/20261023_IR_RMS_A8_CPU_retry1_tianshu');MODEL=Path('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert not ROOT.exists() and not (BASE/'CPU_PREPARE_GATE_32g.json').exists();ROOT.mkdir();selected=[];allnorms=[]
with MODEL.open('rb') as f:
 length=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(length));offset=8+length
 for layer in range(42):
  for role in ('input_layernorm','post_attention_layernorm'):
   name=f'model.layers.{layer}.{role}.weight';meta=header[name];assert meta['dtype']=='BF16' and meta['shape']==[2048];start,end=meta['data_offsets'];assert end-start==4096;f.seek(offset+start);raw=f.read(end-start);assert len(raw)==4096;row=dict(name=name,layer=layer,role=role,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());allnorms.append(row)
   if layer in (0,20,41):
    target=ROOT/f'layer{layer}_{role}.u16';target.write_bytes(raw);selected.append(row|dict(path=str(target)))
assert len(selected)==6 and len(allnorms)==84;checkpoint=sha(MODEL);assert checkpoint=='14fb8e7f0a18d53d1f239773758bf581cee7e456a4523a54622c3a245b64402c'
graph_origin=Path('/root/.cache/vllm/torch_compile_cache/36d2f2d6db/rank_0_0/backbone/computation_graph.py');graph=ROOT/'OBSERVED_MODEL_computation_graph.py';graph.write_bytes(graph_origin.read_bytes());text=graph.read_text();assert 'tianshu_native_attention_decode_scope' in text and 'rms_norm' in text
paths=[*BASE.glob('*.py'),BASE/'PLAN_32g.md',BASE/'run.sh',Path('/usr/local/lib/python3.12/site-packages/vllm/ir/ops/layernorm.py'),Path('/usr/local/lib/python3.12/site-packages/vllm/ir/op.py'),Path('/usr/local/lib/python3.12/site-packages/vllm/model_executor/layers/layernorm.py'),Path('/usr/local/lib/python3.12/site-packages/vllm/model_executor/models/llama.py'),EXP/'tianshu_dense_w8a8_20261020/kernel.py',graph]
for p in BASE.glob('*.py'):ast.parse(p.read_text())
sources={str(p):sha(p) for p in paths};gate=dict(complete=True,CPU_only=True,no_GPU_imports=True,checkpoint_sha256=checkpoint,original_norm_bank_count=84,selected_norms=selected,all_norms=allnorms,K=2048,epsilon=1e-6,M_values=[8,64,128],phases=[0,1,2],residual_modes=[False,True],expected_numeric_cases=108,expected_timing_rounds=120,actual_model_generated_graph=str(graph),actual_model_binary_equivalence_not_asserted=True,standalone_compiled_native_IR_baseline=True,sources=sources,model_not_run=True,adopted=False);(BASE/'CPU_PREPARE_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');assert not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules);print(json.dumps({k:v for k,v in gate.items() if k not in ('sources','all_norms','selected_norms')}))
