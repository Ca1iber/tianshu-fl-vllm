# SPDX-License-Identifier: Apache-2.0
import hashlib,importlib.util,json,struct,sys
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;EXP=BASE.parent;ROOT=Path('/dev/shm/tianshu_W4_gateup_component_v2_20261023');MODEL=Path('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors')
s=importlib.util.spec_from_file_location('_gateup_pack_helpers',EXP/'tianshu_native_dotsaddz_down42_20261023/common.py');h=importlib.util.module_from_spec(s);s.loader.exec_module(h)
def main():
 assert not ROOT.exists() and not (BASE/'CPU_GATE_32g.json').exists();ROOT.mkdir();checkpoint=h.sha(MODEL);assert checkpoint=='14fb8e7f0a18d53d1f239773758bf581cee7e456a4523a54622c3a245b64402c'
 with MODEL.open('rb') as f:n=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(n))
 sigpath=EXP/'tianshu_dense_w8a8_quality_20261020/PACKED_WEIGHT_SIGNATURE_32g.json';signatures={r['prefix']:r for r in json.loads(sigpath.read_text())};files={};banks=[];refs=[]
 def save(path,**arrays):np.savez(path,**arrays);files[str(path)]=h.sha(path)
 for phase in range(3):
  i=np.arange(64*2048,dtype=np.float32).reshape(64,2048);x=h.bf16(np.sin(i*np.float32(.013)+phase)*np.float32(.8)+np.cos(i*np.float32(.019)+phase*.7)*np.float32(.2));save(ROOT/f'X{phase}.npz',X=x)
 for layer in (0,20,41):
  prefix=f'model.layers.{layer}.mlp.gate_up_proj';parts=[]
  for role in ('gate_proj','up_proj'):
   v=header[f'model.layers.{layer}.mlp.{role}.weight'];assert v['dtype']=='BF16' and v['shape']==[6144,2048];parts.append(np.memmap(MODEL,mode='r',dtype='<u2',offset=8+n+v['data_offsets'][0],shape=(6144,2048)))
  w=np.concatenate(parts,axis=0);assert h.ahash(w)==signatures[prefix]['original_BF16_sha256'];wf=h.f32(w)
  sw=np.maximum(abs(wf).max(1),np.float32(1.e-8))/np.float32(127);qw=np.clip(np.rint(wf/sw[:,None]),-127,127).astype(np.int8);assert h.ahash(qw)==signatures[prefix]['packed_sha256'] and h.ahash(sw)==signatures[prefix]['scales_sha256']
  sg=h.bf16(np.maximum(abs(wf.reshape(12288,16,128)).max(2),np.float32(1.e-8))/np.float32(7));q4=(np.clip(np.rint(wf.reshape(12288,16,128)/h.f32(sg)[:,:,None]),-8,7).astype(np.int8)+8).astype(np.uint8).reshape(12288,2048).T.copy();p=h.pack(q4);z=h.bf16(-8*h.f32(sg.T));assert np.array_equal(h.unpack(p),q4)
  path=ROOT/f'bank{layer}.npz';save(path,W8=qw.T.copy(),S8=sw,W4=p,S4=sg.T.copy(),Z4=z);banks.append(dict(layer=layer,prefix=prefix,file=str(path),file_sha256=files[str(path)],original_weight_sha256=h.ahash(w),W8_signature_exact=True,W4_full_roundtrip_exact=True))
  deq=h.f32(h.bf16((q4.astype(np.int16)-8).astype(np.float32)*np.repeat(h.f32(sg.T),128,axis=0))).astype(np.float64)
  for phase in range(3):
   with np.load(ROOT/f'X{phase}.npz') as f:x=h.f32(f['X'])
   sa=np.maximum(abs(x).max(1),np.float32(1.e-8))/np.float32(127);qa=np.clip(np.rint(x/sa[:,None]),-127,127).astype(np.int8);integer=qa.astype(np.float64)@qw.T.astype(np.float64);projection8=h.f32(h.bf16(integer.astype(np.float32)*sa[:,None]*sw[None,:]));projection4=x.astype(np.float64)@deq
   def activate(y):
    y=h.f32(h.bf16(y)).astype(np.float64);a,b=y[:,:6144],y[:,6144:];return a/(1+np.exp(-a))*b
   path=ROOT/f'ref_L{layer}_p{phase}.npz';save(path,projection4=projection4,control=activate(projection8),candidate=activate(projection4));refs.append(dict(layer=layer,phase=phase,file=str(path),sha256=files[str(path)]))
  print(json.dumps(dict(layer=layer,W8_matches_real_r4=True,W4_full_roundtrip=True)),flush=True)
 sources={str(p):h.sha(p) for p in [*BASE.glob('*.py'),BASE/'PLAN_32g.md',BASE/'run.sh',Path(h.__file__),sigpath,EXP/'tianshu_nn_silu_two_dot_20261021/kernel.py',EXP/'tianshu_int8_weight_nn_20261021/kernel_nn.py',Path('/workspace/FlagGems/src/flag_gems/fused/silu_and_mul.py'),Path('/usr/local/corex-4.5.0/lib64/libcuinfer.so.7'),Path('/usr/local/corex-4.5.0/lib64/libcuinfer_blas.so'),Path('/usr/local/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so')]}
 assert h.sha(MODEL)==checkpoint and not any(k.split('.')[0] in ('torch','triton','vllm','flag_gems') for k in sys.modules)
 g=dict(complete=True,CPU_only=True,root=str(ROOT),checkpoint_sha256=checkpoint,sources=sources,files=files,banks=banks,references=refs,baseline_configuration=dict(bm=64,bn=64,bk=128,warps=4,stages=1,mode='two_dot'),layers=[0,20,41],phases=[0,1,2],all_references_full_FP64=True,not_model_quality_or_service_performance=True)
 (BASE/'CPU_GATE_32g.json').write_text(json.dumps(g,indent=2)+'\n')
if __name__=='__main__':main()
