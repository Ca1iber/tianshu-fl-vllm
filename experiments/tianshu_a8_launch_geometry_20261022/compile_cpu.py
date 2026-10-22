# SPDX-License-Identifier: Apache-2.0
import argparse, hashlib, json, os, traceback
from pathlib import Path
BASE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
root=args.output.resolve();assert root.is_relative_to('/workspace/benchmark-results') and not root.exists();root.mkdir()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
files=[Path(__file__),BASE/'kernel.py',BASE/'configs.json']
state=dict(complete=False,CPU_only=True,GPU_executed=False,cases=[],sources={str(p):sha(p) for p in files},runner_pid=os.getpid())
try:
 import triton,kernel
 from triton.compiler import ASTSource
 from triton.backends.compiler import GPUTarget
 target=GPUTarget('corex',71,64)
 for m,k,stride in [(64,2048,2048),(64,6144,6144),(17,259,275)]:
  for rows,warps in json.loads((BASE/'configs.json').read_text()):
   name=f'M{m}_K{k}_R{rows}_W{warps}'
   src=ASTSource(kernel._quantize,signature={'A':'*bf16','Q':'*i8','S':'*fp32'},constexprs=dict(M=m,K=k,ASTRIDE=stride,BLOCK=triton.next_power_of_2(k),ROWS=rows))
   compiled=triton.compile(src,target=target,options=dict(num_warps=warps,num_stages=1,use_sme=0));artifacts={}
   for kind,value in compiled.asm.items():
    dest=root/(name+'.'+kind);dest.write_bytes(value if isinstance(value,bytes) else value.encode());artifacts[kind]=sha(dest)
   state['cases'].append(dict(name=name,compiled=True,hash=compiled.hash,metadata=dict(compiled.metadata._asdict(),target=dict(backend=target.backend,arch=target.arch,warp_size=target.warp_size)),artifacts=artifacts));print(name,'compiled',flush=True)
 assert all(sha(p)==v for p,v in state['sources'].items());state['complete']=True
except BaseException:state['error']=traceback.format_exc()
finally:
 (root/'result.json').write_text(json.dumps(state,indent=2)+'\n');(BASE/'OFFLINE_COMPILE_GATE_32g.json').write_text(json.dumps(state,indent=2)+'\n')
raise SystemExit(0 if state['complete'] else 1)
