# SPDX-License-Identifier: Apache-2.0
import argparse,hashlib,json,traceback
from pathlib import Path
BASE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();root=args.output.resolve();assert root.is_relative_to('/workspace/benchmark-results') and not root.exists();root.mkdir()
state=dict(complete=False,CPU_only=True,GPU_executed=False,cases=[])
try:
 import triton,kernel
 from triton.compiler import ASTSource
 from triton.backends.compiler import GPUTarget
 target=GPUTarget('corex',71,64)
 for block in (256,512,1024):
  for warps in (4,8):
   for diagnostics in (False,True):
    name=f'B{block}W{warps}D{int(diagnostics)}';source=ASTSource(kernel._epilogue,signature={'Integer':'*i32','AS':'*fp32','WS':'*fp32','O':'*bf16','Projected':'*bf16'},constexprs=dict(M=64,I=6144,BLOCK=block,DIAGNOSTICS=diagnostics));compiled=triton.compile(source,target=target,options=dict(num_warps=warps,num_stages=1,use_sme=0));artifacts={}
    for kind,value in compiled.asm.items():
     path=root/(name+'.'+kind);path.write_bytes(value if isinstance(value,bytes) else value.encode());artifacts[kind]=hashlib.sha256(path.read_bytes()).hexdigest()
    state['cases'].append(dict(name=name,hash=compiled.hash,artifacts=artifacts));print(name,'compiled',flush=True)
 state['complete']=True
except BaseException:state['error']=traceback.format_exc()
finally:(BASE/'OFFLINE_COMPILE_GATE_32g.json').write_text(json.dumps(state,indent=2)+'\n');(root/'result.json').write_text(json.dumps(state,indent=2)+'\n')
raise SystemExit(0 if state['complete'] else 1)
