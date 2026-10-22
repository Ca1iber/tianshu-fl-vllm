# SPDX-License-Identifier: Apache-2.0
import argparse,hashlib,json,traceback
from pathlib import Path
BASE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();root=args.output.resolve();assert root.is_relative_to('/workspace/benchmark-results') and not root.exists();root.mkdir();state=dict(complete=False,CPU_only=True,GPU_executed=False,cases=[])
try:
 import triton,kernel
 from triton.compiler import ASTSource
 from triton.backends.compiler import GPUTarget
 target=GPUTarget('corex',71,64)
 for m in (128,512,4096):
  for rows in (1,4):
   for diagnostic in (False,True):
    name=f'M{m}R{rows}D{int(diagnostic)}';src=ASTSource(kernel._epilogue,signature={'Integer':'*i32','AS':'*fp32','WS':'*fp32','O':'*bf16','Projected':'*bf16'},constexprs=dict(M=m,I=6144,ROWS=rows,DIAGNOSTICS=diagnostic));c=triton.compile(src,target=target,options=dict(num_warps=8,num_stages=1,use_sme=0));hashes={}
    for kind,value in c.asm.items():
     path=root/(name+'.'+kind);path.write_bytes(value if isinstance(value,bytes) else value.encode());hashes[kind]=hashlib.sha256(path.read_bytes()).hexdigest()
    state['cases'].append(dict(name=name,hash=c.hash,artifacts=hashes));print(name,'compiled',flush=True)
 state['complete']=True
except BaseException:state['error']=traceback.format_exc()
finally:(BASE/'OFFLINE_COMPILE_GATE_32g.json').write_text(json.dumps(state,indent=2)+'\n');(root/'result.json').write_text(json.dumps(state,indent=2)+'\n')
raise SystemExit(0 if state['complete'] else 1)
