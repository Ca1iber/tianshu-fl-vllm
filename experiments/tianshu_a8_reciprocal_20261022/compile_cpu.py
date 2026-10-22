import argparse,hashlib,json,os,traceback
from pathlib import Path
BASE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.output.resolve();assert root.is_relative_to('/workspace/benchmark-results') and not root.exists();root.mkdir();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();files=[Path(__file__),BASE/'kernel.py'];result=dict(complete=False,CPU_only=True,GPU_executed=False,cases=[],sources={str(p):sha(p) for p in files},runner_pid=os.getpid())
try:
 import triton,kernel
 from triton.compiler import ASTSource
 from triton.backends.compiler import GPUTarget
 target=GPUTarget('corex',71,64)
 for name,m,k,stride,method in [('decode_div',64,2048,2048,1),('decode_fma',64,2048,2048,2),('domain_fma',19803,4096,4096,2),('tail_fma',17,259,275,2)]:
  src=ASTSource(kernel._quantize,signature={'A':'*bf16','Q':'*i8','S':'*fp32'},constexprs=dict(M=m,K=k,ASTRIDE=stride,BLOCK=triton.next_power_of_2(k),METHOD=method));c=triton.compile(src,target=target,options=dict(num_warps=4,num_stages=1,use_sme=0));hashes={}
  for kind,value in c.asm.items():
   path=root/(name+'.'+kind);path.write_bytes(value if isinstance(value,bytes) else value.encode());hashes[kind]=sha(path)
  result['cases'].append(dict(name=name,compiled=True,hash=c.hash,metadata=dict(c.metadata._asdict(),target=dict(backend=target.backend,arch=target.arch,warp_size=target.warp_size)),artifact_hashes=hashes));print(name,'compiled',flush=True)
 assert all(sha(p)==h for p,h in result['sources'].items());result['complete']=True
except BaseException:result['error']=traceback.format_exc()
finally:
 (root/'result.json').write_text(json.dumps(result,indent=2)+'\n');(BASE/'OFFLINE_COMPILE_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n')
raise SystemExit(0 if result['complete'] else 1)
