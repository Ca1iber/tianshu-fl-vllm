import argparse,hashlib,json,os,traceback
from pathlib import Path
BASE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();root=args.output.resolve();assert root.is_relative_to('/workspace/benchmark-results') and not root.exists();root.mkdir();sources={str(p):sha(p) for p in [Path(__file__),BASE/'kernel_w4a8.py']};state=dict(complete=False,CPU_only=True,GPU_executed=False,sources=sources,cases=[],runner_pid=os.getpid())
 try:
  import triton,kernel_w4a8
  from triton.compiler import ASTSource
  from triton.backends.compiler import GPUTarget
  target=GPUTarget('corex',71,64)
  for label,m,n,k,aff in [('tail_sym',17,131,259,False),('down_sym',64,2048,6144,False),('down_aff',64,2048,6144,True)]:
   source=ASTSource(kernel_w4a8._matmul,signature={'A':'*i8','W':'*u8','S':'*fp32','Z':'*u8','AS':'*fp32','O':'*bf16','F':'*fp32','I':'*i32'},constexprs=dict(M=m,N=n,K=k,GROUP=128,AFFINE=aff,BM=64,BN=64,BK=128,DIAGNOSTICS=False));c=triton.compile(source,target=target,options=dict(num_warps=8,num_stages=2,use_sme=0));hashes={}
   for kind,value in c.asm.items():
    dest=root/(label+'.'+kind);dest.write_bytes(value if isinstance(value,bytes) else value.encode());hashes[kind]=sha(dest)
   state['cases'].append(dict(label=label,compiled=True,cache_hash=c.hash,integer_dot_TTIR=('i32' in c.asm['ttir'] and 'tt.dot' in c.asm['ttir']),artifacts=hashes,metadata=dict(c.metadata._asdict(),target=dict(backend=target.backend,arch=target.arch,warp_size=target.warp_size))));print(label,'compiled',flush=True)
  assert all(sha(p)==v for p,v in sources.items());state['complete']=True
 except BaseException:state['error']=traceback.format_exc()
 finally:
  (root/'result.json').write_text(json.dumps(state,indent=2)+'\n');(BASE/'OFFLINE_COMPILE_GATE_32g.json').write_text(json.dumps(state,indent=2)+'\n')
 return 0 if state['complete'] else 1
if __name__=='__main__':raise SystemExit(main())
