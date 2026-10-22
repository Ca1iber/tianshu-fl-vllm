# SPDX-License-Identifier: Apache-2.0
"""Explicit CoreX71 target offline compilation; no driver launch or GPU claims."""
import argparse,hashlib,json,os,sys,traceback
from pathlib import Path
BASE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();root=args.output.resolve()
    assert root.is_relative_to('/workspace/benchmark-results') and not root.exists();root.mkdir()
    result=dict(complete=False,CPU_only=True,actual_GPU_execution=False,GPU_correctness_not_proven=True,performance_not_measured=True,original_model_unchanged=True,runner_pid=os.getpid(),cases=[],sources={str(f):sha(f) for f in (BASE/'kernel_postscale.py',Path(__file__))})
    try:
        import triton
        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource
        import kernel_postscale as kernel
        target=GPUTarget('corex',71,64)
        for label,m,n,k,bits,group,affine in [('w4sym_tail',17,130,259,4,128,False),('w4aff_tail',17,130,259,4,128,True),('w8_tail',17,130,259,8,0,False),('w4_down64',64,2048,6144,4,128,False),('w4_lm8',8,130560,2048,4,128,False)]:
            row=dict(label=label,M=m,N=n,K=k,bits=bits,group=group,affine=affine,GPU_executed=False)
            try:
                source=ASTSource(kernel._matmul,signature={'A':'*bf16','W':'*u8','S':'*fp32','Z':'*u8','O':'*bf16','F':'*fp32'},constexprs=dict(M=m,N=n,K=k,ASTRIDE=k,BITS=bits,GROUP=group,AFFINE=affine,BM=16,BN=64,BK=128,RETURN_FLOAT=False))
                c=triton.compile(source,target=target,options=dict(num_warps=4,num_stages=1,use_sme=0))
                for kind,value in c.asm.items():
                    path=root/(label+'.'+kind)
                    path.write_bytes(value if isinstance(value,bytes) else value.encode())
                row.update(compiled=True,cache_hash=c.hash,metadata=dict(c.metadata._asdict(),target=dict(backend=target.backend,arch=target.arch,warp_size=target.warp_size)),asm_artifact_hashes={kind:sha(root/(label+'.'+kind)) for kind in c.asm},BF16_dot_in_TTIR=('bf16' in c.asm['ttir'] and 'tt.dot' in c.asm['ttir']),packed_shift_in_TTIR=('arith.shrui' in c.asm['ttir']) if bits==4 else None)
            except Exception:
                row.update(compiled=False,error=traceback.format_exc())
            result['cases'].append(row)
            (root/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:row.get(k) for k in ('label','compiled','cache_hash','error')}),flush=True)
        assert all(sha(path)==expected for path,expected in result['sources'].items())
        result.update(complete=all(r['compiled'] for r in result['cases']),no_kernel_launch_called=True)
    except BaseException:result['error']=traceback.format_exc()
    finally:
        (root/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        (BASE/'OFFLINE_COMPILE_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n')
    return 0 if result['complete'] else 1
if __name__=='__main__':raise SystemExit(main())
