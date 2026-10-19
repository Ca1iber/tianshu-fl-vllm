# SPDX-License-Identifier: Apache-2.0
"""Verify exact physical address bijections and inverse maps before GPU execution."""
import argparse,hashlib,json
from pathlib import Path
from source import CHUNKS,k_local,render_attention,render_writer
p=argparse.ArgumentParser();p.add_argument('--attention',type=Path,required=True);p.add_argument('--writer',type=Path,required=True);a=p.parse_args()
records=[]
for x in CHUNKS:
    addresses={}
    for t in range(16):
        for h in range(2):
            for d in range(128):
                i=k_local(t,h,d,x);assert i not in addresses;addresses[i]=(t,h,d)
                assert i//2048==h
                local=i%2048;chunk=local//(16*x);slot=(local//x)%16;inner=local%x
                assert (slot,h,chunk*x+inner)==(t,h,d)
    assert set(addresses)==set(range(4096))
    # K stays within its half of a page, including strided/noncontiguous pages.
    for stride in (8192,16384):
        for block in (0,1,17):
            assert max(block*stride+i for i in addresses)<block*stride+4096
            assert not {block*stride+i for i in addresses}&set(range(block*stride+4096,block*stride+8192))
    at=render_attention(a.attention.read_bytes(),x);wr=render_writer(a.writer.read_bytes(),x)
    records.append(dict(x=x,addresses=4096,inverse_exact=True,attention_sha256=hashlib.sha256(at.encode()).hexdigest(),writer_sha256=hashlib.sha256(wr.encode()).hexdigest()))
# Execute the actual nested Producer class for both branches without a GPU.
import ast,textwrap
from types import SimpleNamespace
source=(Path(__file__).resolve().parent/'probe_layout_v2.py').read_text();tree=ast.parse(source)
node=next(n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name=='Producer')
body=textwrap.dedent('\n'.join(source.splitlines()[node.lineno-1:node.end_lineno]))
ns={'torch':SimpleNamespace(bfloat16='BF16')};exec(compile(body,'<actual_Producer_V2>','exec'),ns)
class Tensor:
    dtype='BF16'
    def __init__(self,shape):self.shape=shape
    def stride(self,i):return (8192,256,128,1)[i]
class Kernel:
    def __getitem__(self,grid):
        def launch(**kw):
            self.grid=grid;self.kw=kw
            return SimpleNamespace(metadata=SimpleNamespace(num_warps=kw['num_warps'],num_stages=kw['num_stages']))
        return launch
launch_fixtures=0
for is3d,n,q,grid,want,tile,warps in ((True,64,64,(96,2,8),(96,2,8),16,1),
    (True,1,1,(1,2,64),(1,2,64),32,4),(False,1,2048,(1025,2),(65,2),128,16)):
    k=Kernel();router=ns['Producer'](k)
    router[grid](BLOCK_SIZE=16,HEAD_SIZE=128,num_query_heads=16,num_queries_per_kv=8,KV_QUANT_MODE=0,
        USE_TD=False,USE_TD_QO=False,query_ptr=Tensor((q,16,128)),key_cache_ptr=Tensor((2,16,2,128)),
        value_cache_ptr=Tensor((2,16,2,128)),IS_3D=is3d,num_seqs=n,BLOCK_M=16,BLOCK_Q=2,NUM_SEGMENTS_PER_SEQ=grid[-1] if is3d else 1)
    assert k.grid==want and k.kw['TILE_SIZE']==tile and k.kw['num_warps']==warps
    assert router.metadata[-1]['grid']==list(want);launch_fixtures+=1
assert launch_fixtures==3

base=Path(__file__).resolve().parent
r=dict(complete=True,CPU_only=True,GPU_measured=False,physical_address_cases=4*4096,actual_Producer_CPU_launch_fixtures=launch_fixtures,records=records,
 sources={n:hashlib.sha256((base/n).read_bytes()).hexdigest() for n in ('source.py','check_cpu_v2.py','probe_layout_v2.py','run_probe_v2.py')},
 scope='Only source transforms and exact page-address bijections; runtime geometry guard, GPU writer, reader and graph tests still required')
(base/'CPU_GATE_V2_32g.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
