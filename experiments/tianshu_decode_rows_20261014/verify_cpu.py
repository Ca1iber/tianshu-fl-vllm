# SPDX-License-Identifier: Apache-2.0
"""Execute the actual row router AST and production reducer router with fake GPU."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace


def main():
    base=Path(__file__).resolve().parent
    source=base/'probe_rows.py'
    tree=ast.parse(source.read_text())
    cls=next(n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name=='PageRouter')
    names=['vllm','vllm.v1','vllm.v1.attention','vllm.v1.attention.ops','vllm.v1.attention.ops.triton_unified_attention']
    for name in names:sys.modules[name]=ModuleType(name)
    native=sys.modules[names[-1]]
    reductions=[]
    class FakeKernel:
        def __init__(self):self.calls=[]
        def __getitem__(self,grid):
            def launch(*a,**kw):
                self.calls.append((grid,kw))
                return SimpleNamespace(metadata=SimpleNamespace(num_warps=kw.get('num_warps'),num_stages=kw.get('num_stages'),shared=0))
            return launch
    native.reduce_segments=FakeKernel()
    parent=base.parent/'tianshu_decode_low_warps_20261014/launch_router.py'
    spec=importlib.util.spec_from_file_location('_larger_rows_cpu_launch',parent)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    env=dict(LaunchRouter=module.LaunchRouter)
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(source),'exec'),env)
    count=0
    for batch in range(1,65):
        for pattern in (0,1,2):
            qlens=[1 if pattern==0 else int((i+pattern)%3!=1) for i in range(batch)]
            queries=sum(qlens)
            if queries==0:continue
            for rows in (16,32,64):
                bq=rows//8;tile=16 if batch in (8,16,32,40,48,56,64) else 32
                kernel=FakeKernel()
                router=env['PageRouter'](kernel,dict(TILE_SIZE=tile,BLOCK_M=rows,BLOCK_Q=bq,num_warps=1,num_stages=1))
                kw=dict(IS_3D=True,BLOCK_M=16,BLOCK_Q=2,num_query_heads=16,num_queries_per_kv=8,HEAD_SIZE=128,KV_QUANT_MODE=0,
                        query_ptr=SimpleNamespace(shape=(queries,16,128)),num_seqs=batch,TILE_SIZE=32,BLOCK_SIZE=16)
                router[(queries//2+batch,2,8)](**kw)
                grid,actual=kernel.calls[-1]
                assert grid==(queries//bq+batch,2,8) and actual['BLOCK_M']==rows and actual['BLOCK_Q']==bq
                starts=[0]
                for q in qlens:starts.append(starts[-1]+q)
                tile_starts=[q//bq+i for i,q in enumerate(starts)]
                covered=[]
                for pid in range(grid[0]):
                    seq=max(i for i,t in enumerate(tile_starts[:-1]) if t<=pid)
                    qstart=(pid-tile_starts[seq])*bq
                    for off in range(bq):
                        if qstart+off<qlens[seq]:covered.append(starts[seq]+qstart+off)
                assert sorted(covered)==list(range(queries)) and len(covered)==len(set(covered))
                native.reduce_segments[(1,)](TILE_SIZE=99)
                assert native.reduce_segments.original.calls[-1][1]['TILE_SIZE']==tile
                assert router.metadata['actual_BLOCK_M']==rows and router.metadata['actual_BLOCK_Q']==bq
                assert router.metadata['producer_and_reducer_tile']==tile
                count+=1
    result=dict(complete=True,cpu_only=True,grid_and_coverage_cases=count,
                actual_router_AST_exercised=True,producer_reducer_tile_preserved=True,
                dense_and_zero_query_padding=True,gpu_compilation_not_tested=True,
                source_sha256={str(p.name):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,parent,Path(__file__))})
    path=base/'results/20261014/prepared_32g/CPU_GRID_AUDIT_32g.json'
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
