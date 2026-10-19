# SPDX-License-Identifier: Apache-2.0
"""Prove encoder identity and disjoint real byte addresses for both mixed formats."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from layout import MODES, geometry

base = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_mixed_planes_CPU_parent', base.parent / 'tianshu_soft8_engine_v2_20261019/source.py')
parent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent)
expected = ast.parse(parent.HELPER).body[0]
actual = next(n for n in ast.parse((base / 'writer.py').read_text()).body
    if isinstance(n, ast.FunctionDef) and n.name == '_soft8_encode_bf16')
assert ast.dump(actual) == ast.dump(expected)
for f in base.glob('*.py'):
    ast.parse(f.read_text())
checked = 0
for mode in MODES:
    g = geometry(mode)
    for factor in (1, 2):
        visited = set()
        for block in range(3):
            for size, offset in ((g['key_bytes'], 0), (g['value_bytes'], g['value_offset'])):
                assert offset % size == 0
                for token in range(16):
                    for head in range(2):
                        for dim in range(128):
                            begin = block * factor * g['page_bytes'] + offset + ((token * 2 + head) * 128 + dim) * size
                            for byte in range(begin, begin + size):
                                assert byte not in visited and block * factor * g['page_bytes'] <= byte < (block * factor + 1) * g['page_bytes']
                                visited.add(byte)
                                checked += 1
        assert len(visited) == 3 * g['page_bytes']

# Execute the actual routing class body in CPU stubs, covering both dimensional branches.
from types import SimpleNamespace
ps=ast.parse((base/'probe_v2.py').read_text())
producer=next(n for n in ast.walk(ps) if isinstance(n,ast.ClassDef) and n.name=='Producer')
policy_path=base.parent/'tianshu_soft8_engine_v2_20261019/dispatch_policy.py'
policy_spec=importlib.util.spec_from_file_location('_mixed_CPU_byte_policy',policy_path)
policy=importlib.util.module_from_spec(policy_spec);policy_spec.loader.exec_module(policy)
class Kernel:
 def __getitem__(self,grid):
  assert len(grid) in (2,3) and all(isinstance(v,int) and v>0 for v in grid)
  def run(**kw):return SimpleNamespace(metadata=SimpleNamespace(num_warps=kw['num_warps'],num_stages=kw['num_stages']),n_regs=0,n_spills=0)
  return run
context=[33];records={};env=dict(choose_config=policy.choose_config,current_context=context,data={'producer_records':records})
exec(compile(ast.fix_missing_locations(ast.Module(body=[producer],type_ignores=[])),str(base/'probe_v2.py'),'exec'),env)
routing_cases=0
for mode in MODES:
 for n in (1,8,16,24,32,64,128,256):
  for ctx in (33,4096):
   context[0]=ctx;p=env['Producer'](Kernel());p.mode=mode
   kw=dict(KV_QUANT_MODE=0,HEAD_SIZE=128,num_seqs=n,IS_3D=n<=64,query_ptr=SimpleNamespace(shape=(n,16,128)),
    key_cache_ptr=SimpleNamespace(dtype='fixture'),value_cache_ptr=SimpleNamespace(dtype='fixture'),
    NUM_SEGMENTS_PER_SEQ=4,TILE_SIZE=128,BLOCK_M=256,BLOCK_Q=32,num_warps=16,num_stages=1)
   p[(n,2,4) if n<=64 else (n,2)](**kw);routing_cases+=1
assert routing_cases==80

sources = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(base.glob('*.py'))}
r = dict(complete=True, CPU_only=True, formats=list(MODES), padded_page_bytes=[geometry(m)['page_bytes'] for m in MODES],
    byte_addresses_checked=checked, actual_Producer_control_flow_cases=routing_cases, no_overlapping_planes_or_pages=True,
    contiguous_and_strided_page_geometry=True, encoder_AST_identical_to_qualified_software8=True,
    sources=sources, GPU_not_run=True, adopted=False)
(base / 'CPU_GATE_V2_32g.json').write_text(json.dumps(r, indent=2) + '\n')
print(json.dumps(r))
