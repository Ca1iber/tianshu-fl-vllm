# SPDX-License-Identifier: Apache-2.0
import ast,hashlib,json,textwrap
from pathlib import Path
from types import SimpleNamespace
from plans import PLANS
from source import prior,encode_unsigned,encode_bits
b=Path(__file__).resolve().parent
assert prior.check_cpu()['complete']
for i in range(65536):assert encode_bits(i)==encode_unsigned(i)
routing=(b/'routing.py').read_text();node=next(n for n in ast.parse(routing).body if isinstance(n,ast.ClassDef) and n.name=='Producer')
ns={'torch':SimpleNamespace(bfloat16='BF16',uint8='U8')};exec(compile(textwrap.dedent('\n'.join(routing.splitlines()[node.lineno-1:node.end_lineno])),'<actual_tuned_Producer>','exec'),ns)
class T:
 def __init__(self,dtype,shape):self.dtype=dtype;self.shape=shape
class K:
 def __getitem__(self,grid):
  def launch(**kw):
   self.grid=grid;self.kw=kw
   return SimpleNamespace(metadata=SimpleNamespace(num_warps=kw['num_warps'],num_stages=kw['num_stages']))
  return launch
for c in PLANS:
 k=K();r=ns['Producer'](k);r.prefill_config=c
 r[(1025,2)](BLOCK_SIZE=16,HEAD_SIZE=128,num_query_heads=16,num_queries_per_kv=8,KV_QUANT_MODE=0,USE_TD=False,USE_TD_QO=False,query_ptr=T('BF16',(2048,16,128)),key_cache_ptr=T('U8',(260,16,2,128)),value_cache_ptr=T('U8',(260,16,2,128)),IS_3D=False,num_seqs=1,BLOCK_M=16,BLOCK_Q=2,NUM_SEGMENTS_PER_SEQ=1)
 assert k.grid==(2048//(c['BLOCK_M']//8)+1,2)
 assert all(k.kw[n]==v for n,v in c.items())
 assert k.kw['BLOCK_Q']==c['BLOCK_M']//8
assert len(PLANS)==15
for p in b.glob('*.py'):ast.parse(p.read_text(),str(p))
r=dict(complete=True,CPU_only=True,all65536_encodings_unchanged=True,actual_prefill_launch_configurations=15,GPU_not_run=True,
 sources={n:hashlib.sha256((b/n).read_bytes()).hexdigest() for n in ('source.py','routing.py','plans.py','check_cpu.py','probe.py','run_probe.py','PREREQUISITE_32g.json')})
(b/'CPU_GATE_32g.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
