# SPDX-License-Identifier: Apache-2.0
import argparse,ast,hashlib,json
from dataclasses import dataclass
from pathlib import Path
from source import encode_bits,encode_unsigned,prior,render_writer
from layout_spec import convert_specs
parser=argparse.ArgumentParser();parser.add_argument('--writer',type=Path,required=True);args=parser.parse_args()
writer_text=render_writer(args.writer.read_bytes());ast.parse(writer_text)
b=Path(__file__).resolve().parent
for bits in range(65536):assert encode_unsigned(bits)==encode_bits(bits)
legacy=prior.check_cpu();assert legacy['complete']
@dataclass(frozen=True)
class Spec:
    dtype:str='bf16';block_size:int=16;num_kv_heads:int=2;head_size:int=128;head_size_v:int=128
    sliding_window:object=None;non_causal:bool=False;attention_chunk_size:object=None;page_size_padded:object=None
    @property
    def page_size_bytes(self):return 2*self.block_size*self.num_kv_heads*self.head_size*(2 if self.dtype=='bf16' else 1)
specs={str(i):Spec() for i in range(42)};new=convert_specs(specs,Spec,'bf16','uint8')
assert all(x.dtype=='uint8' and x.page_size_bytes==8192 for x in new.values())
assert all(x.dtype=='bf16' and x.page_size_bytes==16384 for x in specs.values())
for bad in ({**specs,'0':Spec(block_size=32)},{**specs,'0':Spec(dtype='fp16')},dict(list(specs.items())[:41])):
 try:convert_specs(bad,Spec,'bf16','uint8')
 except AssertionError:pass
 else:raise AssertionError('unsupported spec accepted')
# Execute the actual launch class for 3D and 2D with CPU fixture metadata.
import textwrap
from types import SimpleNamespace
routing=(b/'routing.py').read_text();tree=ast.parse(routing);node=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='Producer')
ns={'torch':SimpleNamespace(bfloat16='BF16',uint8='U8')};exec(compile(textwrap.dedent('\n'.join(routing.splitlines()[node.lineno-1:node.end_lineno])),'<actual_soft8_Producer>','exec'),ns)
class T:
 def __init__(self,dtype,shape):self.dtype=dtype;self.shape=shape
class Kernel:
 def __getitem__(self,grid):
  def launch(**kw):
   self.grid=grid;self.kw=kw
   return SimpleNamespace(metadata=SimpleNamespace(num_warps=kw['num_warps'],num_stages=kw['num_stages']))
  return launch
launch_fixtures=0
for is3d,n,q,grid,want,tile,warps in ((True,64,64,(96,2,8),(96,2,8),16,1),(True,1,1,(1,2,64),(1,2,64),32,4),(False,1,2048,(1025,2),(65,2),128,16)):
 k=Kernel();r=ns['Producer'](k)
 r[grid](BLOCK_SIZE=16,HEAD_SIZE=128,num_query_heads=16,num_queries_per_kv=8,KV_QUANT_MODE=0,USE_TD=False,USE_TD_QO=False,query_ptr=T('BF16',(q,16,128)),key_cache_ptr=T('U8',(2,16,2,128)),value_cache_ptr=T('U8',(2,16,2,128)),IS_3D=is3d,num_seqs=n,BLOCK_M=16,BLOCK_Q=2,NUM_SEGMENTS_PER_SEQ=grid[-1] if is3d else 1)
 assert k.grid==want and k.kw['TILE_SIZE']==tile and k.kw['num_warps']==warps;launch_fixtures+=1
assert launch_fixtures==3

# Physical page packing may use fewer tokens than the logical32 alignment.
packing_fixtures=0
for n,max_len in ((1,32),(1,65),(3,4096),(64,4096),(6,5120)):
 logical_width=(max_len+31)//32*32;physical_width=(max_len+15)//16
 physical_tokens=physical_width*16
 assert physical_tokens<=logical_width and n*physical_tokens*2*128==n*physical_width*16*2*128
 assert physical_tokens>=max_len;packing_fixtures+=1
assert packing_fixtures==5

for p in b.glob('*.py'):ast.parse(p.read_text(),str(p))
r=dict(complete=True,CPU_only=True,all65536_BF16_bit_encodings_equal_legacy=True,all65280_finite_independent_reference=legacy['all65280_finite_BF16_encodings_checked'],all256_decode_values_signed_zeros_checked=True,spec_fixture_layers=42,physical_vs_logical_padding_fixtures=packing_fixtures,actual_Producer_launch_fixtures=launch_fixtures,unsupported_spec_fixtures_rejected=3,GPU_writer_reader_allocator_model_not_run=True,generated_writer_sha256=hashlib.sha256(writer_text.encode()).hexdigest(),
 sources={n:hashlib.sha256((b/n).read_bytes()).hexdigest() for n in ('source.py','layout_spec.py','check_cpu_v2.py','routing.py','probe_pipeline_v2.py','run_probe_v2.py')})
(b/'CPU_GATE_V2_32g.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
