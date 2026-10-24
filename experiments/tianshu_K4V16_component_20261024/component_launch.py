# SPDX-License-Identifier: Apache-2.0
"""Identical qualified dispatch; K4 changes only the accepted quantization ABI."""
import ast,hashlib,importlib.util,sys
from pathlib import Path
import torch
BASE=Path(__file__).resolve().parent
NATIVE=BASE.parent/'tianshu_k8v16_bm8_abi_20261021/native_launch.py'
s=importlib.util.spec_from_file_location('_K4_native_qualified_dispatch',NATIVE)
native=importlib.util.module_from_spec(s);sys.modules[s.name]=native;s.loader.exec_module(native)

class K4Producer:
    def __init__(self,kernel):self.kernel=kernel;self.tile=None;self.records={};self.decode_config=None;self.pure_decode=False
    def __getitem__(self,grid):
        def launch(*args,**kw):
            launch_grid=grid
            assert not args and kw['BLOCK_SIZE']==16 and kw['HEAD_SIZE']==128
            assert kw['num_query_heads']==16 and kw['num_queries_per_kv']==8 and kw['KV_QUANT_MODE']==2
            assert not kw['USE_TD'] and not kw['USE_TD_QO']
            assert kw['query_ptr'].dtype==torch.bfloat16 and kw['key_cache_ptr'].dtype==torch.uint8 and kw['value_cache_ptr'].dtype in (torch.bfloat16, torch.uint8)
            if kw['IS_3D']:
                n=kw['num_seqs'];selected=n in (8,16,32,40,48,56,64)
                tile=16 if selected else 32;warps=1 if selected else 2 if 8<=n<=64 else 4
                assert (kw['BLOCK_M'],kw['BLOCK_Q']) in ((16,2),(8,1))
                if self.decode_config is not None:tile=self.decode_config['TILE_SIZE'];warps=self.decode_config['num_warps']
                kw.update(TILE_SIZE=tile,num_warps=warps,num_stages=1)
            elif self.pure_decode:
                kw.update(BLOCK_M=16,BLOCK_Q=2,TILE_SIZE=16,num_warps=1,num_stages=1)
                launch_grid=(kw['query_ptr'].shape[0]//2+kw['num_seqs'],grid[1])
            else:
                kw.update(BLOCK_M=256,BLOCK_Q=32,TILE_SIZE=128,num_warps=16,num_stages=1)
                launch_grid=(kw['query_ptr'].shape[0]//32+kw['num_seqs'],grid[1])
            kw['KV_CACHE_MODIFIER']='.cv';self.tile=kw['TILE_SIZE'];r=self.kernel[launch_grid](**kw)
            key=(kw['IS_3D'],kw['num_seqs'],kw['NUM_SEGMENTS_PER_SEQ'],self.tile,kw['num_warps'])
            self.records[str(key)]=dict(IS_3D=kw['IS_3D'],sequences=kw['num_seqs'],segments=kw['NUM_SEGMENTS_PER_SEQ'],tile=self.tile,grid=list(launch_grid),warps=r.metadata.num_warps,stages=r.metadata.num_stages,key_dtype=str(kw['key_cache_ptr'].dtype),query_dtype=str(kw['query_ptr'].dtype),value_dtype=str(kw['value_cache_ptr'].dtype),BLOCK_M=kw['BLOCK_M'],BLOCK_Q=kw['BLOCK_Q'],pure_decode=self.pure_decode,n_regs=getattr(r,'n_regs',None),n_spills=getattr(r,'n_spills',None),shared=getattr(r.metadata,'shared',None))
            self.last_record=self.records[str(key)]
            return r
        return launch

def proof():
    original=NATIVE.read_text();candidate=Path(__file__).read_text().replace("kw['KV_QUANT_MODE']==2","kw['KV_QUANT_MODE']==0")
    a=next(x for x in ast.parse(original).body if isinstance(x,ast.ClassDef) and x.name=='Producer')
    b=next(x for x in ast.parse(candidate).body if isinstance(x,ast.ClassDef) and x.name=='K4Producer');b.name='Producer'
    assert ast.dump(a)==ast.dump(b)
    return dict(native_path=str(NATIVE),native_sha256=hashlib.sha256(NATIVE.read_bytes()).hexdigest(),
        only_Producer_change='KV_QUANT_MODE accepted 0 -> 2',all_launch_computation_AST_identical=True,
        original_Reducer_used=True,selected_C33_segments=16,selected_C64_segments=4)

def configuration(n,length):
    return dict(native.choose_config(n,length),TILE_SIZE=16,num_warps=1,num_stages=1)

class Observed:
    def __init__(self,kernel,root,label):self.kernel=kernel;self.root=Path(root);self.label=label;self.records={}
    def __getitem__(self,grid):
        def run(*args,**kw):
            r=self.kernel[grid](*args,**kw);h=hashlib.sha256(r.kernel).hexdigest()
            record=dict(grid=list(grid),IS_3D=kw['IS_3D'],num_seqs=kw['num_seqs'],BLOCK_M=kw['BLOCK_M'],BLOCK_Q=kw['BLOCK_Q'],
                TILE_SIZE=kw['TILE_SIZE'],segments=kw['NUM_SEGMENTS_PER_SEQ'],KV_QUANT_MODE=int(kw['KV_QUANT_MODE']),
                num_warps=r.metadata.num_warps,num_stages=r.metadata.num_stages,use_sme=r.metadata.use_sme,
                n_regs=getattr(r,'n_regs',None),n_spills=getattr(r,'n_spills',None),shared=getattr(r.metadata,'shared',None),
                K_shape=list(kw['key_cache_ptr'].shape),K_stride=list(kw['key_cache_ptr'].stride()),
                V_shape=list(kw['value_cache_ptr'].shape),V_stride=list(kw['value_cache_ptr'].stride()),binary_sha256=h)
            if h not in self.records:
                self.records[h]=record;(self.root/f'{self.label}_{h}.bin').write_bytes(r.kernel)
            self.last=record
            return r
        return run

def attach(reader,label,root):
    assert label in ('current','k4')
    observed=Observed(reader.kernel_unified_attention,root,label)
    producer=(native.Producer if label=='current' else K4Producer)(observed)
    reader.kernel_unified_attention=producer;reader.reduce_segments=native.Reducer(reader.reduce_segments,producer)
    def invoke(kwargs):
        kw=dict(kwargs);assert kw['max_seqlen_q']==1
        n=kw['seqused_k'].numel();assert 1<=n<=64
        cfg=configuration(n,kw['max_seqlen_k']);producer.decode_config=cfg;producer.pure_decode=True
        kw['num_par_softmax_segments']=cfg['segments']
        return reader.unified_attention(**kw)
    return invoke,producer,observed
