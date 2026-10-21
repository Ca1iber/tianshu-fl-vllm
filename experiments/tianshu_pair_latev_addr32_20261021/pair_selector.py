# SPDX-License-Identifier: Apache-2.0
"""One metadata decision selects both producer and reducer; no tensor value reads."""
import hashlib,importlib.util,sys
from pathlib import Path
import torch
import source
GUARD=source.OLD.with_name('guard.py')
spec=importlib.util.spec_from_file_location('_paired_load_existing_offset_guard',GUARD)
guard=importlib.util.module_from_spec(spec);sys.modules[spec.name]=guard;spec.loader.exec_module(guard)

def eligible(kw):
    return bool(kw.get('IS_3D') and 24<kw.get('num_seqs',0)<=64 and kw.get('KV_QUANT_MODE')==0
        and kw.get('HEAD_SIZE')==128 and kw.get('HEAD_SIZE_PADDED',128)==128 and kw.get('BLOCK_SIZE')==16
        and kw.get('BLOCK_M')==16 and kw.get('BLOCK_Q')==2 and kw.get('num_query_heads')==16 and kw.get('num_queries_per_kv')==8
        and kw.get("TILE_SIZE")==16 and kw.get("num_warps")==1 and kw.get("num_stages")==1
        and tuple(kw["query_ptr"].shape)==(kw["num_seqs"],16,128)
        and kw['query_ptr'].dtype==torch.bfloat16 and kw['key_cache_ptr'].dtype==torch.uint8 and kw['value_cache_ptr'].dtype==torch.bfloat16
        and not any(kw.get(k,False) for k in ('USE_TD','USE_TD_QO','USE_ALIBI_SLOPES','USE_QQ_BIAS','USE_SINKS','USE_MM_PREFIX','USE_SOFTCAP','USE_FP8','Q_IS_FP8','SLIDING_WINDOW'))
        and kw.get('CHUNK_LOOKBACK',-1)==-1)

class Selector:
    def __init__(self,wide,candidate,mode,wide_sha,candidate_sha):
        self.wide,self.candidate,self.mode=wide,candidate,mode
        self.wide_sha,self.candidate_sha=wide_sha,candidate_sha;self.last_selection=None;self.last=None
    def __getitem__(self,grid):
        def run(*args,**kw):
            assert not args;safe,bounds=guard.inspect_bounds(kw);supported=eligible(kw)
            selected=self.mode!='current' and supported and safe
            actual=self.candidate if selected else self.wide
            self.last_selection=dict(mode=self.mode,candidate_selected=selected,scope_eligible=supported,signed_element_guard_passed=safe,
                integer_bounds=bounds,active_source_sha256=self.candidate_sha if selected else self.wide_sha,
                address_bits=32 if selected and 'addr32' in self.mode else 64,late_V=bool(selected and 'late_v' in self.mode),
                original_arguments_grid_and_return_preserved=True)
            compiled=actual[grid](**kw)
            self.last=dict(mode=self.mode,grid=list(grid),IS_3D=bool(kw['IS_3D']),actual_tile=kw['TILE_SIZE'],actual_segments=kw['NUM_SEGMENTS_PER_SEQ'],
                short_active_segments=4,num_warps=compiled.metadata.num_warps,num_stages=compiled.metadata.num_stages,
                binary_sha256=hashlib.sha256(compiled.kernel).hexdigest(),kernel_hash=getattr(compiled,'hash',None),
                n_regs=getattr(compiled,'n_regs',None),n_spills=getattr(compiled,'n_spills',None))
            return compiled
        return run

class Reducer:
    def __init__(self,wide,candidate,producer,selector):
        self.wide,self.candidate,self.producer,self.selector=wide,candidate,producer,selector;self.last=None
    def __getitem__(self,grid):
        def run(*args,**kw):
            assert self.selector.last_selection is not None
            narrow=self.selector.last_selection['candidate_selected'] and 'addr32' in self.selector.mode
            actual=self.candidate if narrow else self.wide;kw['TILE_SIZE']=self.producer.tile
            compiled=actual[grid](*args,**kw)
            self.last=dict(address_bits=32 if narrow else 64,producer_and_reducer_guard_decision_consistent=True,
                active_source_sha256=self.selector.candidate_sha if narrow else self.selector.wide_sha,
                binary_sha256=hashlib.sha256(compiled.kernel).hexdigest(),original_arguments_grid_and_return_preserved=True)
            return compiled
        return run
