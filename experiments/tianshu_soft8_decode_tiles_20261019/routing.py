# SPDX-License-Identifier: Apache-2.0
"""Byte-cache routing through actual Triton writer/metadata/forward entrypoints."""
import torch

class Producer:
    def __init__(self,kernel):self.kernel=kernel;self.tile=None;self.records={};self.decode_config=None
    def __getitem__(self,grid):
        def launch(*args,**kw):
            launch_grid=grid
            assert not args and kw['BLOCK_SIZE']==16 and kw['HEAD_SIZE']==128
            assert kw['num_query_heads']==16 and kw['num_queries_per_kv']==8 and kw['KV_QUANT_MODE']==0
            assert not kw['USE_TD'] and not kw['USE_TD_QO']
            assert kw['query_ptr'].dtype==torch.bfloat16 and kw['key_cache_ptr'].dtype==kw['value_cache_ptr'].dtype==torch.uint8
            if kw['IS_3D']:
                n=kw['num_seqs'];selected=n in (8,16,32,40,48,56,64)
                tile=16 if selected else 32;warps=1 if selected else 2 if 8<=n<=64 else 4
                assert kw['BLOCK_M']==16 and kw['BLOCK_Q']==2
                if self.decode_config is not None:tile=self.decode_config['TILE_SIZE'];warps=self.decode_config['num_warps']
                kw.update(TILE_SIZE=tile,num_warps=warps,num_stages=1)
            else:
                kw.update(BLOCK_M=256,BLOCK_Q=32,TILE_SIZE=128,num_warps=16,num_stages=1)
                launch_grid=(kw['query_ptr'].shape[0]//32+kw['num_seqs'],grid[1])
            kw['KV_CACHE_MODIFIER']='.cv';self.tile=kw['TILE_SIZE'];r=self.kernel[launch_grid](**kw)
            key=(kw['IS_3D'],kw['num_seqs'],kw['NUM_SEGMENTS_PER_SEQ'],self.tile,kw['num_warps'])
            self.records[str(key)]=dict(IS_3D=kw['IS_3D'],sequences=kw['num_seqs'],segments=kw['NUM_SEGMENTS_PER_SEQ'],tile=self.tile,grid=list(launch_grid),warps=r.metadata.num_warps,stages=r.metadata.num_stages,key_dtype=str(kw['key_cache_ptr'].dtype),query_dtype=str(kw['query_ptr'].dtype),n_regs=getattr(r,'n_regs',None),n_spills=getattr(r,'n_spills',None))
            self.last_record=self.records[str(key)]
            return r
        return launch

class Reducer:
    def __init__(self,kernel,producer):self.kernel=kernel;self.producer=producer
    def __getitem__(self,grid):
        def launch(*args,**kw):kw['TILE_SIZE']=self.producer.tile;return self.kernel[grid](*args,**kw)
        return launch

def install(backend,hybrid,adaptive,reader,writer):
    assert not hasattr(backend,'_soft8_pipeline_installed')
    producer=Producer(reader.kernel_unified_attention)
    reader.kernel_unified_attention=producer;reader.reduce_segments=Reducer(reader.reduce_segments,producer)
    original_write=backend.triton_reshape_and_cache_flash;original_dispatch=hybrid.dispatch
    counts=dict(encoded_writer=0,decode=0,prefill=0,mixed=0)
    def write(key,value,k,v,slots,dtype,k_scale,v_scale):
        if k.dtype!=torch.uint8:return original_write(key,value,k,v,slots,dtype,k_scale,v_scale)
        assert dtype=='auto' and key.dtype==value.dtype==torch.bfloat16 and v.dtype==torch.uint8
        assert key.ndim==value.ndim==3 and tuple(key.shape[1:])==tuple(value.shape[1:])==(2,128)
        assert k.ndim==v.ndim==4 and tuple(k.shape[1:])==tuple(v.shape[1:])==(16,2,128)
        assert k.stride()==v.stride() and (k.stride(1),k.stride(2),k.stride(3))==(256,128,1)
        counts['encoded_writer']+=1
        return writer.triton_reshape_and_cache_flash(key,value,k,v,slots,dtype,k_scale,v_scale)
    def call(kw):
        kw=dict(kw)
        if kw['max_seqlen_q']==1:kw['num_par_softmax_segments']=(producer.decode_config['segments'] if producer.decode_config is not None else adaptive.choose_segments(kw['seqused_k'].numel()))
        return reader.unified_attention(**kw)
    def dispatch(original,route,kw):
        if kw['k'].dtype!=torch.uint8:return original_dispatch(original,route,kw)
        q,k,v,out=(kw[n] for n in ('q','k','v','out'))
        assert q.ndim==3 and tuple(q.shape[1:])==(16,128) and q.dtype==out.dtype==torch.bfloat16
        assert k.ndim==v.ndim==4 and tuple(k.shape[1:])==tuple(v.shape[1:])==(16,2,128) and v.dtype==torch.uint8
        assert kw['causal'] is True and tuple(kw['window_size'])==(-1,-1) and kw.get('softcap',0.)==0.
        assert int(kw.get('kv_quant_mode',0))==0 and not kw.get('use_td',False) and kw.get('chunk_lookback',-1)==-1
        assert all(kw.get(x) is None for x in ('alibi_slopes','qq_bias','sinks','mm_prefix_range','output_scale'))
        if kw['max_seqlen_q']<=1:counts['decode']+=1;return call(kw)
        if route is None:counts['prefill']+=1;return call(kw)
        nd=route.num_decodes
        if nd:
            decode=dict(kw);decode.update(q=q[:nd],out=out[:nd],cu_seqlens_q=kw['cu_seqlens_q'][:nd+1],max_seqlen_q=1,seqused_k=kw['seqused_k'][:nd],block_table=kw['block_table'][:nd])
            call(decode);counts['mixed']+=1
        else:counts['prefill']+=1
        prefill=dict(kw);prefill.update(q=q[nd:],out=out[nd:],cu_seqlens_q=route.prefill_cu,max_seqlen_q=route.max_prefill_query,seqused_k=kw['seqused_k'][nd:],block_table=kw['block_table'][nd:])
        return call(prefill)
    backend.triton_reshape_and_cache_flash=write;hybrid.dispatch=dispatch
    backend._soft8_pipeline_installed=dict(producer=producer,counts=counts)
    return backend._soft8_pipeline_installed
