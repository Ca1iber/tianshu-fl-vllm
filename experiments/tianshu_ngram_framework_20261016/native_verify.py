# SPDX-License-Identifier: Apache-2.0
"""Correct L0 fallback and guarded two-query split-K using native arithmetic."""
import ast
import hashlib
from pathlib import Path
import textwrap

NATIVE_SHA='bab2c0b9f9a7793ca411796b48ebb4734433160e7e99eabf46ddb7f1c30cb6e0'
CACHE_SHA='8deb8967fdacf354f3b38e6a9687163493f6163764f3e8663dce916250cd1289'


def render(raw):
    assert hashlib.sha256(raw).hexdigest()==NATIVE_SHA
    tree=ast.parse(raw.decode());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='unified_attention')
    before=textwrap.dedent('\n'.join(raw.decode().splitlines()[node.lineno-1:node.end_lineno])+'\n')
    a,b='or max_seqlen_q > 1','or max_seqlen_q > 2'
    assert before.count(a)==1;after=before.replace(a,b)
    assert after.replace(b,a)==before
    return after,dict(original_source_sha256=NATIVE_SHA,generated_method_sha256=hashlib.sha256(after.encode()).hexdigest(),
        only_query_dispatch_limit_changed=True,attention_and_reduction_kernels_unchanged=True)


def install(native,cache,backend,directory,*,enable_split,record=None):
    import torch
    assert hashlib.sha256(Path(cache.__file__).read_bytes()).hexdigest()==CACHE_SHA
    original_kernel=native.kernel_unified_attention
    original_unified=backend.unified_attention
    seen=set()
    def supported(kw):
        return (not kw['IS_3D'] and kw['num_query_heads']==16 and kw['num_queries_per_kv']==8
            and kw['HEAD_SIZE']==128 and kw['KV_QUANT_MODE']==0
            and kw['query_ptr'].dtype==kw['key_cache_ptr'].dtype==kw['value_cache_ptr'].dtype==torch.bfloat16
            and kw['key_cache_ptr'].shape[1:]==(16,2,128)
            and kw['USE_CAUSAL'] and not kw['USE_PER_SEQ_CAUSAL']
            and not kw['USE_SINKS'] and not kw['USE_TD'] and not kw['USE_TD_QO']
            and not kw['USE_ALIBI_SLOPES'] and not kw['USE_SOFTCAP']
            and not kw['USE_QQ_BIAS'] and not kw['USE_MM_PREFIX'] and not kw['SLIDING_WINDOW'])
    class L0Router:
        def __getitem__(self,grid):
            def launch(*args,**kw):
                if args or not supported(kw):return original_kernel[grid](*args,**kw)
                options=dict(kw)
                if kw['query_ptr'].shape[0]>=16*kw['num_seqs']:
                    options.update(BLOCK_M=256,BLOCK_Q=32,TILE_SIZE=128,num_warps=16,num_stages=1)
                    grid2=(kw['query_ptr'].shape[0]//32+kw['num_seqs'],grid[1])
                else:
                    options.update(BLOCK_M=16,BLOCK_Q=2,TILE_SIZE=32,num_warps=4,num_stages=1);grid2=grid
                result=cache.kernel_unified_attention[grid2](**options)
                key=('L0',kw['query_ptr'].shape[0],kw['num_seqs'])
                if record and key not in seen:
                    seen.add(key);record(dict(route='L0_2D',tokens=key[1],num_seqs=key[2],grid=list(grid2),BLOCK_M=options['BLOCK_M'],TILE_SIZE=options['TILE_SIZE'],n_regs=getattr(result,'n_regs',None),n_spills=getattr(result,'n_spills',None),source_sha256=CACHE_SHA))
                return result
            return launch
    native.kernel_unified_attention=L0Router()
    if not enable_split:return native.kernel_unified_attention
    text,derivation=render(Path(native.__file__).read_bytes())
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);path=directory/'generated_split_unified.py'
    if path.exists():assert path.read_text()==text
    else:path.write_text(text)
    namespace=dict(vars(native));exec(compile('from __future__ import annotations\n'+text,str(path),'exec'),namespace)
    split=namespace['unified_attention']
    scratch=torch.empty((128,16,16,128),device='cuda',dtype=torch.float32)
    maxima=torch.empty((128,16,16),device='cuda',dtype=torch.float32);sums=torch.empty_like(maxima)
    def dispatch(*a,**kw):
        if a:return original_unified(*a,**kw)
        q,k,v=kw['q'],kw['k'],kw['v'];n=kw['seqused_k'].numel()
        selected=(kw['max_seqlen_q']==2 and 1<=n<=64 and n<=q.shape[0]<=2*n and q.shape[0]<=128
            and q.shape[1:]==(16,128) and k.shape[1:]==(16,2,128)
            and q.dtype==k.dtype==v.dtype==torch.bfloat16 and int(kw.get('kv_quant_mode',0))==0
            and kw['causal'] and kw['window_size']==(-1,-1) and kw.get('softcap',0.)==0.
            and kw.get('alibi_slopes') is None and kw.get('output_scale') is None
            and kw.get('qq_bias') is None and kw.get('sinks') is None and kw.get('mm_prefix_range') is None
            and kw.get('per_seq_causal') is None and not kw.get('use_td',False))
        if not selected:return original_unified(**kw)
        segments=8 if n>=8 else 16
        options=dict(kw,seq_threshold_3D=64,num_par_softmax_segments=segments,
            softmax_segm_output=scratch,softmax_segm_max=maxima,softmax_segm_expsum=sums)
        result=split(**options)
        key=('split',q.shape[0],n)
        if record and key not in seen:
            seen.add(key);record(dict(route='Q2_split',tokens=q.shape[0],num_seqs=n,segments=segments,
                scratch_capacity_tokens=128,scratch_allocated_segments=16,scratch_bytes=scratch.numel()*4+maxima.numel()*8,
                compact_segment_addressing_preserved=True,derivation=derivation))
        return result
    backend.unified_attention=dispatch
    # Keep all resident buffers alive through capture and graph replay.
    dispatch._tianshu_scratch=(scratch,maxima,sums)
    return dispatch
