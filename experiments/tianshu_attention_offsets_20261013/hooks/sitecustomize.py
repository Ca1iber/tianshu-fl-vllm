# SPDX-License-Identifier: Apache-2.0
"""Guarded small/medium-batch decode offsets; preserve native large batches."""
import importlib.abc
import json
import os
from pathlib import Path
import runpy
import sys

BASE=Path(__file__).resolve().parents[2]
runpy.run_path(str(BASE/'tianshu_buckets75_recheck_20261012/joint_hooks/sitecustomize.py'))


class Loader(importlib.abc.Loader):
    def __init__(self,original):self.original=original
    def create_module(self,spec):return self.original.create_module(spec)
    def exec_module(self,module):
        self.original.exec_module(module)
        helper=runpy.run_path(str(BASE/'tianshu_attention_offsets_20261013/address32.py'))
        root=Path(os.environ['TIANSHU_ADDRESS32_RECORD_DIR']);root.mkdir(parents=True,exist_ok=True)
        candidate,source=helper['load_variant'](module,root/f'generated_{os.getpid()}')
        candidate.kernel_unified_attention=helper['Stage1Kernel'](candidate.kernel_unified_attention)
        original=module.unified_attention;seen=set()
        def unified(*args,**kwargs):
            if args or kwargs.get('max_seqlen_q')!=1:return original(*args,**kwargs)
            q,k,v,out=(kwargs.get(n) for n in ('q','k','v','out'))
            cu=kwargs.get('cu_seqlens_q')
            n=cu.numel()-1 if cu is not None else 0
            scratch=[kwargs.get(name) for name in ('softmax_segm_output','softmax_segm_max','softmax_segm_expsum')]
            supported=(4<=n<=48 and all(x is not None for x in (q,k,v,out)) and q.shape[1:]==(16,128)
                       and str(q.dtype)=='torch.bfloat16' and k.shape[2:]==(2,128)
                       and k.dtype==v.dtype==q.dtype and (kwargs.get('seq_threshold_3D') or 0)>=n
                       and all(x is not None for x in scratch)
                       and kwargs.get('kv_quant_mode',0)==0
                       and kwargs.get('output_scale') is None
                       and all(kwargs.get(name) is None for name in ('q_descale','qq_bias','alibi_slopes','sinks')))
            # TritonAttentionImpl also passes expanded K/V descales for BF16.
            # KVQuantMode.NONE makes their loads dead code in the native kernel.
            if not supported:return original(**kwargs)
            tensors=[q,k,v,out,cu,kwargs['seqused_k'],kwargs['block_table']]+scratch
            try:bounds=helper['assert_safe'](tensors)
            except AssertionError:return original(**kwargs)
            result=candidate.unified_attention(**kwargs)
            key=(n,kwargs.get('num_par_softmax_segments'),tuple(q.stride()),tuple(k.stride()))
            if key not in seen:
                seen.add(key)
                (root/f'route_{os.getpid()}_{len(seen)}.json').write_text(json.dumps(dict(
                    batch=n,segments=key[1],q_stride=key[2],kv_stride=key[3],bounds=bounds,
                    kv_quant_mode=int(kwargs.get('kv_quant_mode',0)),
                    unused_kv_descales_present=[kwargs.get(name) is not None for name in ('k_descale','v_descale')],
                    generated_sha256=source['generated_sha256'],offset_bits=32,
                    pointer_types_unmodified=True,floating_point_body_unmodified=True),indent=2)+'\n')
            return result
        module.unified_attention=unified


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname!='vllm.v1.attention.ops.triton_unified_attention':return None
        for finder in tuple(sys.meta_path):
            if finder is self:continue
            method=getattr(finder,'find_spec',None)
            spec=method(fullname,path,target) if method else None
            if spec is not None:
                assert spec.loader is not None
                spec.loader=Loader(spec.loader);return spec
        return None


if os.getenv('TIANSHU_ADDRESS32_PROBE')=='1':sys.meta_path.insert(0,Finder())
