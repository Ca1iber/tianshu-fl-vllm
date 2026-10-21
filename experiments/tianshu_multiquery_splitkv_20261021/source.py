# SPDX-License-Identifier: Apache-2.0
"""Permit a guarded small multi-query SplitKV route; numerical JIT AST unchanged."""
import ast,hashlib,importlib.util,sys
from pathlib import Path
PARENT=Path('/workspace/benchmark-results/20261020_k8v16_pair_load_GPU_tianshu/current_reader/generated_current.py')
PARENT_SHA='7b94b63e0b33a2775b9c0cd043b51120fbbb6b35a81f5134897e00fbea4cf106'
HELPERS=Path('/usr/local/lib/python3.12/site-packages/vllm/v1/attention/ops/triton_attention_helpers.py')
CONFIGS=[dict(block_m=m,tile=t,warps=w,stages=1) for m,t,w in ((32,32,2),(32,64,2),(32,64,4),(64,32,4),(64,64,4),(64,64,8))]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def render(mode):
    assert mode in ('control','candidate') and sha(PARENT)==PARENT_SHA
    original=PARENT.read_text();text=original;edits=[]
    if mode=='candidate':
        before='        or max_seqlen_q > 1\n'
        after='''        or max_seqlen_q > 8
        or num_queries_per_kv != 8 or head_size != 128
        or num_query_heads != 16 or q.dtype != torch.bfloat16
        or num_par_softmax_segments <= 0 or (num_par_softmax_segments & (num_par_softmax_segments - 1)) != 0
        or k.dtype != torch.uint8 or v.dtype != torch.bfloat16
        or not softmax_segm_output.is_contiguous()
        or not softmax_segm_max.is_contiguous() or not softmax_segm_expsum.is_contiguous()
        or softmax_segm_output.dtype != torch.float32
        or softmax_segm_max.dtype != torch.float32 or softmax_segm_expsum.dtype != torch.float32
        or softmax_segm_output.ndim != 4 or softmax_segm_max.ndim != 3 or softmax_segm_expsum.ndim != 3
        or softmax_segm_output.shape[0] < q.shape[0]
        or softmax_segm_max.shape[0] < q.shape[0] or softmax_segm_expsum.shape[0] < q.shape[0]
        or tuple(softmax_segm_output.shape[1:]) != (num_query_heads, num_par_softmax_segments, head_size_padded)
        or tuple(softmax_segm_max.shape[1:]) != (num_query_heads, num_par_softmax_segments)
        or tuple(softmax_segm_expsum.shape[1:]) != (num_query_heads, num_par_softmax_segments)
'''
        assert text.count(before)==1;text=text.replace(before,after);edits.append((before,after))
    inverse=text
    for a,b in reversed(edits):assert inverse.count(b)==1;inverse=inverse.replace(b,a)
    assert inverse==original
    funcs=lambda t:{n.name:ast.dump(n,include_attributes=False) for n in ast.parse(t).body if isinstance(n,ast.FunctionDef)}
    a,b=funcs(original),funcs(text);assert set(a)==set(b)
    changed=sorted(n for n in a if a[n]!=b[n]);assert changed==(['unified_attention'] if mode=='candidate' else [])
    return text,dict(mode=mode,parent=str(PARENT),parent_sha256=PARENT_SHA,generated_sha256=hashlib.sha256(text.encode()).hexdigest(),
        changed_functions=changed,numerical_JIT_AST_unchanged=True,all_address_widths_and_KV_codec_unchanged=True,
        original_source_reversible=True,scratch_capacity_checked_before_selection=True)
def load(root,mode):
    text,proof=render(mode);root=Path(root);root.mkdir(parents=True,exist_ok=False)
    p=root/'generated_reader.py';p.write_text(text)
    name='_small_multiquery_'+hashlib.sha256(str(p).encode()).hexdigest()[:16]
    s=importlib.util.spec_from_file_location(name,p);module=importlib.util.module_from_spec(s);sys.modules[name]=module;s.loader.exec_module(module)
    return module,dict(proof,generated_source=str(p))
def segments(n):
    assert 1<=n<=64
    return 64 if n==1 else 32 if n<=3 else 16 if n<=8 else 8 if n<=16 else 16 if n<=24 else 4
