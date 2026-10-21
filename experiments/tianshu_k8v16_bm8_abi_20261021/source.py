# SPDX-License-Identifier: Apache-2.0
"""Independent small query-row tile and direct decode mapping, all KV math retained."""
import ast,hashlib,importlib.util,sys
from pathlib import Path
PARENT_SHA='f1df731bae6cf48d5f38eef0a0adb5863ade2145667aab1ac976b9d473bdb002'
MODES=('current','bm8','bm16_direct','bm8_direct')
SHORT_SPLITS={m:4 for m in MODES}
def configuration(mode,n):
 assert mode in MODES and 1<=n<=64
 return dict(TILE_SIZE=16,num_warps=1,num_stages=1)
def configured_segments(mode,n,parent_segments):
 assert mode in MODES and n>0
 return parent_segments

def render(parent,mode):
 parent=Path(parent);original=parent.read_text();assert hashlib.sha256(original.encode()).hexdigest()==PARENT_SHA and mode in MODES
 text=original
 old='''    (
        seq_idx,
        q_block_local_idx,
        cur_batch_in_all_start_index,
        cur_batch_query_len,
        seq_len,
    ) = resolve_seq_and_query_len(
        query_start_len_ptr, seq_lens_ptr, q_block_global_idx, num_seqs, BLOCK_Q
    )
'''
 new='''    if IS_3D:
        seq_idx = q_block_global_idx
        q_block_local_idx = 0
        cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
        cur_batch_query_len = tl.load(query_start_len_ptr + seq_idx + 1) - cur_batch_in_all_start_index
        seq_len = tl.load(seq_lens_ptr + seq_idx)
    else:
        (
            seq_idx,
            q_block_local_idx,
            cur_batch_in_all_start_index,
            cur_batch_query_len,
            seq_len,
        ) = resolve_seq_and_query_len(
            query_start_len_ptr, seq_lens_ptr, q_block_global_idx, num_seqs, BLOCK_Q
        )
'''
 direct=mode.endswith('_direct');small=mode.startswith('bm8')
 if direct:
  assert text.count(old)==1;text=text.replace(old,new)
 anchor='''    grid: tuple[Any, ...]
'''
 insert=''
 if small:insert+='''    if use_3d and num_queries_per_kv == 8 and head_size == 128 and num_seqs > 24:
        BLOCK_M = 8
        BLOCK_Q = 1
        total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs
'''
 if direct:insert+='''    if use_3d:
        total_num_q_blocks = num_seqs
'''
 assert text.count(anchor)==1
 if insert:text=text.replace(anchor,insert+'\n'+anchor)
 # Inverse exact-text proof: all device math beyond sequence resolver unchanged.
 inverse=text
 if insert:inverse=inverse.replace(insert+'\n'+anchor,anchor)
 if direct:inverse=inverse.replace(new,old)
 assert inverse==original
 tree_before,tree_after=ast.parse(original),ast.parse(text)
 before={n.name:ast.dump(n) for n in tree_before.body if isinstance(n,ast.FunctionDef)};after={n.name:ast.dump(n) for n in tree_after.body if isinstance(n,ast.FunctionDef)}
 changed=[name for name in before if before[name]!=after[name]]
 assert set(changed)==({'unified_attention'} if small and not direct else {'unified_attention','kernel_unified_attention'} if direct else set()),changed
 return text,dict(mode=mode,parent_sha256=PARENT_SHA,generated_sha256=hashlib.sha256(text.encode()).hexdigest(),changed_functions=changed,function_ABI_unchanged=True,only_decode_query_row_tile_and_sequence_mapping=True,BM8_only_num_seqs_above24=small,direct_sequence_mapping_all_3D=direct,all_other_functions_AST_unchanged=True,attention_dot_softmax_reducer_codec_and_writer_text_retained=True,short_active_segments=4,segment_capacity_and_KV_writer_and_format_unchanged=True,Math_or_service_not_claimed=True)

def load(parent,directory,mode):
 text,proof=render(parent,mode);directory=Path(directory);directory.mkdir(parents=True,exist_ok=False);path=directory/f'generated_{mode}.py';path.write_text(text)
 name='_BM8_decode_'+hashlib.sha256(str(path).encode()).hexdigest()[:16];s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m,dict(proof,generated_source=str(path))
