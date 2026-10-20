# SPDX-License-Identifier: Apache-2.0
import hashlib,importlib.util,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('_combo_S16_runtime_audit',BASE/'long_runtime_audit.py');parent=importlib.util.module_from_spec(s);s.loader.exec_module(parent)
def read(p):return json.loads(Path(p).read_text())
def audit_runtime(root,state,official=False):
    assert len(state['profile_environment'])==50
    seqs=64 if state['mode']=='candidate' else 256
    pages=8192 if state['mode']=='candidate' else 32768
    result=parent.audit_runtime(root,dict(state,mode='candidate',arena_mode=state['mode'],seq_limit=seqs),official=official)
    rows=[read(p) for p in (root/'small_arena_records').glob('capacity_*.json')];assert rows
    for row in rows:
        assert row['mode']==state['mode'] and row['capacity_pages']==pages and row['original_call_inherited'] and row['original_gather_decode_unchanged'] and row['original_over_capacity_fallback_unchanged']
    defaults=[r for p in (root/'seq_limit_records').glob('seq_default_*.json') for r in read(p)];assert defaults
    for row in defaults:
        assert row['mode']==state['mode'] and row['eligible'] and row['before_max_num_seqs'] is None
        assert row['after_native_max_num_seqs']==256 and row['after_max_num_seqs']==seqs and row['actual_budget']==4096
        assert row['default_applied']==(state['mode']=='candidate') and row['native_budget_policy_called_once']
        assert row['original_arguments_and_return_value_forwarded'] and row['RNG_not_read_or_modified']
    scheduler=[read(p) for p in (root/'seq_limit_records').glob('scheduler_*.json')];assert len(scheduler)==1
    row=scheduler[0]
    assert row['max_num_running_reqs']==row['config_max_num_seqs']==seqs
    assert row['max_num_scheduled_tokens']==row['max_num_batched_tokens']==4096 and row['async_scheduling'] is True
    assert row['max_model_len']==131072 and row['KV_watermark_blocks']==0
    assert row['original_scheduler_and_allocation_methods_unchanged'] and row['arguments_forwarded'] and row['RNG_not_read_or_modified']
    for key,expected in [('native_source','df78076164fe02a63876d7690cefac810f1ba4d8fb58afc15e786c1608064090'),('allocate_slots_source','2e91e77f84dfd9d4bed79f50392be72559207f54e845a3d3402baf88dc529320')]:
        assert row[key+'_sha256']==state['frozen_sources'][row[key]]==hashlib.sha256(Path(row[key]).read_bytes()).hexdigest()==expected
    result.update(mode=state['mode'],max_num_seqs=seqs,arena_capacity_pages=pages,arena_scratch_bytes=(pages+1)*16384,arena_table_bytes=pages*4,
                  native_scheduler_limit_verified=True,actual_arena_reserved_inside_memory_profiling_verified=True,
                  native_staging_fallback_unchanged=True,native_watermark_blocks=0,actual_KV_blocks=row['KV_num_blocks'])
    return result
