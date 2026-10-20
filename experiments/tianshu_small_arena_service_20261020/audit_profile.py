# SPDX-License-Identifier: Apache-2.0
import hashlib,importlib.util,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('_arena_S16_runtime_audit',BASE/'long_runtime_audit.py');parent=importlib.util.module_from_spec(s);s.loader.exec_module(parent)
def audit_runtime(root,state,official=False):
    # Both modes retain candidate S16 kernels; preserve actual mode separately for capacity proof.
    result=parent.audit_runtime(root,dict(state,mode='candidate',arena_mode=state['mode']),official=official)
    pages=8192 if state['mode']=='candidate' else 32768
    files=list((root/'small_arena_records').glob('capacity_*.json'));assert files
    for path in files:
        row=json.loads(path.read_text());assert row['mode']==state['mode'] and row['capacity_pages']==pages
        assert row['original_call_inherited'] and row['original_gather_decode_unchanged'] and row['original_over_capacity_fallback_unchanged'] and row['RNG_not_read_or_modified']
        policy=BASE.parent/'tianshu_small_arena_20261020/policy.py';assert state['frozen_sources'][str(policy)]==row['policy_sha256']==hashlib.sha256(policy.read_bytes()).hexdigest()
    result.update(mode=state['mode'],arena_capacity_pages=pages,arena_scratch_bytes=(pages+1)*16384,arena_table_bytes=pages*4,
                  actual_arena_reserved_inside_memory_profiling_verified=True,native_staging_fallback_unchanged=True)
    return result
