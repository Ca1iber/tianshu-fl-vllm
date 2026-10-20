# SPDX-License-Identifier: Apache-2.0
import importlib.util,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('_headroom_long_runtime_audit',BASE/'long_runtime_audit.py')
parent=importlib.util.module_from_spec(s);s.loader.exec_module(parent)

def audit_runtime(root,state,official=False):
    result=parent.audit_runtime(root,dict(state,mode='candidate'),official=official)
    mode=state['mode'];expected=1/16 if mode=='candidate' else 0.
    rows=[row for path in (root/'kv_headroom_records').glob('watermark_*.json') for row in parent.read(path)]
    assert rows and all(row['mode']==mode for row in rows)
    assert any(row['eligible'] and row['after_watermark']==expected for row in rows)
    assert all(row['budget']==4096 and row['max_num_seqs']==256 and row['RNG_not_read_or_modified'] and row['original_arguments_forwarded'] and row['native_KV_admission_implementation_unchanged'] for row in rows)
    managers=[parent.read(path) for path in (root/'kv_headroom_records').glob('manager_*.json')]
    assert len(managers)==1
    manager=managers[0];assert parent.sha(manager['source'])==manager['source_sha256']==state['frozen_sources'][manager['source']]
    assert manager['watermark_argument']==expected and manager['watermark_blocks']==int(expected*manager['num_blocks'])
    assert manager['max_num_batched_tokens']==4096 and manager['max_model_len']==131072
    assert manager['native_allocate_slots_unmodified'] and manager['arguments_and_return_values_forwarded'] and manager['RNG_not_read_or_modified']
    assert state['profile_environment']['TIANSHU_KV_HEADROOM_MODE']==mode and state['profile_environment']['TIANSHU_LONG_SEGMENT_MODE']=='candidate'
    result.update(native_KV_watermark_verified=True,watermark_ratio=expected,watermark_blocks=manager['watermark_blocks'],
                  cache_blocks=manager['num_blocks'],native_allocation_logic_unchanged=True)
    return result
