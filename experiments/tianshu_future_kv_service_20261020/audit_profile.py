# SPDX-License-Identifier: Apache-2.0
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent
parent_path = BASE.parent / 'tianshu_memory_combo_service_20261020'
sys.path.insert(0, str(parent_path))
spec = importlib.util.spec_from_file_location('_future_native_S16_audit', parent_path / 'long_runtime_audit.py')
parent = importlib.util.module_from_spec(spec); spec.loader.exec_module(parent)


def read(path):
    return json.loads(Path(path).read_text())


def audit_runtime(root, state, official=False):
    assert len(state['profile_environment']) == 52
    result = parent.audit_runtime(root, dict(state, mode='candidate', arena_mode='candidate', seq_limit=64), official=official)
    assert result['actual_FL_Graph_capture_and_replay_verified']
    rows = [read(p) for p in (root / 'small_arena_records').glob('capacity_*.json')]
    assert rows and all(r['mode'] == 'candidate' and r['capacity_pages'] == 8192 for r in rows)
    rows = [read(p) for p in (root / 'seq_limit_records').glob('scheduler_*.json')]
    assert len(rows) == 1
    scheduler = rows[0]
    assert scheduler['max_num_running_reqs'] == scheduler['config_max_num_seqs'] == 64
    assert scheduler['max_num_batched_tokens'] == scheduler['max_num_scheduled_tokens'] == 4096
    assert scheduler['async_scheduling'] is True and scheduler['KV_watermark_blocks'] == 0
    # The inner record is intentionally before installation of our instance wrapper.
    assert scheduler['original_scheduler_and_allocation_methods_unchanged']
    rows = [read(p) for p in (root / 'future_kv_records').glob('future_reservations_*.json')]
    assert len(rows) == 1
    row = rows[0]
    assert row['installed'] and row['mode'] == state['mode']
    assert row['block_size'] == 16 and row['KV_num_blocks'] == scheduler['KV_num_blocks']
    assert row['max_model_len'] == 131072 and row['only_full_attention_single_pool']
    assert row['native_allocator_sha256'] == state['frozen_sources'][row['native_allocator_source']]
    assert row['native_allocator_sha256'] == hashlib.sha256(Path(row['native_allocator_source']).read_bytes()).hexdigest() == '2e91e77f84dfd9d4bed79f50392be72559207f54e845a3d3402baf88dc529320'
    assert row['admission_checks'] >= 194 and row['successful_admissions'] >= 194
    assert row['tracked_requests'] == row['outstanding_blocks'] == 0
    assert row['no_KV_data_tokens_or_RNG_access'] and row['running_allocations_forwarded']
    if state['mode'] == 'control' or state['workload']['input_tokens'] == 4096:
        assert row['gate_rejections'] == 0
    result.update(mode=state['mode'], max_num_seqs=64, arena_capacity_pages=8192,
        native_scheduler_limit_verified=True, native_watermark_blocks=0,
        actual_arena_reserved_inside_memory_profiling_verified=True,
        actual_KV_blocks=scheduler['KV_num_blocks'], future_reservation=row,
        allocator_wrapper_installed_after_inner_native_scheduler_record=True)
    return result
