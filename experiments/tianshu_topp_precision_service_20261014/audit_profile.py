# SPDX-License-Identifier: Apache-2.0
"""Check the actual4096/83-bucket/S8/SQL process and retained source proofs."""
import hashlib
import json
from pathlib import Path


def audit_profile(root, state):
    command = state['server_command']
    assert command[command.index('--max-num-batched-tokens') + 1] == '4096'
    assert command[command.index('--long-prefill-token-threshold') + 1] == '0'
    assert state['max_batched_tokens'] == 4096
    flags = state['environment_flags']
    for key in ('TIANSHU_SQL_REFLECTION_PROBE', 'TIANSHU_EXTENDED_BUCKET_PROBE',
                'TIANSHU_SEGMENT_POLICY_PROBE', 'VLLM_FL_ILUVATAR_PREFILL_BUCKETS',
                'VLLM_FL_ILUVATAR_LOGITS_GATHER'):
        assert flags[key] == '1', key
    assert flags['TIANSHU_KV16_SERVICE_MODE'] == 'candidate'
    assert flags['TIANSHU_CACHE_SERVICE_MODE'] == 'candidate'
    expected = {
        'tianshu_decode_cache_service_20261014/hooks/sitecustomize.py': 'a7ee741af7549715b4d4be016c263e34a9ec00e7f23076a5024bfd4a37234e2a',
        'tianshu_budget8192_buckets_20261014/extend_buckets.py': '3191c0cedb641a96b39c142ac2b2374fdfd8c8e7e79c36ebca035061ac909962',
        'tianshu_sql_workflow_20261014/reflect_one.py': '4cfc6a34e29e0cc220b83d79898c1da4d986e35d4eb0796c37947b1631064c02',
        'tianshu_kv16_segments_20261014/hooks/sitecustomize.py': '71b8368cca267314481f11e4efe5556b083ca5e86d6e80380ba4ab8632b54a19',
    }
    for rel, digest in expected.items():
        source = root / 'source_snapshot/segment_sources' / rel
        assert hashlib.sha256(source.read_bytes()).hexdigest() == digest, rel
    extension = [json.loads(p.read_text()) for p in (root/'extended_bucket_records').glob('extended_*.json')]
    assert extension
    for record in extension:
        assert record['helper_sha256'] == expected['tianshu_budget8192_buckets_20261014/extend_buckets.py']
        assert record['original_bucket_source_sha256'] == '966aad18aad80ef488928cafc8672cbfa2d547a832f9b151346f719ea3abc6db'
        old, new = record['original_points'], record['extended_points']
        assert len(old) == 75 and max(old) == 2048
        assert new == old + list(range(2304,8193,256)) and len(new) == 99
        assert record['changed_constants'] == [[2048,8192]]
        assert record['hidden_size_guard_preserved'] and not record['installed_sources_modified']
    sql = [json.loads(p.read_text()) for p in (root/'sql_reflection_records').glob('route_*.json')]
    assert sql and all(p['reflection_scope'] == 'one_requested_table' and
        p['helper_sha256'] == expected['tianshu_sql_workflow_20261014/reflect_one.py'] and
        not p['configuration_values_modified'] for p in sql)
    assert all(x['cache_service_mode']=='candidate' and x['actual_cache_modifier']=='.cv' for x in state['candidate_routing_records'])
    return dict(complete=True, cache_modifier='.cv', budget=4096, threshold=0, bucket_points=83,
                original75_preserved=True, extra_points=list(range(2304,4097,256)), available_extension_points=99,
                shared_SQL_verified=True, extension_records=len(extension), SQL_records=len(sql))
