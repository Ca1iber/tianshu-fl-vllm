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
    expected = {
        'tianshu_budget4096_buckets_20261014/hooks/sitecustomize.py': '17460d04858268d4dd0ec8716099d4ea062604b4a4abafdfa7789d1d062928be',
        'tianshu_budget4096_buckets_20261014/extend_buckets.py': '96e60c52545bd36e45cabb1de1b8bf35e99e5f3ce8b836f53522ee639617ae3a',
        'tianshu_sql_workflow_20261014/reflect_one.py': '4cfc6a34e29e0cc220b83d79898c1da4d986e35d4eb0796c37947b1631064c02',
        'tianshu_kv16_segments_20261014/hooks/sitecustomize.py': '71b8368cca267314481f11e4efe5556b083ca5e86d6e80380ba4ab8632b54a19',
    }
    for rel, digest in expected.items():
        source = root / 'source_snapshot/segment_sources' / rel
        assert hashlib.sha256(source.read_bytes()).hexdigest() == digest, rel
    extension = [json.loads(p.read_text()) for p in (root/'extended_bucket_records').glob('extended_*.json')]
    assert extension
    for record in extension:
        assert record['helper_sha256'] == expected['tianshu_budget4096_buckets_20261014/extend_buckets.py']
        assert record['original_bucket_source_sha256'] == '966aad18aad80ef488928cafc8672cbfa2d547a832f9b151346f719ea3abc6db'
        old, new = record['original_points'], record['extended_points']
        assert len(old) == 75 and max(old) == 2048
        assert new == old + list(range(2304,4097,256)) and len(new) == 83
        assert record['changed_constants'] == [[2048,4096]]
        assert record['hidden_size_guard_preserved'] and not record['installed_sources_modified']
    sql = [json.loads(p.read_text()) for p in (root/'sql_reflection_records').glob('route_*.json')]
    assert sql and all(p['reflection_scope'] == 'one_requested_table' and
        p['helper_sha256'] == expected['tianshu_sql_workflow_20261014/reflect_one.py'] and
        not p['configuration_values_modified'] for p in sql)
    return dict(complete=True, budget=4096, threshold=0, bucket_points=83,
                original75_preserved=True, extra_points=list(range(2304,4097,256)),
                shared_SQL_verified=True, extension_records=len(extension), SQL_records=len(sql))
