# SPDX-License-Identifier: Apache-2.0
"""Check invalid zero-token forwarding using the installed native allocator body."""
import ast
import argparse
import importlib.util
import hashlib
import json
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace as NS

BASE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('output', type=Path)
parser.add_argument('--policy-file', type=Path, default=BASE.parent / 'tianshu_future_kv_reservation_20261020/policy.py')
parser.add_argument('--require-preserved', action='store_true')
args = parser.parse_args()
spec = importlib.util.spec_from_file_location('_checked_future_policy', args.policy_file)
policy = importlib.util.module_from_spec(spec); spec.loader.exec_module(policy)
Reservation = policy.Reservation
native = Path('/usr/local/lib/python3.12/site-packages/vllm/v1/core/kv_cache_manager.py')
cls = next(n for n in ast.parse(native.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'KVCacheManager')
node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'allocate_slots')
node.decorator_list = []
space = {}
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])), str(native), 'exec'), space)
root = args.output.resolve()
assert root.is_relative_to('/workspace/benchmark-results') and not root.exists()
root.mkdir(parents=True)
rows = []
for mode in ('native', 'control', 'candidate'):
    unit = NS(block_size=16, req_to_blocks={})
    owner = NS(max_model_len=131072, empty_kv_cache_blocks=NS(blocks=((),)),
        block_pool=NS(num_gpu_blocks=1001, get_num_free_blocks=lambda: 1),
        coordinator=NS(single_type_managers=[unit], get_num_blocks_to_allocate=lambda **kw: 200),
        free=lambda request: None, pop_blocks_for_free=lambda request: None)
    owner.allocate_slots = MethodType(space['allocate_slots'], owner)
    if mode != 'native': Reservation(owner, mode, root / mode)
    request = NS(request_id='zero', num_prompt_tokens=128, max_tokens=1024, num_tokens=128,
        num_computed_tokens=0, sampling_params=object(), status=NS(name='WAITING'))
    try:
        answer = owner.allocate_slots(request, 0, full_sequence_must_fit=True)
        rows.append(dict(mode=mode, raised=None, returned_none=answer is None))
    except Exception as e:
        rows.append(dict(mode=mode, raised=type(e).__name__, message=str(e)))
assert rows[0]['raised'] == rows[1]['raised'] == 'ValueError'
result = dict(complete=True, CPU_only=True, adopted=False, native_allocator_AST_body_executed=True,
    native_source_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
    policy_source_sha256=hashlib.sha256(args.policy_file.read_bytes()).hexdigest(),
    zero_token_error_preserved=rows[2]['raised']=='ValueError', rows=rows)
(root/'result.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
if args.require_preserved:
    assert result['zero_token_error_preserved']
    assert rows[0]['message'] == rows[1]['message'] == rows[2]['message']
