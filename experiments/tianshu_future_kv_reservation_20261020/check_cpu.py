# SPDX-License-Identifier: Apache-2.0
"""CPU admission/lifetime simulation with the installed native block predictor AST."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import random
from types import MethodType, SimpleNamespace as NS
from policy import Reservation, supported

BASE = Path(__file__).resolve().parent
NATIVE = Path('/usr/local/lib/python3.12/site-packages/vllm/v1/core')
p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); args = p.parse_args()
root = args.output.resolve(); assert root.is_relative_to('/workspace/benchmark-results') and not root.exists(); root.mkdir(parents=True)
module = ast.parse((NATIVE / 'single_type_kv_cache_manager.py').read_text())
cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == 'SingleTypeKVCacheManager')
node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'get_num_blocks_to_allocate')
node.decorator_list = []
namespace = {'cdiv': lambda n, d: (n + d - 1) // d}
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])), str(NATIVE / 'single_type_kv_cache_manager.py'), 'exec'), namespace)
query = namespace['get_num_blocks_to_allocate']


class Unit:
    def __init__(self):
        self.block_size = 16; self.req_to_blocks = {}; self.num_cached_block = {}
        self._max_admission_blocks_per_request = None; self.use_eagle = False
        self.kv_cache_spec = type('FullAttentionSpec', (), {})()
    @staticmethod
    def get_num_skipped_tokens(total): return 0
    @staticmethod
    def _get_num_evictable_blocks(blocks): return sum(b.ref_cnt == 0 and not b.is_null for b in blocks)


class Manager:
    def __init__(self, capacity=300):
        self.unit = type('FullAttentionManager', (Unit,), {})()
        self.unit.get_num_blocks_to_allocate = MethodType(query, self.unit)
        self.free_blocks = capacity; self.calls = []; self.max_model_len = 131072; self.watermark_blocks = 0
        self.block_pool = NS(num_gpu_blocks=capacity + 1, get_num_free_blocks=lambda: self.free_blocks)
        self.empty_kv_cache_blocks = NS(blocks=((),))
        def predict(request_id, num_tokens, new_computed_blocks, num_encoder_tokens, total_computed_tokens, num_tokens_main_model, apply_admission_cap=False):
            assert num_encoder_tokens == 0 and len(new_computed_blocks) == 1
            return self.unit.get_num_blocks_to_allocate(request_id, num_tokens, new_computed_blocks[0], total_computed_tokens, num_tokens_main_model, apply_admission_cap)
        self.coordinator = NS(single_type_managers=[self.unit], get_num_blocks_to_allocate=predict)
        self.answer = object(); self.exception = None
    def allocate_slots(self, request, num_new_tokens, num_new_computed_tokens=0, new_computed_blocks=None,
                       num_lookahead_tokens=0, num_external_computed_tokens=0, delay_cache_blocks=False,
                       num_encoder_tokens=0, full_sequence_must_fit=False, reserved_blocks=0, has_scheduled_reqs=True):
        self.calls.append(dict(request=request, num_new_tokens=num_new_tokens, full_sequence_must_fit=full_sequence_must_fit,
            reserved_blocks=reserved_blocks, num_lookahead_tokens=num_lookahead_tokens))
        if self.exception is not None: raise self.exception
        prefix = new_computed_blocks.blocks[0] if new_computed_blocks is not None else ()
        computed = request.num_computed_tokens + num_new_computed_tokens
        if full_sequence_must_fit and self.unit.get_num_blocks_to_allocate(request.request_id, request.num_tokens, prefix, computed, request.num_tokens, True) > self.free_blocks:
            return None
        target = min(computed + num_new_tokens + num_lookahead_tokens, self.max_model_len)
        needed = self.unit.get_num_blocks_to_allocate(request.request_id, target, prefix, computed, target)
        if needed > self.free_blocks - reserved_blocks: return None
        held = self.unit.req_to_blocks.setdefault(request.request_id, [])
        for block in prefix:
            if block.ref_cnt == 0 and not block.is_null: self.free_blocks -= 1
            block.ref_cnt += 1
            held.append(block)
        while len(held) * 16 < target:
            held.append(NS(ref_cnt=1, is_null=False)); self.free_blocks -= 1
        self.unit.num_cached_block[request.request_id] = len(held)
        assert self.free_blocks >= 0
        return self.answer
    def free(self, request):
        blocks = self.unit.req_to_blocks.pop(request.request_id, [])
        self.unit.num_cached_block.pop(request.request_id, None)
        for block in blocks:
            block.ref_cnt -= 1
            if block.ref_cnt == 0: self.free_blocks += 1
        return self.answer
    def pop_blocks_for_free(self, request):
        blocks = self.unit.req_to_blocks.pop(request.request_id, [])
        self.unit.num_cached_block.pop(request.request_id, None)
        return blocks


def req(key, prompt=1600, output=160, status='WAITING'):
    return NS(request_id=key, num_prompt_tokens=prompt, max_tokens=output, num_tokens=prompt,
              num_computed_tokens=0, status=NS(name=status), sampling_params=object())


def scheduler(manager):
    return NS(kv_cache_manager=manager, connector=None, ec_connector=None, is_encoder_decoder=False,
        use_eagle=False, num_lookahead_tokens=0, scheduler_reserve_full_isl=True, max_model_len=131072,
        vllm_config=NS(model_config=NS(model='/workspace/MiniCPM5-2B')),
        parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1), dcp_world_size=1, pcp_world_size=1)

rows = []
for mode in ('control', 'candidate'):
    m = Manager(); c = Reservation(m, mode, root / mode)
    a, b, d = (req(x) for x in ('a', 'b', 'd'))
    assert m.allocate_slots(a, 256, full_sequence_must_fit=True) is m.answer
    assert m.allocate_slots(b, 256, full_sequence_must_fit=True) is m.answer
    calls = len(m.calls)
    answer = m.allocate_slots(d, 256, full_sequence_must_fit=True)
    assert (answer is None) == (mode == 'candidate')
    if mode == 'candidate': assert len(m.calls) == calls
    else: assert answer is m.answer
    if mode == 'candidate':
        # Advancing a claim turns virtual blocks into real blocks, never double counts.
        before = m.free_blocks - c.outstanding()
        a.status.name = 'RUNNING'; a.num_computed_tokens = 256
        assert m.allocate_slots(a, 800) is m.answer
        assert m.free_blocks - c.outstanding() == before
        assert m.free(a) is m.answer and 'a' not in c.plans
        assert m.allocate_slots(d, 256, full_sequence_must_fit=True) is m.answer
        reserved_before_pop = c.outstanding()
        popped = m.pop_blocks_for_free(b)
        assert popped and 'b' not in c.plans and c.outstanding() < reserved_before_pop
    c.emit(force=True)
    rows.append(dict(case='partial_prefill_future_admission_lifetime', mode=mode, passed=True, rejections=c.stats['gate_rejections']))

m = Manager(); s = scheduler(m); assert supported(s)
for key, value in [('connector', object()), ('ec_connector', object()), ('use_eagle', True), ('is_encoder_decoder', True), ('num_lookahead_tokens', 1), ('scheduler_reserve_full_isl', False), ('max_model_len', 1024), ('dcp_world_size', 2), ('pcp_world_size', 2)]:
    original = getattr(s, key); setattr(s, key, value); assert not supported(s), key; setattr(s, key, original)
    rows.append(dict(case='setup_fallback_' + key, passed=True))
for key in ('tensor_parallel_size', 'pipeline_parallel_size'):
    setattr(s.parallel_config, key, 2); assert not supported(s); setattr(s.parallel_config, key, 1)
    rows.append(dict(case='setup_fallback_' + key, passed=True))

for label, change in [('running', {'status': NS(name='RUNNING')}), ('pooling', {'sampling_params': None}),
    ('zero_output', {'max_tokens': 0}), ('zero_prompt', {'num_prompt_tokens': 0}), ('invalid_plan', {'num_tokens': 9999})]:
    m = Manager(); c = Reservation(m, 'candidate', root / label); r = req(label)
    for key, value in change.items(): setattr(r, key, value)
    # Some native paths can legitimately reject invalid sizes; check forwarding itself.
    m.allocate_slots(r, 1, full_sequence_must_fit=True)
    assert len(m.calls) == 1 and c.stats['admission_checks'] == 0
    rows.append(dict(case='request_passthrough_' + label, passed=True))
for name in ('num_lookahead_tokens', 'num_external_computed_tokens', 'delay_cache_blocks', 'num_encoder_tokens'):
    m = Manager(); c = Reservation(m, 'candidate', root / name); r = req(name)
    m.allocate_slots(r, 1, full_sequence_must_fit=True, **{name: 1})
    assert len(m.calls) == 1 and c.stats['admission_checks'] == 0
    rows.append(dict(case='call_passthrough_' + name, passed=True))

m = Manager(); c = Reservation(m, 'candidate', root / 'exception'); error = RuntimeError('sentinel'); m.exception = error
try: m.allocate_slots(req('err'), 16, full_sequence_must_fit=True)
except RuntimeError as actual: assert actual is error
else: raise AssertionError('native exception was swallowed')
assert not c.plans
rows.append(dict(case='native_exception_identity', passed=True))

# Actual installed native block predictor, prefix evictability, native reservations,
# and arbitrary allocation lengths checked against a separate integer accounting.
rng = random.Random(20261020)
for i in range(512):
    m = Manager(capacity=600); c = Reservation(m, 'candidate', root / 'random')
    r = req('new', prompt=rng.randrange(64, 2500), output=rng.randrange(1, 2000))
    old_held = rng.randrange(1, 100); old_future = old_held + rng.randrange(0, 200)
    c.plans['old'] = old_future
    m.unit.req_to_blocks['old'] = [NS(ref_cnt=1, is_null=False) for _ in range(old_held)]
    m.free_blocks -= old_held
    prefix_count = rng.randrange(0, min(r.num_prompt_tokens // 16, 30) + 1)
    prefix = [NS(ref_cnt=rng.randrange(0, 2), is_null=False) for _ in range(prefix_count)]
    evictable = sum(b.ref_cnt == 0 for b in prefix)
    native_reserved = rng.randrange(0, 100)
    expected_needed = max((r.num_prompt_tokens + r.max_tokens + 15) // 16 - prefix_count, 0) + evictable
    expected_required = expected_needed + old_future - old_held + native_reserved
    proposal = c.proposal((r, 16), dict(full_sequence_must_fit=True, new_computed_blocks=NS(blocks=(prefix,)), num_new_computed_tokens=16 * prefix_count, reserved_blocks=native_reserved))
    assert proposal is not None and proposal[2] == expected_needed and proposal[4] == expected_required
    answer = m.allocate_slots(r, 16, full_sequence_must_fit=True, new_computed_blocks=NS(blocks=(prefix,)), num_new_computed_tokens=16 * prefix_count, reserved_blocks=native_reserved)
    # Decision comparison uses the pre-call free count; native call identities are separate.
    denied = expected_required > 600 - old_held
    assert (answer is None) == denied
    assert c.stats['gate_rejections'] == int(denied)
rows.append(dict(case='independent_future_prefix_and_reserved_accounting', count=512, passed=True))

sources = [BASE / 'policy.py', BASE / 'check_cpu.py', NATIVE / 'kv_cache_manager.py', NATIVE / 'kv_cache_coordinator.py', NATIVE / 'single_type_kv_cache_manager.py', NATIVE / 'sched/scheduler.py']
gate = dict(complete=True, CPU_only=True, GPU_and_service_not_run=True, adopted=False,
    native_block_predictor_AST_used=True, native_manager_body_not_executed=True,
    request_data_and_RNG_not_accessed=True, random_accounting_cases=512, cases=rows,
    sources={str(x): hashlib.sha256(x.read_bytes()).hexdigest() for x in sources})
(root / 'CPU_GATE_32g.json').write_text(json.dumps(gate, indent=2) + '\n')
(BASE / 'CPU_GATE_32g.json').write_text(json.dumps(gate, indent=2) + '\n')
print(json.dumps(dict(complete=True, CPU_only=True, named_cases=len(rows), random_cases=512)))
