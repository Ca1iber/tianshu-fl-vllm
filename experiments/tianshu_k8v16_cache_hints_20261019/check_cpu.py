# SPDX-License-Identifier: Apache-2.0
"""Check all plans through actual Producer control flow and invariant byte geometry."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from plans import MODES, PLANS, configuration
from layout import geometry

base = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_decode_sweep_CPU_codec', base.parent / 'tianshu_soft8_engine_v2_20261019/source.py')
codec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(codec)
owner = base.parent / 'tianshu_soft8_mixed_planes_20261019/writer.py'
encoder = next(n for n in ast.parse(owner.read_text()).body
    if isinstance(n, ast.FunctionDef) and n.name == '_soft8_encode_bf16')
assert ast.dump(encoder) == ast.dump(ast.parse(codec.HELPER).body[0])
tree = ast.parse((base / 'probe.py').read_text())
node = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == 'Producer')
context = [33]
namespace = dict(configuration=configuration, PLANS=PLANS, current_context=context, data={'producer_records': {}})
exec(compile(ast.Module(body=[node], type_ignores=[]), str(base / 'probe.py'), 'exec'), namespace)


class Kernel:
    arg_names=['KV_CACHE_MODIFIER']
    def __getitem__(self, grid):
        assert len(grid) == 3 and all(x > 0 for x in grid)
        def launch(**kw):
            return SimpleNamespace(metadata=SimpleNamespace(num_warps=kw['num_warps'], num_stages=kw['num_stages']), n_regs=0, n_spills=0, src=SimpleNamespace(constants={(0,):kw['KV_CACHE_MODIFIER']}), hash='CPU-fixture-only')
        return launch


cases = 0
for mode in MODES:
    assert geometry(mode) == dict(key_bytes=1, value_bytes=2, page_bytes=12288, value_offset=4096)
    for n in (1, 8, 16, 24, 32, 64):
        for length in (33, 4096, 5120):
            c = configuration(mode, n, length)
            assert c['TILE_SIZE'] in (16, 32, 64) and c['num_warps'] in (1, 2, 4)
            assert c['segments'] in (2, 4, 8, 16, 32, 64) and c['num_stages'] in (1, 2)
            context[0] = length
            producer = namespace['Producer'](Kernel())
            producer.mode = mode
            kw = dict(KV_QUANT_MODE=0, HEAD_SIZE=128, num_seqs=n, IS_3D=True,
                NUM_SEGMENTS_PER_SEQ=c['segments'], key_cache_ptr=SimpleNamespace(dtype='U8'),
                value_cache_ptr=SimpleNamespace(dtype='BF16'))
            producer[(n, 2, c['segments'])](**kw)
            record = namespace['data']['producer_records'][str((mode, n, length))]
            assert record['actual_compiled_KV_CACHE_MODIFIER']==PLANS[mode]
            assert record['compiled_kernel_hash']=='CPU-fixture-only'
            assert (record['tile'], record['warps'], record['stages']) == (c['TILE_SIZE'], c['num_warps'], c['num_stages'])
            cases += 1
assert len(PLANS)==4 and set(PLANS.values())=={'.cv','.cg','.ca',''}
for n in (1,8,16,24,32,64):
    for c in (33,4096,5120):
        assert all(configuration(m,n,c)==configuration('current',n,c) for m in MODES)
for path in base.glob('*.py'):
    ast.parse(path.read_text())
assert cases == len(PLANS) * 18
r = dict(complete=True, CPU_only=True, plans=len(PLANS), actual_Producer_control_flow_cases=cases,
    cache_hint_is_only_Plan_difference=True, actual_compiled_constexpr_checked=True,
    all_arms_use_same_K8V16_12KiB_layout=True, writer_encoder_AST_identical=True,
    actual_writer_owner_sha256=hashlib.sha256(owner.read_bytes()).hexdigest(),
    sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(base.glob('*.py'))},
    GPU_not_run=True, model_and_Math_not_run=True, adopted=False)
(base / 'CPU_GATE_32g.json').write_text(json.dumps(r, indent=2) + '\n')
print(json.dumps(r))
