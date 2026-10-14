# SPDX-License-Identifier: Apache-2.0
"""Exercise actual producer/tile routers and probe adapter with CPU fake JITs."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

from address32 import ReducerRecorder, bounds, render_source


def main():
    base = Path(__file__).resolve().parent
    router_source = base.parent / "tianshu_decode_low_warps_20261014/launch_router.py"
    spec = importlib.util.spec_from_file_location("_cpu_actual_launch_router", router_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    name = "vllm.v1.attention.ops.triton_unified_attention"
    for i in range(1, len(name.split(".")) + 1):
        part = ".".join(name.split(".")[:i])
        assert part not in sys.modules
        fake = ModuleType(part)
        fake.__path__ = []
        sys.modules[part] = fake
        if i > 1:
            parent, attr = part.rsplit(".", 1)
            setattr(sys.modules[parent], attr, fake)
    native = sys.modules[name]
    calls = []

    class Kernel:
        def __init__(self, kind):
            self.kind = kind
            self.fn = NS(__module__=kind)
        def __getitem__(self, grid):
            def launch(**kwargs):
                calls.append((self.kind, tuple(grid), dict(kwargs)))
                return NS(metadata=NS(num_warps=kwargs.get("num_warps", 4),
                                      num_stages=kwargs.get("num_stages", 3), shared=0),
                          n_regs=85, n_spills=0)
            return launch

    native.reduce_segments = Kernel("reducer64")
    class Tensor:
        shape=(64,16,128)
        def stride(self):return (2048,128,1)
    source = base / "probe_address.py"
    cls = next(n for n in ast.walk(ast.parse(source.read_text()))
               if isinstance(n, ast.ClassDef) and n.name == "PageRouter")
    scope = dict(LaunchRouter=module.LaunchRouter, bounds=bounds, torch=NS(Tensor=Tensor),
                 case_id=0, asm_recorded=set())
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(source), "exec"), scope)
    rows = []
    for mode in ('control','producer32','both32'):
        producer_name='producer64' if mode=='control' else 'producer32'
        reducer_name='reducer32' if mode=='both32' else 'reducer64'
        recorder=ReducerRecorder(Kernel(reducer_name))
        producer=scope['PageRouter'](Kernel(producer_name),
            dict(TILE_SIZE=16,num_warps=1,num_stages=1,KV_CACHE_MODIFIER='.cv'),recorder,mode)
        producer[(96,2,8)](IS_3D=True,BLOCK_M=16,BLOCK_Q=2,num_query_heads=16,
            num_queries_per_kv=8,HEAD_SIZE=128,KV_QUANT_MODE=0,
            query_ptr=Tensor(),num_seqs=64,BLOCK_SIZE=16)
        producer.reduction.original=producer.address_reducer
        native.reduce_segments[(64,16)](TILE_SIZE=32,NUM_SEGMENTS_PER_SEQ=8,
                                       HEAD_SIZE=128,num_query_heads=16)
        assert calls[-2][0]==producer_name and calls[-2][2]['num_warps']==1
        kind,grid,kwargs=calls[-1]
        assert kind==reducer_name and grid==(64,16) and kwargs['TILE_SIZE']==16
        assert 'num_warps' not in kwargs and 'num_stages' not in kwargs
        assert recorder.metadata['actual_num_warps']==4 and recorder.metadata['actual_num_stages']==3
        assert recorder.metadata['source_module']==reducer_name
        assert producer.metadata['producer_source_module']==producer_name
        if mode!='control':assert producer.metadata['address_bounds'] and max(producer.metadata['address_bounds'])<2**31
        rows.append(dict(mode=mode,producer=producer.metadata,reducer=dict(recorder.metadata)))
    # An overflowing masked-lane bound must be rejected before candidate launch.
    class Unsafe(Tensor):
        shape=(2**31,)
        def stride(self):return (1,)
    try:bounds([Unsafe()])
    except AssertionError:pass
    else:raise AssertionError('Unsafe int32 bound accepted')
    raw_root=base.parents[2]/'service-results/20261014_tianshu/decode_cache_cv_math/20261014_decode_cache_cv_math_tianshu'
    candidates=list((raw_root/'decode_coherent_records/cache_generated').glob('*/generated_cache_policy.py'))
    assert len(candidates)==1
    generated=render_source(candidates[0].read_bytes())
    output = dict(complete=True, cpu_only=True, gpu_execution=False,
                  actual_probe_adapter_and_existing_LaunchRouter_executed=True,
                  actual_cv_source32_rendered=True,generated_sha256=hashlib.sha256(generated.encode()).hexdigest(),
                  masked_lane_overflow_rejected=True,
                  fake_native_and_JIT_not_runtime_evidence=True, configurations=rows,
                  sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (Path(__file__), source, base / "address32.py", router_source)})
    dest = base / "results/20261014/prepared_32g/CPU_DISPATCH_AUDIT_32g.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k != "configurations"}))


if __name__ == "__main__":
    main()
