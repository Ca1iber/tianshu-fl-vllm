# SPDX-License-Identifier: Apache-2.0
"""Execute all real composed hooks with no Torch import or GPU initialization."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
from unittest.mock import patch


def main():
    base = Path(__file__).resolve().parent
    experiments = base.parent
    source = base / "hooks/sitecustomize.py"
    original_meta = list(sys.meta_path)
    original_run = runpy.run_path
    runs = []
    assert "torch" not in sys.modules
    for mode in ("control", "candidate"):
        calls = []
        def record(path, *a, **kw):
            calls.append(str(Path(path).resolve()))
            return original_run(path, *a, **kw)
        env = dict(TIANSHU_COMBO_MODE=mode, TIANSHU_MAX_SEQS_MODE=mode,
            TIANSHU_TOPP_PRECISION_SERVICE_MODE=mode,
            TIANSHU_SAMPLER_FUSION_PROBE="1" if mode == "candidate" else "0",
            TIANSHU_CACHE_SERVICE_MODE="candidate", TIANSHU_KV16_SERVICE_MODE="candidate",
            TIANSHU_EXTENDED_BUCKET_PROBE="1", TIANSHU_TOPP_LAUNCH_PROBE="1",
            TIANSHU_DECODE_COHERENT_PROBE="1", TIANSHU_SEGMENT_POLICY_PROBE="1",
            TIANSHU_SQL_REFLECTION_PROBE="1", TIANSHU_NATIVE_PREFILL_PROBE="1",
            TIANSHU_FA2_TILE_PROBE="1", TIANSHU_DECODE_LAUNCH_PROBE="1")
        try:
            with patch.dict(os.environ, env):
                runpy.run_path = record
                original_run(str(source))
            assert runpy.run_path is record
            filenames = [getattr(f.find_spec, "__code__").co_filename for f in sys.meta_path
                if getattr(getattr(f,"find_spec",None),"__code__",None)]
            for rel in ("tianshu_budget16384_20261016/hooks/sitecustomize.py",
                        "tianshu_topp_precision_service_20261014/hooks/top_p_sitecustomize.py",
                        "tianshu_decode_cache_service_20261014/hooks/decode_sitecustomize.py",
                        "tianshu_max_seqs_20261014/hooks/sitecustomize.py"):
                assert str(experiments / rel) in filenames
            assert str(experiments / "tianshu_decode_kv16_w2_20261014/hooks/sitecustomize.py") not in calls
            for rel in ("tianshu_max_seqs_20261014/hooks/sitecustomize.py",
                        "tianshu_budget16384_20261016/hooks/sitecustomize.py"):
                assert calls.count(str(experiments / rel)) == 1
            sampler = str(experiments / "tianshu_sampler_current_20261014/hooks/sitecustomize.py")
            assert (sampler in filenames) == (mode == "candidate")
            assert "torch" not in sys.modules
            runs.append(dict(mode=mode, real_hooks_executed=len(calls), source_order=calls,
                source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in calls},
                only_one_base_execution=True, CV_hook_present=True, strict_TopP_hook_present=True,
                bucket131_hook_present=True, sampler_installed=mode=="candidate", Torch_imported=False))
        finally:
            runpy.run_path = original_run
            sys.meta_path[:] = original_meta
    out = base / "results/20261016/prepared_32g";out.mkdir(parents=True,exist_ok=True)
    result = dict(complete=True,cpu_only=True,actual_hook_code_executed=True,
        GPU_numerical_or_performance_claim=False,run_path_restored_both_modes=True,runs=runs)
    (out / "CPU_COMPOSITION_AUDIT_32g.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="runs"}))


if __name__ == "__main__": main()
