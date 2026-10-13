# SPDX-License-Identifier: Apache-2.0
"""LM-head BF16 NT GEMM: actual vocab, independent CPU checks, Graph timing."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import statistics
import subprocess


def check(actual, reference):
    diff = (actual.float() - reference.float()).abs()
    bad = diff > (.02 + .01 * reference.float().abs())
    return dict(passed=not bool(bad.any()), max_abs=float(diff.max()), failures=int(bad.sum()), atol=.02, rtol=.01)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); root = args.output.resolve(); assert not root.exists()
    assert "No running processes found" in subprocess.check_output(["ixsmi"], text=True)
    deadline = datetime.fromisoformat(json.loads(Path("/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json").read_text())["stop_at_utc"])
    assert (deadline - datetime.now(timezone.utc)).total_seconds() > 1800
    signal.alarm(1800); root.mkdir(parents=True)
    import torch
    import vllm_fl
    import flag_gems
    import inspect
    source = Path(__file__).resolve().parents[1] / "tianshu_splitk_20261012/experimental_mm.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == "86a2244f3fb30a8294d53174bd3aa6080f947b5b80ce43b6212a35b533463d56"
    spec = importlib.util.spec_from_file_location("_tianshu_lm_head_gemm_kernel", source)
    impl = importlib.util.module_from_spec(spec); spec.loader.exec_module(impl)
    (root / "experimental_mm.py").write_bytes(source.read_bytes()); (root / "probe_gemm.py").write_bytes(Path(__file__).read_bytes())
    baseline_source = Path(inspect.getsourcefile(flag_gems.mm))
    (root / "baseline_mm.py").write_bytes(baseline_source.read_bytes())
    model_cfg = json.loads(Path("/workspace/MiniCPM5-2B/config.json").read_text())
    assert model_cfg["vocab_size"] == 130560 and model_cfg["hidden_size"] == 2048
    torch.set_num_threads(8); torch.manual_seed(2026101495)
    torch.backends.cuda.matmul.allow_tf32 = False
    configs = [dict(bm=bm, bn=bn, bk=bk, splits=1, warps=warps, stages=stages)
               for bm,bn,bk,warps,stages in [(16,64,64,4,1),(32,64,64,4,1),(64,64,64,4,1),
                                           (32,128,64,4,1),(64,128,64,8,1),(64,256,32,8,1),
                                           (32,128,64,4,2),(64,128,64,8,2)]]
    data = dict(complete=False, official_benchmark=False, adopted=False, vocab_size=130560,
                hidden_size=2048, kernel_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                baseline_source=dict(path=str(baseline_source),sha256=hashlib.sha256(baseline_source.read_bytes()).hexdigest()), cases=[])
    with torch.inference_mode():
        weight = torch.randn(130560,2048,device="cuda",dtype=torch.bfloat16)
        for rows in (64, 8):
            x = torch.randn(rows,2048,device="cuda",dtype=torch.bfloat16)
            reference = x.float() @ weight.float().T
            cpu_columns = torch.linspace(0,130559,32).long()
            cpu_rows = torch.linspace(0,rows-1,min(rows,8)).long()
            cpu_ref = x[cpu_rows.to("cuda")].cpu().float() @ weight[cpu_columns.to("cuda")].cpu().float().T
            baseline = flag_gems.mm(x, weight.T)
            assert check(baseline,reference)["passed"]
            assert check(baseline[cpu_rows.to("cuda")[:,None],cpu_columns.to("cuda")[None,:]].cpu(),cpu_ref)["passed"]
            for cfg in configs:
                item = dict(rows=rows, configuration=cfg)
                try:
                    scratch = impl.prepare(x,weight,cfg); actual = impl.mm(x,weight,scratch,cfg)
                    item["fp32_check"] = check(actual,reference)
                    item["cpu_check"] = check(actual[cpu_rows.to("cuda")[:,None],cpu_columns.to("cuda")[None,:]].cpu(),cpu_ref)
                    item["native_check"] = check(actual,baseline)
                    if item["fp32_check"]["passed"] and item["cpu_check"]["passed"]:
                        graphs=[]; keep=[]
                        for function in (lambda:flag_gems.mm(x,weight.T),lambda:impl.mm(x,weight,scratch,cfg)):
                            for _ in range(3): result=function()
                            graph=torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):
                                for _ in range(5): result=function()
                            graph.replay(); torch.cuda.synchronize()
                            assert check(result,reference)["passed"]
                            graphs.append(graph); keep.append(result)
                        timings=[[],[]]
                        for repeat in range(7):
                            for selected in ((0,1) if repeat%2==0 else (1,0)):
                                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                                a.record(); graphs[selected].replay(); b.record(); b.synchronize()
                                timings[selected].append(a.elapsed_time(b)/5)
                        medians=[statistics.median(v) for v in timings]
                        item.update(baseline_ms=medians[0],candidate_ms=medians[1],
                                    change_percent=100*(medians[1]/medians[0]-1),timings_ms=timings,
                                    graph_numeric_passed=True)
                except Exception as error:
                    item["error"] = repr(error)
                    if any(x in str(error).lower() for x in ("illegal memory","device-side assert")):raise
                data["cases"].append(item); (root/"result.json").write_text(json.dumps(data,indent=2)+"\n")
                print(json.dumps({k:v for k,v in item.items() if k!="timings_ms"}),flush=True)
    data["complete"]=True; (root/"result.json").write_text(json.dumps(data,indent=2)+"\n")


if __name__=="__main__":main()
