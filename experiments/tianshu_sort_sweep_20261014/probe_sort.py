# SPDX-License-Identifier: Apache-2.0
"""Compare immutable sort snapshots; never replace the installed operator."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys


VARIANTS = {
    "installed_baseline": "baseline_sort.py",
    "empty_bin_only": "empty_bin_only_sort.py",
    "metax_int32_reference": "reference_sources/metax_int32_reference.py",
    "metax_empty_bin": "reference_sources/metax_empty_bin.py",
    "metax_two_bin": "reference_sources/metax_two_bin.py",
    "metax_repeat_digit": "reference_sources/metax_repeat_digit.py",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    assert root.is_relative_to("/workspace/benchmark-results") and not root.exists()
    deadline = datetime.fromisoformat(json.loads(Path(
        "/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json"
    ).read_text())["stop_at_utc"])
    remaining = int((deadline - datetime.now(timezone.utc)).total_seconds())
    assert remaining > 300
    hardware = subprocess.check_output(["ixsmi"], text=True)
    assert "No running processes found" in hardware
    root.mkdir(parents=True)
    (root / "hardware_before.txt").write_text(hardware)
    state = dict(phase="starting", runner_pid=os.getpid(), official_benchmark=False,
                 installed_operator_modified=False, stop_at_utc=deadline.isoformat())

    def save():
        path = root / "state.tmp"
        path.write_text(json.dumps(state, indent=2) + "\n")
        path.replace(root / "state.json")

    def stop(signum, frame):
        raise TimeoutError("probe deadline or external stop: " + str(signum))

    for signum in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, stop)
    signal.alarm(min(1800, remaining))
    save()
    try:
        import torch
        import vllm_fl  # Register the vendor backend in the same environment as serving.
        import flag_gems.ops.sort as installed_sort
        # Some packages export the function instead of the submodule.
        installed_sort = importlib.import_module("flag_gems.ops.sort")
        base = Path(__file__).resolve().parent
        installed_path = Path(installed_sort.__file__).resolve()
        assert installed_path.read_bytes() == (base / VARIANTS["installed_baseline"]).read_bytes()
        snapshots = dict(VARIANTS, probe="probe_sort.py")
        frozen = {}
        for name, relative in snapshots.items():
            path = base / relative
            payload = path.read_bytes()
            frozen[str(path)] = hashlib.sha256(payload).hexdigest()
            dest = root / "source_snapshot" / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(payload)
        frozen[str(installed_path)] = hashlib.sha256(installed_path.read_bytes()).hexdigest()
        modules = {}
        for name, relative in VARIANTS.items():
            module_name = "tianshu_sweep_probe_" + name
            spec = importlib.util.spec_from_file_location(module_name, base / relative)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            modules[name] = module
        torch.set_num_threads(4)
        state.update(phase="running", frozen_sources=frozen, torch_version=torch.__version__)
        save()
        rows = []
        # B64 is an operator control, not evidence that serving selects generic sort at B64.
        cases = [(batch, kind, True) for batch in (1, 4, 7, 64)
                 for kind in ("finite_fp32", "bf16_roundtrip", "duplicate_neginf")]
        cases += [(1, "duplicate_neginf", False)]

        def make_input(batch, kind, seed):
            generator = torch.Generator().manual_seed(seed)
            if kind == "duplicate_neginf":
                values = torch.randint(-8, 9, (batch, 130560), generator=generator).float()
                values[:, ::5] = -float("inf")
                return values
            values = torch.randn((batch, 130560), generator=generator)
            return values.bfloat16().float() if kind == "bf16_roundtrip" else values

        def check(actual, expected):
            values, indices = (item.cpu() for item in actual)
            assert indices.dtype == torch.int64
            assert torch.equal(indices, expected.indices), "stable INT64 index mismatch"
            assert torch.equal(values.view(torch.int32), expected.values.view(torch.int32)), "value bits mismatch"

        for case_id, (batch, kind, descending) in enumerate(cases):
            state.update(case_id=case_id, batch=batch, distribution=kind)
            save()
            cpu_original = make_input(batch, kind, 31400 + case_id)
            cpu_changed = make_input(batch, kind, 72400 + case_id)
            reference = torch.sort(cpu_original, dim=-1, descending=descending, stable=True)
            changed_reference = torch.sort(cpu_changed, dim=-1, descending=descending, stable=True)
            tensor = cpu_original.cuda()
            original = tensor.clone()
            changed = cpu_changed.cuda()
            active = []
            for name, module in modules.items():
                state["variant"] = name
                save()
                invoke = lambda module=module: module.sort_stable(tensor, stable=True, descending=descending)
                for _ in range(2):
                    result = invoke()
                torch.cuda.synchronize()
                check(result, reference)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = invoke()
                graph.replay()
                torch.cuda.synchronize()
                check(captured, reference)
                tensor.copy_(changed)
                graph.replay()
                torch.cuda.synchronize()
                check(captured, changed_reference)
                tensor.copy_(original)
                graph.replay()
                torch.cuda.synchronize()
                check(captured, reference)
                row = dict(case_id=case_id, batch=batch, n=130560, distribution=kind,
                           descending=descending, variant=name, values_bitwise_equal=True,
                           stable_indices_int64_equal=True, changed_input_graph_equal=True,
                           input_sha256=hashlib.sha256(cpu_original.numpy().tobytes()).hexdigest(),
                           changed_input_sha256=hashlib.sha256(cpu_changed.numpy().tobytes()).hexdigest(),
                           event_blocks=[])
                active.append((graph, captured, row))
            # Rotate candidate order to reduce ordering and clock drift effects.
            for block in range(3):
                for offset in range(len(active)):
                    graph, captured, row = active[(block + offset) % len(active)]
                    events = []
                    for _ in range(4):
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        graph.replay()
                        end.record()
                        events.append((start, end))
                    torch.cuda.synchronize()
                    row["event_blocks"].append([start.elapsed_time(end) for start, end in events])
            for graph, captured, row in active:
                row["median_ms"] = statistics.median(x for block in row["event_blocks"] for x in block)
                rows.append(row)
                with (root / "rows.jsonl").open("a") as output:
                    output.write(json.dumps(row) + "\n")
                print(json.dumps({key: row[key] for key in ("case_id", "batch", "distribution", "variant", "median_ms")}), flush=True)
            del active, graph, captured, result, tensor, original, changed
            torch.cuda.empty_cache()
        assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha for path, sha in frozen.items())
        state.update(phase="complete", cases=len(cases), variants=len(VARIANTS), rows=len(rows),
                     numerical_failures=0, changed_input_graph_checks=len(rows),
                     events=sum(len(b) for row in rows for b in row["event_blocks"]))
    except BaseException as error:
        state.update(phase="failed", error=repr(error))
        raise
    finally:
        save()
        (root / "result.json").write_text(json.dumps(state, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
