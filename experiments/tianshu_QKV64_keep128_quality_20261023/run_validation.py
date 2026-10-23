# SPDX-License-Identifier: Apache-2.0
"""Fresh Math105 or original official4K for the screened GEMM/strictTopP profile."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.request

from quality_profile import BENCH_SHA, audit_math, audit_benchmark, audit_runtime, csv_rows, digest, frozen_check, normalized_environment, prepare, read, audit_startup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--stop-at-utc", required=True)
    parser.add_argument("--stage", choices=("math", "official"), required=True)
    parser.add_argument("--math-root", type=Path)
    parser.add_argument("--math64-root", type=Path)
    parser.add_argument("--batch", type=int, choices=(8,64), default=8)
    parser.add_argument("--down-scope", choices=("all","prefill"), default="prefill")
    args = parser.parse_args()
    assert args.stage == "math" and args.down_scope == "prefill", "Math-only QKV/O controller"
    root = args.output.resolve()
    assert root.is_relative_to("/workspace/benchmark-results") and not root.exists()
    root.mkdir(parents=True)
    state = dict(phase="preparing", mode="candidate", seq_limit=64, arena_mode="candidate", actual_budget=4096, original_serve_command=True, runner_pid=os.getpid(), fresh_math_for_original_command_profile=args.stage == "math",
        stage=args.stage, official_benchmark=args.stage == "official", adopted=False, reused_original_accuracy=False,
        started_at_utc=datetime.now(timezone.utc).isoformat(), reference_root=str(args.screen_root.resolve()),
        screen_root=str(args.screen_root.resolve()), output_root=str(root))
    server = evaluator = None

    def save():
        temporary = root / "state.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(root / "state.json")
        print(json.dumps({key: state.get(key) for key in ("phase", "error", "server_pid", "evaluator_pid")}), flush=True)

    def terminate(process):
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)

    class StopRequested(BaseException):
        pass

    def stop(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        for process in (evaluator, server):
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        raise StopRequested(str(signum))

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        signal.signal(signum, stop)
    save()
    try:
        stop_at = datetime.fromisoformat(args.stop_at_utc)
        remaining = int((stop_at - datetime.now(timezone.utc)).total_seconds())
        assert remaining > (3600 if args.stage == "math" else 2400)
        signal.alarm(remaining)
        state["stop_at_utc"] = stop_at.isoformat()
        command, environment, profile, sources = prepare(root, args.screen_root.resolve(), batch=args.batch, scope=args.down_scope)
        state.update(server_command=command, profile_environment=profile, frozen_sources=sources)
        state["controller_sources"] = read(root / "controller_sources.json")
        state["final_profile_kind"] = "qualified_r3_plus_uniform_QKV64_smallprefill_BF16"
        state["quality_batch"]=args.batch
        state["down_mode"]="candidate";state["down_scope"]=args.down_scope
        state["attention_mode"]="candidate"
        state["fresh_math_for_original_command_profile"]=(args.stage=="math" and args.batch==8)
        state["supplementary_C64_quality"]=(args.stage=="math" and args.batch==64)
        state["additional_environment"]={"TRITON_CACHE_DIR":environment["TRITON_CACHE_DIR"]}
        if args.stage == "official":
            assert args.math_root is not None
            assert args.math64_root is not None and args.batch==8
            state["math64_root"]=str(args.math64_root.resolve())
            math_root = args.math_root.resolve()
            quality = read(math_root / "accuracy_audit.json")
            math_state=read(math_root/"result.json")
            if quality.get("CPU_coverage_reconciliation"):
                assert math_state["phase"]=="CPU_recovery_complete" and math_state["original_experiment_passed"] is False
                actual_math_root=Path(quality["actual_math_raw_root"]);original_math_state=read(actual_math_root/"result.json")
                assert original_math_state["phase"]=="audit_failed" and digest(actual_math_root/"result.json")==quality["original_result_sha256"]
                state["math8_quality_root"]=str(math_root)
            else:
                assert math_state["phase"]=="complete";actual_math_root=math_root
            assert quality["complete"] and quality["fresh_math_for_original_command_profile"]
            assert quality["total"] == 105 and quality["accuracy"] > .95
            assert quality["matched_screen_root"] == state["screen_root"]
            assert quality["server_command"] == command and quality["frozen_sources"] == sources
            assert normalized_environment(profile, root) == normalized_environment(quality["profile_environment"], actual_math_root)
            state["math_root"] = str(actual_math_root)
        try:
            connection = socket.create_connection(("127.0.0.1", 9031), timeout=2)
        except OSError:
            pass
        else:
            connection.close()
            raise RuntimeError("Port9031 occupied")
        hardware = subprocess.check_output(["ixsmi"], text=True)
        (root / "hardware_before.txt").write_text(hardware)
        assert "No running processes found" in hardware
        with (root / "server.log").open("w") as log:
            server = subprocess.Popen(command, cwd="/workspace", env=environment, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
        state.update(phase="starting_server", server_pid=server.pid)
        save()
        limit = time.monotonic() + 2400
        while True:
            assert server.poll() is None, "Server exited during startup"
            if time.monotonic() > limit:
                raise TimeoutError("Startup exceeded40minutes")
            try:
                with urllib.request.urlopen("http://127.0.0.1:9031/v1/models", timeout=3) as response:
                    models = json.load(response)
                if any(model["id"] == "minicpm" for model in models["data"]):
                    break
            except Exception:
                pass
            time.sleep(3)
        raw = Path(f"/proc/{server.pid}/environ").read_bytes()
        assert args.stage!="official" or args.batch==8
        observed = dict(value.decode().split("=", 1) for value in raw.split(b"\0") if b"=" in value)
        assert {key: observed.get(key) for key in profile} == profile
        full_profile={**profile,**state["additional_environment"]};assert len(full_profile)==63 and {k:observed.get(k) for k in full_profile}==full_profile
        (root/"observed_full_server_environment.json").write_text(json.dumps(dict(server_pid=server.pid,environment={k:observed[k] for k in full_profile},command=command),indent=2)+"\n")
        (root / "observed_server_environment.json").write_text(json.dumps(dict(
            server_pid=server.pid, environment={key: observed[key] for key in profile},
            command=command, source="actual /proc of this fresh validation API process; actual62 operator-profile keys"), indent=2) + "\n")

        def metrics(name):
            with urllib.request.urlopen("http://127.0.0.1:9031/metrics", timeout=10) as response:
                (root / name).write_bytes(response.read())

        metrics("before_metrics.txt")
        state["startup_contract"]=audit_startup(root,state);save()
        if args.stage == "math":
            evaluation = ["evalscope", "eval", "--model", "minicpm", "--api-url",
            "http://127.0.0.1:9031/v1/chat/completions", "--api-key", "EMPTY", "--eval-type", "openai_api",
            "--datasets", "math_500", "--dataset-args",
            '{"math_500":{"dataset_id":"/workspace/evalscope-datasets/math_500","subset_list":["Level 3"]}}',
            "--eval-batch-size", str(args.batch), "--timeout", "3600", "--generation-config",
            '{"temperature":1.0,"top_p":0.95,"max_tokens":32768}', "--work-dir", str(root / "evalscope"), "--ignore-errors"]
            state["evaluation_command"] = evaluation
            working_directory, log_name = "/workspace", "evalscope.log"
        else:
            benchmark = Path("/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py")
            assert digest(benchmark) == BENCH_SHA
            evaluation = ["python3", "-u", str(benchmark), "--model", "/workspace/MiniCPM5-2B",
                "--served-model-name", "minicpm", "--port", "9031", "--test-cases", "[[4096,1024,64,256]]"]
            state.update(benchmark_command=evaluation, official_script_sha256=BENCH_SHA,
                runs=4, excluded_runs=[1], completed_runs=0)
            working_directory, log_name = str(root), "benchmark.log"
        evaluator_environment = dict(environment)
        evaluator_environment.pop("PYTHONPATH", None)
        with (root / log_name).open("w") as log:
            evaluator = subprocess.Popen(evaluation, cwd=working_directory, env=evaluator_environment,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        state.update(phase="evaluating" if args.stage == "math" else "benchmarking", evaluator_pid=evaluator.pid)
        save()
        while evaluator.poll() is None:
            assert server.poll() is None, "API server exited during validation"
            if args.stage != "math":
                rows = csv_rows(root, "raw_runs_*.csv")
                if len(rows) != state["completed_runs"]:
                    state.update(completed_runs=len(rows), raw_runs=rows)
                    save()
            time.sleep(5)
        exit_key = "evalscope_exit_code" if args.stage == "math" else "benchmark_exit_code"
        state[exit_key] = evaluator.returncode
        assert evaluator.returncode == 0
        if args.stage != "math":
            rows = csv_rows(root, "raw_runs_*.csv")
            summary = csv_rows(root, "summary_*.csv")
            assert len(rows) == 4 and len(summary) == 1
            state.update(completed_runs=4, raw_runs=rows, official_summary=summary[0])
        metrics("after_metrics.txt")
        frozen_check(sources)
        frozen_check(state["controller_sources"])
        state["phase"] = "evaluation_finished_pending_audit"
    except StopRequested as error:
        state.update(phase="stopped", error=str(error))
    except Exception as error:
        state.update(phase="failed", error=repr(error))
    finally:
        terminate(evaluator)
        terminate(server)
        signal.alarm(0)
        (root / "hardware_after_initial.txt").write_text(subprocess.check_output(["ixsmi"], text=True))
        for _ in range(60):
            hardware=subprocess.check_output(["ixsmi"],text=True)
            (root/'hardware_after.txt').write_text(hardware)
            if 'No running processes found' in hardware:break
            time.sleep(1)
        else:state.update(phase='cleanup_failed',error='GPU did not become idle within60seconds')
        save()
        (root / "result.json").write_text(json.dumps(state, indent=2) + "\n")
    if state["phase"] == "evaluation_finished_pending_audit":
        try:
            profile = audit_runtime(root, state, official=args.stage != "math")
            if args.stage == "math":
                accuracy = audit_math(root, state, profile)
                state.update(phase="complete", correct=accuracy["correct"], total=105,
                    accuracy=accuracy["accuracy"], accuracy_passed=accuracy["official_requirement_passed"])
            else:
                comparison = audit_benchmark(root, state, profile, quality if args.stage == "official" else None)
                state.update(phase="complete", total_tok_s=comparison["total_tok_s"], output_tok_s=comparison["output_tok_s"], ttft_ms=comparison["ttft_ms"], original_serve_command_verified=True)
        except Exception as error:
            state.update(phase="audit_failed", error=repr(error))
        save()
        (root / "result.json").write_text(json.dumps(state, indent=2) + "\n")
    if state["phase"] == "complete" and args.stage == "math" and not state["accuracy_passed"]:
        return 2
    return 0 if state["phase"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
