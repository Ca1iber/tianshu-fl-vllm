#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_int8_prefill_budget_large_20261021
root=/workspace/benchmark-results/20261021_int8_prefill_budget_large_GPU_tianshu
runner=/workspace/benchmark-results/20261021_int8_prefill_budget_large_runner_tianshu
test ! -e "$root";test ! -e "$runner";mkdir "$runner";printf 'GPU_component\n' > "$runner/phase.txt"
trap 'code=$?;if ((code!=0));then cp "$runner/phase.txt" "$runner/failed_phase.txt";printf "failed\n" > "$runner/phase.txt";printf "%s\n" "$code" > "$runner/exit_code.txt";fi' EXIT
ixsmi > "$runner/hardware_before.txt";grep -q 'No running processes found' "$runner/hardware_before.txt"
timeout -k 20s 1800s env -u PYTHONPATH OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python3 -u "$base/probe_gpu.py" --output "$root" --stop-at-utc 2026-10-21T15:58:00+00:00 > "$runner/probe.log" 2>&1
ixsmi > "$runner/hardware_after.txt";grep -q 'No running processes found' "$runner/hardware_after.txt"
printf 'CPU_recheck\n' > "$runner/phase.txt"
timeout -k 20s 900s env -u PYTHONPATH CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python3 -u experiments/tianshu_int8_prefill_budget_large_results_20261021/verify_gpu.py "$root" > "$runner/CPU_verify.log" 2>&1
printf 'complete\n' > "$runner/phase.txt";date -Is > "$runner/complete.txt"
