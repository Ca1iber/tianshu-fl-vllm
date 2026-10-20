#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_gateup_bf16_dispatch_20261020
root=/workspace/benchmark-results/20261020_gateup_bf16_layout_GPU_tianshu
runner=/workspace/benchmark-results/20261020_gateup_bf16_layout_runner_tianshu
ixsmi | grep -q 'No running processes found'
test ! -e "$root";test ! -e "$runner";mkdir "$runner"
printf '%s\n' "$$" > "$runner/runner_pid.txt";printf 'GPU_diagnostic\n' > "$runner/phase.txt"
trap 'code=$?;if ((code!=0));then cp "$runner/phase.txt" "$runner/failed_phase.txt";printf "failed\n" > "$runner/phase.txt";printf "%s\n" "$code" > "$runner/exit_code.txt";fi' EXIT
timeout -k 30s 1800s env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -u "$base/probe_layout_diagnostic.py" --output "$root" --stop-at-utc 2026-10-21T15:58:00+00:00 > "$runner/component.log" 2>&1
printf 'CPU_verification\n' > "$runner/phase.txt"
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 "$base/verify_layout_cpu.py" "$root" > "$runner/verification.log" 2>&1
ixsmi > "$runner/hardware_after.txt";grep -q 'No running processes found' "$runner/hardware_after.txt"
printf 'complete\n' > "$runner/phase.txt";date -Is > "$runner/complete.txt"
