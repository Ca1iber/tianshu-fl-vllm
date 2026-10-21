#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
prior=/workspace/benchmark-results/20261021_prefill_budget8192_validation_runner_tianshu
phase=$(cat "$prior/phase.txt")
case "$phase" in complete|failed_or_quality_rejected) ;; *) printf 'Previous quality/formal queue still active: %s\n' "$phase" >&2;exit 1;; esac
ixsmi | grep -q 'No running processes found'
base=/workspace/vllm-plugin-FL/experiments/tianshu_static_compile_20261021
runner=/workspace/benchmark-results/20261021_static_compile_runner_tianshu
test ! -e "$runner";mkdir "$runner"
trap 'printf "failed\n" > "$runner/phase.txt"' ERR
for mode in control candidate; do
 printf '%s\n' "$mode" > "$runner/phase.txt"
 python3 "$base/run_service_trial.py" --mode "$mode" --output "/workspace/benchmark-results/20261021_static_compile_service_${mode}_tianshu" --stop-at-utc '2026-10-21T15:58:00+00:00' > "$runner/${mode}.log" 2>&1
done
printf 'CPU_verify\n' > "$runner/phase.txt"
CUDA_VISIBLE_DEVICES=-1 python3 "$base/verify_pair.py" > "$runner/verify.log" 2>&1
printf 'complete\n' > "$runner/phase.txt"
