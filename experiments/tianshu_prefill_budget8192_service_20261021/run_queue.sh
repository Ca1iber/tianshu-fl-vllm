#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=/workspace/vllm-plugin-FL/experiments/tianshu_prefill_budget8192_service_20261021
runner=/workspace/benchmark-results/20261021_prefill_budget8192_runner_tianshu
mkdir -p "$runner"
trap 'printf "failed\n" > "$runner/phase.txt"' ERR
for mode in control candidate; do
  printf '%s\n' "$mode" > "$runner/phase.txt"
  python3 "$base/run_service_trial.py" --mode "$mode" --output "/workspace/benchmark-results/20261021_prefill_budget8192_service_${mode}_tianshu" --stop-at-utc '2026-10-21T15:58:00+00:00' > "$runner/${mode}.log" 2>&1
done
printf 'CPU_verification\n' > "$runner/phase.txt"
CUDA_VISIBLE_DEVICES=-1 python3 "$base/verify_pair.py" > "$runner/verify.log" 2>&1
printf 'complete\n' > "$runner/phase.txt"
