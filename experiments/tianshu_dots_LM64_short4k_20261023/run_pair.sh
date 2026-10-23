#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_dots_LM64_short4k_20261023
sha256sum --check --status "$base/EXECUTION.sha256"
runner=/workspace/benchmark-results/20261023_dots_LM64_short4k_runner_tianshu
mkdir "$runner"
for arm in control candidate;do
python3 "$base/run_service_trial.py" --output "/workspace/benchmark-results/20261023_dots_LM64_short4k_${arm}_tianshu" --mode "$arm" --scope prefill --input-len 4096 --stop-at-utc 2026-10-23T15:58:00+00:00 > "$runner/${arm}.log" 2>&1
done
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 "$base/verify_pair_cpu.py" --scope prefill > "$runner/comparison.log" 2>&1
