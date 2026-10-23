#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
sha256sum --check --status experiments/tianshu_LM_pad64_short4k_20261023/EXECUTION.sha256
short_runner=/workspace/benchmark-results/20261023_LM_pad64_short4k_runner_tianshu
mkdir "$short_runner"
for arm in control candidate;do
python3 experiments/tianshu_LM_pad64_short4k_20261023/run_service_trial.py --output "/workspace/benchmark-results/20261023_LM_pad64_short4k_${arm}_tianshu" --mode "$arm" --scope prefill --input-len 4096 --stop-at-utc 2026-10-23T15:58:00+00:00 > "$short_runner/${arm}.log" 2>&1
done
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python3 experiments/tianshu_LM_pad64_short4k_20261023/verify_pair_cpu.py --scope prefill > "$short_runner/comparison.log" 2>&1
