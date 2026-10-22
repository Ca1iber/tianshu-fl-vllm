#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
down_scope=${1:-prefill}
case "$down_scope" in all|prefill) ;; *) exit 2 ;; esac
short_runner="/workspace/benchmark-results/20261022_native_int8_down_${down_scope}_short4k_v2_runner_tianshu"
mkdir "$short_runner"
for arm in control candidate; do
python3 experiments/tianshu_native_int8_down_short4k_v2_20261022/run_service_trial.py --output "/workspace/benchmark-results/20261022_native_int8_down_${down_scope}_short4k_v2_${arm}_tianshu" --mode "$arm" --scope "$down_scope" --input-len 4096 --stop-at-utc 2026-10-22T15:58:00+00:00 > "$short_runner/${arm}.log" 2>&1
done
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python3 experiments/tianshu_native_int8_down_short4k_v2_20261022/verify_pair_cpu.py --scope "$down_scope" > "$short_runner/comparison.log" 2>&1
