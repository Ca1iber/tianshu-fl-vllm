#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
down_scope=${1:-all}
case "$down_scope" in all|prefill) ;; *) exit 2 ;; esac
quality_runner="/workspace/benchmark-results/20261022_native_int8_down_${down_scope}_quality_runner_tianshu"
mkdir "$quality_runner"
python3 experiments/tianshu_native_int8_down_quality_20261022/run_validation.py --output "/workspace/benchmark-results/20261022_native_int8_down_${down_scope}_math64_tianshu" --screen-root /workspace/vllm-plugin-FL/experiments/tianshu_native_int8_down42_20261022 --stop-at-utc 2026-10-22T15:58:00+00:00 --stage math --batch 64 --down-scope "$down_scope" > "$quality_runner/math64.log" 2>&1
python3 experiments/tianshu_native_int8_down_quality_20261022/run_validation.py --output "/workspace/benchmark-results/20261022_native_int8_down_${down_scope}_math8_tianshu" --screen-root /workspace/vllm-plugin-FL/experiments/tianshu_native_int8_down42_20261022 --stop-at-utc 2026-10-22T15:58:00+00:00 --stage math --batch 8 --down-scope "$down_scope" > "$quality_runner/math8.log" 2>&1
