#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_RMS_A8_GU_quality_v2_20261023
sha256sum --check --status "$base/EXECUTION.sha256"
available=$(df -B1 --output=avail /workspace | tail -n 1);test "$available" -ge 2147483648
bash experiments/tianshu_native_int8_attention_prefill_release_20261022/restore_frozen_cache.sh --check
runner=/workspace/benchmark-results/20261023_RMS_A8_GU_v2_quality_runner_tianshu
mkdir "$runner"
for batch in 64 8;do
 python3 "$base/run_validation.py" --output "/workspace/benchmark-results/20261023_RMS_A8_GU_v2_math${batch}_tianshu" --screen-root /workspace/vllm-plugin-FL/experiments/tianshu_native_int8_attention42_20261022 --stop-at-utc 2026-10-23T15:58:00+00:00 --stage math --batch "$batch" --down-scope prefill > "$runner/math${batch}.log" 2>&1
done
