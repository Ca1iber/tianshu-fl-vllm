#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
sha256sum --check --status experiments/tianshu_native_int8_down_quality_v2_retry_20261022/EXECUTION.sha256
available_bytes=$(df -B1 --output=avail /workspace | tail -n 1)
test "$available_bytes" -ge 5368709120
bash experiments/tianshu_native_int8_prefill_release_v2_20261022/restore_frozen_cache.sh --check
quality_runner=/workspace/benchmark-results/20261022_native_int8_down_prefill_quality_v2_retry1_runner_tianshu
mkdir "$quality_runner"
python3 experiments/tianshu_native_int8_down_quality_v2_20261022/run_validation.py --output /workspace/benchmark-results/20261022_native_int8_down_prefill_math64_v2_retry1_tianshu --screen-root /workspace/vllm-plugin-FL/experiments/tianshu_native_int8_down42_20261022 --stop-at-utc 2026-10-22T15:58:00+00:00 --stage math --batch 64 --down-scope prefill > "$quality_runner/math64.log" 2>&1
python3 experiments/tianshu_native_int8_down_quality_v2_20261022/run_validation.py --output /workspace/benchmark-results/20261022_native_int8_down_prefill_math8_v2_retry1_tianshu --screen-root /workspace/vllm-plugin-FL/experiments/tianshu_native_int8_down42_20261022 --stop-at-utc 2026-10-22T15:58:00+00:00 --stage math --batch 8 --down-scope prefill > "$quality_runner/math8.log" 2>&1
