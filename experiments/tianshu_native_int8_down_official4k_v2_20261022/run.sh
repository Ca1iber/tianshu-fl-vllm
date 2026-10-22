#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
python3 experiments/tianshu_native_int8_down_official4k_v2_20261022/run_validation.py --stage official --batch 8 --down-scope prefill --screen-root /workspace/vllm-plugin-FL/experiments/tianshu_native_int8_down42_20261022 --math-root /workspace/benchmark-results/20261022_native_int8_down_prefill_math8_v2_retry1_tianshu --math64-root /workspace/benchmark-results/20261022_native_int8_down_prefill_math64_v2_retry1_tianshu --output /workspace/benchmark-results/20261022_native_int8_down_prefill_official4k_v2_tianshu --stop-at-utc 2026-10-22T15:58:00+00:00
