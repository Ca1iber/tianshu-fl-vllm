#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
sha256sum --check --status experiments/tianshu_native_int8_attention_decode64_official4k_20261022/EXECUTION.sha256
python3 experiments/tianshu_native_int8_attention_decode64_official4k_20261022/run_validation.py --stage official --batch 8 --down-scope prefill --screen-root /workspace/vllm-plugin-FL/experiments/tianshu_native_int8_attention64_20261022 --math-root /workspace/benchmark-results/20261022_native_int8_attention_decode64_math8_tianshu --math64-root /workspace/benchmark-results/20261022_native_int8_attention_decode64_math64_tianshu --output /workspace/benchmark-results/20261022_native_int8_attention_decode64_official4k_tianshu --stop-at-utc 2026-10-22T15:58:00+00:00
