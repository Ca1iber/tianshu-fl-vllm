#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_dots_Down64_official4k_v2_20261023
sha256sum --check --status "$base/EXECUTION.sha256"
bash experiments/tianshu_native_int8_attention_prefill_release_20261022/restore_frozen_cache.sh --check
python3 "$base/run_validation.py" --output /workspace/benchmark-results/20261023_dots_Down64_v2_official4k_tianshu --screen-root /workspace/vllm-plugin-FL/experiments/tianshu_native_int8_attention42_20261022 --stop-at-utc 2026-10-23T15:58:00+00:00 --stage official --batch 8 --down-scope prefill --math-root /workspace/benchmark-results/20261023_dots_Down64_v2_math8_tianshu --math64-root /workspace/benchmark-results/20261023_dots_Down64_v2_math64_tianshu
