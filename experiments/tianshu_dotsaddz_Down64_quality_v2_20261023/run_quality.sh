#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_dotsaddz_Down64_quality_v2_20261023
sha256sum --check --status "$base/EXECUTION.sha256"
sha256sum --check --status experiments/tianshu_dotsaddz_Down64_service_v2_20261023/EXECUTION.sha256
available_bytes=$(df -B1 --output=avail /workspace | tail -n 1)
test "$available_bytes" -ge 2147483648
bash experiments/tianshu_native_int8_attention_prefill_release_20261022/restore_frozen_cache.sh --check
runner=/workspace/benchmark-results/20261023_dots_Down64_v2_quality_runner_tianshu
mkdir "$runner"
python3 "$base/run_validation.py" --output /workspace/benchmark-results/20261023_dots_Down64_v2_math64_tianshu --screen-root /workspace/vllm-plugin-FL/experiments/tianshu_native_int8_attention42_20261022 --stop-at-utc 2026-10-23T15:58:00+00:00 --stage math --batch 64 --down-scope prefill > "$runner/math64.log" 2>&1
jq -e '.phase=="complete" and .accuracy_passed and .correct>=100 and .total==105' /workspace/benchmark-results/20261023_dots_Down64_v2_math64_tianshu/result.json >/dev/null
python3 "$base/run_validation.py" --output /workspace/benchmark-results/20261023_dots_Down64_v2_math8_tianshu --screen-root /workspace/vllm-plugin-FL/experiments/tianshu_native_int8_attention42_20261022 --stop-at-utc 2026-10-23T15:58:00+00:00 --stage math --batch 8 --down-scope prefill > "$runner/math8.log" 2>&1
sha256sum --check --status "$base/EXECUTION.sha256"
sha256sum --check --status experiments/tianshu_dotsaddz_Down64_service_v2_20261023/EXECUTION.sha256
