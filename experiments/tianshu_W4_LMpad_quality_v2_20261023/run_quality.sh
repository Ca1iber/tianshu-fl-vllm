#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_W4_LMpad_quality_v2_20261023
sha256sum --check --status "$base/EXECUTION.sha256"
sha256sum --check --status experiments/tianshu_W4_LMpad_service_v2_20261023/EXECUTION.sha256
available_bytes=$(df -B1 --output=avail /workspace | tail -n1)
test "$available_bytes" -ge 1073741824
bash experiments/tianshu_native_int8_attention_prefill_release_20261022/restore_frozen_cache.sh --check
ixsmi | grep 'No running processes found'
runner=/workspace/benchmark-results/20261023_W4_LMpad_quality_v2_runner_tianshu
mkdir "$runner"
for batch in 8 64;do
python3 "$base/run_validation.py" --output "/workspace/benchmark-results/20261023_W4_LMpad_v2_math${batch}_tianshu" --screen-root /workspace/vllm-plugin-FL/experiments/tianshu_native_int8_attention42_20261022 --stop-at-utc 2026-10-23T15:58:00+00:00 --stage math --batch "$batch" --down-scope prefill > "$runner/math${batch}.log" 2>&1
jq -e '.phase=="complete" and .accuracy_passed and .correct>=100 and .total==105' "/workspace/benchmark-results/20261023_W4_LMpad_v2_math${batch}_tianshu/result.json" >/dev/null
done
