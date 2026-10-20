#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
root=/workspace/benchmark-results/20261020_future_kv_validation_pipeline_tianshu
math=/workspace/benchmark-results/20261020_future_kv_math_tianshu
official=/workspace/benchmark-results/20261020_future_kv_official4k16k_tianshu
screen=/workspace/benchmark-results/20261020_future_kv_service16k_v2_tianshu
test ! -e "$root"
test ! -e "$math"
test ! -e "$official"
jq -e '.complete and .model_requests_rerun==false' /workspace/benchmark-results/20261020_decode_metadata_service4k_tianshu/recovery_complete.json > /dev/null
ixsmi | grep -q 'No running processes found'
mkdir "$root"
printf '%s\n' "$$" > "$root/runner_pid.txt"
date -Is > "$root/started.txt"
printf 'math\n' > "$root/phase.txt"
timeout -k 60s 14400s python3 -u experiments/tianshu_future_kv_validation_20261020/run_validation.py --output "$math" --screen-root "$screen" --stop-at-utc 2026-10-20T15:58:00+00:00 --stage math > "$root/math.log" 2>&1
jq -e '.complete and .total==105 and .accuracy>0.95' "$math/accuracy_audit.json" > "$root/quality_gate.txt"
printf 'official\n' > "$root/phase.txt"
timeout -k 60s 10800s python3 -u experiments/tianshu_future_kv_validation_20261020/run_validation.py --output "$official" --screen-root "$screen" --math-root "$math" --stop-at-utc 2026-10-20T15:58:00+00:00 --stage official > "$root/official.log" 2>&1
jq -e '.complete and .successful_requests==1536' "$official/comparison.json" > "$root/official_gate.txt"
printf 'complete\n' > "$root/phase.txt"
date -Is > "$root/ended.txt"
printf 'complete\n' > "$root/complete.txt"
