#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
pipeline=/workspace/benchmark-results/20261020_pair_K_validation_pipeline_tianshu
pid=$(cat "$pipeline/runner_pid.txt")
cutoff=$(date -u -d '2026-10-20 15:58:00 UTC' +%s)
printf 'Waiting for Math/official4K runner PID %s\n' "$pid"
while kill -0 "$pid" 2>/dev/null; do
  test "$((cutoff-$(date -u +%s)))" -gt 1200
  sleep 15
done
test -f "$pipeline/complete.txt"
test "$(cat "$pipeline/phase.txt")" = complete
jq -e '.complete and .total==105 and .accuracy>0.95' /workspace/benchmark-results/20261020_pair_K_math_tianshu/accuracy_audit.json >/dev/null
jq -e '.complete and .requests_successful==1024' /workspace/benchmark-results/20261020_pair_K_official4k_tianshu/comparison.json >/dev/null
ixsmi | grep -q 'No running processes found'
exec bash experiments/tianshu_pair_short_segments_20261020/run_component.sh
