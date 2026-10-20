#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
wait_pid=$(cat /workspace/benchmark-results/20261020_pair_short_segments_queue.pid)
cutoff=$(date -u -d '2026-10-20 15:58:00 UTC' +%s)
printf 'Waiting for paired-segment queue PID %s\n' "$wait_pid"
while kill -0 "$wait_pid" 2>/dev/null; do
  if ps -p "$wait_pid" -o stat= | grep -q '^Z'; then break; fi
  test "$((cutoff-$(date -u +%s)))" -gt 1200
  sleep 15
done
test "$(cat /workspace/benchmark-results/20261020_pair_short_segments_runner_tianshu/phase.txt)" = complete
jq -e '.complete and .requests_successful==1024' /workspace/benchmark-results/20261020_pair_K_official4k_tianshu/comparison.json >/dev/null
ixsmi | grep -q 'No running processes found'
exec bash experiments/tianshu_dense_w8a8_20261020/run_component.sh
