#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
screen=/workspace/benchmark-results/20261020_k8v16_pair_service4k_v2_tianshu
screen_pid=$(cat "$screen/runner_pid.txt")
cutoff=$(date -u -d '2026-10-20 15:58:00 UTC' +%s)
printf 'Waiting for screen runner PID %s\n' "$screen_pid"
while kill -0 "$screen_pid" 2>/dev/null; do
  test "$((cutoff-$(date -u +%s)))" -gt 1200
  sleep 15
done
test -f "$screen/complete.txt"
test "$(cat "$screen/phase.txt")" = complete
jq -e '.complete and .independently_rechecked' "$screen/independent_screen_gate.json" >/dev/null
ixsmi | grep -q 'No running processes found'
exec bash experiments/tianshu_k8v16_word_decode_20261020/run_component.sh
