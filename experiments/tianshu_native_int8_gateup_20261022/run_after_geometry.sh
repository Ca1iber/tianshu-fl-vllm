#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
geometry_pid=$(cat /workspace/benchmark-results/20261022_a8_geometry_runner_tianshu/gpu_pid)
geometry_result=/workspace/benchmark-results/20261022_a8_geometry_GPU_tianshu/result.json
native_runner=/workspace/benchmark-results/20261022_native_int8_gateup_runner_tianshu
dealine_epoch=$(date -u -d '2026-10-22T15:58:00+00:00' +%s)
if [ -r "/proc/$geometry_pid/stat" ]; then
 geometry_tick=$(awk '{print $22}' "/proc/$geometry_pid/stat")
 printf 'Waiting on geometry GPU PID=%s start_tick=%s\n' "$geometry_pid" "$geometry_tick"
 while [ -r "/proc/$geometry_pid/stat" ]; do
  read -r geometry_state actual_tick < <(awk '{print $3,$22}' "/proc/$geometry_pid/stat")
  if [ "$actual_tick" != "$geometry_tick" ] || [ "$geometry_state" = Z ]; then break; fi
  test "$(date -u +%s)" -lt "$dealine_epoch"
  sleep 5
 done
fi
jq -e '.complete and .phase=="complete"' "$geometry_result" >/dev/null
ixsmi > "$native_runner/hardware_queue_release.txt"
grep -q 'No running processes found' "$native_runner/hardware_queue_release.txt"
printf 'Geometry GPU terminal and complete; launching native full component at %s\n' "$(date -Is)"
exec bash experiments/tianshu_native_int8_gateup_20261022/run.sh
