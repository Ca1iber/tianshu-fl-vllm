#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
pipeline=/workspace/benchmark-results/20261020_dense_w8a8_validation_pipeline_tianshu
while test -e /proc/76900;do
  state=$(awk '{print $3}' /proc/76900/stat)
  test "$state" = Z && break
  sleep 20
done
test -f "$pipeline/complete.txt"
test "$(cat "$pipeline/phase.txt")" = complete
ixsmi | grep -q 'No running processes found'
exec bash experiments/tianshu_gateup_bf16_dispatch_20261020/run_component.sh
