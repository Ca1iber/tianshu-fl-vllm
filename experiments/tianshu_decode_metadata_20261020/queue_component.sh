#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
root=/workspace/benchmark-results/20261020_decode_metadata_queue_tianshu
test ! -e "$root"
mkdir "$root"
printf '%s\n' "$$" > "$root/runner_pid.txt"
date -Is > "$root/started.txt"
# Wait only for the known, currently running exclusive benchmark controller.
while ps -p 41865 -o stat= | grep -qv Z; do
  sleep 15
done
test -f /workspace/benchmark-results/20261020_future_kv_service16k_v2_tianshu/complete.txt
bash experiments/tianshu_decode_metadata_20261020/run_component.sh
printf 'complete\n' > "$root/complete.txt"
