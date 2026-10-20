#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
root=/workspace/benchmark-results/20261020_decode_metadata_service4k_tianshu
test ! -e "$root"
test -f /workspace/benchmark-results/20261020_decode_metadata_GPU_tianshu/complete.txt
ixsmi | grep -q 'No running processes found'
mkdir "$root"
printf '%s\n' "$$" > "$root/runner_pid.txt"
date -Is > "$root/started.txt"
sha256sum experiments/tianshu_decode_metadata_results_20261020/run_4k_screen.sh experiments/tianshu_decode_metadata_results_20261020/verify_screen.py > "$root/controller_sources.sha256"
for mode in control candidate; do
  printf '%s\n' "$mode" > "$root/phase.txt"
  env -u PYTHONPATH timeout -k 30s 3600s python3 -u experiments/tianshu_decode_metadata_service_20261020/run_service_trial.py --output "$root/$mode" --mode "$mode" --input-len 4096 --stop-at-utc 2026-10-20T15:58:00+00:00 > "$root/$mode.log" 2>&1
  jq -e '.phase=="complete"' "$root/$mode/result.json" > /dev/null
done
python3 experiments/tianshu_decode_metadata_results_20261020/verify_screen.py --root "$root" --label 4K > "$root/audit.log" 2>&1
sha256sum -c "$root/controller_sources.sha256" > "$root/controller_verified.txt"
ixsmi > "$root/hardware_after.txt"
grep -q 'No running processes found' "$root/hardware_after.txt"
date -Is > "$root/ended.txt"
printf 'complete\n' > "$root/phase.txt"
printf 'complete\n' > "$root/complete.txt"
