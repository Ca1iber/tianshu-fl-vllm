#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
queue=/workspace/benchmark-results/20261020_rope_kv_fusion_queue_tianshu
output=/workspace/benchmark-results/20261020_rope_kv_fusion_GPU_tianshu
official=/workspace/benchmark-results/20261020_future_kv_official4k16k_tianshu
pipeline=/workspace/benchmark-results/20261020_future_kv_validation_pipeline_tianshu
cutoff=$(date -u -d '2026-10-20 15:58:00 UTC' +%s)
mkdir "$queue"
printf '%s\n' "$$" > "$queue/runner_pid.txt"
trap 'component_exit_status=$?; if (( component_exit_status!=0 )); then cp "$queue/phase.txt" "$queue/failed_phase.txt"; printf "failed\n" > "$queue/phase.txt"; printf "%s\n" "$component_exit_status" > "$queue/exit_code.txt"; fi' EXIT
printf '%s\n' waiting_for_official > "$queue/phase.txt"
controller_pid=$(cat "$pipeline/runner_pid.txt")
printf '%s\n' "$controller_pid" > "$queue/predecessor_pid.txt"
while true; do
  now=$(date -u +%s)
  if (( cutoff-now<1800 )); then printf '%s\n' insufficient_remaining_time > "$queue/phase.txt"; exit 2; fi
  status=$(ps -p "$controller_pid" -o stat= || true)
  if test -z "$status" || [[ "$status" == Z* ]]; then break; fi
  sleep 10
done
# Only the outer pipeline creates complete.txt, after both stages pass.
test -f "$pipeline/complete.txt"
jq -e '.phase=="complete" and .completed_runs==8' "$official/result.json" >/dev/null
jq -e '.complete and .successful_requests==1536' "$official/comparison.json" >/dev/null
ixsmi > "$queue/hardware_before.txt"
grep -q 'No running processes found' "$queue/hardware_before.txt"
test ! -e "$output"
sha256sum -c experiments/tianshu_rope_kv_fusion_20261020/SOURCE_PREREQUISITE.sha256 > "$queue/prerequisites.txt"
printf '%s\n' running_component > "$queue/phase.txt"
remaining=$((cutoff-$(date -u +%s)))
limit=1800
if (( remaining<limit )); then limit=$remaining; fi
timeout -k 30s "${limit}s" env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -u experiments/tianshu_rope_kv_fusion_20261020/probe_gpu.py --output "$output" --stop-at-utc 2026-10-20T15:58:00+00:00 > "$queue/component.log" 2>&1
printf '%s\n' independently_verifying > "$queue/phase.txt"
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' python3 experiments/tianshu_rope_kv_fusion_20261020/verify_gpu.py "$output" > "$queue/verification.log" 2>&1
ixsmi > "$queue/hardware_after.txt"
grep -q 'No running processes found' "$queue/hardware_after.txt"
printf '%s\n' complete > "$queue/phase.txt"
date -Is > "$queue/complete.txt"
