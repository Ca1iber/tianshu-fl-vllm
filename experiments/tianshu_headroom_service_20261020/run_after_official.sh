#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_headroom_service_20261020
pipeline=/workspace/benchmark-results/20261020_long_segments_validation_pipeline_tianshu
official=/workspace/benchmark-results/20261020_long_segments_official4k16k_tianshu
queue=/workspace/benchmark-results/20261020_headroom_service_queue_tianshu
output=/workspace/benchmark-results/20261020_headroom_service16k_tianshu
mkdir "$queue"
trap 'status=$?; printf "%s\n" "$status" > "$queue/exit_code.txt"; date --iso-8601=seconds > "$queue/ended.txt"' EXIT
printf '%s\n' "$$" > "$queue/queue_pid.txt"
date --iso-8601=seconds > "$queue/queued.txt"
sha256sum "$base"/*.py "$base"/hooks/*.py "$base"/CPU_GATE_32g.json "$base"/batch_policy.json "$base"/run_after_official.sh > "$queue/frozen_entrypoints.sha256"
latest_start=$(date -u -d '2026-10-20T14:58:00+00:00' +%s)
while [ ! -f "$pipeline/complete.txt" ]; do
  [ "$(date -u +%s)" -lt "$latest_start" ] || { echo 'Less than one hour remains; not starting GPU.' >&2; exit 2; }
  if [ -f "$official/state.json" ]; then
    phase=$(jq -r '.phase' "$official/state.json")
    case "$phase" in failed|audit_failed|stopped|cleanup_failed) echo "Official stage ended as $phase; not starting GPU." >&2; exit 3;; esac
  fi
  sleep 15
done
jq -e '.complete and .official_benchmark and .one_profile_for_both_workloads and .fresh_math_accuracy>0.95 and (.workloads|length)==2 and all(.workloads[];.passes_public_performance_gates)' "$official/comparison.json" > "$queue/preceding_official_gate.txt"
sha256sum --check "$queue/frozen_entrypoints.sha256" > "$queue/source_verification.txt"
date --iso-8601=seconds > "$queue/started_gpu.txt"
env -u PYTHONPATH VLLM_PLUGINS=fl timeout -k 30s 4500s python3 -u "$base/run_service_pair.py" --input-len 16384 --output "$output" --stop-at-utc 2026-10-20T15:58:00+00:00 > "$queue/service.log" 2>&1
jq -e '.complete' "$output/comparison.json" > "$queue/comparison_complete.txt"
date --iso-8601=seconds > "$queue/complete.txt"
