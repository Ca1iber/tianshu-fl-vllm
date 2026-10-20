#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_k8v16_address32_20261020
queue=/workspace/benchmark-results/20261020_k8v16_address32_queue_tianshu
prior=/workspace/benchmark-results/20261020_small_arena_service_pipeline_tianshu
mkdir "$queue"
date -Iseconds > "$queue/queued.txt"
printf '%s\n' "$$" > "$queue/runner_pid.txt"
sha256sum "$base"/*.py "$base"/CPU_GATE_32g.json "$base"/GUARD_CPU_GATE_32g.json "$base"/GPU_PREPARATION_32g.json > "$queue/frozen_entrypoints.sha256"
end_epoch=$(date -u -d '2026-10-20 15:58:00 UTC' +%s)
while [ ! -f "$prior/complete.txt" ]; do
    prior_pid=$(cat "$prior/runner_pid.txt")
    if ! ps -p "$prior_pid" -o args= | grep -Fq 'tianshu_small_arena_service_20261020/run_screens.sh'; then
        date -Iseconds > "$queue/preceding_process_ended_without_complete.txt"
        exit 1
    fi
    if (( end_epoch - $(date -u +%s) < 1200 )); then
        date -Iseconds > "$queue/skipped_insufficient_time.txt"
        exit 0
    fi
    sleep 30
done
sha256sum --check --quiet "$prior/frozen_sources.sha256"
sha256sum --check --quiet "$queue/frozen_entrypoints.sha256"
jq -e '.complete and .all384_screen_requests_successful' /workspace/benchmark-results/20261020_small_arena_service16k_tianshu/comparison.json > "$queue/preceding_gate.txt"
if (( end_epoch - $(date -u +%s) < 1200 )); then
    date -Iseconds > "$queue/skipped_insufficient_time.txt"
    exit 0
fi
date -Iseconds > "$queue/started_gpu.txt"
timeout -k 20s 900s env -u PYTHONPATH VLLM_PLUGINS=fl python3 -u "$base/probe.py" --output /workspace/benchmark-results/20261020_k8v16_address32_decode_tianshu --reader-source /workspace/benchmark-results/20261019_k8v16_budget_default_official4k_tianshu/decode_coherent_records/cache_generated/97987/generated_cache_policy.py --stop-at-utc 2026-10-20T15:58:00+00:00 > "$queue/component.log" 2>&1
sha256sum --check --quiet "$queue/frozen_entrypoints.sha256"
date -Iseconds > "$queue/complete.txt"
