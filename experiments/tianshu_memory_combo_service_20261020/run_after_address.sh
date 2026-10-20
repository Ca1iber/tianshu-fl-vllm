#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_memory_combo_service_20261020
queue=/workspace/benchmark-results/20261020_memory_combo_queue_tianshu
prior=/workspace/benchmark-results/20261020_k8v16_address32_queue_tianshu
mkdir "$queue"
date -Iseconds > "$queue/queued.txt"
printf '%s\n' "$$" > "$queue/runner_pid.txt"
sha256sum "$base"/*.py "$base"/hooks/*.py "$base"/CPU_GATE_32g.json "$base"/COMPOSITION_32g.json "$base"/run_screens.sh > "$queue/frozen_entrypoints.sha256"
end_epoch=$(date -u -d '2026-10-20 15:58:00 UTC' +%s)
while [ ! -f "$prior/complete.txt" ]; do
    prior_pid=$(cat "$prior/runner_pid.txt")
    if ! ps -p "$prior_pid" -o args= | grep -Fq 'tianshu_k8v16_address32_20261020/run_after_service.sh'; then
        date -Iseconds > "$queue/preceding_process_terminal.txt"
        break
    fi
    if (( end_epoch - $(date -u +%s) < 7200 )); then
        date -Iseconds > "$queue/skipped_insufficient_time.txt"
        exit 0
    fi
    sleep 30
done
while ! ixsmi | grep -q 'No running processes found'; do
    if (( end_epoch - $(date -u +%s) < 7200 )); then
        date -Iseconds > "$queue/skipped_GPU_busy_until_deadline.txt"
        exit 0
    fi
    sleep 30
done
sha256sum --check --quiet "$queue/frozen_entrypoints.sha256"
if (( end_epoch - $(date -u +%s) < 7200 )); then
    date -Iseconds > "$queue/skipped_insufficient_time.txt"
    exit 0
fi
date -Iseconds > "$queue/started_gpu.txt"
timeout -k 80s 6600s bash "$base/run_screens.sh" > "$queue/service.log" 2>&1
date -Iseconds > "$queue/complete.txt"
