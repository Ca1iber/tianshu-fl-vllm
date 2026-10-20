#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_small_arena_service_20261020
pipeline=/workspace/benchmark-results/20261020_small_arena_service_pipeline_tianshu
mkdir "$pipeline"
date -Iseconds > "$pipeline/started.txt"
printf '%s\n' "$$" > "$pipeline/runner_pid.txt"
jq -r '.frozen_sources|to_entries[]|"\(.value)  \(.key)"' "$base/COMPOSITION_32g.json" > "$pipeline/frozen_sources.sha256"
sha256sum --check --quiet "$pipeline/frozen_sources.sha256"
timeout -k 70s 1800s env -u PYTHONPATH python3 -u "$base/run_service_pair.py" --output /workspace/benchmark-results/20261020_small_arena_service4k_tianshu --input-len 4096 --stop-at-utc 2026-10-20T15:58:00+00:00 > "$pipeline/4k.log" 2>&1
jq -e '.complete and .all384_screen_requests_successful and .gain_percent >= -1' /workspace/benchmark-results/20261020_small_arena_service4k_tianshu/comparison.json > "$pipeline/4k_gate.txt"
sha256sum --check --quiet "$pipeline/frozen_sources.sha256"
end_epoch=$(date -u -d '2026-10-20 15:58:00 UTC' +%s)
now_epoch=$(date -u +%s)
if (( end_epoch - now_epoch < 5400 )); then
    date -Iseconds > "$pipeline/16k_skipped_insufficient_time.txt"
    exit 0
fi
timeout -k 70s 4500s env -u PYTHONPATH python3 -u "$base/run_service_pair.py" --output /workspace/benchmark-results/20261020_small_arena_service16k_tianshu --input-len 16384 --stop-at-utc 2026-10-20T15:58:00+00:00 > "$pipeline/16k.log" 2>&1
sha256sum --check --quiet "$pipeline/frozen_sources.sha256"
jq -e '.complete and .all384_screen_requests_successful' /workspace/benchmark-results/20261020_small_arena_service16k_tianshu/comparison.json > "$pipeline/16k_gate.txt"
date -Iseconds > "$pipeline/complete.txt"
