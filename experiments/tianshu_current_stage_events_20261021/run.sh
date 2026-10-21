#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_current_stage_events_20261021
runner=/workspace/benchmark-results/20261021_current_stage_events_runner_tianshu
root=/workspace/benchmark-results/20261021_current_stage_events_tianshu
test ! -e "$runner";test ! -e "$root"
jq -e '.complete and .CUDA_not_initialized' "$base/CPU_GATE_32g.json" >/dev/null
ixsmi | grep -q 'No running processes found'
mkdir "$runner";printf '%s\n' "$$" > "$runner/runner_pid.txt";printf 'stage_diagnostic_GPU\n' > "$runner/phase.txt"
trap 'code=$?;printf "%s\n" "$code" > "$runner/exit_code.txt";if (( code!=0 ));then cp "$runner/phase.txt" "$runner/failed_phase.txt";printf "failed\n" > "$runner/phase.txt";fi' EXIT
sha256sum "$base"/*.py "$base"/*.sh "$base/hooks_candidate/sitecustomize.py" "$base/CPU_GATE_32g.json" "$base/PLAN_32g.md" > "$runner/frozen_entrypoints.sha256"
timeout -k 30s 1200s env -u PYTHONPATH python3 -u "$base/run_profile.py" --output "$root" --stop-at-utc 2026-10-21T15:58:00+00:00 > "$runner/service.log" 2>&1
sha256sum --check --quiet "$runner/frozen_entrypoints.sha256"
jq -e '.phase=="complete" and .summary.complete' "$root/result.json" >/dev/null
printf 'complete\n' > "$runner/phase.txt";date -Is > "$runner/complete.txt"
