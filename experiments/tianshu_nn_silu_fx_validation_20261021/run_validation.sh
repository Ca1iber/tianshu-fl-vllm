#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_nn_silu_fx_validation_20261021
runner=/workspace/benchmark-results/20261021_nn_silu_fx_math8_runner_tianshu
root=/workspace/benchmark-results/20261021_nn_silu_fx_math8_tianshu
remaining=$(( $(date -u -d '2026-10-21 15:58:00 UTC' +%s) - $(date -u +%s) ))
test "$remaining" -gt 7200
test ! -e "$runner";test ! -e "$root"
jq -e '.complete and .CUDA_not_initialized' "$base/CPU_GATE_32g.json" >/dev/null
ixsmi | grep -q 'No running processes found'
mkdir "$runner";printf '%s\n' "$$" > "$runner/runner_pid.txt";printf '%s\n' math8 > "$runner/phase.txt"
trap 'exit_code=$?; if (( exit_code!=0 )); then cp "$runner/phase.txt" "$runner/failed_phase.txt"; printf "failed\n" > "$runner/phase.txt"; printf "%s\n" "$exit_code" > "$runner/exit_code.txt"; fi' EXIT
sha256sum "$base"/*.py "$base"/*.sh "$base/CPU_GATE_32g.json" "$base/PLAN_32g.md" > "$runner/frozen_entrypoints.sha256"
env -u PYTHONPATH python3 -u "$base/run_validation.py" --stage math --screen-root experiments/tianshu_nn_silu_fx_screen_20261021 --output "$root" --stop-at-utc 2026-10-21T15:58:00+00:00 > "$runner/math.log" 2>&1
sha256sum --check --quiet "$runner/frozen_entrypoints.sha256"
printf '%s\n' complete > "$runner/phase.txt";date -Is > "$runner/complete.txt"
