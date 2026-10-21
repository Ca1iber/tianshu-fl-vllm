#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_nn_silu_fx_observer_20261021
root=/workspace/benchmark-results/20261021_nn_silu_fx_observer_GPU_tianshu
runner=/workspace/benchmark-results/20261021_nn_silu_fx_observer_runner_tianshu
remaining=$(( $(date -u -d '2026-10-21 15:58:00 UTC' +%s) - $(date -u +%s) ))
test "$remaining" -gt 1800
test ! -e "$root";test ! -e "$runner"
jq -e '.complete and .CUDA_not_initialized' "$base/CPU_GATE_32g.json" >/dev/null
ixsmi | grep -q 'No running processes found'
mkdir "$runner"
printf '%s\n' "$$" > "$runner/runner_pid.txt"
printf '%s\n' starting > "$runner/phase.txt"
trap 'exit_code=$?; if (( exit_code!=0 )); then cp "$runner/phase.txt" "$runner/failed_phase.txt"; printf "failed\n" > "$runner/phase.txt"; printf "%s\n" "$exit_code" > "$runner/exit_code.txt"; fi' EXIT
sha256sum "$base"/*.py "$base"/hooks/*.py "$base"/*.sh "$base/CPU_GATE_32g.json" "$base/PLAN_32g.md" > "$runner/frozen_entrypoints.sha256"
timeout -k 30s 1830s env -u PYTHONPATH python3 -u "$base/probe_service.py" --output "$root" --stop-at-utc 2026-10-21T15:58:00+00:00 > "$runner/probe.log" 2>&1
for (( i=0; i<60; i++ ));do
  ixsmi > "$runner/hardware_after.txt"
  if grep -q 'No running processes found' "$runner/hardware_after.txt";then break;fi
  sleep 1
done
grep -q 'No running processes found' "$runner/hardware_after.txt"
sha256sum --check --quiet "$runner/frozen_entrypoints.sha256"
printf '%s\n' complete > "$runner/phase.txt"
date -Is > "$runner/complete.txt"
