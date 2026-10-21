#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_stage_prefill_int8_20261021
root=/workspace/benchmark-results/20261021_stage_prefill_int8_tianshu
runner=/workspace/benchmark-results/20261021_stage_prefill_int8_runner_tianshu
remaining=$(( $(date -u -d '2026-10-21 15:58:00 UTC' +%s) - $(date -u +%s) ))
test "$remaining" -gt 2400
test ! -e "$root"
test ! -e "$runner"
jq -e '.complete and .CUDA_not_initialized' "$base/CPU_GATE_32g.json" > /dev/null
ixsmi | grep -q 'No running processes found'
mkdir "$runner"
printf '%s\n' "$$" > "$runner/runner_pid.txt"
printf '%s\n' profiling > "$runner/phase.txt"
trap 'exit_code=$?; if (( exit_code!=0 )); then cp "$runner/phase.txt" "$runner/failed_phase.txt"; printf "failed\n" > "$runner/phase.txt"; printf "%s\n" "$exit_code" > "$runner/exit_code.txt"; fi' EXIT
sha256sum "$base"/*.py "$base/hooks"/*.py "$base"/*.sh "$base/CPU_GATE_32g.json" "$base/PLAN_32g.md" > "$runner/frozen_entrypoints.sha256"
limit=2400
if (( remaining<limit )); then limit=$remaining; fi
timeout -k 30s "${limit}s" env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -u "$base/run_profile.py" --output "$root" --stop-at-utc 2026-10-21T15:58:00+00:00 > "$runner/profile.log" 2>&1
printf '%s\n' independent_CPU_verification > "$runner/phase.txt"
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 "$base/verify_profile.py" "$root" > "$runner/verification.log" 2>&1
ixsmi > "$runner/hardware_after.txt"
grep -q 'No running processes found' "$runner/hardware_after.txt"
sha256sum --check --quiet "$runner/frozen_entrypoints.sha256"
printf '%s\n' complete > "$runner/phase.txt"
date -Is > "$runner/complete.txt"
