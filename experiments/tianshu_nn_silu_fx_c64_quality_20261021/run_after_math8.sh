#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_nn_silu_fx_c64_quality_20261021
runner=/workspace/benchmark-results/20261021_nn_silu_fx_c64_quality_runner_tianshu
root=/workspace/benchmark-results/20261021_nn_silu_fx_c64_quality_tianshu
math8=/workspace/benchmark-results/20261021_nn_silu_fx_math8_tianshu
remaining=$(( $(date -u -d '2026-10-21 15:58:00 UTC' +%s) - $(date -u +%s) ))
test "$remaining" -gt 7200
test ! -e "$runner";test ! -e "$root"
jq -e '.complete and .CUDA_not_initialized' "$base/CPU_GATE_32g.json" >/dev/null
mkdir "$runner";printf '%s\n' "$$" > "$runner/runner_pid.txt";printf '%s\n' waiting_for_math8 > "$runner/phase.txt"
trap 'exit_code=$?; if (( exit_code!=0 )); then cp "$runner/phase.txt" "$runner/failed_phase.txt"; printf "failed\n" > "$runner/phase.txt"; printf "%s\n" "$exit_code" > "$runner/exit_code.txt"; fi' EXIT
sha256sum "$base"/*.py "$base"/*.sh "$base/CPU_GATE_32g.json" "$base/PLAN_32g.md" > "$runner/frozen_entrypoints.sha256"
while true;do
 status=$(ps -p 247467 -o stat= | tr -d ' ' || true)
 if [ -z "$status" ] || [[ "$status" == Z* ]];then break;fi
 remaining=$(( $(date -u -d '2026-10-21 15:58:00 UTC' +%s) - $(date -u +%s) ))
 test "$remaining" -gt 5400
 sleep 15
done
jq -e '.phase=="complete" and .accuracy_passed and .total==105 and .accuracy>0.95' "$math8/result.json" >/dev/null
jq -e '.complete and .total==105 and .accuracy>0.95' "$math8/accuracy_audit.json" >/dev/null
ixsmi | grep -q 'No running processes found'
sha256sum --check --quiet "$runner/frozen_entrypoints.sha256"
printf '%s\n' C64_math > "$runner/phase.txt"
env -u PYTHONPATH python3 -u "$base/run_quality.py" --stage math --screen-root experiments/tianshu_nn_silu_fx_screen_20261021 --output "$root" --stop-at-utc 2026-10-21T15:58:00+00:00 > "$runner/quality.log" 2>&1
sha256sum --check --quiet "$runner/frozen_entrypoints.sha256"
jq -e '.complete and .total==105 and .accuracy>0.95 and .actual_C64_fused_replay_verified' "$base/C64_QUALITY_GATE_32g.json" >/dev/null
ixsmi | grep -q 'No running processes found'
printf '%s\n' official4K > "$runner/phase.txt"
validation=experiments/tianshu_nn_silu_fx_validation_20261021
sha256sum --check --quiet /workspace/benchmark-results/20261021_nn_silu_fx_math8_runner_tianshu/frozen_entrypoints.sha256
env -u PYTHONPATH python3 -u "$validation/run_validation.py" --stage official --screen-root experiments/tianshu_nn_silu_fx_screen_20261021 --math-root "$math8" --output /workspace/benchmark-results/20261021_nn_silu_fx_official4k_tianshu --stop-at-utc 2026-10-21T15:58:00+00:00 > "$runner/official4k.log" 2>&1
sha256sum --check --quiet "$runner/frozen_entrypoints.sha256"
ixsmi > "$runner/hardware_after.txt";grep -q 'No running processes found' "$runner/hardware_after.txt"
printf '%s\n' complete > "$runner/phase.txt";date -Is > "$runner/complete.txt"
