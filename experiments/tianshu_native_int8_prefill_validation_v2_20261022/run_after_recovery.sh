#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_native_int8_prefill_validation_v2_20261022
screen=experiments/tianshu_native_int8_prefill_results_v2_20261022
recovery=experiments/tianshu_native_int8_prefill_math8_recovery_20261022
runner=/workspace/benchmark-results/20261022_native_int8_prefill_validation_v2_runner_tianshu
math8=/workspace/benchmark-results/20261022_native_int8_prefill_math8_CPU_recovery_tianshu
math64=/workspace/benchmark-results/20261022_native_int8_prefill_math64_v2_tianshu
official=/workspace/benchmark-results/20261022_native_int8_prefill_official4k_v2_tianshu
stop=2026-10-22T15:58:00+00:00
test ! -e "$runner";mkdir "$runner";printf '%s\n' "$$" > "$runner/runner.pid"
printf 'validate_CPU_recovery\n' > "$runner/phase.txt"
trap 'code=$?; if (( code!=0 )); then cp "$runner/phase.txt" "$runner/failed_phase.txt"; printf "failed\n" > "$runner/phase.txt"; printf "%s\n" "$code" > "$runner/exit_code.txt"; fi' EXIT
sha256sum "$base"/*.py "$base"/*.sh "$base/PLAN_32g.md" "$recovery"/*.py "$recovery/PLAN_32g.md" "$recovery/CPU_RECOVERY_GATE_32g.json" > "$runner/frozen_entrypoints.sha256"
jq -e '.phase=="CPU_recovery_complete" and .original_experiment_passed==false and .correct>=100 and .total==105 and .actual_prefill_precision_coverage==false and .full_inherited_checks_retained' "$math8/result.json" >/dev/null
jq -e '.complete and .correct>=100 and .total==105 and .CPU_coverage_reconciliation and .actual_native_prefill_bank_count==0 and .C64_native_INT8_all42_fresh_quality_required' "$math8/accuracy_audit.json" >/dev/null
ixsmi | grep -q 'No running processes found'
printf 'CPU_prepared_profiles\n' > "$runner/phase.txt"
env -u PYTHONPATH python3 -u "$base/check_prepared_profiles_cpu.py" --output /workspace/benchmark-results/20261022_native_int8_prefill_validation_v2_CPU_tianshu > "$runner/CPU.log" 2>&1
sha256sum --check --quiet "$runner/frozen_entrypoints.sha256"
remaining=$(( $(date -u -d "$stop" +%s) - $(date -u +%s) ));test "$remaining" -gt 5400
printf 'Math64\n' > "$runner/phase.txt"
env -u PYTHONPATH python3 -u "$base/run_validation.py" --stage math --batch 64 --screen-root "$screen" --output "$math64" --stop-at-utc "$stop" > "$runner/math64.log" 2>&1
jq -e '.phase=="complete" and .accuracy_passed and .correct>=100 and .total==105' "$math64/result.json" >/dev/null
jq -e '.complete and .correct>=100 and .total==105 and .actual_native_prefill_bank_count==42 and (.actual_native_prefill_M_values|index(4096)!=null)' "$math64/accuracy_audit.json" >/dev/null
sha256sum --check --quiet "$runner/frozen_entrypoints.sha256"
ixsmi | grep -q 'No running processes found'
remaining=$(( $(date -u -d "$stop" +%s) - $(date -u +%s) ));test "$remaining" -gt 2400
printf 'official4K\n' > "$runner/phase.txt"
env -u PYTHONPATH python3 -u "$base/run_validation.py" --stage official --batch 8 --screen-root "$screen" --math-root "$math8" --math64-root "$math64" --output "$official" --stop-at-utc "$stop" > "$runner/official4k.log" 2>&1
jq -e '.phase=="complete" and .completed_runs==4' "$official/result.json" >/dev/null
sha256sum --check --quiet "$runner/frozen_entrypoints.sha256"
ixsmi > "$runner/hardware_after.txt";grep -q 'No running processes found' "$runner/hardware_after.txt"
printf 'complete\n' > "$runner/phase.txt";date -Is > "$runner/completed.txt"
