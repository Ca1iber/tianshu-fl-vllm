#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_m_range_prefill_validation_20261021
screen=experiments/tianshu_m_range_prefill_results_20261021
runner=/workspace/benchmark-results/20261021_m_range_prefill_validation_runner_tianshu
pair=/workspace/benchmark-results/20261021_m_range_prefill_pair_runner_tianshu
math8=/workspace/benchmark-results/20261021_m_range_prefill_math8_tianshu
c64=/workspace/benchmark-results/20261021_m_range_prefill_c64_quality_tianshu
formal=/workspace/benchmark-results/20261021_m_range_prefill_official4k_tianshu
test ! -e "$runner";test ! -e "$math8";test ! -e "$c64";test ! -e "$formal"
jq -e '.complete and .CUDA_not_initialized and .runtime_source_count==471' "$base/CPU_GATE_32g.json" >/dev/null
mkdir "$runner";printf '%s\n' "$$" > "$runner/runner_pid.txt"
printf 'waiting_for_pair\n' > "$runner/phase.txt"
trap 'code=$?;printf "%s\n" "$code" > "$runner/exit_code.txt";if (( code!=0 ));then cp "$runner/phase.txt" "$runner/failed_phase.txt";printf "failed\n" > "$runner/phase.txt";fi' EXIT
sha256sum "$base"/*.py "$base"/*.sh "$base/CPU_GATE_32g.json" "$base/PLAN_32g.md" "$screen/verify_pair_cpu.py" "$screen/PLAN_32g.md" > "$runner/frozen_entrypoints.sha256"
while true;do
 status=$(ps -p 279266 -o stat= | tr -d ' ' || true)
 if [ -z "$status" ] || [[ "$status" == Z* ]];then break;fi
 test "$(date -u +%s)" -lt "$(date -u -d '2026-10-21 13:15:00 UTC' +%s)"
 sleep 15
done
test "$(cat "$pair/phase.txt")" = GPU_pair_complete_pending_CPU_comparison
printf 'CPU_comparison\n' > "$runner/phase.txt"
env CUDA_VISIBLE_DEVICES='' VLLM_PLUGINS='' PYTHONPATH='' python3 -u "$screen/verify_pair_cpu.py" > "$runner/CPU_comparison.log" 2>&1
sed -i 's/\r$//' "$screen/KEY_ROUNDS_32g.csv"
if ! jq -e '.complete and .candidate_passes_perf_screen' "$screen/SCREEN_GATE_32g.json" >/dev/null;then
 printf 'complete_candidate_not_promoted\n' > "$runner/phase.txt";exit 0
fi
sha256sum --check --quiet "$runner/frozen_entrypoints.sha256"
remaining=$(( $(date -u -d '2026-10-21 15:58:00 UTC' +%s)-$(date -u +%s) ));test "$remaining" -gt 7200
ixsmi | grep -q 'No running processes found'
printf 'Math8\n' > "$runner/phase.txt"
timeout -k 30s 4800s env -u PYTHONPATH python3 -u "$base/run_validation.py" --stage math --screen-root "$screen" --output "$math8" --stop-at-utc 2026-10-21T15:58:00+00:00 > "$runner/math8.log" 2>&1
sha256sum --check --quiet "$runner/frozen_entrypoints.sha256"
jq -e '.phase=="complete" and .accuracy_passed and .total==105 and .accuracy>0.95' "$math8/result.json" >/dev/null
jq -e '.complete and .total==105 and .accuracy>0.95' "$math8/accuracy_audit.json" >/dev/null
remaining=$(( $(date -u -d '2026-10-21 15:58:00 UTC' +%s)-$(date -u +%s) ));test "$remaining" -gt 5400
ixsmi | grep -q 'No running processes found'
printf 'C64_quality\n' > "$runner/phase.txt"
timeout -k 30s 4800s env -u PYTHONPATH python3 -u "$base/run_c64_quality.py" --stage math --screen-root "$screen" --output "$c64" --stop-at-utc 2026-10-21T15:58:00+00:00 > "$runner/C64_quality.log" 2>&1
sha256sum --check --quiet "$runner/frozen_entrypoints.sha256"
jq -e '.complete and .total==105 and .accuracy>0.95 and .actual_C64_fused_replay_verified' "$base/C64_QUALITY_GATE_32g.json" >/dev/null
remaining=$(( $(date -u -d '2026-10-21 15:58:00 UTC' +%s)-$(date -u +%s) ));test "$remaining" -gt 2400
ixsmi | grep -q 'No running processes found'
printf 'official4K\n' > "$runner/phase.txt"
timeout -k 30s 2400s env -u PYTHONPATH python3 -u "$base/run_validation.py" --stage official --screen-root "$screen" --math-root "$math8" --output "$formal" --stop-at-utc 2026-10-21T15:58:00+00:00 > "$runner/official4K.log" 2>&1
sha256sum --check --quiet "$runner/frozen_entrypoints.sha256"
ixsmi > "$runner/hardware_after.txt";grep -q 'No running processes found' "$runner/hardware_after.txt"
printf 'complete\n' > "$runner/phase.txt";date -Is > "$runner/complete.txt"
