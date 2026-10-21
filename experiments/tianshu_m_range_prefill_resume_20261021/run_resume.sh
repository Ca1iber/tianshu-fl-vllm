#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_m_range_prefill_resume_20261021
screen=experiments/tianshu_m_range_prefill_results_v2_20261021
recovery=experiments/tianshu_m_range_math8_coverage_recovery_20261021
runner=/workspace/benchmark-results/20261021_m_range_prefill_resume_runner_tianshu
math8=/workspace/benchmark-results/20261021_m_range_prefill_v2_math8_tianshu
recovered=/workspace/benchmark-results/20261021_m_range_math8_CPU_recovery_tianshu
c64=/workspace/benchmark-results/20261021_m_range_prefill_resume_c64_quality_tianshu
formal=/workspace/benchmark-results/20261021_m_range_prefill_resume_official4k_tianshu
test ! -e "$runner";test ! -e "$c64";test ! -e "$formal"
jq -e '.complete and .CUDA_not_initialized and .runtime_source_count==471' "$base/CPU_GATE_32g.json" >/dev/null
jq -e '.complete and .CPU_only and (.original_experiment_passed==false) and .correct==103 and .C64_actual_M4096_all42_quality_still_required' "$recovery/MATH8_RECOVERY_GATE_32g.json" >/dev/null
mkdir "$runner";printf '%s\n' "$$" > "$runner/runner_pid.txt"
trap 'code=$?;printf "%s\n" "$code" > "$runner/exit_code.txt";if (( code!=0 ));then cp "$runner/phase.txt" "$runner/failed_phase.txt";printf "failed\n" > "$runner/phase.txt";fi' EXIT
printf 'preflight\n' > "$runner/phase.txt"
sha256sum "$base"/*.py "$base"/*.sh "$base/CPU_GATE_32g.json" "$base/PLAN_32g.md" "$recovery/verify_cpu.py" "$recovery/MATH8_RECOVERY_GATE_32g.json" "$recovery/PLAN_32g.md" > "$runner/frozen_entrypoints.sha256"
remaining=$(( $(date -u -d '2026-10-21 15:58:00 UTC' +%s)-$(date -u +%s) ));test "$remaining" -gt 5400
ixsmi | grep -q 'No running processes found'
printf 'C64_quality\n' > "$runner/phase.txt"
timeout -k 30s 4800s env -u PYTHONPATH python3 -u "$base/run_c64_quality.py" --stage math --screen-root "$screen" --output "$c64" --stop-at-utc 2026-10-21T15:58:00+00:00 > "$runner/C64_quality.log" 2>&1
sha256sum --check --quiet "$runner/frozen_entrypoints.sha256"
jq -e '.complete and .total==105 and .accuracy>0.95 and .actual_C64_fused_replay_verified and .actual_M4096_range_fused_all42_verified' "$base/C64_QUALITY_GATE_32g.json" >/dev/null
remaining=$(( $(date -u -d '2026-10-21 15:58:00 UTC' +%s)-$(date -u +%s) ));test "$remaining" -gt 2400
ixsmi | grep -q 'No running processes found'
printf 'official4K\n' > "$runner/phase.txt"
timeout -k 30s 2400s env -u PYTHONPATH python3 -u "$base/run_validation.py" --stage official --screen-root "$screen" --math-root "$math8" --math-recovery-root "$recovered" --output "$formal" --stop-at-utc 2026-10-21T15:58:00+00:00 > "$runner/official4K.log" 2>&1
sha256sum --check --quiet "$runner/frozen_entrypoints.sha256"
ixsmi > "$runner/hardware_after.txt";grep -q 'No running processes found' "$runner/hardware_after.txt"
printf 'complete\n' > "$runner/phase.txt";date -Is > "$runner/complete.txt"
