#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_k8v16_bm8_validation_v2_20261021
screen=experiments/tianshu_k8v16_bm8_service_results_v2_20261021
screen_runner=/workspace/benchmark-results/20261021_BM8_candidate_v2_runner_tianshu
runner=/workspace/benchmark-results/20261021_BM8_quality_v2_runner_tianshu
math8=/workspace/benchmark-results/20261021_BM8_v2_math8_tianshu
c64=/workspace/benchmark-results/20261021_BM8_v2_c64_quality_tianshu
formal=/workspace/benchmark-results/20261021_BM8_v2_official4k_tianshu
test ! -e "$runner";test ! -e "$math8";test ! -e "$c64";test ! -e "$formal"
mkdir "$runner";printf '%s\n' "$$" > "$runner/runner_pid.txt";printf 'waiting_for_screen\n' > "$runner/phase.txt"
trap 'code=$?;printf "%s\n" "$code" > "$runner/exit_code.txt";if (( code!=0 ));then cp "$runner/phase.txt" "$runner/failed_phase.txt";printf "failed\n" > "$runner/phase.txt";fi' EXIT
sha256sum "$base"/*.py "$base"/*.sh "$base/CPU_GATE_32g.json" "$base/PLAN_32g.md" > "$runner/frozen_entrypoints.sha256"
while true;do state=$(ps -p 308178 -o stat= || true);if test -z "$state" || [[ "$state" == Z* ]];then break;fi;sleep 10;done
sha256sum --check --quiet "$runner/frozen_entrypoints.sha256"
test "$(cat "$screen_runner/phase.txt")" == complete
jq -e '.complete and .all_runtime_audits_rechecked' "$screen/SCREEN_GATE_32g.json" >/dev/null
if ! jq -e '.candidate_passes_perf_screen' "$screen/SCREEN_GATE_32g.json" >/dev/null;then printf 'complete_screen_rejected_no_quality_GPU\n' > "$runner/phase.txt";date -Is > "$runner/complete.txt";exit 0;fi
jq -e '.complete and .CUDA_not_initialized and .runtime_source_count==497' "$base/CPU_GATE_32g.json" >/dev/null
remaining=$(( $(date -u -d '2026-10-21 15:58:00 UTC' +%s)-$(date -u +%s) ));test "$remaining" -gt 4800
ixsmi | grep -q 'No running processes found'
printf 'official_C8_Math\n' > "$runner/phase.txt"
timeout -k 30s 3600s env -u PYTHONPATH python3 -u "$base/run_validation.py" --stage math --screen-root "$screen" --output "$math8" --stop-at-utc 2026-10-21T15:58:00+00:00 > "$runner/Math8.log" 2>&1
sha256sum --check --quiet "$runner/frozen_entrypoints.sha256"
jq -e '.phase=="complete" and .accuracy>0.95' "$math8/result.json" >/dev/null
remaining=$(( $(date -u -d '2026-10-21 15:58:00 UTC' +%s)-$(date -u +%s) ));test "$remaining" -gt 3600
ixsmi | grep -q 'No running processes found'
printf 'actual_C64_Math\n' > "$runner/phase.txt"
timeout -k 30s 3600s env -u PYTHONPATH python3 -u "$base/run_c64_quality.py" --stage math --screen-root "$screen" --output "$c64" --stop-at-utc 2026-10-21T15:58:00+00:00 > "$runner/C64_Math.log" 2>&1
sha256sum --check --quiet "$runner/frozen_entrypoints.sha256"
jq -e '.complete and .accuracy>0.95 and .BM8_actual_C64_replay_verified' "$base/C64_QUALITY_GATE_32g.json" >/dev/null
remaining=$(( $(date -u -d '2026-10-21 15:58:00 UTC' +%s)-$(date -u +%s) ));test "$remaining" -gt 2400
ixsmi | grep -q 'No running processes found'
printf 'official4K\n' > "$runner/phase.txt"
timeout -k 30s 2400s env -u PYTHONPATH python3 -u "$base/run_validation.py" --stage official --screen-root "$screen" --math-root "$math8" --output "$formal" --stop-at-utc 2026-10-21T15:58:00+00:00 > "$runner/official4K.log" 2>&1
sha256sum --check --quiet "$runner/frozen_entrypoints.sha256"
ixsmi > "$runner/hardware_after.txt";grep -q 'No running processes found' "$runner/hardware_after.txt"
printf 'complete\n' > "$runner/phase.txt";date -Is > "$runner/complete.txt"
