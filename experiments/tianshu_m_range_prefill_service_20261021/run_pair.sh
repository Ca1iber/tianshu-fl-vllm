#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_m_range_prefill_service_20261021
runner=/workspace/benchmark-results/20261021_m_range_prefill_pair_runner_tianshu
test ! -e "$runner"
test "$(date -u +%s)" -lt "$(date -u -d '2026-10-21 14:00:00 UTC' +%s)"
jq -e '.complete and .CPU_only and .actual_entry_prepare_passed' "$base/CPU_GATE_32g.json" >/dev/null
ixsmi | grep -q 'No running processes found'
mkdir "$runner";printf '%s\n' "$$" > "$runner/runner_pid.txt"
trap 'code=$?; printf "%s\n" "$code" > "$runner/exit_code.txt"; if (( code!=0 )); then cp "$runner/phase.txt" "$runner/failed_phase.txt"; printf "failed\n" > "$runner/phase.txt"; fi' EXIT
find "$base" -type f \( -name '*.py' -o -name '*.sh' -o -name 'CPU_GATE_32g.json' -o -name 'selected_plan.json' -o -name 'PLAN_32g.md' \) -print0 | sort -z | xargs -0 sha256sum > "$runner/frozen_entrypoints.sha256"
for arm in control candidate; do
 printf '%s\n' "$arm" > "$runner/phase.txt"
 root=/workspace/benchmark-results/20261021_m_range_prefill_pair_${arm}_tianshu
 test ! -e "$root"
 timeout -k 30s 1800s env -u PYTHONPATH python3 -u "$base/run_service_trial.py" --mode "$arm" --output "$root" --stop-at-utc 2026-10-21T15:58:00+00:00 > "$runner/$arm.log" 2>&1
 sha256sum --check --quiet "$runner/frozen_entrypoints.sha256"
done
printf 'GPU_pair_complete_pending_CPU_comparison\n' > "$runner/phase.txt"
