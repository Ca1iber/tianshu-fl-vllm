#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
ti_base=experiments/tianshu_r5_BM8_fresh128_20261024
ti_quality=/dev/shm/tianshu_r5_BM8_runs_20261024
ti_pair=$ti_quality/fresh128_pair
trap 'printf "screen_pipeline_exit_code=%s\n" "$?"' EXIT
ixsmi | grep -q 'No running processes found'
ti_quality_pid=$(cat "$ti_quality/quality_pipeline.pid")
ti_status=$(ps -p "$ti_quality_pid" -o stat= || true)
if test -n "$ti_status" && [[ "$ti_status" != Z* ]]; then
 printf 'Quality pipeline PID %s is still live; no performance process started.\n' "$ti_quality_pid"
 exit 3
fi
for ti_batch in 8 64; do
 jq -e '.phase=="complete" and .evaluation_exit_code==0 and .accuracy_passed and .total==105 and .bm8_mode=="candidate"' "$ti_quality/math${ti_batch}/result.json" > /dev/null
 jq -e '.complete and .fresh_entire_105 and .all_caps_in_denominator and .above95 and .total==105' "$ti_quality/math${ti_batch}/accuracy_audit.json" > /dev/null
 jq -e '.complete and .this_batch_above95 and .source_count==1023 and .environment_count==67' "experiments/tianshu_r5_BM8_results_20261024/C${ti_batch}_QUALITY_GATE_32g.json" > /dev/null
done
jq -e -s '.[0].frozen_sources==.[1].frozen_sources and .[0].server_command==.[1].server_command' "$ti_quality/math8/result.json" "$ti_quality/math64/result.json" > /dev/null
ti_available=$(df --output=avail -B1 /workspace | tail -1)
if test "$ti_available" -lt 67108864; then
 printf 'Workspace free bytes=%s; inspect and archive rebuildable FX cache while GPU idle before starting.\n' "$ti_available"
 exit 4
fi
sha256sum --check --status "$ti_base/EXECUTION.sha256"
test ! -e "$ti_pair"
mkdir "$ti_pair"
cp "$ti_base/EXECUTION.sha256" "$ti_pair/controllers.sha256"
for ti_arm in control candidate; do
 printf '%s_running\n' "$ti_arm" > "$ti_pair/phase.txt"
 ixsmi | grep -q 'No running processes found'
 python3 -u "$ti_base/run_model.py" --output "$ti_pair/$ti_arm" --stage screen --batch 64 --mode "$ti_arm"
 jq -e '.phase=="complete" and .evaluation_exit_code==0' "$ti_pair/$ti_arm/result.json" > /dev/null
 printf '%s_screen_complete\n' "$ti_arm"
done
printf 'both_screens_complete_pending_comparison\n' > "$ti_pair/phase.txt"
sha256sum --check "$ti_pair/controllers.sha256" > "$ti_pair/controllers_verified.txt"
python3 "$ti_base/compare_screens_cpu.py"
printf 'BM8_r5_fresh128_complete_pending_review\n' > "$ti_pair/phase.txt"
