#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
ti_pair=/workspace/benchmark-results/20261024_W4_O64_affine_screen_pair_tianshu
ti_component=experiments/tianshu_W4_QKV64_fitted_component_20261024
ti_status=/workspace/benchmark-results/20261024_W4_QKV64_post_pair_phase.txt
trap 'printf "post_pair_exit_code=%s\n" "$?"' EXIT
printf 'waiting_for_pair_pipeline_pid3866\n' > "$ti_status"
while true; do
 ti_stat=$(ps -p 3866 -o stat= | tr -d '[:space:]' || true)
 if test -z "$ti_stat" || [[ "$ti_stat" == Z* ]]; then break; fi
 sleep 20
done
test "$(cat "$ti_pair/phase.txt")" = both_screens_complete_pending_comparison
ixsmi | grep -q 'No running processes found'
printf 'CPU_verifying_O64_pair\n' > "$ti_status"
python3 -u experiments/tianshu_W4_O64_affine_results_20261024/compare_screens_cpu.py
ixsmi | grep -q 'No running processes found'
printf 'QKV_fitted_GPU_component\n' > "$ti_status"
python3 -u "$ti_component/probe_gpu.py"
printf 'QKV_fitted_independent_CPU_verification\n' > "$ti_status"
python3 -u "$ti_component/verify_cpu.py"
jq -e '
 .complete and .all_full_FP64_numeric_passed and
 ([.timings[]|select(.mode=="graph" and .variant=="W4_fitted")][0]) as $fit |
 ([.timings[]|select(.mode=="graph" and .variant=="W4_minmax")][0]) as $base |
 ([.projection_errors[]|select(.variant=="W4_fitted")|.relative_L2_vs_original_FP64]|add/length) as $fit_error |
 ([.projection_errors[]|select(.variant=="W4_minmax")|.relative_L2_vs_original_FP64]|add/length) as $base_error |
 $fit.component_advancement_gate and ($fit.median_42banks_ms/$base.median_42banks_ms-1)*100<=1 and $fit_error<$base_error
' "$ti_component/INDEPENDENT_CPU_GATE_32g.json" > /dev/null
printf 'QKV_fitted_component_qualified_pending_model_work\n' > "$ti_status"
printf 'No automatic model integration or adoption; fresh combined-model Math still required.\n'
