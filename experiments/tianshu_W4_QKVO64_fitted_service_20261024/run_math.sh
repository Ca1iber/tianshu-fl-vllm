#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
ti_dir=experiments/tianshu_W4_QKVO64_fitted_service_20261024
ti_phase=/workspace/benchmark-results/20261024_W4_QKVO64_fitted_v2_quality_phase.txt
ixsmi | grep -q 'No running processes found'
jq -e '.complete==true and .model_ready==true and .model_quality_qualified==false' "$ti_dir/CPU_GATE_32g.json" > /dev/null
trap 'printf "quality_pipeline_exit_code=%s\n" "$?"' EXIT
for ti_batch in 64 8; do
 ti_root=/workspace/benchmark-results/20261024_W4_QKVO64_fitted_v2_math${ti_batch}_tianshu
 test ! -e "$ti_root"
 printf 'fresh_math%s_running\n' "$ti_batch" > "$ti_phase"
 python3 -u "$ti_dir/run_model.py" --output "$ti_root" --stage math --batch "$ti_batch" --mode candidate
 jq -e '.phase=="complete" and .evaluation_exit_code==0 and .accuracy_passed==true and .total==105' "$ti_root/result.json" > /dev/null
 jq -e '.complete==true and .above95==true and .fresh_entire_105==true and .all_caps_in_denominator==true' "$ti_root/accuracy_audit.json" > /dev/null
 printf 'fresh_math%s_passed\n' "$ti_batch"
done
jq -e -s '.[0].frozen_sources==.[1].frozen_sources and .[0].server_command==.[1].server_command' /workspace/benchmark-results/20261024_W4_QKVO64_fitted_v2_math64_tianshu/result.json /workspace/benchmark-results/20261024_W4_QKVO64_fitted_v2_math8_tianshu/result.json > /dev/null
printf 'both_fresh_full105_quality_batches_passed_pending_review\n' > "$ti_phase"
