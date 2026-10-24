#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
ti_model=experiments/tianshu_r5_BM8_KV_20261024
ti_results=experiments/tianshu_r5_BM8_results_20261024
ti_root=/dev/shm/tianshu_r5_BM8_runs_20261024
ixsmi | grep -q 'No running processes found'
jq -e '.complete and .model_ready and .original997_exact_subset' "$ti_model/CPU_GATE_32g.json" > /dev/null
trap 'printf "quality_pipeline_exit_code=%s\n" "$?"' EXIT
for ti_batch in 8 64; do
 test ! -e "$ti_root/math${ti_batch}"
 printf 'fresh_math%s_running\n' "$ti_batch" > "$ti_root/phase.txt"
 python3 -u "$ti_model/run_model.py" --output "$ti_root/math${ti_batch}" --stage math --batch "$ti_batch" --mode candidate
 jq -e '.phase=="complete" and .evaluation_exit_code==0 and .accuracy_passed==true and .total==105' "$ti_root/math${ti_batch}/result.json" > /dev/null
 jq -e '.complete==true and .above95==true and .fresh_entire_105==true and .all_caps_in_denominator==true' "$ti_root/math${ti_batch}/accuracy_audit.json" > /dev/null
 python3 "$ti_results/collect_quality_cpu.py" --batch "$ti_batch"
 printf 'fresh_math%s_passed\n' "$ti_batch" > "$ti_root/phase.txt"
done
jq -e -s '.[0].frozen_sources==.[1].frozen_sources and .[0].server_command==.[1].server_command' "$ti_root/math64/result.json" "$ti_root/math8/result.json" > /dev/null
printf 'both_fresh_full105_quality_batches_passed_pending_review\n' > "$ti_root/phase.txt"
