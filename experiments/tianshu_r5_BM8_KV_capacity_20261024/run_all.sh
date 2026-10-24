#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
ti_model=experiments/tianshu_r5_BM8_KV_capacity_20261024
ti_root=/dev/shm/tianshu_r5_BM8_capacity_runs_20261024
ixsmi | grep -q 'No running processes found'
jq -e '.complete and .model_ready and .native_CPU_allocation_verified' "$ti_model/CPU_GATE_32g.json" > /dev/null
trap 'printf "capacity_pipeline_exit_code=%s\n" "$?"' EXIT
for ti_batch in 8 64; do
 test ! -e "$ti_root/math${ti_batch}"
 printf 'fresh_math%s_running\n' "$ti_batch" > "$ti_root/phase.txt"
 python3 -u "$ti_model/run_model.py" --output "$ti_root/math${ti_batch}" --stage math --batch "$ti_batch" --mode candidate
 jq -e '.phase=="complete" and .evaluation_exit_code==0 and .accuracy_passed and .total==105' "$ti_root/math${ti_batch}/result.json" > /dev/null
 printf 'fresh_math%s_passed\n' "$ti_batch" > "$ti_root/phase.txt"
done
for ti_arm in control candidate; do
 printf 'official_%s_running\n' "$ti_arm" > "$ti_root/phase.txt"
 python3 -u "$ti_model/run_model.py" --output "$ti_root/official_${ti_arm}" --stage official --batch 64 --mode "$ti_arm"
 jq -e '.phase=="complete" and .evaluation_exit_code==0' "$ti_root/official_${ti_arm}/result.json" > /dev/null
 printf 'official_%s_passed\n' "$ti_arm" > "$ti_root/phase.txt"
done
printf 'both_official_arms_complete_pending_comparison\n' > "$ti_root/phase.txt"
python3 "$ti_model/compare_cpu.py"
printf 'all_complete_pending_review\n' > "$ti_root/phase.txt"
