#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
ti_base=experiments/tianshu_BM8_official4k_pair_20261024
ti_meta=/dev/shm/tianshu_BM8_official4k_pair_20261024
ti_followup=/dev/shm/tianshu_official_followup_20261024
ixsmi | grep -q 'No running processes found'
sha256sum --check --status "$ti_base/EXECUTION.sha256"
test ! -e "$ti_meta"
mkdir -p "$ti_meta" "$ti_followup"
cp "$ti_base/EXECUTION.sha256" "$ti_meta/controllers.sha256"
trap 'printf "official_pair_exit_code=%s\n" "$?"' EXIT
for ti_arm in control candidate; do
 if test "$ti_arm" = control; then ti_root=$ti_followup/4k_r5_control_bm8_inactive; else ti_root=$ti_followup/4k_bm8; fi
 test ! -e "$ti_root"
 printf '%s_official4K_running\n' "$ti_arm" > "$ti_meta/phase.txt"
 ixsmi | grep -q 'No running processes found'
 python3 -u "$ti_base/run_model.py" --output "$ti_root" --stage official --batch 64 --mode "$ti_arm"
 jq -e '.phase=="complete" and .evaluation_exit_code==0' "$ti_root/result.json" > /dev/null
done
printf 'both_official4K_complete_pending_comparison\n' > "$ti_meta/phase.txt"
sha256sum --check "$ti_meta/controllers.sha256" > "$ti_meta/controllers_verified.txt"
python3 "$ti_base/compare_official_cpu.py"
printf 'paired_official4K_complete_pending16K_selection\n' > "$ti_meta/phase.txt"
