#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
quant_profile_queue=/workspace/benchmark-results/20261022_quant_ixkn_queue_tianshu
mkdir -p "$quant_profile_queue"
state() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$quant_profile_queue/state.txt"; }
trap 'code=$?; state "EXIT $code"' EXIT
quant_previous_pid=$(cat /workspace/benchmark-results/20261022_quant_nt_serial_queue_tianshu/pid)
state "WAIT_NT $quant_previous_pid"
while test -r "/proc/$quant_previous_pid/stat"; do
 quant_process_state=$(awk '{print $3}' "/proc/$quant_previous_pid/stat")
 if test "$quant_process_state" = Z; then break; fi
 test "$(date -u +%s)" -lt "$(date -ud '2026-10-22 15:58:00' +%s)"
 sleep 5
done
grep -q QUEUE_COMPLETE /workspace/benchmark-results/20261022_quant_nt_serial_queue_tianshu/state.txt
jq -e '.complete==true' experiments/tianshu_w8a16_nt_postscale_20261022/INDEPENDENT_CPU_GATE_32g.json
jq -e '.complete==true' experiments/tianshu_w4a16_nt_group_postscale_20261022/INDEPENDENT_CPU_GATE_32g.json
while read -r profile_role profile_m profile_format profile_path; do
 test "$(date -u +%s)" -lt "$(date -ud '2026-10-22 15:43:00' +%s)"
 profile_raw="/workspace/benchmark-results/20261022_ixkn_${profile_role}_M${profile_m}_${profile_format}_${profile_path}_tianshu"
 mkdir "$profile_raw"
 ixsmi > "$profile_raw/hardware_before.txt"
 grep -q 'No running processes found' "$profile_raw/hardware_before.txt"
 state "PROFILE $profile_role $profile_m $profile_format $profile_path"
 timeout --signal=INT --kill-after=30s 900s /usr/local/corex-4.5.0/bin/ixkn-cli --profile-from-start off --set detailed --cache-control none --csv -o "$profile_raw/metrics" --log-file "$profile_raw/profiler.csv" python3 experiments/tianshu_quant_ixkn_20261022/profile_one.py --role "$profile_role" --m "$profile_m" --format "$profile_format" --path "$profile_path" --output "$profile_raw/app" --stop-at-utc 2026-10-22T15:58:00+00:00 > "$profile_raw/app.log" 2>&1 < /dev/null
 jq -e '.complete==true' "$profile_raw/app/result.json"
 test -f "$profile_raw/metrics.ixkn-rep"
 CUDA_VISIBLE_DEVICES='' timeout 30s /usr/local/corex-4.5.0/bin/ixkn-cli --import-profile "$profile_raw/metrics.ixkn-rep" --csv --page details > "$profile_raw/metrics_details.csv" 2> "$profile_raw/import.log"
 ixsmi > "$profile_raw/hardware_after.txt"
 state "COMPLETE $profile_role $profile_m $profile_format $profile_path"
done <<'CASES'
down 64 W8_channel candidate
down 64 W8_channel BF16
down 64 W8_channel W8A8
down 64 W4_aff128 candidate
lm_head 8 W8_channel candidate
lm_head 8 W8_channel BF16
lm_head 64 W8_channel candidate
lm_head 64 W8_channel W8A8
CASES
state QUEUE_COMPLETE
