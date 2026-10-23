#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_prefill_RMS_QKV_sequence_v2_20261023
sha256sum --check --status "$base/EXECUTION.sha256"
printf 'Waiting for frozen quality PID 37294; no concurrent GPU work.\n'
while [ -e /proc/37294/stat ] && [ "$(awk '{print $3}' /proc/37294/stat)" != Z ];do
 tr '\0' ' ' < /proc/37294/cmdline | grep -F 'tianshu_prefill_RMS_QKV_quality_v2_20261023/run_quality.sh' >/dev/null
 remaining=$(( $(date -u -d '2026-10-23 15:58:00 UTC' +%s) - $(date -u +%s) ));test "$remaining" -gt 2400
 sleep 10
done
if ! test -f experiments/tianshu_prefill_RMS_QKV_quality_results_v2_20261023/QUALITY_GATE_32g.json;then printf 'Quality did not produce independent complete gate; short GPU skipped.\n';exit 0;fi
if ! jq -e '.complete and .model_quality_qualified and .full_profile_and_accuracy_audit_recomputed_exact and .C8_correct/105>0.95 and .C64_correct/105>0.95' experiments/tianshu_prefill_RMS_QKV_quality_results_v2_20261023/QUALITY_GATE_32g.json >/dev/null;then printf 'Quality gate failed; short GPU skipped.\n';exit 0;fi
ixsmi | grep 'No running processes found'
remaining=$(( $(date -u -d '2026-10-23 15:58:00 UTC' +%s) - $(date -u +%s) ));test "$remaining" -gt 2400
available=$(df -B1 --output=avail /workspace | tail -n 1);test "$available" -ge 1073741824
printf 'Independent dual quality passed; starting one fresh short pair, no formal auto-run.\n'
bash experiments/tianshu_prefill_RMS_QKV_short4k_v2_20261023/run_pair.sh
printf 'Short pair and independent CPU verification complete; formal requires positive short gate.\n'
