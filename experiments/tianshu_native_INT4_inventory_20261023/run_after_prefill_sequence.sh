#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_native_INT4_inventory_20261023
sha256sum --check --status "$base/EXECUTION.sha256"
printf 'Waiting for prefill sequence PID41925; independent INT4 functional smoke only.\n'
while [ -e /proc/41925/stat ] && [ "$(awk '{print $3}' /proc/41925/stat)" != Z ];do
 tr '\0' ' ' < /proc/41925/cmdline | grep -F 'tianshu_prefill_RMS_QKV_sequence_v3_20261023/run_after_quality.sh' >/dev/null
 remaining=$(( $(date -u -d '2026-10-23 15:58:00 UTC' +%s) - $(date -u +%s) ));test "$remaining" -gt 300
 sleep 10
done
ixsmi | grep 'No running processes found'
bash "$base/run_smoke.sh"
printf 'INT4 functional smoke complete; no model or performance claim.\n'
