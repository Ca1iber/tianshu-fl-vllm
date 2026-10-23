#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_native_W4_formats_20261023
sha256sum --check --status "$base/EXECUTION.sha256"
printf 'Waiting for INT4 micro-smoke PID43163, no concurrent GPU.\n'
while [ -e /proc/43163/stat ] && [ "$(awk '{print $3}' /proc/43163/stat)" != Z ];do
 tr '\0' ' ' < /proc/43163/cmdline | grep -F 'tianshu_native_INT4_inventory_20261023/run_after_prefill_sequence.sh' >/dev/null
 remaining=$(( $(date -u -d '2026-10-23 15:58:00 UTC' +%s) - $(date -u +%s) ));test "$remaining" -gt 900
 sleep 10
done
ixsmi | grep 'No running processes found'
bash "$base/run.sh"
