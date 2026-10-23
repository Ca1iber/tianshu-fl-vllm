#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_native_W4_formats_20261023
sha256sum --check --status "$base/EXECUTION.sha256"
sha256sum --check --status "$base/INPUTS.sha256"
remaining=$(( $(date -u -d '2026-10-23 15:58:00 UTC' +%s) - $(date -u +%s) ));test "$remaining" -gt 900
ixsmi | grep 'No running processes found'
runner=/workspace/benchmark-results/20261023_native_W4_formats_runner_tianshu
mkdir "$runner"
for variant in generic_low_even generic_high_even awq_natural_zeros_packed awq_natural_zeros_bf16 awq_natural_offsets_bf16 awq_permuted_zeros_packed awq_permuted_zeros_bf16 awq_permuted_offsets_bf16;do
 if [[ "$variant" == generic_* ]] && ! jq -e '.complete and .functionality_passed' experiments/tianshu_native_INT4_inventory_20261023/FUNCTIONAL_GATE_32g.json >/dev/null;then printf '%s: prior generic functional smoke failed or missing, skipped\n' "$variant" >> "$runner/skipped.txt";continue;fi
 ixsmi | grep 'No running processes found'
 if timeout --signal=TERM --kill-after=10s 45s python3 "$base/probe_gpu.py" --variant "$variant" > "$runner/${variant}.log" 2>&1;then status=0;else status=$?;fi
 printf '%s exit=%s\n' "$variant" "$status" >> "$runner/process_status.txt"
 ixsmi > "$runner/${variant}_hardware_after.txt"
 grep 'No running processes found' "$runner/${variant}_hardware_after.txt"
done
CUDA_VISIBLE_DEVICES='' python3 "$base/verify_cpu.py" > "$runner/independent_CPU.log" 2>&1
sha256sum --check --status "$base/EXECUTION.sha256"
sha256sum --check --status "$base/INPUTS.sha256"
