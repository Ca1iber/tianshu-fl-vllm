#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_dense_w8a8_validation_20261020
screen=/workspace/benchmark-results/20261020_dense_w8a8_service4k_v4_tianshu
pid=$(cat /workspace/benchmark-results/20261020_dense_w8a8_service4k_v4_launcher.pid)
cutoff=$(date -u -d '2026-10-20 15:58:00 UTC' +%s)
printf 'Waiting for exact W8A8 screen launcher PID %s\n' "$pid"
while kill -0 "$pid" 2>/dev/null;do
    if ps -p "$pid" -o stat= | grep -q '^Z';then break;fi
    test "$((cutoff-$(date -u +%s)))" -gt 6000
    sleep 15
done
test "$(cat "$screen/phase.txt")" = complete
test -f "$screen/complete.txt"
gate=experiments/tianshu_dense_w8a8_service_20261020/DIAGNOSTIC_4K_GATE_32g.json
if ! jq -e '.complete and .independently_rechecked and .gain_percent>=1 and .spread_percent.control<1 and .spread_percent.candidate<1' "$gate" >/dev/null;then
    printf 'No qualifying stable >=1%% service gain; fresh Math and official benchmark not started.\n'
    exit 0
fi
ixsmi | grep -q 'No running processes found'
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 "$base/check_profile_cpu.py"
exec bash "$base/run_math_and_official.sh"
