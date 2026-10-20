#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
diagnostic_pid=$(cat /workspace/benchmark-results/20261020_dense_runtime_m_M64_diagnostic.pid)
while ps -p "$diagnostic_pid" -o args= 2>/dev/null | grep -Fq 'diagnose_m64.py';do sleep 5;done
ixsmi | grep -q 'No running processes found'
exec bash experiments/tianshu_lm_precision_service_20261020/run_pair.sh
