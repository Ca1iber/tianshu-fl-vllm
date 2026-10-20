#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
while ps -p 96729 -o stat= 2>/dev/null | grep -qv '^Z';do sleep 5;done
ixsmi | grep -q 'No running processes found'
exec bash experiments/tianshu_lm_precision_service_20261020/run_pair.sh
