#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_k8vi8_heads_v2_20261021
runner=/workspace/benchmark-results/20261021_k8vi8_heads_v2_runner_tianshu
test ! -e "$runner";mkdir "$runner";printf 'component\n' > "$runner/phase.txt"
trap 'code=$?;if ((code!=0));then cp "$runner/phase.txt" "$runner/failed_phase.txt";printf "failed\n" > "$runner/phase.txt";printf "%s\n" "$code" > "$runner/exit_code.txt";fi' EXIT
ixsmi | grep -q 'No running processes found'
timeout -k 20s 1800s env -u PYTHONPATH python3 -u "$base/probe_gpu.py" --output /workspace/benchmark-results/20261021_k8vi8_heads_v2_GPU_tianshu --stop-at-utc 2026-10-21T15:58:00+00:00 > "$runner/probe.log" 2>&1
printf 'CPU_recheck\n' > "$runner/phase.txt"
env -u PYTHONPATH python3 "$base/verify_gpu.py" --root /workspace/benchmark-results/20261021_k8vi8_heads_v2_GPU_tianshu > "$runner/verify.log" 2>&1
printf 'complete\n' > "$runner/phase.txt";date -Is > "$runner/complete.txt"
