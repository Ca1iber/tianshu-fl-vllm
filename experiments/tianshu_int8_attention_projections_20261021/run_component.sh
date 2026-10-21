#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_int8_attention_projections_20261021
runner=/workspace/benchmark-results/20261021_int8_attention_projections_runner_tianshu
root=/workspace/benchmark-results/20261021_int8_attention_projections_GPU_tianshu
test ! -e "$runner";test ! -e "$root";mkdir "$runner";printf 'component\n' > "$runner/phase.txt"
trap 'code=$?;if ((code!=0));then cp "$runner/phase.txt" "$runner/failed_phase.txt";printf "failed\n" > "$runner/phase.txt";printf "%s\n" "$code" > "$runner/exit_code.txt";fi' EXIT
ixsmi > "$runner/hardware_before.txt";grep -q 'No running processes found' "$runner/hardware_before.txt"
sha256sum "$base"/*.py "$base"/*.sh "$base"/*.json > "$runner/frozen_entrypoints.sha256"
timeout -k 20s 1800s env -u PYTHONPATH OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python3 -u "$base/probe_gpu.py" --output "$root" --stop-at-utc 2026-10-21T15:58:00+00:00 > "$runner/probe.log" 2>&1
ixsmi > "$runner/hardware_after.txt";grep -q 'No running processes found' "$runner/hardware_after.txt"
sha256sum --check --quiet "$runner/frozen_entrypoints.sha256"
printf 'awaiting_independent_CPU_recheck\n' > "$runner/phase.txt";date -Is > "$runner/GPU_complete.txt"
