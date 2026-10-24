#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
ti_dir=experiments/tianshu_W4_QKVO64_fitted_service_20261024
test ! -e "$ti_dir/CPU_GATE_32g.json"
ixsmi | grep -q 'No running processes found'
printf '14fb8e7f0a18d53d1f239773758bf581cee7e456a4523a54622c3a245b64402c  /workspace/MiniCPM5-2B/model-00000-of-00001.safetensors\n' > "$ti_dir/CHECKPOINT_CPU.sha256"
sha256sum -c "$ti_dir/CHECKPOINT_CPU.sha256"
python3 -m py_compile "$ti_dir/candidate_profile.py" "$ti_dir/audit.py" "$ti_dir/prepare_profile_cpu.py" "$ti_dir/run_model.py"
python3 -u "$ti_dir/prepare_profile_cpu.py"
