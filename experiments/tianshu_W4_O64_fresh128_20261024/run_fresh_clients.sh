#!/usr/bin/env bash
set -euo pipefail
ti_root=$1
ti_here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
test -f "$ti_root/state.json"
mkdir "$ti_root/client_rounds" "$ti_root/benchmark_results"
cat > "$ti_root/client_configuration.json" <<'JSON'
{"case":[4096,1024,64,128],"seeds":[2026102401,2026102402,2026102403,2026102404],"excluded_rounds":[1],"max_prefix_hit_fraction":0.01,"max_preemptions":0,"save_full_input_bodies":false}
JSON
for ti_round in 1 2 3 4; do
 ti_seed=$((2026102400+ti_round))
 curl --fail --silent --show-error --max-time 10 http://127.0.0.1:9031/metrics > "$ti_root/client_rounds/before_${ti_round}.txt"
 vllm bench serve --backend vllm --model minicpm --tokenizer /workspace/MiniCPM5-2B --endpoint /v1/completions --host localhost --port 9031 --dataset-name random --ignore-eos --random-input-len 4096 --random-output-len 1024 --max-concurrency 64 --num-prompts 128 --seed "$ti_seed" > "$ti_root/client_rounds/round_${ti_round}.log" 2>&1
 curl --fail --silent --show-error --max-time 10 http://127.0.0.1:9031/metrics > "$ti_root/client_rounds/after_${ti_round}.txt"
 python3 "$ti_here/parse_client_cpu.py" "$ti_root" "$ti_round"
done
