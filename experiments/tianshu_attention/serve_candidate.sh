#!/usr/bin/env bash
set -euo pipefail
export VLLM_PLUGINS=fl
export VLLM_FL_USE_FLAGGEMS_ATTN=1
export FLAGGEMS_ILUVATAR_COMPACT_VARLEN=1
export VLLM_FL_ILUVATAR_SPLIT_MIXED_ATTN=1
exec vllm serve /workspace/MiniCPM5-2B \
  --port 9031 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --served-model-name minicpm \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072
