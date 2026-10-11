#!/usr/bin/env bash
# BI-V150 adaptive Decode profile validated at102/105 MATH-500 (>95%).
set -euo pipefail

export VLLM_PLUGINS=fl
export VLLM_FL_ILUVATAR_HYBRID_ATTENTION=1
export VLLM_FL_ILUVATAR_ADAPTIVE_DECODE_SEGMENTS=1
export FLAGGEMS_ILUVATAR_COMPACT_VARLEN=1
export VLLM_FL_USE_FLAGGEMS_ATTN=0
export VLLM_FL_ILUVATAR_SPLIT_MIXED_ATTN=0
export VLLM_FL_ILUVATAR_PREFILL_BUCKETS=0
export VLLM_FL_ILUVATAR_LOGITS_GATHER=0

exec vllm serve "${TIANSHU_MODEL_PATH:-/workspace/MiniCPM5-2B}" \
  --port "${TIANSHU_PORT:-9031}" \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --served-model-name minicpm \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072 "$@"
