#!/usr/bin/env bash

set -euo pipefail

PROJECT="/root/autodl-tmp/projects/single-gpu-specdec"
VENV="/root/autodl-tmp/venvs/vllm"

MODEL_PATH="/root/autodl-tmp/huggingface/hub/models--Qwen--Qwen3-14B-AWQ/snapshots/31c69efc29464b6bb0aee1398b5a7b50a99340c3"
DRAFT_PATH="/root/autodl-tmp/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"

source "$VENV/bin/activate"

export HF_HOME="/root/autodl-tmp/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$PROJECT/logs/specdec_server_${TIMESTAMP}.log"

echo "Starting speculative decoding server"
echo "Target: $MODEL_PATH"
echo "Draft:  $DRAFT_PATH"
echo "K:      5"
echo "Log:    $LOG_FILE"

vllm serve "$MODEL_PATH" \
  --served-model-name "Qwen/Qwen3-14B-AWQ" \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype auto \
  --max-model-len 8192 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.85 \
  --generation-config vllm \
  --no-enable-prefix-caching \
  --speculative-config "{\"method\":\"draft_model\",\"model\":\"${DRAFT_PATH}\",\"num_speculative_tokens\":5}" \
  2>&1 | tee "$LOG_FILE"
