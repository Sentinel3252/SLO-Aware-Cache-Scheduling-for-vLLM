#!/usr/bin/env bash
set -euo pipefail

: "${MODEL:=Qwen/Qwen2.5-0.5B-Instruct}"
: "${CACHEFLOW_MODE:=caras}"
: "${CACHEFLOW_CANDIDATE_WINDOW:=16}"
: "${CACHEFLOW_MAX_WAIT_MS:=2000}"
: "${CACHEFLOW_PREFILL_TOKENS_PER_MS:=4.0}"

export CACHEFLOW_MODE CACHEFLOW_CANDIDATE_WINDOW CACHEFLOW_MAX_WAIT_MS
export CACHEFLOW_PREFILL_TOKENS_PER_MS

python -m cacheflow.tools.check_vllm_compat
exec vllm serve "$MODEL" \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --stream-interval 1 \
  --no-async-scheduling \
  --scheduler-cls cacheflow.vllm_adapter.CARASScheduler \
  "$@"
