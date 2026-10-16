#!/usr/bin/env bash
set -euo pipefail
interval=${1:?interval}
export HF_HOME=/model-cache HF_HUB_CACHE=/model-cache/hub
export HF_TOKEN_PATH=/root/.cache/huggingface/token
nohup setsid sglang serve \
  --model-path openai/gpt-oss-120b --revision b5c939de8f754692c1647ca79fbf85e8c1e70f8a \
  --served-model-name openai/gpt-oss-120b \
  --reasoning-parser gpt-oss --tool-call-parser gpt-oss \
  --tp 1 --context-length 8192 --chunked-prefill-size 4096 \
  --mem-fraction-static 0.75 --max-running-requests 16 \
  --disable-cuda-graph --stream-interval "$interval" \
  --host 0.0.0.0 --port 8000 \
  > "/work/sglang-i${interval}.log" 2>&1 < /dev/null &
echo "$!" > /work/server.pid
