#!/usr/bin/env bash
set -euo pipefail
interval=${1:?interval}
tags=${2:?true or false}
export PYTHONPATH=/work/overlay HF_HOME=/model-cache HF_HUB_CACHE=/model-cache/hub
export HF_TOKEN_PATH=/root/.cache/huggingface/token
export DYN_FILE_KV=/work/discovery DYN_NAMESPACE=gpt-oss-unified
mkdir -p "$DYN_FILE_KV"
nohup setsid python3 -m dynamo.sglang \
  --namespace gpt-oss-unified --discovery-backend file \
  --request-plane tcp --response-plane tcp --event-plane zmq \
  --dyn-tool-call-parser harmony --dyn-reasoning-parser gpt_oss \
  --dyn-structural-tag "$tags" --reasoning-parser gpt-oss \
  --model-path openai/gpt-oss-120b --revision b5c939de8f754692c1647ca79fbf85e8c1e70f8a --served-model-name openai/gpt-oss-120b \
  --tp 1 --context-length 8192 --chunked-prefill-size 4096 \
  --mem-fraction-static 0.75 --max-running-requests 16 \
  --disable-cuda-graph --stream-interval "$interval" \
  > "/work/worker-i${interval}-tags-${tags}.log" 2>&1 < /dev/null &
echo "$!" > /work/server.pid
for version in 1 2; do
  port=8000
  frontend_overlay=/work/overlay-v2
  if [ "$version" = 1 ]; then port=8001; frontend_overlay=/work/overlay; fi
  PYTHONPATH="$frontend_overlay" DYN_PARSER_VERSION="$version" nohup setsid python3 -m dynamo.frontend \
    --namespace gpt-oss-unified --discovery-backend file \
    --request-plane tcp --response-plane tcp --event-plane zmq \
    --http-port "$port" \
    > "/work/frontend-v${version}-i${interval}-tags-${tags}.log" 2>&1 < /dev/null &
  echo "$!" > "/work/frontend-v${version}.pid"
done
