#!/usr/bin/env bash
# Smoke tests for dynamo-demo-server. Hits each endpoint and prints a one-line summary.
# Usage: scripts/smoke.sh [http://127.0.0.1:3000]

set -u
BASE="${1:-http://127.0.0.1:3000}"
PASS=0; FAIL=0

check() {
  local name="$1"; shift
  local body
  body=$(curl -fsS "$@" 2>&1) && {
    echo "  PASS  $name"
    PASS=$((PASS+1))
  } || {
    echo "  FAIL  $name"
    echo "        $body" | head -c 200
    echo
    FAIL=$((FAIL+1))
  }
}

echo "== $BASE =="

check "health" "$BASE/health"

check "tokenize" -H 'Content-Type: application/json' -X POST "$BASE/v1/tokenize" \
  -d '{"text":"hello world"}'

check "detokenize" -H 'Content-Type: application/json' -X POST "$BASE/v1/detokenize" \
  -d '{"token_ids":[1,15043,3186]}'

check "chat completions" -H 'Content-Type: application/json' -X POST "$BASE/v1/chat/completions" \
  -d '{"model":"echo","messages":[{"role":"user","content":"hello"}]}'

check "completions" -H 'Content-Type: application/json' -X POST "$BASE/v1/completions" \
  -d '{"model":"echo","prompt":"hello world"}'

check "anthropic messages" -H 'Content-Type: application/json' -X POST "$BASE/v1/messages" \
  -d '{"model":"echo","messages":[{"role":"user","content":"hello"}],"max_tokens":50}'

check "responses" -H 'Content-Type: application/json' -X POST "$BASE/v1/responses" \
  -d '{"model":"echo","input":"hello"}'

check "tool-parse (hermes)" -H 'Content-Type: application/json' -X POST "$BASE/v1/tool-parse" \
  -d '{"text":"<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"SF\"}}</tool_call>","parser":"hermes"}'

check "reasoning-parse (deepseek_r1)" -H 'Content-Type: application/json' -X POST "$BASE/v1/reasoning-parse" \
  -d '{"text":"<think>let me think</think>the answer is 42","parser":"deepseek_r1"}'

echo
echo "== $PASS passed, $FAIL failed =="
[ "$FAIL" -eq 0 ]
