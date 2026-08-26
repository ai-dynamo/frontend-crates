#!/usr/bin/env bash
set -euo pipefail

cluster_name=${AGENT_SANDBOX_KIND_CLUSTER:-agent-rt-sandbox}
agent_sandbox_version=${AGENT_SANDBOX_VERSION:-v0.5.3}
repo_root=$(git rev-parse --show-toplevel)
service_image=agent-rt-sandbox-service:local
sandbox_image=agent-rt-python-sandbox:local
token=local-sandbox-token-0123456789abcdef

for command in docker kind kubectl curl jq rg; do
  command -v "$command" >/dev/null || {
    echo "missing required command: $command" >&2
    exit 1
  }
done

if ! kind get clusters | rg -Fxq "$cluster_name"; then
  kind create cluster --name "$cluster_name"
fi

docker build -f "$repo_root/agent-sandbox-service/images/service/Dockerfile" -t "$service_image" "$repo_root"
docker build -f "$repo_root/agent-sandbox-service/images/python-sandbox/Dockerfile" -t "$sandbox_image" "$repo_root"
kind load docker-image --name "$cluster_name" "$service_image" "$sandbox_image"

kubectl apply -f "https://github.com/kubernetes-sigs/agent-sandbox/releases/download/${agent_sandbox_version}/sandbox-with-extensions.yaml"
kubectl -n agent-sandbox-system wait --for=condition=Available deployment --all --timeout=180s
kubectl apply -k "$repo_root/agent-sandbox-service/deploy/kubernetes/local"
kubectl -n agent-rt-sandbox-system rollout restart deployment/agent-rt-sandbox-service
kubectl -n agent-rt-sandbox-system rollout status statefulset/postgres --timeout=180s
kubectl -n agent-rt-sandbox-system rollout status deployment/agent-rt-sandbox-service --timeout=180s
kubectl -n agent-sandboxes wait --for=condition=Ready pod -l agents.x-k8s.io/warm-pool-sandbox --timeout=180s

kubectl -n agent-rt-sandbox-system port-forward service/agent-rt-sandbox-service 18090:8090 >/tmp/agent-rt-sandbox-port-forward.log 2>&1 &
port_forward_pid=$!
trap 'kill "$port_forward_pid" 2>/dev/null || true' EXIT

for _ in $(seq 1 30); do
  if curl -fsS http://127.0.0.1:18090/healthz >/dev/null; then
    break
  fi
  sleep 1
done

start_request='{
  "api_version": "v1",
  "scope": {"tenant_id": "tenant-a", "principal_id": "principal-a"},
  "workspace_id": "kind-workspace",
  "execution_id": "kind-execution",
  "profile": "python-deny-egress",
  "command": {
    "argv": ["python", "-c", "from pathlib import Path; Path(\"/workspace/result.txt\").write_text(\"done\"); print(42)"],
    "cwd": "/workspace",
    "env": {},
    "stdin": [],
    "artifact_paths": ["result.txt"]
  },
  "limits": {
    "timeout_millis": 10000,
    "max_output_bytes": 1024,
    "max_artifact_bytes": 4096
  }
}'
scope_headers=(
  -H "Authorization: Bearer ${token}"
  -H "x-agent-sandbox-tenant-id: tenant-a"
  -H "x-agent-sandbox-principal-id: principal-a"
  -H "content-type: application/json"
)

curl -fsS "${scope_headers[@]}" -d "$start_request" http://127.0.0.1:18090/v1/executions | jq .

lookup_request='{
  "scope": {"tenant_id": "tenant-a", "principal_id": "principal-a"},
  "workspace_id": "kind-workspace",
  "profile": "python-deny-egress",
  "execution_id": "kind-execution"
}'
for _ in $(seq 1 60); do
  outcome=$(curl -fsS "${scope_headers[@]}" -d "$lookup_request" http://127.0.0.1:18090/v1/executions:lookup)
  state=$(jq -r '.state // "missing"' <<<"$outcome")
  if [[ "$state" == "succeeded" ]]; then
    jq -e '.exit_code == 0 and .stdout == [52, 50, 10] and .artifacts[0].name == "result.txt"' <<<"$outcome" >/dev/null
    artifact_id=$(jq -r '.artifacts[0].artifact_id' <<<"$outcome")
    artifact_request=$(jq -n \
      --argjson execution "$lookup_request" \
      --arg artifact_id "$artifact_id" \
      '{execution: $execution, artifact_id: $artifact_id}')
    artifact=$(curl -fsS "${scope_headers[@]}" -d "$artifact_request" http://127.0.0.1:18090/v1/artifacts:read)
    jq -e '.metadata.name == "result.txt" and .bytes_base64 == "ZG9uZQ=="' <<<"$artifact" >/dev/null
    echo "Kubernetes sandbox execution succeeded"
    exit 0
  fi
  if [[ "$state" =~ ^(failed|cancelled|timed_out|outcome_unknown)$ ]]; then
    jq . <<<"$outcome" >&2
    exit 1
  fi
  sleep 1
done

echo "sandbox execution did not finish" >&2
exit 1
