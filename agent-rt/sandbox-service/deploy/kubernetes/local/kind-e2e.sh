#!/usr/bin/env bash
set -euo pipefail

cluster_name=${AGENT_SANDBOX_KIND_CLUSTER:-agent-rt-sandbox}
agent_sandbox_version=${AGENT_SANDBOX_VERSION:-v0.5.3}
repo_root=$(git rev-parse --show-toplevel)
service_image=agent-rt-sandbox-service:local
sandbox_image=agent-rt-python-sandbox:local
token=local-sandbox-token-0123456789abcdef
run_id=$(date +%s%N)
workspace_id=kind-workspace-${run_id}
execution_id=kind-execution-${run_id}

for command in docker kind kubectl curl jq rg; do
  command -v "$command" >/dev/null || {
    echo "missing required command: $command" >&2
    exit 1
  }
done

if ! kind get clusters | rg -Fxq "$cluster_name"; then
  kind create cluster --name "$cluster_name"
fi

docker build -f "$repo_root/agent-rt/sandbox-service/images/service/Dockerfile" -t "$service_image" "$repo_root"
docker build -f "$repo_root/agent-rt/sandbox-service/images/python-sandbox/Dockerfile" -t "$sandbox_image" "$repo_root"
kind load docker-image --name "$cluster_name" "$service_image" "$sandbox_image"

kubectl apply -f "https://github.com/kubernetes-sigs/agent-sandbox/releases/download/${agent_sandbox_version}/sandbox-with-extensions.yaml"
kubectl -n agent-sandbox-system wait --for=condition=Available deployment --all --timeout=180s
kubectl apply -k "$repo_root/agent-rt/sandbox-service/deploy/kubernetes/local"
kubectl -n agent-rt-sandbox-system rollout restart deployment/agent-rt-sandbox-service
kubectl -n agent-rt-sandbox-system rollout status statefulset/postgres --timeout=180s
kubectl -n agent-rt-sandbox-system rollout status deployment/agent-rt-sandbox-service --timeout=180s
kubectl -n agent-sandboxes wait --for=condition=Ready pod -l agents.x-k8s.io/warm-pool-sandbox --timeout=180s

kubectl -n agent-rt-sandbox-system port-forward service/agent-rt-sandbox-service 18090:8090 >/tmp/agent-rt-sandbox-port-forward.log 2>&1 &
port_forward_pid=$!
trap 'kill "$port_forward_pid" 2>/dev/null || true' EXIT

service_ready=false
for _ in $(seq 1 30); do
  if curl -fsS http://127.0.0.1:18090/healthz >/dev/null; then
    service_ready=true
    break
  fi
  sleep 1
done
if [[ "$service_ready" != "true" ]]; then
  echo "sandbox service did not become ready" >&2
  exit 1
fi

start_request='{
  "api_version": "v1",
  "scope": {"tenant_id": "tenant-a", "principal_id": "principal-a"},
  "workspace_id": "kind-workspace-placeholder",
  "execution_id": "kind-execution-placeholder",
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
start_request=$(jq -c \
  --arg workspace_id "$workspace_id" \
  --arg execution_id "$execution_id" \
  '.workspace_id = $workspace_id | .execution_id = $execution_id' <<<"$start_request")
scope_headers=(
  -H "Authorization: Bearer ${token}"
  -H "x-agent-sandbox-tenant-id: tenant-a"
  -H "x-agent-sandbox-principal-id: principal-a"
  -H "content-type: application/json"
)

curl -fsS "${scope_headers[@]}" -d "$start_request" http://127.0.0.1:18090/v1/executions | jq .

lookup_request='{
  "scope": {"tenant_id": "tenant-a", "principal_id": "principal-a"},
  "workspace_id": "kind-workspace-placeholder",
  "profile": "python-deny-egress",
  "execution_id": "kind-execution-placeholder"
}'
lookup_request=$(jq -c \
  --arg workspace_id "$workspace_id" \
  --arg execution_id "$execution_id" \
  '.workspace_id = $workspace_id | .execution_id = $execution_id' <<<"$lookup_request")
execution_succeeded=false
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
    sandbox_id=$(jq -r '.provider_sandbox_id' <<<"$outcome")
    execution_succeeded=true
    break
  fi
  if [[ "$state" =~ ^(failed|cancelled|timed_out|outcome_unknown)$ ]]; then
    jq . <<<"$outcome" >&2
    exit 1
  fi
  sleep 1
done
if [[ "$execution_succeeded" != "true" ]]; then
  echo "sandbox execution did not finish" >&2
  exit 1
fi

cancel_execution_id=kind-cancel-${run_id}
cancel_start_request=$(jq -c \
  --arg execution_id "$cancel_execution_id" \
  '.execution_id = $execution_id
    | .command.argv = ["python", "-c", "import time; time.sleep(60)"]
    | .command.artifact_paths = []
    | .limits.timeout_millis = 60000' <<<"$start_request")
curl -fsS "${scope_headers[@]}" -d "$cancel_start_request" http://127.0.0.1:18090/v1/executions >/dev/null
cancel_lookup_request=$(jq -c \
  --arg execution_id "$cancel_execution_id" \
  '.execution_id = $execution_id' <<<"$lookup_request")
curl -fsS "${scope_headers[@]}" -d "$cancel_lookup_request" http://127.0.0.1:18090/v1/executions:cancel >/dev/null

execution_cancelled=false
for _ in $(seq 1 30); do
  outcome=$(curl -fsS "${scope_headers[@]}" -d "$cancel_lookup_request" http://127.0.0.1:18090/v1/executions:lookup)
  state=$(jq -r '.state // "missing"' <<<"$outcome")
  if [[ "$state" == "cancelled" ]]; then
    execution_cancelled=true
    break
  fi
  if [[ "$state" =~ ^(succeeded|failed|timed_out|outcome_unknown)$ ]]; then
    jq . <<<"$outcome" >&2
    exit 1
  fi
  sleep 1
done
if [[ "$execution_cancelled" != "true" ]]; then
  echo "sandbox execution was not cancelled" >&2
  exit 1
fi

workspace_request=$(jq -n \
  --arg workspace_id "$workspace_id" \
  '{scope: {tenant_id: "tenant-a", principal_id: "principal-a"}, workspace_id: $workspace_id, profile: "python-deny-egress"}')
curl -fsS "${scope_headers[@]}" -d "$workspace_request" http://127.0.0.1:18090/v1/workspaces:delete >/dev/null
kubectl -n agent-sandboxes wait --for=delete "sandbox/${sandbox_id}" --timeout=60s

echo "Kubernetes sandbox execution, cancellation, artifact recovery, and cleanup succeeded"
