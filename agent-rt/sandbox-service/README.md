# agent-rt-sandbox-service

Authenticated, multi-replica execution service for the Kubernetes Agent Sandbox provider.

The service owns Kubernetes credentials, `SandboxClaim` lifecycle, sandboxd process/file access, and fenced PostgreSQL execution state. Inference frontends call it through the provider-neutral `SandboxProvider` contract and never receive Kubernetes credentials.
