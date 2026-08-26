# agent-rt-sandbox

Framework-neutral contracts for an external sandbox execution plane used by `agent-rt` runtime tools.

This crate does not put Kubernetes credentials or sandbox lifecycle into `agent-rt`. A provider service implements `SandboxProvider`; the agent frontend calls that service through a `ToolExecutor` adapter. Kubernetes Agent Sandbox is the first provider, while hosted systems such as Modal can implement the same contract.
