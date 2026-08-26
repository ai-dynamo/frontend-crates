# dynamo-agent-rt-mcp

Deployment-configured outbound MCP tool execution for `dynamo-agent-rt`.

This crate is deliberately a client adapter, not an MCP gateway or control plane. A deployment supplies one trusted Streamable HTTP endpoint, credentials, and a fixed tool descriptor allowlist. The executor verifies the server's advertised tool schemas before it sends a call. Model-generated input can select only a route already admitted by the host.

`ToolRunner` in `dynamo-agent-rt` remains responsible for authorization, idempotency keys, durable journaling, timeouts around dispatch, and recovery. This crate owns only MCP transport and result normalization. The initial contract admits read-only tools, so recovery can safely re-execute an interrupted request.

Not supported in the initial contract:

- client-supplied MCP endpoints or credentials
- inbound `/mcp` APIs
- dynamic tool discovery or schema injection
- stdio transports
- persisted MCP session identifiers
- MCP tasks, elicitation, binary content, or embedded resources
