<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-agent-rt

`dynamo-agent-rt` provides the protocol-neutral dependency seams and Responses-specific state model needed to add durable agent turns to an inference frontend.

The crate owns continuation state and orchestration contracts. It does not own HTTP ingress, authentication, model rendering, tokenization, routing, engine adapters, or external tool credentials.

Runtime instances retain native protocol DTOs through `AgentProtocol`. The initial families are OpenAI Responses and Anthropic Messages (including Claude Code traffic); there is no shared lossy agent-request IR.

The initial public seams are:

- `CheckpointStore` for atomic turn claims and fenced state transitions.
- `InferenceInvoker` for sending a fully materialized native Responses request to an inference system.
- `RequestMaterializer` and `ContinuationPolicy` for protocol hydration and replaceable inheritance behavior.
- `AnthropicRequestMaterializer` for Claude Code's native full-history Messages requests.
- `ToolExecutor` for already-authorized runtime-owned tool calls.
- `RuntimeSelector`, `IdGenerator`, and `Clock` for replaceable frontend policy and deterministic tests.
- `RequestFingerprinter` for stable idempotency without requiring protocol DTO equality.

Implementations can embed the runtime in a local frontend or invoke a private Kubernetes inference route without changing the state model.
