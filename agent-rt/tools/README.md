<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-agent-tools

`dynamo-agent-tools` contains deployment-facing implementations of `dynamo-agent-rt`'s external tool contracts. Tool credentials and provider networking stay here rather than entering Dynamo or the runtime state machine.

The first implementation is a bounded, read-only web-search executor. Model-generated arguments select only the query and result count. Deployment configuration owns the provider, credential, endpoint, locale, timeout, concurrency, and response limits.
