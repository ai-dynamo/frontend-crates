-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: Apache-2.0

CREATE TABLE IF NOT EXISTS agent_sandbox_executions (
    tenant_id TEXT NOT NULL,
    principal_id TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    profile TEXT NOT NULL,
    execution_id TEXT NOT NULL,
    request_fingerprint TEXT NOT NULL,
    provider_sandbox_id TEXT NOT NULL,
    request_json TEXT NOT NULL,
    record_json TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN (
        'pending', 'running', 'succeeded', 'failed', 'cancelled', 'timed_out', 'outcome_unknown'
    )),
    lease_owner_id TEXT,
    lease_fence BIGINT,
    lease_deadline_unix_millis BIGINT,
    cancel_requested BOOLEAN NOT NULL DEFAULT FALSE,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (tenant_id, principal_id, workspace_id, profile, execution_id),
    CHECK (
        (lease_owner_id IS NULL AND lease_fence IS NULL AND lease_deadline_unix_millis IS NULL)
        OR
        (lease_owner_id IS NOT NULL AND lease_fence IS NOT NULL AND lease_deadline_unix_millis IS NOT NULL)
    )
);

CREATE INDEX IF NOT EXISTS agent_sandbox_executions_lease_deadline_idx
    ON agent_sandbox_executions (lease_deadline_unix_millis)
    WHERE state IN ('pending', 'running');

CREATE TABLE IF NOT EXISTS agent_sandbox_artifacts (
    tenant_id TEXT NOT NULL,
    principal_id TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    profile TEXT NOT NULL,
    execution_id TEXT NOT NULL,
    artifact_id TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    bytes BYTEA NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (
        tenant_id, principal_id, workspace_id, profile, execution_id, artifact_id
    ),
    FOREIGN KEY (tenant_id, principal_id, workspace_id, profile, execution_id)
        REFERENCES agent_sandbox_executions (
            tenant_id, principal_id, workspace_id, profile, execution_id
        ) ON DELETE CASCADE
);
