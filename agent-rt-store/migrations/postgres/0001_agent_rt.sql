CREATE TABLE IF NOT EXISTS agent_rt_checkpoints (
    protocol TEXT NOT NULL,
    response_id TEXT NOT NULL,
    parent_response_id TEXT,
    tenant_id TEXT NOT NULL,
    principal_id TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    request_fingerprint BYTEA NOT NULL,
    state TEXT NOT NULL CHECK (state IN (
        'in_flight', 'awaiting_client_tool_output', 'tool_started',
        'outcome_unknown', 'completed', 'failed'
    )),
    version BIGINT NOT NULL CHECK (version >= 0),
    request_json TEXT NOT NULL,
    response_json TEXT,
    lease_turn_id TEXT,
    lease_deadline BIGINT,
    PRIMARY KEY (protocol, response_id),
    UNIQUE (protocol, tenant_id, principal_id, idempotency_key),
    CHECK ((lease_turn_id IS NULL) = (lease_deadline IS NULL))
);

CREATE TABLE IF NOT EXISTS agent_rt_checkpoint_output_items (
    protocol TEXT NOT NULL,
    response_id TEXT NOT NULL,
    sequence BIGINT NOT NULL CHECK (sequence >= 0),
    item_json TEXT NOT NULL,
    PRIMARY KEY (protocol, response_id, sequence),
    FOREIGN KEY (protocol, response_id)
        REFERENCES agent_rt_checkpoints (protocol, response_id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS agent_rt_tool_journal (
    tenant_id TEXT NOT NULL,
    principal_id TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    request_json TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN (
        'started', 'completed', 'failed', 'outcome_unknown'
    )),
    result_json TEXT,
    failure_json TEXT,
    PRIMARY KEY (tenant_id, principal_id, idempotency_key)
);
