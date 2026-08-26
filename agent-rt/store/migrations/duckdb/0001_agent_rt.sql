CREATE TABLE IF NOT EXISTS agent_rt_checkpoints (
    protocol VARCHAR NOT NULL,
    response_id VARCHAR NOT NULL,
    parent_response_id VARCHAR,
    tenant_id VARCHAR NOT NULL,
    principal_id VARCHAR NOT NULL,
    idempotency_key VARCHAR NOT NULL,
    request_fingerprint BLOB NOT NULL,
    state VARCHAR NOT NULL,
    version BIGINT NOT NULL,
    request_json VARCHAR NOT NULL,
    response_json VARCHAR,
    lease_turn_id VARCHAR,
    lease_deadline BIGINT,
    PRIMARY KEY (protocol, response_id),
    UNIQUE (protocol, tenant_id, principal_id, idempotency_key)
);

CREATE TABLE IF NOT EXISTS agent_rt_checkpoint_output_items (
    protocol VARCHAR NOT NULL,
    response_id VARCHAR NOT NULL,
    sequence BIGINT NOT NULL,
    item_json VARCHAR NOT NULL,
    PRIMARY KEY (protocol, response_id, sequence)
);

CREATE TABLE IF NOT EXISTS agent_rt_tool_journal (
    tenant_id VARCHAR NOT NULL,
    principal_id VARCHAR NOT NULL,
    idempotency_key VARCHAR NOT NULL,
    request_json VARCHAR NOT NULL,
    state VARCHAR NOT NULL,
    result_json VARCHAR,
    failure_json VARCHAR,
    PRIMARY KEY (tenant_id, principal_id, idempotency_key)
);
