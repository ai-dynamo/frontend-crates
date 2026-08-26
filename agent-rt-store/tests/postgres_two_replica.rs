// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;
use std::time::Duration;

use dynamo_agent_rt::{
    AuthorizationScope, BeginTurn, BeginTurnResult, CheckpointStore, CommitTurn, IdempotencyKey,
    LeaseDeadline, OpenAiResponses, RequestFingerprint, ResponseId, RuntimeAuthorization,
    RuntimeLimits, ToolClaimResult, ToolExecutionRequest, ToolExecutionResult, ToolJournal,
    ToolJournalOutcome, TurnId, TurnState,
};
use dynamo_agent_rt_store::{PostgresStore, PostgresStoreError, StoreInvariantError};
use postgresql_embedded::PostgreSQL;
use serde_json::json;

fn authorization(tenant: &str) -> RuntimeAuthorization {
    RuntimeAuthorization {
        scope: AuthorizationScope {
            tenant_id: tenant.to_owned(),
            principal_id: "principal-a".to_owned(),
        },
        permitted_connectors: BTreeSet::new(),
        limits: RuntimeLimits::default(),
    }
}

fn begin(
    response_id: &str,
    turn_id: &str,
    idempotency_key: &str,
    deadline: u64,
) -> BeginTurn<OpenAiResponses> {
    BeginTurn {
        response_id: ResponseId::from(response_id),
        turn_id: TurnId::from(turn_id),
        parent_response_id: None,
        authorization: authorization("tenant-a"),
        idempotency_key: IdempotencyKey::from(idempotency_key),
        request_fingerprint: RequestFingerprint::new([42; 32]),
        request: Default::default(),
        lease_deadline: LeaseDeadline(deadline),
    }
}

fn tool_request() -> ToolExecutionRequest {
    ToolExecutionRequest {
        response_id: ResponseId::from("response-a"),
        call_id: "call-a".to_owned(),
        connector: "search".to_owned(),
        operation: "query".to_owned(),
        profile: "default".to_owned(),
        arguments: json!({"query": "rust"}),
        scope: authorization("tenant-a").scope,
        idempotency_key: IdempotencyKey::from("tool-concurrent"),
        attempt: 0,
    }
}

async fn database_now(store: &PostgresStore<OpenAiResponses>) -> u64 {
    let client = store.pool().get().await.unwrap();
    let row = client
        .query_one(
            "SELECT FLOOR(EXTRACT(EPOCH FROM clock_timestamp()) * 1000)::BIGINT",
            &[],
        )
        .await
        .unwrap();
    u64::try_from(row.get::<_, i64>(0)).unwrap()
}

#[tokio::test]
async fn two_replicas_serialize_claims_and_fence_expired_owners() {
    let mut postgres = PostgreSQL::default();
    postgres.setup().await.unwrap();
    postgres.start().await.unwrap();
    postgres.create_database("agent_rt_test").await.unwrap();
    let database_url = postgres.settings().url("agent_rt_test").to_string();

    let replica_a = PostgresStore::<OpenAiResponses>::connect_no_tls(&database_url, 4)
        .await
        .unwrap();
    let replica_b = PostgresStore::<OpenAiResponses>::connect_no_tls(&database_url, 4)
        .await
        .unwrap();
    let now = database_now(&replica_a).await;

    let left = begin("response-a", "turn-a", "concurrent", now + 30_000);
    let right = BeginTurn {
        response_id: ResponseId::from("response-b"),
        turn_id: TurnId::from("turn-b"),
        ..left.clone()
    };
    let (left, right) = tokio::join!(replica_a.begin_turn(left), replica_b.begin_turn(right));
    let outcomes = [left.unwrap(), right.unwrap()];
    assert_eq!(
        outcomes
            .iter()
            .filter(|result| matches!(result, BeginTurnResult::Acquired(_)))
            .count(),
        1
    );
    assert_eq!(
        outcomes
            .iter()
            .filter(|result| matches!(result, BeginTurnResult::Existing(_)))
            .count(),
        1
    );

    let now = database_now(&replica_a).await;
    let original = begin("takeover", "old-owner", "takeover", now + 150);
    let BeginTurnResult::Acquired(old_lease) =
        replica_a.begin_turn(original.clone()).await.unwrap()
    else {
        panic!("expected original lease");
    };
    tokio::time::sleep(Duration::from_millis(250)).await;
    let now = database_now(&replica_b).await;
    let replacement = BeginTurn {
        response_id: ResponseId::from("ignored"),
        turn_id: TurnId::from("new-owner"),
        lease_deadline: LeaseDeadline(now + 30_000),
        ..original
    };
    let BeginTurnResult::Acquired(new_lease) = replica_b.begin_turn(replacement).await.unwrap()
    else {
        panic!("expected replacement lease");
    };
    assert_eq!(new_lease.response_id.as_str(), "takeover");
    assert_eq!(new_lease.version.0, 1);

    let stale = replica_a
        .commit_turn(CommitTurn {
            lease: old_lease,
            next_state: TurnState::Failed,
            append_output_items: Vec::new(),
            response: None,
        })
        .await
        .unwrap_err();
    assert!(matches!(
        stale,
        PostgresStoreError::Invariant(
            StoreInvariantError::LeaseMismatch | StoreInvariantError::VersionConflict
        )
    ));
    replica_b
        .commit_turn(CommitTurn {
            lease: new_lease,
            next_state: TurnState::Completed,
            append_output_items: Vec::new(),
            response: None,
        })
        .await
        .unwrap();

    let request = tool_request();
    let key = request.journal_key();
    let (left, right) = tokio::join!(
        replica_a.claim(request.clone()),
        replica_b.claim(request.clone())
    );
    let claims = [left.unwrap(), right.unwrap()];
    assert_eq!(
        claims
            .iter()
            .filter(|claim| matches!(claim, ToolClaimResult::Acquired(_)))
            .count(),
        1
    );
    assert_eq!(
        claims
            .iter()
            .filter(|claim| matches!(claim, ToolClaimResult::Existing(_)))
            .count(),
        1
    );
    replica_a
        .finish(
            key,
            ToolJournalOutcome::Completed(ToolExecutionResult {
                output: json!({"answer": 42}),
            }),
        )
        .await
        .unwrap();
    let ToolClaimResult::Existing(replayed) = replica_b.claim(request).await.unwrap() else {
        panic!("completed tool call was reclaimed");
    };
    assert_eq!(replayed.result.unwrap().output["answer"], 42);

    drop(replica_a);
    drop(replica_b);
    postgres.stop().await.unwrap();
}
