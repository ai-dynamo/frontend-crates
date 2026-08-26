// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::time::Duration;

use agent_rt_sandbox::{
    Artifact, ArtifactRef, ClaimExecution, ExecutionClaimResult, ExecutionId, ExecutionState,
    ExecutionStore, PostgresExecutionStore, PostgresExecutionStoreError, SANDBOX_API_VERSION,
    SandboxCommand, SandboxLimits, SandboxProfile, StartExecution, WorkspaceId,
};
use dynamo_agent_rt::AuthorizationScope;
use postgresql_embedded::PostgreSQL;

fn request(execution_id: &str) -> StartExecution {
    StartExecution {
        api_version: SANDBOX_API_VERSION.to_owned(),
        scope: AuthorizationScope {
            tenant_id: "tenant-a".to_owned(),
            principal_id: "principal-a".to_owned(),
        },
        workspace_id: WorkspaceId("workspace-a".to_owned()),
        execution_id: ExecutionId(execution_id.to_owned()),
        profile: SandboxProfile("python-deny-egress".to_owned()),
        command: SandboxCommand {
            argv: vec!["python".to_owned(), "-c".to_owned(), "print(42)".to_owned()],
            cwd: Some("/workspace".to_owned()),
            env: BTreeMap::new(),
            stdin: Vec::new(),
            artifact_paths: vec!["result.txt".to_owned()],
        },
        limits: SandboxLimits {
            timeout_millis: 10_000,
            max_output_bytes: 1_024,
            max_artifact_bytes: 4_096,
        },
    }
}

fn claim(request: StartExecution, owner_id: &str, lease_millis: u64) -> ClaimExecution {
    ClaimExecution {
        request,
        provider_sandbox_id: "sandbox-a".to_owned(),
        owner_id: owner_id.to_owned(),
        now_unix_millis: 0,
        lease_millis,
    }
}

#[tokio::test]
async fn two_replicas_fence_owners_and_share_terminal_artifacts() {
    let mut postgres = PostgreSQL::default();
    postgres.setup().await.unwrap();
    postgres.start().await.unwrap();
    postgres
        .create_database("agent_sandbox_test")
        .await
        .unwrap();
    let database_url = postgres.settings().url("agent_sandbox_test").to_string();

    let replica_a = PostgresExecutionStore::connect_no_tls(&database_url, 4)
        .await
        .unwrap();
    let replica_b = PostgresExecutionStore::connect_no_tls(&database_url, 4)
        .await
        .unwrap();
    let concurrent_request = request("execution-a");
    let (left, right) = tokio::join!(
        replica_a.claim(claim(concurrent_request.clone(), "replica-a", 30_000)),
        replica_b.claim(claim(concurrent_request, "replica-b", 30_000)),
    );
    let claims = [left.unwrap(), right.unwrap()];
    assert_eq!(
        claims
            .iter()
            .filter(|claim| matches!(claim, ExecutionClaimResult::Acquired(_)))
            .count(),
        1
    );
    assert_eq!(
        claims
            .iter()
            .filter(|claim| matches!(claim, ExecutionClaimResult::Existing(_)))
            .count(),
        1
    );

    let takeover = request("takeover");
    let ExecutionClaimResult::Acquired(old_lease) = replica_a
        .claim(claim(takeover.clone(), "replica-a", 150))
        .await
        .unwrap()
    else {
        panic!("expected original lease")
    };
    tokio::time::sleep(Duration::from_millis(250)).await;
    let ExecutionClaimResult::Acquired(new_lease) = replica_b
        .claim(claim(takeover.clone(), "replica-b", 30_000))
        .await
        .unwrap()
    else {
        panic!("expected replacement lease")
    };
    assert_eq!(new_lease.fence, 2);
    assert!(matches!(
        replica_a.mark_running(&old_lease, 0).await,
        Err(PostgresExecutionStoreError::StaleLease)
    ));

    let running = replica_b.mark_running(&new_lease, 0).await.unwrap();
    let artifact = Artifact {
        metadata: ArtifactRef {
            artifact_id: "artifact-a".to_owned(),
            name: "result.txt".to_owned(),
            size_bytes: 4,
            media_type: Some("text/plain".to_owned()),
        },
        bytes: b"done".to_vec(),
    };
    let mut record = running.record;
    record.state = ExecutionState::Succeeded;
    record.exit_code = Some(0);
    record.stdout = b"42\n".to_vec();
    record.artifacts = vec![artifact.metadata.clone()];
    replica_b
        .finish(&new_lease, 0, record, vec![artifact.clone()])
        .await
        .unwrap();

    let loaded = replica_a
        .load(&new_lease.execution, 0)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(loaded.record.state, ExecutionState::Succeeded);
    assert_eq!(
        replica_a
            .read_artifact(&new_lease.execution, "artifact-a")
            .await
            .unwrap(),
        Some(artifact)
    );

    drop(replica_a);
    drop(replica_b);
    postgres.stop().await.unwrap();
}
