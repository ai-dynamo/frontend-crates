// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use thiserror::Error;

use crate::{
    ResponseId, RuntimeAuthorization, RuntimeToolCall, RuntimeToolResult, ToolClaimResult,
    ToolExecutionFailure, ToolExecutionRequest, ToolExecutionResult, ToolExecutor,
    ToolFailureDisposition, ToolFailurePolicy, ToolIdempotencyKeyProvider, ToolJournal,
    ToolJournalOutcome, ToolJournalRecord, ToolJournalState,
};

#[derive(Debug, Error)]
pub enum ToolRunError<JournalError, ExecutorError>
where
    JournalError: std::error::Error + Send + Sync + 'static,
    ExecutorError: std::error::Error + Send + Sync + 'static,
{
    #[error("connector {0} is not authorized for this turn")]
    UnauthorizedConnector(String),
    #[error("tool journal failed: {0}")]
    Journal(JournalError),
    #[error("tool execution failed: {error}; journal finalization: {journal_error:?}")]
    Executor {
        error: ExecutorError,
        outcome_unknown: bool,
        journal_error: Option<JournalError>,
    },
    #[error("tool recovery lookup failed: {error}; journal finalization: {journal_error:?}")]
    RecoveryLookup {
        error: ExecutorError,
        journal_error: Option<JournalError>,
    },
    #[error("tool side-effect outcome is unknown; journal finalization: {journal_error:?}")]
    OutcomeUnknown { journal_error: Option<JournalError> },
    #[error("tool execution previously failed: {0:?}")]
    PersistedFailure(ToolExecutionFailure),
    #[error(
        "tool execution exceeded its {limit_millis}ms limit; journal finalization: {journal_error:?}"
    )]
    ExecutionTimedOut {
        limit_millis: u64,
        journal_error: Option<JournalError>,
    },
    #[error(
        "tool recovery lookup exceeded its {limit_millis}ms limit; journal finalization: {journal_error:?}"
    )]
    RecoveryTimedOut {
        limit_millis: u64,
        journal_error: Option<JournalError>,
    },
    #[error("tool journal record is internally inconsistent")]
    CorruptJournal,
    #[error(
        "tool output is {actual_bytes} bytes, exceeding limit {limit_bytes}; journal finalization: {journal_error:?}"
    )]
    OutputTooLarge {
        actual_bytes: u64,
        limit_bytes: u64,
        journal_error: Option<JournalError>,
    },
    #[error("tool completed but its result could not be journaled: {0}")]
    JournalAfterExecution(JournalError),
}

impl<JournalError, ExecutorError> ToolRunError<JournalError, ExecutorError>
where
    JournalError: std::error::Error + Send + Sync + 'static,
    ExecutorError: std::error::Error + Send + Sync + 'static,
{
    /// Whether retrying the public turn could duplicate an unresolved side effect.
    pub fn requires_unknown_outcome(&self) -> bool {
        match self {
            Self::UnauthorizedConnector(_) | Self::Journal(_) | Self::PersistedFailure(_) => false,
            Self::Executor {
                outcome_unknown,
                journal_error,
                ..
            } => *outcome_unknown || journal_error.is_some(),
            Self::RecoveryLookup { .. }
            | Self::ExecutionTimedOut { .. }
            | Self::RecoveryTimedOut { .. }
            | Self::OutcomeUnknown { .. }
            | Self::CorruptJournal
            | Self::JournalAfterExecution(_) => true,
            Self::OutputTooLarge { journal_error, .. } => journal_error.is_some(),
        }
    }
}

/// Protocol-independent execution of one already-routed runtime tool call.
pub struct ToolRunner<J, E, K, F> {
    journal: J,
    executor: E,
    idempotency_keys: K,
    failure_policy: F,
}

impl<J, E, K, F> ToolRunner<J, E, K, F> {
    pub fn new(journal: J, executor: E, idempotency_keys: K, failure_policy: F) -> Self {
        Self {
            journal,
            executor,
            idempotency_keys,
            failure_policy,
        }
    }

    pub fn journal(&self) -> &J {
        &self.journal
    }
}

impl<J, E, K, F> ToolRunner<J, E, K, F>
where
    J: ToolJournal,
    E: ToolExecutor,
    K: ToolIdempotencyKeyProvider,
    F: ToolFailurePolicy<E::Error>,
{
    pub async fn run(
        &self,
        response_id: &ResponseId,
        call: RuntimeToolCall,
        authorization: &RuntimeAuthorization,
        attempt: u32,
    ) -> Result<RuntimeToolResult, ToolRunError<J::Error, E::Error>> {
        if !authorization.permits_connector(&call.connector) {
            return Err(ToolRunError::UnauthorizedConnector(call.connector));
        }
        let request = ToolExecutionRequest {
            response_id: response_id.clone(),
            call_id: call.call_id.clone(),
            connector: call.connector.clone(),
            operation: call.operation.clone(),
            profile: call.profile.clone(),
            arguments: call.arguments.clone(),
            scope: authorization.scope.clone(),
            idempotency_key: self
                .idempotency_keys
                .idempotency_key(response_id, &call, attempt),
            attempt,
        };
        let key = request.journal_key();

        match self
            .journal
            .claim(request.clone())
            .await
            .map_err(ToolRunError::Journal)?
        {
            ToolClaimResult::Existing(record) => {
                return self.recover_existing(*record, call, authorization).await;
            }
            ToolClaimResult::Acquired(_) => {}
        }

        let execution = tokio::time::timeout(
            Duration::from_millis(authorization.limits.max_external_work_millis),
            self.executor.execute(request),
        )
        .await;
        let execution = match execution {
            Ok(execution) => execution,
            Err(_) => {
                let journal_error = self
                    .journal
                    .finish(key, ToolJournalOutcome::OutcomeUnknown)
                    .await
                    .err();
                return Err(ToolRunError::ExecutionTimedOut {
                    limit_millis: authorization.limits.max_external_work_millis,
                    journal_error,
                });
            }
        };

        match execution {
            Ok(result) => {
                self.validate_output(&key, &result, authorization).await?;
                let record = self
                    .journal
                    .finish(key, ToolJournalOutcome::Completed(result))
                    .await
                    .map_err(ToolRunError::JournalAfterExecution)?;
                let result = record.result.ok_or(ToolRunError::CorruptJournal)?;
                Ok(RuntimeToolResult { call, result })
            }
            Err(error) => {
                let (outcome, outcome_unknown) = match self.failure_policy.classify(&error) {
                    ToolFailureDisposition::Failed(failure) => {
                        (ToolJournalOutcome::Failed(failure), false)
                    }
                    ToolFailureDisposition::OutcomeUnknown => {
                        (ToolJournalOutcome::OutcomeUnknown, true)
                    }
                };
                let journal_error = self.journal.finish(key, outcome).await.err();
                Err(ToolRunError::Executor {
                    error,
                    outcome_unknown,
                    journal_error,
                })
            }
        }
    }

    async fn recover_existing(
        &self,
        record: ToolJournalRecord,
        call: RuntimeToolCall,
        authorization: &RuntimeAuthorization,
    ) -> Result<RuntimeToolResult, ToolRunError<J::Error, E::Error>> {
        match record.state {
            ToolJournalState::Completed => {
                let result = record.result.ok_or(ToolRunError::CorruptJournal)?;
                Ok(RuntimeToolResult { call, result })
            }
            ToolJournalState::Failed => Err(ToolRunError::PersistedFailure(
                record.failure.ok_or(ToolRunError::CorruptJournal)?,
            )),
            ToolJournalState::OutcomeUnknown => Err(ToolRunError::OutcomeUnknown {
                journal_error: None,
            }),
            ToolJournalState::Started => {
                let key = record.request.journal_key();
                let lookup = tokio::time::timeout(
                    Duration::from_millis(authorization.limits.max_external_work_millis),
                    self.executor.lookup(&record.request),
                )
                .await;
                let lookup = match lookup {
                    Ok(lookup) => lookup,
                    Err(_) => {
                        let journal_error = self
                            .journal
                            .finish(key, ToolJournalOutcome::OutcomeUnknown)
                            .await
                            .err();
                        return Err(ToolRunError::RecoveryTimedOut {
                            limit_millis: authorization.limits.max_external_work_millis,
                            journal_error,
                        });
                    }
                };
                match lookup {
                    Ok(Some(result)) => {
                        self.validate_output(&key, &result, authorization).await?;
                        let record = self
                            .journal
                            .finish(key, ToolJournalOutcome::Completed(result))
                            .await
                            .map_err(ToolRunError::JournalAfterExecution)?;
                        let result = record.result.ok_or(ToolRunError::CorruptJournal)?;
                        Ok(RuntimeToolResult { call, result })
                    }
                    Ok(None) => {
                        let journal_error = self
                            .journal
                            .finish(key, ToolJournalOutcome::OutcomeUnknown)
                            .await
                            .err();
                        Err(ToolRunError::OutcomeUnknown { journal_error })
                    }
                    Err(error) => {
                        let journal_error = self
                            .journal
                            .finish(key, ToolJournalOutcome::OutcomeUnknown)
                            .await
                            .err();
                        Err(ToolRunError::RecoveryLookup {
                            error,
                            journal_error,
                        })
                    }
                }
            }
        }
    }

    async fn validate_output(
        &self,
        key: &crate::ToolJournalKey,
        result: &ToolExecutionResult,
        authorization: &RuntimeAuthorization,
    ) -> Result<(), ToolRunError<J::Error, E::Error>> {
        let actual_bytes = u64::try_from(result.output.to_string().len()).unwrap_or(u64::MAX);
        let limit_bytes = authorization.limits.max_tool_output_bytes;
        if actual_bytes <= limit_bytes {
            return Ok(());
        }

        let journal_error = self
            .journal
            .finish(
                key.clone(),
                ToolJournalOutcome::Failed(ToolExecutionFailure {
                    code: "output_too_large".to_owned(),
                    message: format!("tool output was {actual_bytes} bytes"),
                    retryable: false,
                }),
            )
            .await
            .err();
        Err(ToolRunError::OutputTooLarge {
            actual_bytes,
            limit_bytes,
            journal_error,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use serde_json::json;
    use thiserror::Error;

    use crate::{
        AuthorizationScope, Blake3ToolIdempotencyKeys, BoxFuture, ConservativeToolFailurePolicy,
        InMemoryToolJournal, ResponseId, RuntimeAuthorization, RuntimeLimits, RuntimeToolCall,
        ToolExecutionRequest, ToolExecutionResult, ToolExecutor, ToolIdempotencyKeyProvider,
        ToolJournal, ToolJournalState,
    };

    use super::{ToolRunError, ToolRunner};

    #[derive(Debug, Error)]
    #[error("executor failed")]
    struct MockExecutorError;

    struct MockExecutor {
        executes: Arc<AtomicUsize>,
        fail: bool,
        output: serde_json::Value,
    }

    impl ToolExecutor for MockExecutor {
        type Error = MockExecutorError;

        fn execute(
            &self,
            _request: ToolExecutionRequest,
        ) -> BoxFuture<'_, Result<ToolExecutionResult, Self::Error>> {
            self.executes.fetch_add(1, Ordering::SeqCst);
            Box::pin(async move {
                if self.fail {
                    Err(MockExecutorError)
                } else {
                    Ok(ToolExecutionResult {
                        output: self.output.clone(),
                        is_error: false,
                    })
                }
            })
        }

        fn lookup(
            &self,
            _request: &ToolExecutionRequest,
        ) -> BoxFuture<'_, Result<Option<ToolExecutionResult>, Self::Error>> {
            Box::pin(async { Ok(None) })
        }
    }

    struct PendingExecutor;

    impl ToolExecutor for PendingExecutor {
        type Error = MockExecutorError;

        fn execute(
            &self,
            _request: ToolExecutionRequest,
        ) -> BoxFuture<'_, Result<ToolExecutionResult, Self::Error>> {
            Box::pin(std::future::pending())
        }

        fn lookup(
            &self,
            _request: &ToolExecutionRequest,
        ) -> BoxFuture<'_, Result<Option<ToolExecutionResult>, Self::Error>> {
            Box::pin(std::future::pending())
        }
    }

    fn authorization(max_tool_output_bytes: u64) -> RuntimeAuthorization {
        RuntimeAuthorization {
            scope: AuthorizationScope {
                tenant_id: "tenant".to_owned(),
                principal_id: "principal".to_owned(),
            },
            permitted_connectors: BTreeSet::from(["search".to_owned()]),
            limits: RuntimeLimits {
                max_tool_output_bytes,
                ..RuntimeLimits::default()
            },
        }
    }

    fn call(connector: &str) -> RuntimeToolCall {
        RuntimeToolCall {
            call_id: "call-1".to_owned(),
            connector: connector.to_owned(),
            operation: "query".to_owned(),
            profile: "default".to_owned(),
            arguments: json!({"query": "rust"}),
        }
    }

    fn runner(
        executes: Arc<AtomicUsize>,
        fail: bool,
        output: serde_json::Value,
    ) -> ToolRunner<
        InMemoryToolJournal,
        MockExecutor,
        Blake3ToolIdempotencyKeys,
        ConservativeToolFailurePolicy,
    > {
        ToolRunner::new(
            InMemoryToolJournal::default(),
            MockExecutor {
                executes,
                fail,
                output,
            },
            Blake3ToolIdempotencyKeys,
            ConservativeToolFailurePolicy,
        )
    }

    #[test]
    fn execution_profile_changes_the_idempotency_key() {
        let mut first = call("sandbox");
        first.profile = "python-deny-egress".to_owned();
        let mut second = first.clone();
        second.profile = "python-public-egress".to_owned();

        let keys = Blake3ToolIdempotencyKeys;
        assert_ne!(
            keys.idempotency_key(&ResponseId::from("resp-1"), &first, 0),
            keys.idempotency_key(&ResponseId::from("resp-1"), &second, 0)
        );
    }

    #[tokio::test]
    async fn completed_call_is_replayed_without_reexecution() {
        let executes = Arc::new(AtomicUsize::new(0));
        let runner = runner(executes.clone(), false, json!({"answer": 42}));
        let response_id = ResponseId::from("resp-1");
        let authorization = authorization(1024);

        let first = runner
            .run(&response_id, call("search"), &authorization, 0)
            .await
            .unwrap();
        let second = runner
            .run(&response_id, call("search"), &authorization, 0)
            .await
            .unwrap();

        assert_eq!(first, second);
        assert_eq!(executes.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn unauthorized_connector_is_never_executed() {
        let executes = Arc::new(AtomicUsize::new(0));
        let runner = runner(executes.clone(), false, json!(null));

        let error = runner
            .run(
                &ResponseId::from("resp-1"),
                call("filesystem"),
                &authorization(1024),
                0,
            )
            .await
            .unwrap_err();
        assert!(matches!(
            &error,
            ToolRunError::UnauthorizedConnector(connector) if connector == "filesystem"
        ));
        assert!(!error.requires_unknown_outcome());
        assert_eq!(executes.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn uncertain_executor_failure_is_not_retried() {
        let executes = Arc::new(AtomicUsize::new(0));
        let runner = runner(executes.clone(), true, json!(null));
        let response_id = ResponseId::from("resp-1");
        let authorization = authorization(1024);
        let runtime_call = call("search");

        let error = runner
            .run(&response_id, runtime_call.clone(), &authorization, 0)
            .await
            .unwrap_err();
        assert!(matches!(&error, ToolRunError::Executor { .. }));
        assert!(error.requires_unknown_outcome());
        assert!(matches!(
            runner
                .run(&response_id, runtime_call.clone(), &authorization, 0)
                .await,
            Err(ToolRunError::OutcomeUnknown { .. })
        ));
        assert_eq!(executes.load(Ordering::SeqCst), 1);

        let key = crate::ToolJournalKey {
            scope: authorization.scope,
            idempotency_key: Blake3ToolIdempotencyKeys.idempotency_key(
                &response_id,
                &runtime_call,
                0,
            ),
        };
        assert_eq!(
            runner.journal().load(&key).await.unwrap().unwrap().state,
            ToolJournalState::OutcomeUnknown
        );
    }

    #[tokio::test]
    async fn execution_timeout_is_persisted_as_outcome_unknown() {
        let runner = ToolRunner::new(
            InMemoryToolJournal::default(),
            PendingExecutor,
            Blake3ToolIdempotencyKeys,
            ConservativeToolFailurePolicy,
        );
        let response_id = ResponseId::from("resp-1");
        let runtime_call = call("search");
        let mut authorization = authorization(1024);
        authorization.limits.max_external_work_millis = 1;

        let error = runner
            .run(&response_id, runtime_call.clone(), &authorization, 0)
            .await
            .unwrap_err();
        assert!(matches!(error, ToolRunError::ExecutionTimedOut { .. }));
        let key = crate::ToolJournalKey {
            scope: authorization.scope,
            idempotency_key: Blake3ToolIdempotencyKeys.idempotency_key(
                &response_id,
                &runtime_call,
                0,
            ),
        };
        assert_eq!(
            runner.journal().load(&key).await.unwrap().unwrap().state,
            ToolJournalState::OutcomeUnknown
        );
    }

    #[tokio::test]
    async fn oversized_output_is_failed_durably() {
        let executes = Arc::new(AtomicUsize::new(0));
        let runner = runner(executes.clone(), false, json!("abcd"));
        let response_id = ResponseId::from("resp-1");
        let authorization = authorization(5);
        let runtime_call = call("search");

        let error = runner
            .run(&response_id, runtime_call.clone(), &authorization, 0)
            .await
            .unwrap_err();
        assert!(matches!(
            &error,
            ToolRunError::OutputTooLarge {
                actual_bytes: 6,
                limit_bytes: 5,
                ..
            }
        ));
        assert!(!error.requires_unknown_outcome());
        assert_eq!(executes.load(Ordering::SeqCst), 1);

        let key = crate::ToolJournalKey {
            scope: authorization.scope,
            idempotency_key: Blake3ToolIdempotencyKeys.idempotency_key(
                &response_id,
                &runtime_call,
                0,
            ),
        };
        let record = runner.journal().load(&key).await.unwrap().unwrap();
        assert_eq!(record.state, ToolJournalState::Failed);
        assert_eq!(record.failure.unwrap().code, "output_too_large");
    }
}
