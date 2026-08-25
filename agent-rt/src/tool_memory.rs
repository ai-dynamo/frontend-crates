// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Mutex;

use thiserror::Error;

use crate::{
    BoxFuture, ToolClaimResult, ToolExecutionRequest, ToolJournal, ToolJournalKey,
    ToolJournalOutcome, ToolJournalRecord, ToolJournalState,
};

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum InMemoryToolJournalError {
    #[error("tool journal mutex is poisoned")]
    Poisoned,
    #[error("tool idempotency key was reused with a different request")]
    IdempotencyConflict,
    #[error("tool journal record was not found")]
    NotFound,
    #[error("tool journal record is already in terminal state {0:?}")]
    AlreadyFinished(ToolJournalState),
}

/// Single-process tool journal for tests and local proofs of concept.
#[derive(Debug, Default)]
pub struct InMemoryToolJournal {
    records: Mutex<HashMap<ToolJournalKey, ToolJournalRecord>>,
}

impl ToolJournal for InMemoryToolJournal {
    type Error = InMemoryToolJournalError;

    fn claim(
        &self,
        request: ToolExecutionRequest,
    ) -> BoxFuture<'_, Result<ToolClaimResult, Self::Error>> {
        Box::pin(async move {
            let key = request.journal_key();
            let mut records = self
                .records
                .lock()
                .map_err(|_| InMemoryToolJournalError::Poisoned)?;
            if let Some(existing) = records.get(&key) {
                if existing.request != request {
                    return Err(InMemoryToolJournalError::IdempotencyConflict);
                }
                return Ok(ToolClaimResult::Existing(Box::new(existing.clone())));
            }

            let record = ToolJournalRecord {
                request,
                state: ToolJournalState::Started,
                result: None,
                failure: None,
            };
            records.insert(key, record.clone());
            Ok(ToolClaimResult::Acquired(Box::new(record)))
        })
    }

    fn load(
        &self,
        key: &ToolJournalKey,
    ) -> BoxFuture<'_, Result<Option<ToolJournalRecord>, Self::Error>> {
        let key = key.clone();
        Box::pin(async move {
            self.records
                .lock()
                .map_err(|_| InMemoryToolJournalError::Poisoned)
                .map(|records| records.get(&key).cloned())
        })
    }

    fn finish(
        &self,
        key: ToolJournalKey,
        outcome: ToolJournalOutcome,
    ) -> BoxFuture<'_, Result<ToolJournalRecord, Self::Error>> {
        Box::pin(async move {
            let mut records = self
                .records
                .lock()
                .map_err(|_| InMemoryToolJournalError::Poisoned)?;
            let record = records
                .get_mut(&key)
                .ok_or(InMemoryToolJournalError::NotFound)?;
            if record.state != ToolJournalState::Started {
                return Err(InMemoryToolJournalError::AlreadyFinished(
                    record.state.clone(),
                ));
            }

            match outcome {
                ToolJournalOutcome::Completed(result) => {
                    record.state = ToolJournalState::Completed;
                    record.result = Some(result);
                }
                ToolJournalOutcome::Failed(failure) => {
                    record.state = ToolJournalState::Failed;
                    record.failure = Some(failure);
                }
                ToolJournalOutcome::OutcomeUnknown => {
                    record.state = ToolJournalState::OutcomeUnknown;
                }
            }
            Ok(record.clone())
        })
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::{
        AuthorizationScope, IdempotencyKey, ResponseId, ToolClaimResult, ToolExecutionFailure,
        ToolExecutionRequest, ToolExecutionResult, ToolJournal, ToolJournalOutcome,
        ToolJournalState,
    };

    use super::{InMemoryToolJournal, InMemoryToolJournalError};

    fn request(operation: &str) -> ToolExecutionRequest {
        ToolExecutionRequest {
            response_id: ResponseId::from("resp-1"),
            call_id: "call-1".to_owned(),
            connector: "search".to_owned(),
            operation: operation.to_owned(),
            arguments: json!({"query": "rust"}),
            scope: AuthorizationScope {
                tenant_id: "tenant".to_owned(),
                principal_id: "principal".to_owned(),
            },
            idempotency_key: IdempotencyKey::from("tool-idem-1"),
            attempt: 0,
        }
    }

    #[tokio::test]
    async fn claim_is_idempotent_and_detects_request_reuse() {
        let journal = InMemoryToolJournal::default();
        assert!(matches!(
            journal.claim(request("query")).await.unwrap(),
            ToolClaimResult::Acquired(_)
        ));
        assert!(matches!(
            journal.claim(request("query")).await.unwrap(),
            ToolClaimResult::Existing(_)
        ));
        assert_eq!(
            journal.claim(request("different")).await.unwrap_err(),
            InMemoryToolJournalError::IdempotencyConflict
        );
    }

    #[tokio::test]
    async fn terminal_outcome_is_persisted_once() {
        let journal = InMemoryToolJournal::default();
        let request = request("query");
        let key = request.journal_key();
        journal.claim(request).await.unwrap();
        let completed = journal
            .finish(
                key.clone(),
                ToolJournalOutcome::Completed(ToolExecutionResult {
                    output: json!({"answer": 42}),
                }),
            )
            .await
            .unwrap();
        assert_eq!(completed.state, ToolJournalState::Completed);
        assert_eq!(completed.result.unwrap().output["answer"], 42);

        assert_eq!(
            journal
                .finish(
                    key,
                    ToolJournalOutcome::Failed(ToolExecutionFailure {
                        code: "late".to_owned(),
                        message: "late failure".to_owned(),
                        retryable: false,
                    }),
                )
                .await
                .unwrap_err(),
            InMemoryToolJournalError::AlreadyFinished(ToolJournalState::Completed)
        );
    }

    #[tokio::test]
    async fn unknown_outcome_is_terminal_and_recoverable() {
        let journal = InMemoryToolJournal::default();
        let request = request("query");
        let key = request.journal_key();
        journal.claim(request).await.unwrap();
        journal
            .finish(key.clone(), ToolJournalOutcome::OutcomeUnknown)
            .await
            .unwrap();

        assert_eq!(
            journal.load(&key).await.unwrap().unwrap().state,
            ToolJournalState::OutcomeUnknown
        );
    }
}
