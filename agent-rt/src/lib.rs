// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stateful OpenAI Responses runtime contracts.
//!
//! This crate deliberately stops before HTTP ingress, authentication, request
//! rendering, routing, engine selection, and tool-worker implementation. Those
//! concerns are supplied through narrow traits so the same runtime can be
//! embedded in different frontends and inference deployments.

#![cfg_attr(docsrs, feature(doc_cfg))]

mod authorization;
mod checkpoint;
mod clock;
mod ids;
mod inference;
mod materialize;
mod memory;
mod selector;
mod tool;

pub use authorization::{AuthorizationScope, RuntimeAuthorization, RuntimeLimits};
pub use checkpoint::{
    BeginTurn, BeginTurnResult, CheckpointRecord, CheckpointStore, CheckpointVersion, CommitTurn,
    CommitTurnResult, LeaseDeadline, LoadChain, RenewLease, TurnLease, TurnState,
};
pub use clock::{Clock, SystemClock};
pub use ids::{IdGenerator, IdempotencyKey, ResponseId, TurnId, UuidGenerator};
pub use inference::{
    InferenceIntent, InferenceInvoker, InferenceOutput, InferenceRequest, InvocationContext,
    ModelStepKind,
};
pub use materialize::{
    ContinuationPolicy, InheritContinuationControls, MaterializationError, MaterializedTurn,
    PolicyRequestMaterializer, RequestMaterializer, ResponsesRequestMaterializer,
};
pub use memory::{InMemoryCheckpointStore, InMemoryStoreError};
pub use selector::{RuntimeSelector, StatefulRequestSelector};
pub use tool::{ToolExecutionRequest, ToolExecutionResult, ToolExecutor};

/// A boxed future used by object-safe asynchronous runtime traits.
pub type BoxFuture<'a, T> = std::pin::Pin<Box<dyn std::future::Future<Output = T> + Send + 'a>>;

/// A boxed stream used by inference implementations.
pub type BoxStream<'a, T> = std::pin::Pin<Box<dyn futures::Stream<Item = T> + Send + 'a>>;
