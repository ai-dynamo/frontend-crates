// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::Dialect;
use serde_json::{Value, json};

/// Sanitized, transport-independent error. The HTTP owner adds request-ID headers.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
#[error("{message}")]
pub struct DecisionError {
    pub dialect: Dialect,
    pub status: u16,
    pub code: String,
    pub message: String,
}

impl DecisionError {
    pub fn new(
        dialect: Dialect,
        status: u16,
        code: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            dialect,
            status,
            code: code.into(),
            message: message.into(),
        }
    }
    pub fn validation(dialect: Dialect, message: impl Into<String>) -> Self {
        Self::new(
            dialect,
            if dialect == Dialect::Jev { 422 } else { 400 },
            "invalid_request_error",
            message,
        )
    }
    pub fn unsupported(dialect: Dialect, message: impl Into<String>) -> Self {
        Self::new(
            dialect,
            if dialect == Dialect::Jev { 422 } else { 400 },
            "unsupported_capability",
            message,
        )
    }
    pub fn execution(dialect: Dialect, message: impl Into<String>) -> Self {
        Self::new(dialect, 502, "invalid_executor_result", message)
    }
    pub fn response_body(&self) -> Value {
        match self.dialect {
            Dialect::Jev
                if self.code == "invalid_request_error"
                    || self.code == "unsupported_capability" =>
            {
                json!({
                    "detail":[{"loc":["body"],"msg":self.message,"type":self.code}]
                })
            }
            Dialect::Jev => json!({"detail":self.message,"code":self.code}),
            Dialect::OpenAi => {
                json!({"error":{"message":self.message,"type":self.code,"param":null,"code":self.code}})
            }
        }
    }
}
