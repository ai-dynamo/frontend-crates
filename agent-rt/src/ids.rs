// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};

macro_rules! string_id {
    ($name:ident) => {
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(String);

        impl $name {
            pub fn new(value: impl Into<String>) -> Self {
                Self(value.into())
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }

            pub fn into_inner(self) -> String {
                self.0
            }
        }

        impl From<String> for $name {
            fn from(value: String) -> Self {
                Self(value)
            }
        }

        impl From<&str> for $name {
            fn from(value: &str) -> Self {
                Self(value.to_owned())
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str(&self.0)
            }
        }
    };
}

string_id!(ResponseId);
string_id!(TurnId);
string_id!(IdempotencyKey);

/// Generates externally visible and internal runtime identifiers.
pub trait IdGenerator: Send + Sync + 'static {
    fn response_id(&self) -> ResponseId;
    fn turn_id(&self) -> TurnId;
}

/// UUID-backed identifier generator suitable for production frontends.
#[derive(Debug, Clone, Copy, Default)]
pub struct UuidGenerator;

impl IdGenerator for UuidGenerator {
    fn response_id(&self) -> ResponseId {
        ResponseId::new(format!("resp_{}", uuid::Uuid::new_v4().simple()))
    }

    fn turn_id(&self) -> TurnId {
        TurnId::new(format!("turn_{}", uuid::Uuid::new_v4().simple()))
    }
}
