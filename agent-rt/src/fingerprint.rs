// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde_json::Value;

use crate::{AgentProtocol, RequestFingerprint};

/// Produces the stable digest used for scoped idempotency checks.
pub trait RequestFingerprinter<P>: Send + Sync + 'static
where
    P: AgentProtocol,
{
    type Error: std::error::Error + Send + Sync + 'static;

    fn fingerprint(&self, request: &P::Request) -> Result<RequestFingerprint, Self::Error>;
}

/// Canonical JSON fingerprint suitable for native protocol DTOs.
///
/// Object keys are sorted explicitly so the digest is independent of map
/// implementation and insertion order. Deployments can replace this with a
/// protocol-specific zero-copy implementation through `RequestFingerprinter`.
#[derive(Debug, Clone, Copy, Default)]
pub struct CanonicalJsonFingerprinter;

impl<P> RequestFingerprinter<P> for CanonicalJsonFingerprinter
where
    P: AgentProtocol,
{
    type Error = serde_json::Error;

    fn fingerprint(&self, request: &P::Request) -> Result<RequestFingerprint, Self::Error> {
        let value = serde_json::to_value(request)?;
        let mut hasher = blake3::Hasher::new();
        hash_value(&value, &mut hasher);
        Ok(RequestFingerprint::new(*hasher.finalize().as_bytes()))
    }
}

fn hash_value(value: &Value, hasher: &mut blake3::Hasher) {
    match value {
        Value::Null => {
            hasher.update(&[0]);
        }
        Value::Bool(boolean) => {
            hasher.update(&[1, u8::from(*boolean)]);
        }
        Value::Number(number) => {
            hasher.update(&[2]);
            hash_bytes(number.to_string().as_bytes(), hasher);
        }
        Value::String(string) => {
            hasher.update(&[3]);
            hash_bytes(string.as_bytes(), hasher);
        }
        Value::Array(values) => {
            hasher.update(&[4]);
            hash_len(values.len(), hasher);
            for value in values {
                hash_value(value, hasher);
            }
        }
        Value::Object(object) => {
            hasher.update(&[5]);
            hash_len(object.len(), hasher);
            let mut entries = object.iter().collect::<Vec<_>>();
            entries.sort_unstable_by_key(|(key, _)| *key);
            for (key, value) in entries {
                hash_bytes(key.as_bytes(), hasher);
                hash_value(value, hasher);
            }
        }
    }
}

fn hash_bytes(bytes: &[u8], hasher: &mut blake3::Hasher) {
    hash_len(bytes.len(), hasher);
    hasher.update(bytes);
}

fn hash_len(len: usize, hasher: &mut blake3::Hasher) {
    hasher.update(&(len as u64).to_le_bytes());
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use dynamo_protocols::types::responses::{CreateResponse, InputParam};

    use crate::OpenAiResponses;

    use super::{CanonicalJsonFingerprinter, RequestFingerprinter};

    fn fingerprint(request: &CreateResponse) -> crate::RequestFingerprint {
        <CanonicalJsonFingerprinter as RequestFingerprinter<OpenAiResponses>>::fingerprint(
            &CanonicalJsonFingerprinter,
            request,
        )
        .unwrap()
    }

    #[test]
    fn object_insertion_order_does_not_change_the_fingerprint() {
        let mut left_metadata = HashMap::new();
        left_metadata.insert("a".to_owned(), "one".to_owned());
        left_metadata.insert("b".to_owned(), "two".to_owned());
        let mut right_metadata = HashMap::new();
        right_metadata.insert("b".to_owned(), "two".to_owned());
        right_metadata.insert("a".to_owned(), "one".to_owned());

        let left = CreateResponse {
            input: InputParam::Text("hello".to_owned()),
            metadata: Some(left_metadata),
            ..Default::default()
        };
        let right = CreateResponse {
            input: InputParam::Text("hello".to_owned()),
            metadata: Some(right_metadata),
            ..Default::default()
        };

        assert_eq!(fingerprint(&left), fingerprint(&right));
    }

    #[test]
    fn request_changes_produce_a_different_fingerprint() {
        let first = CreateResponse {
            input: InputParam::Text("first".to_owned()),
            ..Default::default()
        };
        let second = CreateResponse {
            input: InputParam::Text("second".to_owned()),
            ..Default::default()
        };

        assert_ne!(fingerprint(&first), fingerprint(&second));
    }
}
