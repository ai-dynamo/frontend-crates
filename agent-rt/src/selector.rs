// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_protocols::types::responses::CreateResponse;

/// Frontend policy deciding whether a request needs stateful orchestration.
pub trait RuntimeSelector: Send + Sync + 'static {
    fn requires_runtime(&self, request: &CreateResponse) -> bool;
}

/// Selects the runtime for persisted or continuation requests.
///
/// Runtime-owned tool policy can wrap or replace this selector later without
/// changing the state runtime.
#[derive(Debug, Clone, Copy, Default)]
pub struct StatefulRequestSelector;

impl RuntimeSelector for StatefulRequestSelector {
    fn requires_runtime(&self, request: &CreateResponse) -> bool {
        request.store == Some(true) || request.previous_response_id.is_some()
    }
}

#[cfg(test)]
mod tests {
    use dynamo_protocols::types::responses::CreateResponse;

    use super::{RuntimeSelector, StatefulRequestSelector};

    #[test]
    fn selects_only_persisted_or_continuation_requests() {
        let selector = StatefulRequestSelector;
        assert!(!selector.requires_runtime(&CreateResponse::default()));

        let stored = CreateResponse {
            store: Some(true),
            ..Default::default()
        };
        assert!(selector.requires_runtime(&stored));

        let continuation = CreateResponse {
            previous_response_id: Some("resp_parent".to_owned()),
            ..Default::default()
        };
        assert!(selector.requires_runtime(&continuation));
    }
}
