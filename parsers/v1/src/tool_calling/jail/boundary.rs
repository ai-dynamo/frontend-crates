// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The jail's boundary rules, moved from dynamo's jail adapter
//! (`OpenAIPreprocessor::apply_tool_calling_jail` in dynamo
//! `lib/llm/src/preprocessor.rs`).
//!
//! The adapter's metadata handling (metrics, `nvext`, prompt logprobs) stays in
//! dynamo. These are the parts that do not depend on dynamo's types: the
//! terminal-error contract, and the request check on `n` that the jail needs.

use std::fmt;
use std::sync::{Arc, Mutex};

use futures::{Stream, StreamExt};

use crate::chat_stream::ChatStreamItem;

/// Run `jail` under the terminal-error contract.
///
/// The jail cannot tell a stream that ended on a backend error from one that
/// finished: it sees a plain end of stream either way. On that end it can
/// build a `tool_calls` finish chunk from the arguments it holds, although the
/// request failed. The unified parser adapters already handle a terminal error
/// this way: they yield the error and drop everything after it. This gives the
/// jail the same contract:
///
/// * The first error item (see [`ChatStreamItem::is_error`]) is latched and
///   ends the jail's input there. The jail sees a plain end of stream and may
///   still finalize.
/// * Output that the jail emits after the latch is dropped, so nothing that
///   finalize builds reaches the caller.
/// * The original error item is yielded last.
///
/// `jail` is the stage under the contract. It gets the input stream without
/// errors and returns the jailed stream. dynamo passes its own adapter here,
/// which converts its items for [`super::apply_tool_calling_jail`] and adds its
/// metadata back. Output that `jail` emits after its input ends (for example
/// that metadata) is also dropped once an error is latched.
pub fn apply_jail_terminal_error_contract<T, S, J, O>(
    stream: S,
    jail: J,
) -> impl Stream<Item = T> + Send
where
    T: ChatStreamItem,
    S: Stream<Item = T> + Send + 'static,
    J: FnOnce(futures::stream::BoxStream<'static, T>) -> O,
    O: Stream<Item = T> + Send + 'static,
{
    let terminal_error: Arc<Mutex<Option<T>>> = Arc::new(Mutex::new(None));
    let terminal_error_in = Arc::clone(&terminal_error);
    let stream = stream.take_while(move |a| {
        let is_error = a.is_error();
        if is_error {
            *terminal_error_in
                .lock()
                .expect("jail terminal error poisoned") = Some(a.clone());
        }
        std::future::ready(!is_error)
    });

    // Once an upstream error is latched, drop any output the jail synthesized
    // while it observed the shortened stream.
    let terminal_error_out = Arc::clone(&terminal_error);
    let jailed_output = jail(stream.boxed()).take_while(move |_| {
        let stop = terminal_error_out
            .lock()
            .expect("jail terminal error poisoned")
            .is_some();
        std::future::ready(!stop)
    });

    async_stream::stream! {
        tokio::pin!(jailed_output);
        while let Some(response) = jailed_output.next().await {
            yield response;
        }

        let terminal_error = {
            terminal_error
                .lock()
                .expect("jail terminal error poisoned")
                .take()
        };
        if let Some(error) = terminal_error {
            yield error;
        }
    }
}

/// A request asked for more than one choice and for a per-choice
/// `nvext.extra_fields` entry that the jail cannot keep apart per choice.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JailChoiceCountError {
    /// The `nvext.extra_fields` entry that needs `n = 1`.
    pub field: String,
}

impl fmt::Display for JailChoiceCountError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "legacy tool-call parsing requires n = 1 when nvext.extra_fields requests choice-specific field `{}`",
            self.field
        )
    }
}

impl std::error::Error for JailChoiceCountError {}

/// Refuse `n > 1` on the jail route when `extra_fields` asks for a field that
/// is specific to one choice.
///
/// The jail can fold several input chunks into one output chunk, so it cannot
/// keep per-choice response metadata apart. dynamo reports the error as an
/// invalid argument (HTTP 400).
pub fn validate_jail_nvext_choice_count(
    n: u8,
    extra_fields: Option<&[String]>,
    is_jail_route: bool,
) -> Result<(), JailChoiceCountError> {
    if n <= 1 || !is_jail_route {
        return Ok(());
    }

    const CHOICE_SPECIFIC_FIELDS: [&str; 3] = ["engine_data", "routed_experts", "stop_reason"];
    if let Some(field) = extra_fields.and_then(|fields| {
        fields
            .iter()
            .find(|field| CHOICE_SPECIFIC_FIELDS.contains(&field.as_str()))
    }) {
        return Err(JailChoiceCountError {
            field: field.clone(),
        });
    }

    Ok(())
}
