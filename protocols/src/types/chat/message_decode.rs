// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One-pass decode of [`ChatCompletionRequestMessage`].
//!
//! A derived `#[serde(tag = "role")]` enum buffers each message to find its
//! tag, and a `#[serde(flatten)]` guard around the message buffers it again.
//! Clients write `role` first, so this decoder reads the first key: when it is
//! `role`, it decodes the role's message straight from the input. A message
//! that puts another key first goes through a derived tagged enum, which
//! buffers it once.
//!
//! Both paths run the same guard, [`decode_role`]: it refuses a second
//! `role`, and `tools` or `partial` on a role that cannot carry them.
//!
//! A value that no field reads is skipped without being parsed, as serde
//! skips an unknown field anywhere else. So a number out of range, or a lone
//! surrogate escape, in such a value is dropped, where a buffered message
//! refused it.

use std::fmt;
use std::marker::PhantomData;

use serde::de::value::{
    BorrowedStrDeserializer, BytesDeserializer, MapAccessDeserializer, SeqAccessDeserializer,
    StringDeserializer,
};
use serde::de::{
    self, DeserializeSeed, IgnoredAny, IntoDeserializer, MapAccess, SeqAccess, Visitor,
};
use serde::{Deserialize, Deserializer};

use super::{
    ChatCompletionRequestAssistantMessage, ChatCompletionRequestDeveloperMessage,
    ChatCompletionRequestFunctionMessage, ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessage, ChatCompletionRequestToolMessage,
    ChatCompletionRequestUserMessage,
};

/// A role's message type, and the Kimi fields that role may carry.
///
/// Moonshot's negative tests expect a request error for `tools` off a system
/// message and `partial` off an assistant message, so a role that cannot
/// carry one refuses it rather than letting serde drop it as unknown.
trait Role {
    const NAME: &'static str;
    const TAKES_TOOLS: bool = false;
    const TAKES_PARTIAL: bool = false;
}

impl Role for ChatCompletionRequestDeveloperMessage {
    const NAME: &'static str = "developer";
}

impl Role for ChatCompletionRequestSystemMessage {
    const NAME: &'static str = "system";
    const TAKES_TOOLS: bool = true;
}

impl Role for ChatCompletionRequestUserMessage {
    const NAME: &'static str = "user";
}

impl Role for ChatCompletionRequestAssistantMessage {
    const NAME: &'static str = "assistant";
    const TAKES_PARTIAL: bool = true;
}

impl Role for ChatCompletionRequestToolMessage {
    const NAME: &'static str = "tool";
}

impl Role for ChatCompletionRequestFunctionMessage {
    const NAME: &'static str = "function";
}

/// The value of `role`. The derived identifier keeps serde's own errors for
/// an unknown role and for a role that is not a string.
#[derive(Deserialize)]
#[serde(variant_identifier, rename_all = "lowercase")]
enum RoleTag {
    Developer,
    System,
    User,
    Assistant,
    Tool,
    Function,
}

/// The path for a message whose first key is not `role`, and for a message
/// sent as an array: serde's tagged enum buffers it once to find the role.
#[derive(Deserialize)]
#[serde(tag = "role", rename_all = "lowercase")]
enum Tagged {
    Developer(Guarded<ChatCompletionRequestDeveloperMessage>),
    System(Guarded<ChatCompletionRequestSystemMessage>),
    User(Guarded<ChatCompletionRequestUserMessage>),
    Assistant(Guarded<ChatCompletionRequestAssistantMessage>),
    Tool(Guarded<ChatCompletionRequestToolMessage>),
    Function(Guarded<ChatCompletionRequestFunctionMessage>),
}

impl From<Tagged> for ChatCompletionRequestMessage {
    fn from(tagged: Tagged) -> Self {
        match tagged {
            Tagged::Developer(Guarded(message)) => Self::Developer(message),
            Tagged::System(Guarded(message)) => Self::System(message),
            Tagged::User(Guarded(message)) => Self::User(message),
            Tagged::Assistant(Guarded(message)) => Self::Assistant(message),
            Tagged::Tool(Guarded(message)) => Self::Tool(message),
            Tagged::Function(Guarded(message)) => Self::Function(message),
        }
    }
}

impl<'de> Deserialize<'de> for ChatCompletionRequestMessage {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // `deserialize_any`, as the derived tagged enum asks, so a message
        // sent as an array still reaches the tagged enum's sequence form.
        deserializer.deserialize_any(MessageVisitor)
    }
}

struct MessageVisitor;

impl<'de> Visitor<'de> for MessageVisitor {
    type Value = ChatCompletionRequestMessage;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("internally tagged enum ChatCompletionRequestMessage")
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let first = map.next_key::<Key<'de>>()?;
        if first.as_ref().and_then(Key::as_str) == Some("role") {
            use ChatCompletionRequestMessage as M;
            return match map.next_value::<RoleTag>()? {
                RoleTag::Developer => decode_role(map).map(M::Developer),
                RoleTag::System => decode_role(map).map(M::System),
                RoleTag::User => decode_role(map).map(M::User),
                RoleTag::Assistant => decode_role(map).map(M::Assistant),
                RoleTag::Tool => decode_role(map).map(M::Tool),
                RoleTag::Function => decode_role(map).map(M::Function),
            };
        }
        Tagged::deserialize(MapAccessDeserializer::new(Replay { first, map })).map(Into::into)
    }

    fn visit_seq<A>(self, seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        Tagged::deserialize(SeqAccessDeserializer::new(seq)).map(Into::into)
    }
}

/// A role's message, decoded through [`decode_role`]: the tagged enum's
/// variants, which serde hands the buffered message less its `role`.
struct Guarded<T>(T);

impl<'de, T> Deserialize<'de> for Guarded<T>
where
    T: Role + Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct GuardedVisitor<T>(PhantomData<T>);

        impl<'de, T> Visitor<'de> for GuardedVisitor<T>
        where
            T: Role + Deserialize<'de>,
        {
            type Value = Guarded<T>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                write!(formatter, "a {} message", T::NAME)
            }

            fn visit_map<A>(self, map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                decode_role(map).map(Guarded)
            }
        }

        deserializer.deserialize_map(GuardedVisitor(PhantomData))
    }
}

/// Decodes the rest of a message, after its `role`, as `T`.
///
/// A `tools` or `partial` that `T`'s role cannot carry is held back from `T`
/// and refused once `T` has decoded, so `T`'s own errors come first, as they
/// did when a flattened guard struct held the two fields. A null one is
/// accepted, as before.
fn decode_role<'de, T, A>(map: A) -> Result<T, A::Error>
where
    T: Role + Deserialize<'de>,
    A: MapAccess<'de>,
{
    let mut guard = Guard {
        map,
        tools: (!T::TAKES_TOOLS).then_some(Seen::Absent),
        partial: (!T::TAKES_PARTIAL).then_some(Seen::Absent),
    };
    let message = T::deserialize(MapAccessDeserializer::new(&mut guard))?;
    forbid(guard.tools, "tools", "system", T::NAME)?;
    forbid(guard.partial, "partial", "assistant", T::NAME)?;
    Ok(message)
}

fn forbid<E: de::Error>(
    seen: Option<Seen>,
    field: &str,
    allowed_role: &str,
    actual_role: &str,
) -> Result<(), E> {
    if seen == Some(Seen::Value) {
        return Err(E::custom(format!(
            "`{field}` is only accepted on {allowed_role} messages, not on role {actual_role}"
        )));
    }
    Ok(())
}

/// What a guarded field held so far.
#[derive(Clone, Copy, PartialEq)]
enum Seen {
    Absent,
    Null,
    Value,
}

/// The map of one message, after its `role`, as its role's decoder sees it.
///
/// A guarded field (`Some` in `tools` or `partial`) is consumed here and never
/// reaches the decoder. Every other key passes through.
struct Guard<A> {
    map: A,
    tools: Option<Seen>,
    partial: Option<Seen>,
}

impl<'de, A> MapAccess<'de> for Guard<A>
where
    A: MapAccess<'de>,
{
    type Error = A::Error;

    fn next_key_seed<K>(&mut self, seed: K) -> Result<Option<K::Value>, Self::Error>
    where
        K: DeserializeSeed<'de>,
    {
        while let Some(key) = self.map.next_key::<Key<'de>>()? {
            let guarded = match key.as_str() {
                // A struct's fields have string names, so a flattened struct
                // skipped any other key. Skip it too.
                None => {
                    self.map.next_value::<IgnoredAny>()?;
                    continue;
                }
                Some("role") => return Err(de::Error::duplicate_field("role")),
                Some("tools") => self.tools.as_mut().map(|seen| (seen, "tools")),
                Some("partial") => self.partial.as_mut().map(|seen| (seen, "partial")),
                Some(_) => None,
            };
            let Some((seen, field)) = guarded else {
                return key.feed(seed).map(Some);
            };
            if *seen != Seen::Absent {
                return Err(de::Error::duplicate_field(field));
            }
            *seen = match self.map.next_value::<Option<IgnoredAny>>()? {
                Some(IgnoredAny) => Seen::Value,
                None => Seen::Null,
            };
        }
        Ok(None)
    }

    fn next_value_seed<V>(&mut self, seed: V) -> Result<V::Value, Self::Error>
    where
        V: DeserializeSeed<'de>,
    {
        self.map.next_value_seed(seed)
    }
}

/// A message's map with its first key, already read, put back in front.
struct Replay<'de, A> {
    first: Option<Key<'de>>,
    map: A,
}

impl<'de, A> MapAccess<'de> for Replay<'de, A>
where
    A: MapAccess<'de>,
{
    type Error = A::Error;

    fn next_key_seed<K>(&mut self, seed: K) -> Result<Option<K::Value>, Self::Error>
    where
        K: DeserializeSeed<'de>,
    {
        match self.first.take() {
            Some(key) => key.feed(seed).map(Some),
            None => self.map.next_key_seed(seed),
        }
    }

    fn next_value_seed<V>(&mut self, seed: V) -> Result<V::Value, Self::Error>
    where
        V: DeserializeSeed<'de>,
    {
        self.map.next_value_seed(seed)
    }
}

/// A map key, borrowed from the input when the input allows it. A format
/// such as MessagePack can also send a number, a bool, or bytes as a key;
/// that rare key is boxed, so a key stays the size of a `String`.
enum Key<'de> {
    Borrowed(&'de str),
    Owned(String),
    Other(Box<OtherKey>),
}

/// A map key that is not a string.
enum OtherKey {
    U64(u64),
    I64(i64),
    F64(f64),
    Bool(bool),
    Bytes(Vec<u8>),
}

impl<'de> Key<'de> {
    fn as_str(&self) -> Option<&str> {
        match self {
            Key::Borrowed(key) => Some(key),
            Key::Owned(key) => Some(key),
            _ => None,
        }
    }

    /// Hands the key to `seed` as the value it was read as.
    fn feed<K, E>(self, seed: K) -> Result<K::Value, E>
    where
        K: DeserializeSeed<'de>,
        E: de::Error,
    {
        match self {
            Key::Borrowed(key) => seed.deserialize(BorrowedStrDeserializer::new(key)),
            Key::Owned(key) => seed.deserialize(StringDeserializer::new(key)),
            Key::Other(key) => match *key {
                OtherKey::U64(key) => seed.deserialize(key.into_deserializer()),
                OtherKey::I64(key) => seed.deserialize(key.into_deserializer()),
                OtherKey::F64(key) => seed.deserialize(key.into_deserializer()),
                OtherKey::Bool(key) => seed.deserialize(key.into_deserializer()),
                OtherKey::Bytes(key) => seed.deserialize(BytesDeserializer::new(&key)),
            },
        }
    }
}

impl<'de> Deserialize<'de> for Key<'de> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct KeyVisitor;

        impl<'de> Visitor<'de> for KeyVisitor {
            type Value = Key<'de>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a map key")
            }

            fn visit_u64<E>(self, key: u64) -> Result<Self::Value, E> {
                Ok(Key::Other(Box::new(OtherKey::U64(key))))
            }

            fn visit_i64<E>(self, key: i64) -> Result<Self::Value, E> {
                Ok(Key::Other(Box::new(OtherKey::I64(key))))
            }

            fn visit_f64<E>(self, key: f64) -> Result<Self::Value, E> {
                Ok(Key::Other(Box::new(OtherKey::F64(key))))
            }

            fn visit_bool<E>(self, key: bool) -> Result<Self::Value, E> {
                Ok(Key::Other(Box::new(OtherKey::Bool(key))))
            }

            fn visit_char<E>(self, key: char) -> Result<Self::Value, E> {
                Ok(Key::Owned(key.to_string()))
            }

            fn visit_borrowed_str<E>(self, key: &'de str) -> Result<Self::Value, E> {
                Ok(Key::Borrowed(key))
            }

            fn visit_str<E>(self, key: &str) -> Result<Self::Value, E> {
                Ok(Key::Owned(key.to_owned()))
            }

            fn visit_string<E>(self, key: String) -> Result<Self::Value, E> {
                Ok(Key::Owned(key))
            }

            fn visit_borrowed_bytes<E>(self, key: &'de [u8]) -> Result<Self::Value, E> {
                Ok(std::str::from_utf8(key).map_or_else(
                    |_| Key::Other(Box::new(OtherKey::Bytes(key.to_vec()))),
                    Key::Borrowed,
                ))
            }

            fn visit_bytes<E>(self, key: &[u8]) -> Result<Self::Value, E> {
                Ok(std::str::from_utf8(key).map_or_else(
                    |_| Key::Other(Box::new(OtherKey::Bytes(key.to_vec()))),
                    |key| Key::Owned(key.to_owned()),
                ))
            }
        }

        // `deserialize_any`, as serde's tagged enum reads each key, so a key
        // that is not a string reaches the decoder as what it is.
        deserializer.deserialize_any(KeyVisitor)
    }
}

#[cfg(test)]
mod tests {
    /// A key is read and moved for every field of every message, so it
    /// stays the size of a `String`. At 32 bytes, a 200-message request
    /// took 5 percent longer to decode from named MessagePack.
    #[test]
    fn a_key_stays_the_size_of_a_string() {
        assert_eq!(
            std::mem::size_of::<super::Key<'static>>(),
            std::mem::size_of::<String>()
        );
    }
}
