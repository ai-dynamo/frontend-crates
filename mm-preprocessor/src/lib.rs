// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Model-family multimodal preprocessing for LLM inference serving — a
//! Rust replacement for the image pipelines behind HF `AutoProcessor`.
//!
//! Model families implement [`processor::MmFamilyProcessor`] (decoded media →
//! named tensors, prompt geometry as data, position encodings, pixel-free
//! token accounting), selected through [`registry::ProcessorSpec`] — resolved
//! from the HF config files ([`registry::spec_from_model_dir`]) or handed
//! over pre-resolved. The crate also carries what routers and engines must
//! agree on: token accounting and content-hash identity
//! ([`content_hash_bytes`], [`content_hash_canonical_image`]). The
//! feature-gated `fetch` module is an optional trusted-source compatibility
//! helper. Request orchestration — concurrency, caps, URL security policy,
//! failure policy, packing — stays in the consumer's driver, as on the Python
//! path; the README maps the boundary.
//!
//! Bit-exactness is the contract: the resize kernels ([`image::resize`]) and
//! each family's normalize/patchify reproduce the mirrored HF processor's
//! arithmetic exactly, so an engine can swap this crate in without output
//! drift.
//!
//! [`MmError`] preserves a small failure taxonomy so each consumer can choose
//! its own fallback or rejection policy without matching display strings.
//!
//! The crate reads no environment variables and owns no threads until a
//! consumer arms the rayon pool; see [`execution`].

pub mod execution;
#[cfg(feature = "fetch")]
pub mod fetch;
pub mod image;
pub mod models;
pub mod processor;
pub mod registry;
pub mod token_layout;

/// A multimodal preprocessing failure.
#[derive(Debug)]
#[non_exhaustive]
pub enum MmError {
    Unsupported {
        message: String,
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
    InvalidInput {
        message: String,
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
    LimitExceeded {
        message: String,
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
    Internal {
        message: String,
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
}

impl MmError {
    pub fn invalid_input(message: impl Into<String>) -> Self {
        Self::InvalidInput {
            message: message.into(),
            source: None,
        }
    }

    pub fn invalid_input_with_source(
        message: impl Into<String>,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::InvalidInput {
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }

    pub fn unsupported(message: impl Into<String>) -> Self {
        Self::Unsupported {
            message: message.into(),
            source: None,
        }
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal {
            message: message.into(),
            source: None,
        }
    }
}

impl std::fmt::Display for MmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported { message, .. }
            | Self::InvalidInput { message, .. }
            | Self::LimitExceeded { message, .. }
            | Self::Internal { message, .. } => f.write_str(message),
        }
    }
}

impl std::error::Error for MmError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        let source = match self {
            Self::Unsupported { source, .. }
            | Self::InvalidInput { source, .. }
            | Self::LimitExceeded { source, .. }
            | Self::Internal { source, .. } => source.as_deref(),
        };
        source.map(|source| source as &(dyn std::error::Error + 'static))
    }
}

pub type Result<T> = std::result::Result<T, MmError>;

/// BLAKE3 over encoded media bytes, truncated to a big-endian `u64`.
pub fn content_hash_bytes(data: &[u8]) -> u64 {
    let digest = blake3::hash(data);
    u64::from_be_bytes(digest.as_bytes()[..8].try_into().unwrap())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum MediaDtype {
    U8,
}

/// Dynamo's canonical XXH3-64 identity for a decoded image.
///
/// Hashes tensor rank, dimensions, dtype, and contiguous RGB bytes. Other
/// modalities have separate identity contracts; in particular, Dynamo's
/// video identity also covers decoded metadata.
pub fn content_hash_canonical_image(shape: &[usize], dtype: MediaDtype, data: &[u8]) -> u64 {
    let mut hasher = xxhash_rust::xxh3::Xxh3::new();
    // Rank and dimensions are fixed-width so distinct shapes cannot alias
    // after concatenation.
    hasher.update(&(shape.len() as u64).to_le_bytes());
    for &dim in shape {
        hasher.update(&(dim as u64).to_le_bytes());
    }
    let dtype_byte: u8 = match dtype {
        MediaDtype::U8 => 0,
    };
    hasher.update(&[dtype_byte]);
    hasher.update(data);
    hasher.digest()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both identities are cross-process contracts — a router hashes and an
    /// engine looks the key up — so they are pinned to the values the mirrored
    /// implementations produce, not merely to being stable here.
    #[test]
    fn identities_match_the_mirrored_implementations() {
        // Dynamo's `canonical_content_hash` (lib/llm preprocessor::media::rdma).
        assert_eq!(
            content_hash_canonical_image(&[1, 2, 3], MediaDtype::U8, &[0, 1, 2, 3, 4, 5]),
            0x7a9b_bcb1_1a89_8630
        );
        // SGLang's fallback: BLAKE3 of b"abc", first 8 bytes big-endian.
        assert_eq!(content_hash_bytes(b"abc"), 0x6437_b3ac_3846_5133);
    }

    /// Shape and dtype are hashed alongside the pixels, so two buffers that
    /// differ only in geometry cannot collide into one cache entry.
    #[test]
    fn canonical_image_identity_covers_geometry() {
        let data = [0u8, 1, 2, 3, 4, 5];
        let hash = |shape: &[usize]| content_hash_canonical_image(shape, MediaDtype::U8, &data);
        assert_ne!(hash(&[1, 2, 3]), hash(&[3, 2, 1]));
        assert_ne!(hash(&[1, 2, 3]), hash(&[2, 3]));
    }
}
