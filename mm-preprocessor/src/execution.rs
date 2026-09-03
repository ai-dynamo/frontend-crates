// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The crate's only parallelism seam.
//!
//! All fan-out goes through the functions below. By default they run inline
//! on the calling thread — servers already provide cross-request concurrency
//! and shouldn't have a library spawning pools behind their back. Consumers
//! that call in from few threads (e.g. a Python processor) can call
//! [`init_pool`] once to fan out on a crate-owned rayon pool instead.
//! Hosts that already parallelize preprocessing should leave it unarmed to
//! avoid nested parallelism.
//!
//! "Inline" means *on the caller*, not a one-thread pool: `install` on a
//! 1-sized pool would serialize every concurrent request in the process.
//!
//! Results are identical either way — the fan-outs are order-preserving maps
//! and disjoint-slice writes, never reductions.
//!
//! The `parallel` cargo feature (default on) only controls whether rayon is
//! linked; disabling it drops [`init_pool`] and forces the inline path.

/// Arm the crate's CPU pool: from the first call on, the helpers below fan
/// out on it instead of running inline. `threads == 0` picks the default size
/// `min(available_parallelism, 8)`. Repeating the resolved thread count is
/// idempotent; requesting a different count after initialization returns
/// [`MmError::InvalidInput`](crate::MmError::InvalidInput).
#[cfg(feature = "parallel")]
pub fn init_pool(threads: usize) -> crate::Result<()> {
    let _ = threads;
    todo!("arm the OnceLock-backed rayon pool")
}

/// Map `items`, short-circuiting on the first error. Output order matches input
/// order. CPU-bound work: decode, resize, patchify (engines may also reuse
/// this seam for their own per-item fan-out, e.g. hashing).
pub fn try_map<'a, T, R, E>(
    items: &'a [T],
    f: impl Fn(&'a T) -> Result<R, E> + Send + Sync,
) -> Result<Vec<R>, E>
where
    T: Send + Sync,
    R: Send,
    E: Send,
{
    let _ = (items, &f);
    todo!("rayon par_iter when the pool is armed, inline iterator otherwise")
}

/// Apply `f(chunk_index, chunk)` over disjoint `chunk_size`-element windows of
/// `buf`. The final chunk is short when `chunk_size` does not divide the length.
pub fn for_chunks_mut<T: Send>(
    buf: &mut [T],
    chunk_size: usize,
    f: impl Fn(usize, &mut [T]) + Send + Sync,
) {
    let _ = (buf, chunk_size, &f);
    todo!("rayon par_chunks_mut when the pool is armed, inline otherwise")
}

/// Run `f` with the CPU pool already entered, so nested [`for_chunks_mut`]
/// calls inside it reuse this entry instead of injecting a job each. Use it to
/// wrap a multi-stage leaf (e.g. the two passes of a separable resize) that
/// would otherwise pay per-stage pool entry.
pub fn in_pool<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    let _ = &f;
    todo!("pool().install when the pool is armed, direct call otherwise")
}
