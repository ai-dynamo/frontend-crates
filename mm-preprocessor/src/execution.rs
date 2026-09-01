// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The crate's only parallelism seam.
//!
//! Every fan-out in the crate goes through the functions below, and whether
//! any of them actually fans out is a **runtime** decision made in exactly one
//! place: until a consumer arms the crate's rayon pool with [`init_pool`],
//! everything runs inline on the calling thread and the crate owns no
//! threads.
//!
//! * **inline (the default)**: a server supplies concurrency across requests
//!   and owns its own core budget (it may pin threads), so a library that
//!   silently spawned pools behind its back would fight it.
//! * **armed**: work fans out on the crate-owned pool. A consumer calling in
//!   from one or two threads (e.g. a Python processor with the GIL released)
//!   arms the pool once at startup to get intra-call parallelism.
//!
//! Note that "inline" means *on the caller*, not a one-thread pool:
//! `ThreadPool::install` injects work into the pool and blocks the caller, so
//! sizing a pool to 1 would serialize every concurrent request in the process
//! instead of just declining to fan out.
//!
//! Results are identical either way — the fan-outs are order-preserving maps
//! and writes into disjoint slices, never reductions.
//!
//! The `parallel` cargo feature (default: **on**) only controls whether rayon
//! is linked at all; disabling it (`default-features = false`) drops the
//! dependency and [`init_pool`] with it, forcing the inline path at compile
//! time.

/// Arm the crate's CPU pool: from the first call on, the helpers below fan
/// out on it instead of running inline. `threads == 0` picks the default size
/// `min(available_parallelism, 8)`. Idempotent — the first caller wins, and
/// once armed the size is fixed.
#[cfg(feature = "parallel")]
pub fn init_pool(threads: usize) {
    let _ = threads;
    todo!("PR2: arm the OnceLock-backed rayon pool")
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
    todo!("PR2: rayon par_iter when the pool is armed, inline iterator otherwise")
}

/// Apply `f(chunk_index, chunk)` over disjoint `chunk_size`-element windows of
/// `buf`. The final chunk is short when `chunk_size` does not divide the length.
pub fn for_chunks_mut<T: Send>(
    buf: &mut [T],
    chunk_size: usize,
    f: impl Fn(usize, &mut [T]) + Send + Sync,
) {
    let _ = (buf, chunk_size, &f);
    todo!("PR2: rayon par_chunks_mut when the pool is armed, inline otherwise")
}

/// Run `f` with the CPU pool already entered, so nested [`for_chunks_mut`]
/// calls inside it reuse this entry instead of injecting a job each. Use it to
/// wrap a multi-stage leaf (e.g. the two passes of a separable resize) that
/// would otherwise pay per-stage pool entry.
pub fn in_pool<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    let _ = &f;
    todo!("PR2: pool().install when the pool is armed, direct call otherwise")
}
