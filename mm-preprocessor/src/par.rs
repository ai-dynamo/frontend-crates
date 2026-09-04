// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The crate's only parallelism seam.
//!
//! Every fan-out in the crate goes through the functions below, so whether
//! this crate owns worker threads at all is decided in exactly one place: the
//! `parallel` cargo feature.
//!
//! * **feature on**: work is fanned out on the crate's rayon pool. A consumer
//!   calling in from one or two worker threads (e.g. a Python processor with
//!   the GIL released) gets intra-call parallelism.
//! * **feature off**: rayon is not even a dependency, and everything runs
//!   inline on the calling thread. A server supplies concurrency across
//!   requests and owns its own core budget (it may pin threads), so a library
//!   that silently spawns its own pools would fight it.
//!
//! Note that "sequential" here means *inline on the caller*, not a one-thread
//! pool: `ThreadPool::install` injects work into the pool and blocks the
//! caller, so sizing a pool to 1 would serialize every concurrent request in
//! the process instead of just declining to fan out.
//!
//! Results are identical either way — the fan-outs are order-preserving maps
//! and writes into disjoint slices, never reductions.

#[cfg(feature = "parallel")]
use std::sync::OnceLock;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "parallel")]
static POOL_SIZE: OnceLock<usize> = OnceLock::new();

/// Size the crate's CPU pool before its first use. Idempotent — the first
/// caller wins, and once the pool exists the size is fixed; zero is ignored.
/// Never called: `min(available_parallelism, 8)`.
#[cfg(feature = "parallel")]
pub fn init_pool(threads: usize) {
    if threads > 0 {
        let _ = POOL_SIZE.set(threads);
    }
}

/// CPU pool: decode, resize, patchify, hash. Capped at 8 by default because
/// the work is compute-bound — never run blocking I/O on it, or one request's
/// remote fetches stall every other request's preprocessing.
#[cfg(feature = "parallel")]
fn pool() -> &'static rayon::ThreadPool {
    static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        let n = POOL_SIZE
            .get()
            .copied()
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(8, |c| c.get().min(8)));
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .thread_name(|i| format!("dyn-mm-{i}"))
            .build()
            .expect("failed to build rayon pool")
    })
}

/// Map `items`, short-circuiting on the first error. Output order matches input
/// order. CPU-bound work: decode, resize, patchify, hash.
#[cfg(feature = "parallel")]
pub fn try_map<'a, T, R, E>(
    items: &'a [T],
    f: impl Fn(&'a T) -> Result<R, E> + Send + Sync,
) -> Result<Vec<R>, E>
where
    T: Send + Sync,
    R: Send,
    E: Send,
{
    pool().install(|| items.par_iter().map(f).collect())
}

#[cfg(not(feature = "parallel"))]
pub fn try_map<'a, T, R, E>(
    items: &'a [T],
    f: impl Fn(&'a T) -> Result<R, E> + Send + Sync,
) -> Result<Vec<R>, E>
where
    T: Send + Sync,
    R: Send,
    E: Send,
{
    items.iter().map(f).collect()
}

/// Apply `f(chunk_index, chunk)` over disjoint `chunk_size`-element windows of
/// `buf`. The final chunk is short when `chunk_size` does not divide the length.
#[cfg(feature = "parallel")]
pub fn for_chunks_mut<T: Send>(
    buf: &mut [T],
    chunk_size: usize,
    f: impl Fn(usize, &mut [T]) + Send + Sync,
) {
    pool().install(|| {
        buf.par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(index, chunk)| f(index, chunk));
    });
}

#[cfg(not(feature = "parallel"))]
pub fn for_chunks_mut<T: Send>(
    buf: &mut [T],
    chunk_size: usize,
    f: impl Fn(usize, &mut [T]) + Send + Sync,
) {
    for (index, chunk) in buf.chunks_mut(chunk_size).enumerate() {
        f(index, chunk);
    }
}

/// Run `f` with the CPU pool already entered, so nested [`for_chunks_mut`]
/// calls inside it reuse this entry instead of injecting a job each. Use it to
/// wrap a multi-stage leaf (e.g. the two passes of a separable resize) that
/// would otherwise pay per-stage pool entry.
#[cfg(feature = "parallel")]
pub fn in_pool<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    pool().install(f)
}

#[cfg(not(feature = "parallel"))]
pub fn in_pool<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    f()
}
