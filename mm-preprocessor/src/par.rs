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
//! Results are identical either way — the fan-outs are order-preserving maps
//! and writes into disjoint slices, never reductions.

/// Size the crate's CPU pool before its first use. Idempotent — the first
/// caller wins, and once the pool exists the size is fixed; zero is ignored.
/// Never called: `min(available_parallelism, 8)`.
#[cfg(feature = "parallel")]
pub fn init_pool(threads: usize) {
    let _ = threads;
    todo!("PR2: OnceLock-backed rayon pool sizing")
}

/// Map `items`, short-circuiting on the first error. Output order matches input
/// order. CPU-bound work: decode, resize, patchify, hash.
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
    todo!("PR2: rayon par_iter under `parallel`, inline iterator otherwise")
}

/// Apply `f(chunk_index, chunk)` over disjoint `chunk_size`-element windows of
/// `buf`. The final chunk is short when `chunk_size` does not divide the length.
pub fn for_chunks_mut<T: Send>(
    buf: &mut [T],
    chunk_size: usize,
    f: impl Fn(usize, &mut [T]) + Send + Sync,
) {
    let _ = (buf, chunk_size, &f);
    todo!("PR2: rayon par_chunks_mut under `parallel`, inline otherwise")
}

/// Run `f` with the CPU pool already entered, so nested [`for_chunks_mut`]
/// calls inside it reuse this entry instead of injecting a job each. Use it to
/// wrap a multi-stage leaf (e.g. the two passes of a separable resize) that
/// would otherwise pay per-stage pool entry.
pub fn in_pool<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    let _ = &f;
    todo!("PR2: pool().install under `parallel`, direct call otherwise")
}
