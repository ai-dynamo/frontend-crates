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
//! Successful results are identical either way — the fan-outs are order-preserving maps
//! and disjoint-slice writes, never reductions.
//!
//! The `parallel` cargo feature (default on) only controls whether rayon is
//! linked; disabling it drops [`init_pool`] and forces the inline path.

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "parallel")]
static POOL: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();

/// Arm the crate's CPU pool: from the first call on, the helpers below fan
/// out on it instead of running inline. `threads == 0` picks the default size
/// `min(available_parallelism, 8)`. Repeating the resolved thread count is
/// idempotent; requesting a different count after initialization returns
/// [`MmError::InvalidInput`](crate::MmError::InvalidInput).
#[cfg(feature = "parallel")]
pub fn init_pool(threads: usize) -> crate::Result<()> {
    let threads = if threads == 0 {
        std::thread::available_parallelism().map_or(8, |n| n.get().min(8))
    } else {
        threads
    };
    let pool = match POOL.get() {
        Some(pool) => pool,
        None => {
            let built = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .thread_name(|i| format!("dyn-mm-{i}"))
                .build()
                .map_err(|error| {
                    crate::MmError::internal_with_source("cannot build mm pool", error)
                })?;
            POOL.get_or_init(|| built)
        }
    };
    if pool.current_num_threads() != threads {
        return Err(crate::MmError::invalid_input(format!(
            "mm pool already initialized with {} thread(s), requested {threads}",
            pool.current_num_threads()
        )));
    }
    Ok(())
}

#[cfg(feature = "parallel")]
fn armed() -> Option<&'static rayon::ThreadPool> {
    POOL.get()
}

/// The armed pool when the caller is not one of its threads: a caller that
/// would block on the pool from inside it runs inline.
#[cfg(feature = "parallel")]
fn armed_outside() -> Option<&'static rayon::ThreadPool> {
    armed().filter(|pool| pool.current_thread_index().is_none())
}

/// Map `items`, stopping when an error is observed. With the pool armed,
/// which error is returned is unspecified, and other items may already have
/// run. Successful output order matches input order.
pub fn try_map<'a, T, R, E>(
    items: &'a [T],
    f: impl Fn(&'a T) -> Result<R, E> + Send + Sync,
) -> Result<Vec<R>, E>
where
    T: Send + Sync,
    R: Send,
    E: Send,
{
    #[cfg(feature = "parallel")]
    if let Some(pool) = armed() {
        return pool.install(|| items.par_iter().map(&f).collect());
    }
    items.iter().map(f).collect()
}

/// The number of threads a caller may fan out over: the pool size once armed,
/// else 1, and 1 on a pool thread.
pub fn threads() -> usize {
    #[cfg(feature = "parallel")]
    if let Some(pool) = armed_outside() {
        return pool.current_num_threads();
    }
    1
}

/// Run `f` on the pool without waiting for it, or inline while unarmed. A
/// panic in `f` aborts the process while the pool is armed.
#[cfg(feature = "video-decode")]
pub(crate) fn spawn(f: impl FnOnce() + Send + 'static) {
    #[cfg(feature = "parallel")]
    if let Some(pool) = armed() {
        return pool.spawn(f);
    }
    f()
}

/// Apply `f(chunk_index, chunk)` over disjoint `chunk_size`-element windows of
/// `buf`. The final chunk is short when `chunk_size` does not divide the length.
pub fn for_chunks_mut<T: Send>(
    buf: &mut [T],
    chunk_size: usize,
    f: impl Fn(usize, &mut [T]) + Send + Sync,
) {
    #[cfg(feature = "parallel")]
    if let Some(pool) = armed() {
        return pool.install(|| {
            buf.par_chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(index, chunk)| f(index, chunk));
        });
    }
    for (index, chunk) in buf.chunks_mut(chunk_size).enumerate() {
        f(index, chunk);
    }
}

/// Run `f` with the CPU pool already entered, so nested [`for_chunks_mut`]
/// calls inside it reuse this entry instead of injecting a job each. Use it to
/// wrap a multi-stage leaf (e.g. the two passes of a separable resize) that
/// would otherwise pay per-stage pool entry.
pub fn in_pool<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    #[cfg(feature = "parallel")]
    if let Some(pool) = armed() {
        return pool.install(f);
    }
    f()
}

/// Run `produce` on each of `items` and hand the results to `consume` in
/// input order, on the calling thread. With the pool armed, items are
/// produced on it, at most `ahead` past the one being consumed; `items`
/// itself is iterated on the caller. Returns the first error in input order,
/// whether `produce` or `consume` made it; later items may already have run.
pub fn try_for_each_ordered<T, R, E>(
    items: impl IntoIterator<Item = T>,
    ahead: usize,
    produce: impl Fn(T) -> Result<R, E> + Send + Sync,
    mut consume: impl FnMut(R) -> Result<(), E>,
) -> Result<(), E>
where
    T: Send,
    R: Send,
    E: Send,
{
    #[cfg(feature = "parallel")]
    if let Some(pool) = armed_outside() {
        use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
        use std::sync::atomic::{AtomicBool, Ordering};
        let stop = AtomicBool::new(false);
        let (produce, stop) = (&produce, &stop);
        let (sender, receiver) = std::sync::mpsc::channel();
        return pool.in_place_scope(|scope| {
            let mut items = items.into_iter().enumerate();
            let spawn = |(index, item): (usize, T)| {
                let sender = sender.clone();
                scope.spawn(move |_| {
                    // Every job sends once, so the caller never waits on a
                    // job that panicked or was skipped.
                    let result = (!stop.load(Ordering::Relaxed))
                        .then(|| catch_unwind(AssertUnwindSafe(|| produce(item))));
                    let _ = sender.send((index, result));
                });
            };
            let mut pending = 0;
            for item in items.by_ref().take(ahead.max(1)) {
                spawn(item);
                pending += 1;
            }
            let mut ready = std::collections::BTreeMap::new();
            let mut next = 0;
            let outcome = loop {
                if pending == 0 {
                    break Ok(());
                }
                let Some(result) = ready.remove(&next) else {
                    let (index, result) = receiver.recv().expect("a job sends before it ends");
                    ready.insert(index, result);
                    continue;
                };
                pending -= 1;
                next += 1;
                let consumed = match result.expect("no job is skipped before the stop") {
                    Ok(produced) => produced.and_then(&mut consume),
                    Err(panic) => {
                        stop.store(true, Ordering::Relaxed);
                        resume_unwind(panic);
                    }
                };
                if consumed.is_err() {
                    break consumed;
                }
                if let Some(item) = items.next() {
                    spawn(item);
                    pending += 1;
                }
            };
            stop.store(true, Ordering::Relaxed);
            outcome
        });
    }
    items
        .into_iter()
        .try_for_each(|item| produce(item).and_then(&mut consume))
}
