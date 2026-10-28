// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One CPU policy and allocation admission owner for the complete publication.

use anyhow::{Context, Result, ensure};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::{Condvar, Mutex},
};

/// Detected process resources, before starting worker threads.
#[derive(Debug)]
pub struct Policy {
    pub detected_cpus: usize,
    pub workers: usize,
    pub memory_bytes: u64,
    pub allocator_arenas: Option<usize>,
}

fn ancestors() -> Vec<PathBuf> {
    let mut result = Vec::new();
    if let Ok(text) = fs::read_to_string("/proc/self/cgroup")
        && let Some(group) = text.lines().find_map(|line| line.strip_prefix("0::"))
    {
        let root = Path::new("/sys/fs/cgroup");
        let group = root.join(group.trim_start_matches('/'));
        for dir in group.ancestors().take_while(|p| p.starts_with(root)) {
            result.push(dir.to_path_buf());
        }
    }
    result
}

fn quota_cpus(text: &str) -> Result<Option<usize>> {
    let values = text.split_whitespace().collect::<Vec<_>>();
    ensure!(values.len() == 2, "invalid cgroup CPU quota");
    if values[0] == "max" {
        return Ok(None);
    }
    let quota: u64 = values[0].parse()?;
    let period: u64 = values[1].parse()?;
    ensure!(quota > 0 && period > 0, "invalid cgroup CPU quota");
    Ok(Some(usize::try_from(quota.div_ceil(period))?.max(1)))
}

impl Policy {
    /// Respect affinity and ancestor quotas, retaining the configured pool size
    /// when memory admission must limit the number of simultaneous large jobs.
    pub fn detect(requested: &str) -> Result<Self> {
        let mut cpus = std::thread::available_parallelism()?.get();
        let memory_text = fs::read_to_string("/proc/meminfo")
            .context("memory limit detection requires /proc/meminfo")?;
        let mut memory = memory_text
            .lines()
            .find_map(|line| line.strip_prefix("MemAvailable:"))
            .context("MemAvailable is missing")?
            .split_whitespace()
            .next()
            .context("MemAvailable is empty")?
            .parse::<u64>()?
            .checked_mul(1024)
            .context("memory count overflow")?;
        for dir in ancestors() {
            if let Ok(text) = fs::read_to_string(dir.join("cpu.max"))
                && let Some(limit) = quota_cpus(&text)?
            {
                cpus = cpus.min(limit);
            }
            if let Ok(text) = fs::read_to_string(dir.join("memory.max"))
                && text.trim() != "max"
            {
                let maximum: u64 = text.trim().parse()?;
                let current: u64 = fs::read_to_string(dir.join("memory.current"))?
                    .trim()
                    .parse()?;
                memory = memory.min(maximum.saturating_sub(current));
            }
        }
        let mut address_space = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        ensure!(
            unsafe { libc::getrlimit(libc::RLIMIT_AS, &mut address_space) } == 0,
            "could not inspect process memory limit"
        );
        if address_space.rlim_cur != libc::RLIM_INFINITY {
            memory = memory.min(address_space.rlim_cur);
        }
        let workers = if requested == "auto" {
            cpus
        } else {
            requested
                .parse()
                .context("workers must be auto or a positive integer")?
        };
        ensure!(workers > 0, "workers must be auto or a positive integer");
        // Leave room for catalogs, allocator bookkeeping and runtime assets.
        let reserved = (memory / 8).min(1024 * 1024 * 1024);
        memory = memory.saturating_sub(reserved);
        ensure!(memory > 0, "no memory available for publication jobs");
        #[cfg(all(target_os = "linux", target_env = "gnu"))]
        let arenas = {
            let arenas =
                i32::try_from(workers).context("worker count exceeds allocator arena range")?;
            // The host can export MALLOC_ARENA_MAX=2. Override it before the
            // shared pool starts so large independent jobs do not contend.
            ensure!(
                unsafe { libc::mallopt(libc::M_ARENA_MAX, arenas) } != 0,
                "could not configure allocator arenas"
            );
            Some(workers)
        };
        #[cfg(not(all(target_os = "linux", target_env = "gnu")))]
        let arenas = None;
        Ok(Self {
            detected_cpus: cpus,
            workers,
            memory_bytes: memory,
            allocator_arenas: arenas,
        })
    }

    /// Create the only Rayon pool used by the publication pipeline.
    pub fn pool(&self) -> Result<rayon::ThreadPool> {
        Ok(rayon::ThreadPoolBuilder::new()
            .num_threads(self.workers)
            .build()?)
    }
}

/// Admission limits allocations, never worker configuration. Callers reserve
/// before loading a complete independent job and release before queuing another.
pub struct MemoryBudget {
    limit: u64,
    used: Mutex<u64>,
    released: Condvar,
}

/// A reservation releases even when a parse/validation/write fails.
pub struct Reservation<'a> {
    budget: &'a MemoryBudget,
    bytes: u64,
}

impl MemoryBudget {
    pub fn new(limit: u64) -> Self {
        Self {
            limit,
            used: Mutex::new(0),
            released: Condvar::new(),
        }
    }

    pub fn reserve(&self, bytes: u64) -> Result<Reservation<'_>> {
        ensure!(
            bytes <= self.limit,
            "publication job requires {bytes} bytes but only {} bytes are available",
            self.limit
        );
        let mut used = self
            .used
            .lock()
            .map_err(|_| anyhow::anyhow!("memory admission lock poisoned"))?;
        while bytes > self.limit - *used {
            used = self
                .released
                .wait(used)
                .map_err(|_| anyhow::anyhow!("memory admission lock poisoned"))?;
        }
        *used += bytes;
        Ok(Reservation {
            budget: self,
            bytes,
        })
    }
}

impl Drop for Reservation<'_> {
    fn drop(&mut self) {
        if let Ok(mut used) = self.budget.used.lock() {
            *used -= self.bytes;
            self.budget.released.notify_all();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quotas_and_admission_do_not_silently_change_worker_policy() -> Result<()> {
        assert_eq!(quota_cpus("max 100000")?, None);
        assert_eq!(quota_cpus("150000 100000")?, Some(2));
        assert!(quota_cpus("1000 0").is_err());
        let budget = MemoryBudget::new(100);
        assert!(budget.reserve(101).is_err());
        let reservation = budget.reserve(100)?;
        assert_eq!(*budget.used.lock().unwrap(), 100);
        drop(reservation);
        assert_eq!(*budget.used.lock().unwrap(), 0);
        Ok(())
    }

    #[test]
    fn reservation_release_notifies_waiters_and_allows_reuse() -> Result<()> {
        for limit in [1, 100, u64::MAX] {
            let budget = MemoryBudget::new(limit);
            let reservation = budget.reserve(limit)?;
            let timeout = std::time::Duration::from_secs(5);
            let (waiting, ready) = std::sync::mpsc::sync_channel(1);
            std::thread::scope(|scope| -> Result<()> {
                let waiter = scope.spawn(|| -> Result<()> {
                    let mut used = budget.used.lock().unwrap();
                    waiting.send(()).unwrap();
                    // Holding used until wait registers prevents a lost notification.
                    let deadline = std::time::Instant::now() + timeout;
                    while *used != 0 {
                        let remaining =
                            deadline.saturating_duration_since(std::time::Instant::now());
                        let (next, outcome) =
                            budget.released.wait_timeout(used, remaining).unwrap();
                        used = next;
                        assert!(!outcome.timed_out(), "reservation release did not notify");
                    }
                    assert_eq!(*used, 0);
                    drop(used);
                    let reused = budget.reserve(limit)?;
                    assert_eq!(*budget.used.lock().unwrap(), limit);
                    drop(reused);
                    Ok(())
                });
                let started = ready.recv_timeout(timeout);
                drop(reservation);
                let result = waiter.join().expect("reservation waiter panicked");
                assert!(started.is_ok(), "reservation waiter did not start");
                result
            })?;
            assert_eq!(*budget.used.lock().unwrap(), 0);
        }
        Ok(())
    }
}
