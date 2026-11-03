// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Directory leases and durable journals compatible with retained Python writers.
use crate::{
    codec::json_bytes,
    source::{no_links, relative},
};
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::{
    ffi::CString,
    fs::{self, File, OpenOptions},
    io::Write,
    os::fd::AsRawFd,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

const JOURNAL: &str = ".conformance-transaction.json";

pub struct Lease {
    _file: File,
}
impl Lease {
    fn lock(parent: &Path, operation: i32) -> Result<Self> {
        no_links(parent)?;
        let file = File::open(parent)?;
        ensure!(file.metadata()?.is_dir(), "lease parent is not a directory");
        ensure!(
            unsafe { libc::flock(file.as_raw_fd(), operation) } == 0,
            "directory lease failed: {}",
            std::io::Error::last_os_error()
        );
        Ok(Self { _file: file })
    }

    pub fn acquire(parent: &Path) -> Result<Self> {
        let lease = Self::lock(parent, libc::LOCK_EX)?;
        recover(parent)?;
        Ok(lease)
    }

    /// Hold stable source bytes alongside other readers. Recovery always takes
    /// a fresh exclusive lease; never upgrade a held shared lease in place.
    pub fn read(parent: &Path) -> Result<Self> {
        loop {
            let lease = Self::lock(parent, libc::LOCK_SH)?;
            let journal = parent.join(JOURNAL);
            no_links(&journal)?;
            if !journal.exists() {
                return Ok(lease);
            }
            drop(lease);
            drop(Self::acquire(parent)?);
        }
    }
}

pub fn sync_tree(path: &Path) -> Result<()> {
    no_links(path)?;
    let metadata = fs::symlink_metadata(path)?;
    if metadata.is_dir() {
        for entry in fs::read_dir(path)? {
            sync_tree(&entry?.path())?;
        }
    } else {
        ensure!(metadata.is_file(), "nonregular transaction payload");
    }
    File::open(path)?.sync_all()?;
    Ok(())
}

pub fn unique(parent: &Path, prefix: &str) -> Result<PathBuf> {
    let path = parent.join(format!(
        "{prefix}{}-{}",
        std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
    ));
    fs::create_dir(&path)?;
    Ok(path)
}

fn is_generated_rust_transaction_dir(name: &str, prefix: &str) -> bool {
    let Some(suffix) = name.strip_prefix(prefix) else {
        return false;
    };
    suffix.split_once('-').is_some_and(|(pid, timestamp)| {
        pid.parse::<u32>().is_ok_and(|pid| pid > 0)
            && timestamp.len() == 19
            && timestamp.bytes().all(|byte| byte.is_ascii_digit())
    })
}

fn is_generated_backup_dir(name: &str) -> bool {
    let prefix = ".conformance-transaction-backup-";
    let Some(suffix) = name.strip_prefix(prefix) else {
        return false;
    };
    is_generated_rust_transaction_dir(name, prefix)
        || (suffix.len() == 8
            && suffix
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_'))
}

fn journal(parent: &Path, value: &Value) -> Result<()> {
    let temporary = parent.join(format!("{JOURNAL}.{}.tmp", std::process::id()));
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary)?;
    file.write_all(&json_bytes(value)?)?;
    file.sync_all()?;
    fs::rename(&temporary, parent.join(JOURNAL))?;
    File::open(parent)?.sync_all()?;
    Ok(())
}

fn path(parent: &Path, value: &Value) -> Result<PathBuf> {
    let value = value.as_str().context("missing transaction path")?;
    relative(value)?;
    let path = parent.join(value);
    no_links(&path)?;
    Ok(path)
}

fn remove(path: &Path) -> Result<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            ensure!(!metadata.file_type().is_symlink(), "transaction symlink");
            if metadata.is_dir() {
                fs::remove_dir_all(path)?;
            } else {
                fs::remove_file(path)?;
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => (),
        Err(error) => return Err(error.into()),
    }
    Ok(())
}

fn exchange(left: &Path, right: &Path) -> Result<()> {
    let left = CString::new(left.as_os_str().as_encoded_bytes())?;
    let right = CString::new(right.as_os_str().as_encoded_bytes())?;
    ensure!(
        unsafe {
            libc::renameat2(
                libc::AT_FDCWD,
                left.as_ptr(),
                libc::AT_FDCWD,
                right.as_ptr(),
                libc::RENAME_EXCHANGE,
            )
        } == 0,
        "atomic directory exchange failed: {}",
        std::io::Error::last_os_error()
    );
    Ok(())
}

fn backup_existing(source: &Path, destination: &Path) -> Result<()> {
    no_links(source)?;
    let metadata = fs::symlink_metadata(source)?;
    if metadata.is_dir() {
        fs::create_dir(destination)?;
        for entry in fs::read_dir(source)? {
            let entry = entry?;
            backup_existing(&entry.path(), &destination.join(entry.file_name()))?;
        }
    } else {
        ensure!(metadata.is_file(), "nonregular existing publication");
        fs::hard_link(source, destination)?;
    }
    Ok(())
}

fn restore(backup: &Path, destination: &Path, candidate: &Path) -> Result<()> {
    // Keep the rollback snapshot immutable so recovery can itself be interrupted.
    remove(candidate)?;
    backup_existing(backup, candidate)?;
    sync_tree(candidate)?;
    if destination.exists() {
        exchange(candidate, destination)?;
    } else {
        fs::rename(candidate, destination)?;
    }
    File::open(
        destination
            .parent()
            .context("missing rollback destination parent")?,
    )?
    .sync_all()?;
    remove(candidate)
}

pub fn recover(parent: &Path) -> Result<()> {
    recover_preserving(parent, &[])
}

fn cleanup_orphan_transactions(parent: &Path, active_candidates: &[PathBuf]) -> Result<()> {
    let mut changed = false;
    for entry in fs::read_dir(parent)? {
        let entry = entry?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        let is_internal = is_generated_rust_transaction_dir(name, ".native-publication-stage-")
            || is_generated_backup_dir(name);
        let is_active = active_candidates.iter().any(|candidate| {
            candidate
                .strip_prefix(parent)
                .ok()
                .and_then(|relative| relative.components().next())
                .is_some_and(|component| component.as_os_str() == name)
        });
        if !is_internal || is_active {
            continue;
        }
        let kind = entry.file_type()?;
        ensure!(
            kind.is_dir(),
            "transaction orphan is not a directory: {}",
            entry.path().display()
        );
        fs::remove_dir_all(entry.path())?;
        changed = true;
    }
    if changed {
        File::open(parent)?.sync_all()?;
    }
    Ok(())
}

fn recover_preserving(parent: &Path, active_candidates: &[PathBuf]) -> Result<()> {
    let journal_path = parent.join(JOURNAL);
    no_links(&journal_path)?;
    if !journal_path.exists() {
        return cleanup_orphan_transactions(parent, active_candidates);
    }
    let value: Value = serde_json::from_slice(&fs::read(&journal_path)?)?;
    ensure!(
        value["schema_version"] == 1
            && [Some("prepared"), Some("committed")].contains(&value["phase"].as_str()),
        "invalid transaction journal"
    );
    let backup_root = path(parent, &value["backup_root"])?;
    let backup_name = backup_root.file_name().and_then(|name| name.to_str());
    ensure!(
        backup_root.parent() == Some(parent) && backup_name.is_some_and(is_generated_backup_dir),
        "invalid backup root"
    );
    let states = value["states"]
        .as_array()
        .context("invalid transaction states")?;
    let mut destinations = Vec::new();
    for state in states {
        let destination = path(parent, &state["destination"])?;
        let candidate = path(parent, &state["candidate"])?;
        let backup = path(parent, &state["backup"])?;
        let candidate_root = candidate
            .strip_prefix(parent)?
            .components()
            .next()
            .and_then(|component| component.as_os_str().to_str());
        ensure!(
            backup.parent() == Some(backup_root.as_path())
                && candidate_root.is_some_and(|name| {
                    !name.starts_with(".native-publication-stage-")
                        || is_generated_rust_transaction_dir(name, ".native-publication-stage-")
                })
                && destination != journal_path
                && !destination.starts_with(&backup_root)
                && !candidate.starts_with(&destination)
                && !destination.starts_with(&candidate),
            "invalid transaction paths"
        );
        ensure!(
            destinations
                .iter()
                .all(|p: &PathBuf| !p.starts_with(&destination) && !destination.starts_with(p)),
            "overlapping transaction destinations"
        );
        destinations.push(destination);
        ensure!(state["had_original"].is_boolean(), "invalid original state");
    }
    if value["phase"] == "prepared" {
        for (index, state) in states.iter().enumerate().rev() {
            let destination = path(parent, &state["destination"])?;
            let backup = path(parent, &state["backup"])?;
            if state["had_original"] == true {
                if backup.exists() {
                    restore(
                        &backup,
                        &destination,
                        &backup_root.join(format!(".restore-{index:02}")),
                    )?;
                } else {
                    ensure!(destination.exists(), "transaction original unavailable");
                }
            } else {
                ensure!(!backup.exists(), "unexpected transaction backup");
                if state["retain_on_rollback"] != true {
                    remove(&destination)?;
                }
            }
            File::open(destination.parent().context("missing destination parent")?)?.sync_all()?;
        }
    }
    for state in states {
        let candidate = path(parent, &state["candidate"])?;
        let first = candidate
            .strip_prefix(parent)?
            .components()
            .next()
            .context("empty candidate")?;
        if first.as_os_str().to_str().is_some_and(|name| {
            is_generated_rust_transaction_dir(name, ".native-publication-stage-")
        }) {
            remove(&parent.join(first.as_os_str()))?;
        }
    }
    remove(&backup_root)?;
    fs::remove_file(journal_path)?;
    File::open(parent)?.sync_all()?;
    cleanup_orphan_transactions(parent, active_candidates)
}

pub fn publish(parent: &Path, pairs: &[(PathBuf, PathBuf)]) -> Result<()> {
    publish_checked(parent, pairs, &[], || Ok(()))
}

pub fn publish_checked(
    parent: &Path,
    pairs: &[(PathBuf, PathBuf)],
    retain_on_rollback: &[PathBuf],
    before_commit: impl FnOnce() -> Result<()>,
) -> Result<()> {
    let active_candidates = pairs
        .iter()
        .map(|(candidate, _)| candidate.clone())
        .collect::<Vec<_>>();
    recover_preserving(parent, &active_candidates)?;
    ensure!(!pairs.is_empty(), "empty publication transaction");
    for (index, (candidate, destination)) in pairs.iter().enumerate() {
        for (other_candidate, other_destination) in &pairs[..index] {
            ensure!(
                !destination.starts_with(other_destination)
                    && !other_destination.starts_with(destination)
                    && !candidate.starts_with(other_candidate)
                    && !other_candidate.starts_with(candidate),
                "overlapping publication paths"
            );
        }
        sync_tree(candidate)?;
    }
    let backup_root = unique(parent, ".conformance-transaction-backup-")?;
    let prepare = (|| -> Result<Vec<Value>> {
        let mut states = Vec::new();
        for (index, (candidate, destination)) in pairs.iter().enumerate() {
            no_links(candidate)?;
            no_links(destination)?;
            ensure!(
                candidate.starts_with(parent) && destination.starts_with(parent),
                "transaction outside parent"
            );
            ensure!(
                !candidate.starts_with(destination) && !destination.starts_with(candidate),
                "overlapping candidate and destination"
            );
            let backup = backup_root.join(format!(
                "{index:02}-{}",
                destination
                    .file_name()
                    .context("invalid destination")?
                    .to_string_lossy()
            ));
            if destination.exists() {
                backup_existing(destination, &backup)?;
            }
            states.push(json!({"candidate":candidate.strip_prefix(parent)?.to_str(),"destination":destination.strip_prefix(parent)?.to_str(),"backup":backup.strip_prefix(parent)?.to_str(),"had_original":destination.exists()}));
        }
        for destination in retain_on_rollback {
            ensure!(
                pairs
                    .iter()
                    .any(|(_, pair_destination)| pair_destination == destination),
                "rollback-retained path is not a transaction destination"
            );
            let state = states
                .iter_mut()
                .find(|state| parent.join(state["destination"].as_str().unwrap()) == *destination)
                .context("missing rollback-retained transaction state")?;
            state["retain_on_rollback"] = json!(true);
        }
        sync_tree(&backup_root)?;
        before_commit()?;
        Ok(states)
    })();
    let states = match prepare {
        Ok(states) => states,
        Err(error) => {
            remove(&backup_root).context("failed to clean incomplete transaction backup")?;
            return Err(error);
        }
    };
    let mut document = json!({"schema_version":1,"phase":"prepared","backup_root":backup_root.strip_prefix(parent)?.to_str(),"states":states});
    journal(parent, &document)?;
    let result = (|| -> Result<()> {
        for state in &states {
            let candidate = path(parent, &state["candidate"])?;
            let destination = path(parent, &state["destination"])?;
            if state["had_original"] == true {
                exchange(&candidate, &destination)?;
            } else {
                fs::rename(&candidate, &destination)?;
            }
            File::open(candidate.parent().context("missing candidate parent")?)?.sync_all()?;
            File::open(destination.parent().context("missing destination parent")?)?.sync_all()?;
        }
        document["phase"] = json!("committed");
        journal(parent, &document)
    })();
    if let Err(error) = result {
        // A visible commit must retain its journal if durability is uncertain.
        let current: Value = serde_json::from_slice(&fs::read(parent.join(JOURNAL))?)?;
        if current["phase"] == "prepared" {
            recover(parent).context("publication rollback failed")?;
        }
        return Err(error);
    }
    recover(parent)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovery_reclaims_unjournaled_publication_orphans() -> Result<()> {
        let parent = tempfile::tempdir()?;
        let stale_stage = parent
            .path()
            .join(".native-publication-stage-123-1700000000000000000");
        let stale_backup = parent
            .path()
            .join(".conformance-transaction-backup-123-1700000000000000000");
        let stale_python_backup = parent
            .path()
            .join(".conformance-transaction-backup-a1b2c3d4");
        let user_stage = parent.path().join(".native-publication-stage-notes");
        let random_suffix_stage = parent.path().join(".native-publication-stage-a1b2c3d4");
        let short_numeric_stage = parent.path().join(".native-publication-stage-1-2");
        let user_backup = parent.path().join(".conformance-transaction-backup-manual");
        let short_numeric_backup = parent.path().join(".conformance-transaction-backup-1-2");
        let active_stage = parent
            .path()
            .join(".native-publication-stage-123-1700000000000000001");
        fs::create_dir(&stale_stage)?;
        fs::write(stale_stage.join("partial-pack.bin"), b"partial")?;
        fs::create_dir(&stale_backup)?;
        fs::write(stale_backup.join("partial-copy.bin"), b"partial")?;
        fs::create_dir(&stale_python_backup)?;
        fs::write(stale_python_backup.join("partial-copy.bin"), b"partial")?;
        fs::create_dir(&user_stage)?;
        fs::write(user_stage.join("keep"), b"user data")?;
        fs::create_dir(&random_suffix_stage)?;
        fs::write(random_suffix_stage.join("keep"), b"user data")?;
        fs::create_dir(&short_numeric_stage)?;
        fs::write(short_numeric_stage.join("keep"), b"user data")?;
        fs::create_dir(&user_backup)?;
        fs::write(user_backup.join("keep"), b"user data")?;
        fs::create_dir(&short_numeric_backup)?;
        fs::write(short_numeric_backup.join("keep"), b"user data")?;
        fs::create_dir(&active_stage)?;
        fs::write(active_stage.join("candidate"), b"active")?;

        recover_preserving(parent.path(), std::slice::from_ref(&active_stage))?;

        assert!(!stale_stage.exists());
        assert!(!stale_backup.exists());
        assert!(!stale_python_backup.exists());
        assert_eq!(fs::read(user_stage.join("keep"))?, b"user data");
        assert_eq!(fs::read(random_suffix_stage.join("keep"))?, b"user data");
        assert_eq!(fs::read(short_numeric_stage.join("keep"))?, b"user data");
        assert_eq!(fs::read(user_backup.join("keep"))?, b"user data");
        assert_eq!(fs::read(short_numeric_backup.join("keep"))?, b"user data");
        assert_eq!(fs::read(active_stage.join("candidate"))?, b"active");

        let lease_orphan = unique(parent.path(), ".native-publication-stage-")?;
        drop(Lease::acquire(parent.path())?);
        assert!(!lease_orphan.exists());
        Ok(())
    }

    #[test]
    fn failed_backup_preparation_removes_partial_backup_tree() -> Result<()> {
        let parent = tempfile::tempdir()?;
        let candidate = unique(parent.path(), ".native-publication-stage-")?;
        fs::write(candidate.join("candidate"), b"new")?;
        let destination = parent.path().join("existing");
        fs::create_dir(&destination)?;
        let outside = parent.path().join("outside");
        fs::write(&outside, b"outside")?;
        std::os::unix::fs::symlink(&outside, destination.join("linked"))?;

        assert!(
            publish_checked(
                parent.path(),
                &[(candidate, destination.clone())],
                &[],
                || Ok(())
            )
            .is_err()
        );

        let orphans = fs::read_dir(parent.path())?
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry
                    .file_name()
                    .to_str()
                    .is_some_and(|name| name.starts_with(".conformance-transaction-backup-"))
            })
            .count();
        assert_eq!(orphans, 0);
        assert!(destination.join("linked").is_symlink());
        Ok(())
    }

    #[test]
    fn recovery_rejects_similarly_named_user_directories() -> Result<()> {
        for (backup_root_is_user_dir, candidate_name) in [
            (true, None),
            (false, Some(".native-publication-stage-notes")),
            (false, Some(".native-publication-stage-1-2")),
        ] {
            let parent = tempfile::tempdir()?;
            let backup_root = if backup_root_is_user_dir {
                parent.path().join(".conformance-transaction-backup-manual")
            } else {
                unique(parent.path(), ".conformance-transaction-backup-")?
            };
            let candidate_root = match candidate_name {
                Some(name) => parent.path().join(name),
                None => unique(parent.path(), ".native-publication-stage-")?,
            };
            fs::create_dir_all(&backup_root)?;
            fs::write(backup_root.join("keep"), b"user backup")?;
            fs::create_dir_all(&candidate_root)?;
            fs::write(candidate_root.join("keep"), b"user stage")?;
            let backup = backup_root.join("00-data");
            let journal_value = json!({
                "schema_version": 1,
                "phase": "committed",
                "backup_root": backup_root.strip_prefix(parent.path())?.to_str(),
                "states": [{
                    "candidate": candidate_root.strip_prefix(parent.path())?.to_str(),
                    "destination": "destination",
                    "backup": backup.strip_prefix(parent.path())?.to_str(),
                    "had_original": false
                }]
            });
            journal(parent.path(), &journal_value)?;

            assert!(recover(parent.path()).is_err());
            assert_eq!(fs::read(backup_root.join("keep"))?, b"user backup");
            assert_eq!(fs::read(candidate_root.join("keep"))?, b"user stage");
        }
        Ok(())
    }

    #[test]
    fn source_readers_share_ownership_and_exclude_writers() -> Result<()> {
        let parent = tempfile::tempdir()?;
        let lease = Lease::read(parent.path())?;
        let peer = File::open(parent.path())?;
        assert_eq!(
            unsafe { libc::flock(peer.as_raw_fd(), libc::LOCK_SH | libc::LOCK_NB) },
            0
        );
        assert_ne!(
            unsafe { libc::flock(peer.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
            0
        );
        drop(lease);
        assert_eq!(
            unsafe { libc::flock(peer.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
            0
        );
        Ok(())
    }

    #[test]
    fn interrupted_partial_data_recovery_is_repeatable() -> Result<()> {
        let parent = unique(&std::env::temp_dir(), "native-partial-recovery-")?;
        let result = (|| -> Result<()> {
            let stage = unique(&parent, ".native-publication-stage-")?;
            let backup_root = unique(&parent, ".conformance-transaction-backup-")?;
            let mut states = Vec::new();
            for index in 0..3 {
                let destination = parent.join(format!("data-{index}"));
                let candidate = stage.join(format!("data-{index}"));
                let backup = backup_root.join(format!("{index:02}-data"));
                fs::write(&candidate, b"new")?;
                if index != 0 {
                    fs::write(&destination, b"old")?;
                    backup_existing(&destination, &backup)?;
                    exchange(&candidate, &destination)?;
                } else {
                    fs::rename(&candidate, &destination)?;
                }
                states.push(json!({"candidate":candidate.strip_prefix(&parent)?.to_str(),"destination":destination.strip_prefix(&parent)?.to_str(),"backup":backup.strip_prefix(&parent)?.to_str(),"had_original":index != 0}));
            }
            journal(
                &parent,
                &json!({"schema_version":1,"phase":"prepared","backup_root":backup_root.file_name().unwrap().to_str(),"states":states}),
            )?;
            // Recovery restores in reverse order. Interrupt twice after its exchange,
            // leaving one restored output and the other outputs still installed.
            let saved = backup_root.join("02-data");
            let destination = parent.join("data-2");
            let disposable = backup_root.join(".restore-02");
            for _ in 0..2 {
                remove(&disposable)?;
                backup_existing(&saved, &disposable)?;
                sync_tree(&disposable)?;
                exchange(&disposable, &destination)?;
                assert_eq!(fs::read(&destination)?, b"old");
                assert_eq!(fs::read(&saved)?, b"old");
                assert_eq!(fs::read(parent.join("data-1"))?, b"new");
            }
            recover(&parent)?;
            recover(&parent)?;
            assert!(!parent.join("data-0").exists());
            for index in 1..3 {
                assert_eq!(fs::read(parent.join(format!("data-{index}")))?, b"old");
            }
            assert!(!stage.exists());
            assert!(!backup_root.exists());
            assert!(!parent.join(JOURNAL).exists());
            Ok(())
        })();
        fs::remove_dir_all(parent)?;
        result
    }

    #[test]
    fn exchange_keeps_complete_output_visible_and_prepared_journal_rolls_back() -> Result<()> {
        let parent = unique(&std::env::temp_dir(), "native-exchange-test-")?;
        let result = (|| -> Result<()> {
            let destination = parent.join("site");
            fs::create_dir(&destination)?;
            fs::write(destination.join("value"), b"old")?;
            let candidate = unique(&parent, ".native-publication-stage-")?;
            fs::write(candidate.join("value"), b"new")?;
            let backup_root = unique(&parent, ".conformance-transaction-backup-")?;
            let saved = backup_root.join("00-site");
            backup_existing(&destination, &saved)?;
            sync_tree(&backup_root)?;
            let document = json!({"schema_version":1,"phase":"prepared","backup_root":backup_root.file_name().unwrap().to_str(),"states":[{"candidate":candidate.file_name().unwrap().to_str(),"destination":"site","backup":saved.strip_prefix(&parent)?.to_str(),"had_original":true}]});
            journal(&parent, &document)?;
            exchange(&candidate, &destination)?;
            // This is the persistent state at an interruption after the install syscall.
            assert_eq!(fs::read(destination.join("value"))?, b"new");
            assert_eq!(fs::read(saved.join("value"))?, b"old");
            let interrupted = backup_root.join(".restore-00");
            backup_existing(&saved, &interrupted)?;
            exchange(&interrupted, &destination)?;
            assert_eq!(fs::read(destination.join("value"))?, b"old");
            assert_eq!(fs::read(saved.join("value"))?, b"old");
            recover(&parent)?;
            assert_eq!(fs::read(destination.join("value"))?, b"old");
            assert!(!candidate.exists());
            let rejected = unique(&parent, ".native-publication-stage-")?;
            fs::write(rejected.join("value"), b"rejected")?;
            assert!(
                publish_checked(
                    &parent,
                    &[(rejected, destination.clone())],
                    &[],
                    || anyhow::bail!("source changed after staging")
                )
                .is_err()
            );
            assert_eq!(fs::read(destination.join("value"))?, b"old");
            let names = fs::read_dir(&parent)?
                .map(|entry| entry.map(|entry| entry.file_name()))
                .collect::<std::io::Result<Vec<_>>>()?;
            assert!(!names.iter().any(|name| {
                name.to_str()
                    .is_some_and(|name| name.starts_with(".conformance-transaction-backup-"))
            }));
            Ok(())
        })();
        fs::remove_dir_all(parent)?;
        result
    }

    #[test]
    fn rollback_retains_a_generation_already_referenced_by_a_visible_catalog() -> Result<()> {
        let parent = unique(&std::env::temp_dir(), "native-retained-generation-")?;
        let result = (|| -> Result<()> {
            let generation_name = "viewer-data/generations/immutable";
            let generation = parent.join(generation_name);
            fs::create_dir_all(generation.parent().context("generation has no parent")?)?;
            let catalog = parent.join("catalog.json");
            fs::write(&catalog, b"old catalog")?;

            let backup_root = unique(&parent, ".conformance-transaction-backup-")?;
            let backup_catalog = backup_root.join("01-catalog.json");
            backup_existing(&catalog, &backup_catalog)?;
            let stage = unique(&parent, ".native-publication-stage-")?;
            let generation_candidate = stage.join("generation");
            fs::create_dir(&generation_candidate)?;
            fs::write(
                generation_candidate.join("capture.json.gz"),
                b"new generation",
            )?;
            let catalog_candidate = stage.join("catalog.json");
            fs::write(&catalog_candidate, b"new catalog -> immutable generation")?;
            let states = json!([
                {"candidate":generation_candidate.strip_prefix(&parent)?.to_str(),"destination":generation_name,"backup":backup_root.join("00-generation").strip_prefix(&parent)?.to_str(),"had_original":false,"retain_on_rollback":true},
                {"candidate":catalog_candidate.strip_prefix(&parent)?.to_str(),"destination":"catalog.json","backup":backup_catalog.strip_prefix(&parent)?.to_str(),"had_original":true}
            ]);
            fs::rename(&generation_candidate, &generation)?;
            exchange(&catalog_candidate, &catalog)?;
            journal(
                &parent,
                &json!({"schema_version":1,"phase":"prepared","backup_root":backup_root.strip_prefix(&parent)?.to_str(),"states":states}),
            )?;

            assert_eq!(fs::read(&catalog)?, b"new catalog -> immutable generation");
            assert_eq!(
                fs::read(generation.join("capture.json.gz"))?,
                b"new generation"
            );
            recover(&parent)?;
            assert_eq!(fs::read(&catalog)?, b"old catalog");
            assert_eq!(
                fs::read(generation.join("capture.json.gz"))?,
                b"new generation"
            );
            assert!(!parent.join(JOURNAL).exists());
            Ok(())
        })();
        fs::remove_dir_all(parent)?;
        result
    }
}
