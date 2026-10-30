// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::{Context, Result};
use openai_harmony::{HarmonyEncoding, HarmonyEncodingName, load_harmony_encoding};
use sha2::{Digest, Sha256};

#[cfg(test)]
const VOCAB_CACHE_KEY: &str = "fb374d419588a4632f3f557e76b4b70aebbca790";
const VOCAB_SHA256: &str = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d";

static HARMONY_GPT_OSS_ENCODING: OnceLock<Result<HarmonyEncoding>> = OnceLock::new();

/// Loads the GPT-OSS Harmony encoding once per process, coordinated across processes sharing its cache.
pub fn harmony_gpt_oss_encoding() -> &'static Result<HarmonyEncoding> {
    HARMONY_GPT_OSS_ENCODING.get_or_init(|| {
        // Harmony's blocking HTTP client creates and drops a Tokio runtime, so load it outside async runtimes.
        std::thread::spawn(|| {
            let cache_dir = resolve_cache_dir();
            with_cache_lock_at(&cache_dir, || {
                load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss)
                    .context("loading GPT-OSS Harmony encoding")
            })
        })
        .join()
        .unwrap_or_else(|_| Err(anyhow::anyhow!("Harmony encoding loader thread panicked")))
    })
}

fn resolve_cache_dir() -> PathBuf {
    std::env::var("TIKTOKEN_RS_CACHE_DIR")
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("tiktoken-rs-cache"))
}

fn with_cache_lock_at<T>(cache_dir: &Path, load: impl FnOnce() -> Result<T>) -> Result<T> {
    with_cache_lock_at_using(cache_dir, VOCAB_SHA256, load, |lock_path| {
        open_lock_file(lock_path)
    })
}

fn with_cache_lock_at_using<T>(
    cache_dir: &Path,
    expected_vocab_sha256: &str,
    load: impl FnOnce() -> Result<T>,
    open_lock_file: impl FnOnce(&Path) -> std::io::Result<std::fs::File>,
) -> Result<T> {
    fs::create_dir_all(cache_dir)
        .with_context(|| format!("creating Harmony vocabulary cache directory {cache_dir:?}"))?;
    let lock_path = cache_dir.join("harmony-vocab.lock");
    let lock_file = match open_lock_file(&lock_path) {
        Ok(lock_file) => lock_file,
        Err(error)
            if matches!(
                error.kind(),
                std::io::ErrorKind::PermissionDenied | std::io::ErrorKind::ReadOnlyFilesystem
            ) =>
        {
            if lock_path.exists() {
                // Existing read-only caches can still share an already-created lock file.
                OpenOptions::new()
                    .read(true)
                    .open(&lock_path)
                    .with_context(|| {
                        format!("opening Harmony vocabulary lock file {lock_path:?}")
                    })?
            } else {
                // A verified cached vocabulary can serve as a stable lock inode on read-only mounts.
                open_verified_vocab_lock(cache_dir, expected_vocab_sha256)?
            }
        }
        Err(error) => {
            return Err(error)
                .with_context(|| format!("opening Harmony vocabulary lock file {lock_path:?}"));
        }
    };
    lock_file
        .lock()
        .with_context(|| format!("locking Harmony vocabulary cache {cache_dir:?}"))?;

    let result = load();
    drop(lock_file);
    result
}

fn open_lock_file(lock_path: &Path) -> std::io::Result<File> {
    OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(lock_path)
}

fn open_verified_vocab_lock(cache_dir: &Path, expected_sha256: &str) -> Result<File> {
    let explicit_vocab_path = std::env::var_os("TIKTOKEN_ENCODINGS_BASE")
        .map(|base| PathBuf::from(base).join("o200k_base.tiktoken"));
    let vocab_paths = if let Some(vocab_path) = explicit_vocab_path {
        vec![vocab_path]
    } else {
        let entries = fs::read_dir(cache_dir)
            .with_context(|| format!("listing Harmony vocabulary cache directory {cache_dir:?}"))?
            .collect::<std::io::Result<Vec<_>>>()?;
        let mut paths = Vec::new();
        for entry in entries {
            if entry.file_type()?.is_file() {
                paths.push(entry.path());
            }
        }
        paths.sort();
        paths
    };

    for vocab_path in vocab_paths {
        let Ok(mut vocab_file) = File::open(&vocab_path) else {
            continue;
        };
        if !matches_vocab_checksum(&mut vocab_file, expected_sha256)? {
            continue;
        }
        vocab_file
            .lock()
            .with_context(|| format!("locking Harmony vocabulary cache file {vocab_path:?}"))?;
        if matches_vocab_checksum(&mut vocab_file, expected_sha256)? {
            return Ok(vocab_file);
        }
    }

    anyhow::bail!(
        "no checksum-verified Harmony vocabulary file was found in cache directory {cache_dir:?}"
    )
}

fn matches_vocab_checksum(vocab_file: &mut File, expected_sha256: &str) -> Result<bool> {
    let mut bytes = Vec::new();
    vocab_file.seek(SeekFrom::Start(0))?;
    vocab_file.read_to_end(&mut bytes)?;
    Ok(format!("{:x}", Sha256::digest(&bytes)) == expected_sha256)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io;
    use std::path::{Path, PathBuf};
    use std::process::{Child, Command, ExitStatus, Stdio};
    use std::thread;
    use std::time::{Duration, Instant};

    use anyhow::Result;

    use super::{VOCAB_CACHE_KEY, with_cache_lock_at, with_cache_lock_at_using};

    const TEST_VOCAB_SHA256: &str =
        "5d459a81205fa7ba34f57d3af8e41660c76527de1759589b487b96491f3d3793";

    struct ChildGuard(Child);

    impl ChildGuard {
        fn kill(&mut self) -> io::Result<()> {
            self.0.kill()
        }

        fn try_wait(&mut self) -> io::Result<Option<ExitStatus>> {
            self.0.try_wait()
        }

        fn wait(&mut self) -> io::Result<ExitStatus> {
            self.0.wait()
        }
    }

    impl Drop for ChildGuard {
        fn drop(&mut self) {
            if matches!(self.0.try_wait(), Ok(None)) {
                let _ = self.0.kill();
                let _ = self.0.wait();
            }
        }
    }

    fn wait_for_file(path: &Path) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while !path.exists() {
            assert!(Instant::now() < deadline, "timed out waiting for {path:?}");
            thread::yield_now();
        }
    }

    fn assert_file_stays_absent(path: &Path) {
        let deadline = Instant::now() + Duration::from_millis(200);
        while Instant::now() < deadline {
            assert!(
                !path.exists(),
                "unexpectedly entered the held critical section"
            );
            thread::yield_now();
        }
    }

    fn spawn_helper(
        mode: &str,
        cache_dir: &PathBuf,
        marker_dir: &PathBuf,
        read_only_fs: bool,
    ) -> ChildGuard {
        ChildGuard(
            Command::new(std::env::current_exe().unwrap())
                .arg("--exact")
                .arg("tests::subprocess_lock_helper")
                .arg("--nocapture")
                .env("DYNAMO_HARMONY_LOCK_TEST_MODE", mode)
                .env("DYNAMO_HARMONY_LOCK_TEST_CACHE", cache_dir)
                .env("DYNAMO_HARMONY_LOCK_TEST_MARKERS", marker_dir)
                .env(
                    "DYNAMO_HARMONY_LOCK_TEST_READ_ONLY_FS",
                    read_only_fs.to_string(),
                )
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .unwrap(),
        )
    }

    #[test]
    fn subprocess_lock_helper() {
        let Ok(mode) = std::env::var("DYNAMO_HARMONY_LOCK_TEST_MODE") else {
            return;
        };
        let cache_dir = PathBuf::from(std::env::var_os("DYNAMO_HARMONY_LOCK_TEST_CACHE").unwrap());
        let marker_dir =
            PathBuf::from(std::env::var_os("DYNAMO_HARMONY_LOCK_TEST_MARKERS").unwrap());
        let name = mode.replace(':', "-");
        match mode.as_str() {
            "hold" | "waiter" | "other-cache" => {
                if mode == "waiter" {
                    fs::write(marker_dir.join("waiter.attempting"), b"attempting").unwrap();
                }
                let load = || {
                    fs::write(marker_dir.join(format!("{name}.entered")), b"entered")?;
                    if mode == "hold" {
                        let release = marker_dir.join("hold.release");
                        let deadline = Instant::now() + Duration::from_secs(10);
                        while !release.exists() {
                            assert!(Instant::now() < deadline, "holder release timed out");
                            thread::yield_now();
                        }
                    }
                    Ok(())
                };
                let result = if std::env::var("DYNAMO_HARMONY_LOCK_TEST_READ_ONLY_FS")
                    .is_ok_and(|value| value == "true")
                {
                    super::with_cache_lock_at_using(&cache_dir, TEST_VOCAB_SHA256, load, |_| {
                        Err(io::Error::from(io::ErrorKind::ReadOnlyFilesystem))
                    })
                } else {
                    with_cache_lock_at(&cache_dir, load)
                };
                result.unwrap();
            }
            _ => panic!("unknown subprocess lock mode {mode}"),
        }
    }

    #[test]
    fn locks_are_exclusive_across_processes_and_independent_by_cache() -> Result<()> {
        let directory = tempfile::tempdir()?;
        for read_only_fs in [false, true] {
            let scenario = if read_only_fs { "vocab-lock" } else { "cache" };
            let markers = directory.path().join(format!("{scenario}-markers"));
            let first_cache = directory.path().join(format!("{scenario}-first"));
            let second_cache = directory.path().join(format!("{scenario}-second"));
            fs::create_dir_all(&markers)?;
            if read_only_fs {
                fs::create_dir_all(&first_cache)?;
                fs::create_dir_all(&second_cache)?;
                fs::write(first_cache.join(VOCAB_CACHE_KEY), b"test vocabulary")?;
                fs::write(second_cache.join(VOCAB_CACHE_KEY), b"test vocabulary")?;
            }

            let mut holder = spawn_helper("hold", &first_cache, &markers, read_only_fs);
            wait_for_file(&markers.join("hold.entered"));

            let mut other_cache =
                spawn_helper("other-cache", &second_cache, &markers, read_only_fs);
            wait_for_file(&markers.join("other-cache.entered"));
            assert!(other_cache.wait()?.success());

            let mut waiter = spawn_helper("waiter", &first_cache, &markers, read_only_fs);
            wait_for_file(&markers.join("waiter.attempting"));
            let waiter_status = waiter.try_wait()?;
            assert!(
                waiter_status.is_none(),
                "same-cache process did not block on the held {scenario} lock"
            );
            assert_file_stays_absent(&markers.join("waiter.entered"));

            fs::write(markers.join("hold.release"), b"release")?;
            assert!(holder.wait()?.success());
            assert!(waiter.wait()?.success());
            assert!(markers.join("waiter.entered").exists());
            if read_only_fs {
                assert!(!first_cache.join("harmony-vocab.lock").exists());
                assert!(!second_cache.join("harmony-vocab.lock").exists());
            } else {
                assert!(first_cache.join("harmony-vocab.lock").exists());
                assert!(second_cache.join("harmony-vocab.lock").exists());
            }
        }
        Ok(())
    }

    #[test]
    fn lock_is_released_after_load_error_and_process_termination() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let markers = directory.path().join("markers");
        let cache = directory.path().join("cache");
        fs::create_dir_all(&markers)?;

        let error = with_cache_lock_at::<()>(&cache, || anyhow::bail!("injected loader failure"));
        assert!(error.is_err());
        with_cache_lock_at(&cache, || Ok(()))?;

        let mut holder = spawn_helper("hold", &cache, &markers, false);
        wait_for_file(&markers.join("hold.entered"));
        holder.kill()?;
        assert!(!holder.wait()?.success());

        let next = with_cache_lock_at(&cache, || Ok(()));
        assert!(
            next.is_ok(),
            "OS did not release the lock after holder termination"
        );
        assert!(cache.join("harmony-vocab.lock").exists());

        let read_only_cache = directory.path().join("read-only-cache");
        fs::create_dir(&read_only_cache)?;
        fs::write(read_only_cache.join(VOCAB_CACHE_KEY), b"test vocabulary")?;
        let mut holder = spawn_helper("hold", &read_only_cache, &markers, true);
        wait_for_file(&markers.join("hold.entered"));
        holder.kill()?;
        assert!(!holder.wait()?.success());
        let released = with_cache_lock_at_using(
            &read_only_cache,
            TEST_VOCAB_SHA256,
            || Ok(()),
            |_| Err(io::Error::from(io::ErrorKind::ReadOnlyFilesystem)),
        );
        assert!(
            released.is_ok(),
            "OS did not release the vocabulary lock after holder termination"
        );
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn read_only_cache_without_lock_file_can_load_existing_vocabulary() -> Result<()> {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir()?;
        let cache = directory.path().join("cache");
        fs::create_dir(&cache)?;
        fs::write(cache.join(VOCAB_CACHE_KEY), b"test vocabulary")?;
        fs::set_permissions(&cache, fs::Permissions::from_mode(0o555))?;

        let result = with_cache_lock_at_using(
            &cache,
            TEST_VOCAB_SHA256,
            || Ok("loaded"),
            |_| Err(io::Error::from(io::ErrorKind::ReadOnlyFilesystem)),
        );

        fs::set_permissions(&cache, fs::Permissions::from_mode(0o755))?;
        assert_eq!(result?, "loaded");
        assert!(!cache.join("harmony-vocab.lock").exists());
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn read_only_cache_can_lock_an_existing_read_only_lock_file() -> Result<()> {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir()?;
        let cache = directory.path().join("cache");
        fs::create_dir(&cache)?;
        let lock_path = cache.join("harmony-vocab.lock");
        fs::write(&lock_path, b"")?;
        fs::set_permissions(&lock_path, fs::Permissions::from_mode(0o444))?;
        fs::set_permissions(&cache, fs::Permissions::from_mode(0o555))?;

        let result = with_cache_lock_at_using(
            &cache,
            TEST_VOCAB_SHA256,
            || Ok("loaded"),
            |_| Err(io::Error::from(io::ErrorKind::ReadOnlyFilesystem)),
        );

        fs::set_permissions(&cache, fs::Permissions::from_mode(0o755))?;
        assert_eq!(result?, "loaded");
        assert!(lock_path.exists());
        Ok(())
    }

    #[test]
    fn read_only_filesystem_without_lock_file_can_load_existing_vocabulary() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let cache = directory.path().join("cache");
        fs::create_dir(&cache)?;

        fs::write(cache.join(VOCAB_CACHE_KEY), b"test vocabulary")?;
        let result = with_cache_lock_at_using(
            &cache,
            TEST_VOCAB_SHA256,
            || Ok("loaded"),
            |_| Err(io::Error::from(io::ErrorKind::ReadOnlyFilesystem)),
        );

        assert_eq!(result?, "loaded");
        assert!(!cache.join("harmony-vocab.lock").exists());
        Ok(())
    }

    #[test]
    fn read_only_cache_finds_a_custom_url_cache_entry() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let cache = directory.path().join("cache");
        fs::create_dir(&cache)?;
        fs::write(
            cache.join("custom-url-derived-cache-key"),
            b"test vocabulary",
        )?;
        let result = with_cache_lock_at_using(
            &cache,
            TEST_VOCAB_SHA256,
            || Ok("loaded"),
            |_| Err(io::Error::from(io::ErrorKind::ReadOnlyFilesystem)),
        );

        assert_eq!(result?, "loaded");
        assert!(!cache.join("harmony-vocab.lock").exists());
        Ok(())
    }

    #[test]
    fn read_only_cache_without_a_verified_vocabulary_fails_closed() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let cache = directory.path().join("cache");
        fs::create_dir(&cache)?;
        fs::write(cache.join(VOCAB_CACHE_KEY), b"partial vocabulary")?;
        let mut loaded = false;

        let result = with_cache_lock_at_using(
            &cache,
            TEST_VOCAB_SHA256,
            || {
                loaded = true;
                Ok(())
            },
            |_| Err(io::Error::from(io::ErrorKind::ReadOnlyFilesystem)),
        );

        assert!(result.is_err());
        assert!(
            !loaded,
            "loader ran without a lock on a verified vocabulary"
        );
        assert!(!cache.join("harmony-vocab.lock").exists());
        Ok(())
    }
}
