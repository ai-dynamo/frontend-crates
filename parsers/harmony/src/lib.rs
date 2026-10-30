// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fs::{self, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::{Context, Result};
use openai_harmony::{HarmonyEncoding, HarmonyEncodingName, load_harmony_encoding};

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
    fs::create_dir_all(cache_dir)
        .with_context(|| format!("creating Harmony vocabulary cache directory {cache_dir:?}"))?;
    let lock_path = cache_dir.join("harmony-vocab.lock");
    let lock_file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&lock_path)
        .with_context(|| format!("opening Harmony vocabulary lock file {lock_path:?}"))?;
    lock_file
        .lock()
        .with_context(|| format!("locking Harmony vocabulary cache {cache_dir:?}"))?;

    let result = load();
    drop(lock_file);
    result
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

    use super::with_cache_lock_at;

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

    fn spawn_helper(mode: &str, cache_dir: &PathBuf, marker_dir: &PathBuf) -> ChildGuard {
        ChildGuard(
            Command::new(std::env::current_exe().unwrap())
                .arg("--exact")
                .arg("tests::subprocess_lock_helper")
                .arg("--nocapture")
                .env("DYNAMO_HARMONY_LOCK_TEST_MODE", mode)
                .env("DYNAMO_HARMONY_LOCK_TEST_CACHE", cache_dir)
                .env("DYNAMO_HARMONY_LOCK_TEST_MARKERS", marker_dir)
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
                let result = with_cache_lock_at(&cache_dir, || {
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
                });
                result.unwrap();
            }
            _ => panic!("unknown subprocess lock mode {mode}"),
        }
    }

    #[test]
    fn locks_are_exclusive_across_processes_and_independent_by_cache() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let markers = directory.path().join("markers");
        let first_cache = directory.path().join("first");
        let second_cache = directory.path().join("second");
        fs::create_dir_all(&markers)?;

        let mut holder = spawn_helper("hold", &first_cache, &markers);
        wait_for_file(&markers.join("hold.entered"));

        let mut other_cache = spawn_helper("other-cache", &second_cache, &markers);
        wait_for_file(&markers.join("other-cache.entered"));
        assert!(other_cache.wait()?.success());

        let mut waiter = spawn_helper("waiter", &first_cache, &markers);
        wait_for_file(&markers.join("waiter.attempting"));
        let waiter_status = waiter.try_wait()?;
        assert!(
            waiter_status.is_none(),
            "same-cache process did not block on the held lock"
        );
        assert_file_stays_absent(&markers.join("waiter.entered"));

        fs::write(markers.join("hold.release"), b"release")?;
        assert!(holder.wait()?.success());
        assert!(waiter.wait()?.success());
        assert!(markers.join("waiter.entered").exists());
        assert!(first_cache.join("harmony-vocab.lock").exists());
        assert!(second_cache.join("harmony-vocab.lock").exists());
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

        let mut holder = spawn_helper("hold", &cache, &markers);
        wait_for_file(&markers.join("hold.entered"));
        holder.kill()?;
        assert!(!holder.wait()?.success());

        let next = with_cache_lock_at(&cache, || Ok(()));
        assert!(
            next.is_ok(),
            "OS did not release the lock after holder termination"
        );
        assert!(cache.join("harmony-vocab.lock").exists());
        Ok(())
    }
}
