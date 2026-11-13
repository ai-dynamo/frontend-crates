// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Return true in the configured child so the caller runs its original assertions.
// The parent only launches the child; it never changes its own cached quote policy.
pub fn quote_aware_process(test_name: &str) -> bool {
    const CHILD: &str = "DYNAMO_PARSERS_QUOTE_AWARE_TEST";
    if std::env::var(CHILD).as_deref() == Ok(test_name) {
        return true;
    }
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", test_name, "--nocapture"])
        .env("DYNAMO_PARSERS_DISABLE_QUOTE_AWARE_PARSING", "0")
        .env(CHILD, test_name)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success() && stdout.lines().any(|line| line == "running 1 test"),
        "{test_name} with quote-aware compatibility enabled\nstdout:\n{}\nstderr:\n{}",
        stdout,
        String::from_utf8_lossy(&output.stderr)
    );
    false
}
