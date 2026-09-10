// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Arming the pool is process-global state, so everything about it lives in
//! this one test binary (its counterpart, the unarmed guard, has its own).

#![cfg(all(feature = "parallel", target_os = "linux"))]

use dynamo_mm_preprocessor::{MmError, execution};

#[test]
fn init_pool_arms_fanout_and_pins_the_thread_count() {
    execution::init_pool(2).unwrap();

    let mut buf = vec![0usize; 8];
    execution::for_chunks_mut(&mut buf, 2, |i, chunk| chunk.fill(i));
    assert_eq!(buf, [0, 0, 1, 1, 2, 2, 3, 3]);

    let named: Vec<String> = std::fs::read_dir("/proc/self/task")
        .expect("procfs")
        .filter_map(|entry| {
            let comm = entry.ok()?.path().join("comm");
            std::fs::read_to_string(comm).ok()
        })
        .map(|name| name.trim().to_string())
        .filter(|name| name.starts_with("dyn-mm"))
        .collect();
    assert_eq!(
        named.len(),
        2,
        "expected the armed pool's threads: {named:?}"
    );

    // Repeating the resolved count is idempotent; changing it is refused.
    execution::init_pool(2).unwrap();
    assert!(matches!(
        execution::init_pool(3),
        Err(MmError::InvalidInput { .. })
    ));
}
