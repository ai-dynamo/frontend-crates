// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=proto/process.proto");
    if std::env::var_os("CARGO_FEATURE_SANDBOXD_CLIENT").is_some() {
        tonic_prost_build::configure()
            .build_server(false)
            .compile_protos(&["proto/process.proto"], &["proto"])?;
    }
    Ok(())
}
