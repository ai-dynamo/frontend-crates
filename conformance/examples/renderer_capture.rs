// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result, bail};
use std::path::PathBuf;

fn main() -> Result<()> {
    let mut root = PathBuf::from(".");
    let mut output = None;
    let mut model = None;
    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--root" => root = args.next().context("--root needs a path")?.into(),
            "--output" => {
                output = Some(PathBuf::from(args.next().context("--output needs a path")?))
            }
            "--model" => model = Some(args.next().context("--model needs an ID")?),
            "--help" => {
                println!("renderer_capture --root REPO --output NEW_DIRECTORY [--model ID]");
                return Ok(());
            }
            _ => bail!("unknown renderer capture argument: {argument}"),
        }
    }
    dynamo_conformance_fixtures_v2::renderer_runner::capture(
        &root,
        &output.context("--output is required")?,
        model.as_deref(),
    )
}
