// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use clap::Parser;
use std::{
    io::{BufWriter, Write},
    path::PathBuf,
};

#[derive(Parser)]
#[command(
    name = "conformance-publish",
    about = "Native conformance site publisher"
)]
struct Args {
    #[arg(long, default_value = ".")]
    root: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value = "")]
    revision: String,
    #[arg(long)]
    index_unified: bool,
    #[arg(long, conflicts_with = "index_unified")]
    validate_unified: bool,
    #[arg(long)]
    candidate_tree: Option<String>,
    #[arg(long)]
    data_only: bool,
    #[arg(long, default_value = "auto")]
    workers: String,
    #[arg(long)]
    include_history: bool,
    #[arg(long, requires = "include_history")]
    history_source: Option<PathBuf>,
    #[arg(long)]
    cache_dir: Option<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    // This CLI owns its process and sets the publication timezone before threads start.
    unsafe {
        std::env::set_var("TZ", "America/Los_Angeles");
    }
    let args = Args::parse();
    if args.index_unified || args.validate_unified {
        let catalog = if args.index_unified {
            conformance_publisher::catalog::index_cached(
                &args.root,
                &args.workers,
                args.cache_dir.as_deref(),
            )?
        } else {
            conformance_publisher::publication::validate(&args.root, &args.workers)?
        };
        let parent = args.output.parent().unwrap_or(std::path::Path::new("."));
        let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
        {
            let mut writer = BufWriter::new(temporary.as_file_mut());
            serde_json::to_writer(&mut writer, &catalog)?;
            writer.flush()?;
        }
        temporary.as_file().sync_all()?;
        temporary.persist(&args.output)?;
        return Ok(());
    }
    conformance_publisher::publication::publish_mode_cached(
        conformance_publisher::publication::Options {
            root: args.root,
            output: args.output,
            revision: args.revision,
            candidate_tree: args.candidate_tree,
            data_only: args.data_only,
            workers: args.workers,
        },
        args.include_history,
        args.history_source.as_deref(),
        args.cache_dir.as_deref(),
    )
}
