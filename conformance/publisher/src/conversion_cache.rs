// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{
    codec::{json_bytes, sha256},
    source,
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::{fs, io::Write, path::Path};

pub fn read(directory: &Path, key: &str) -> Option<Value> {
    if !source::hex(key, 64) {
        return None;
    }
    let path = directory.join(format!("{key}.json"));
    source::no_links(&path).ok()?;
    if fs::metadata(&path).ok()?.len() >= source::MAX_ARTIFACT {
        return None;
    }
    let entry: Value = serde_json::from_slice(&fs::read(path).ok()?).ok()?;
    let Value::Object(mut entry) = entry else {
        return None;
    };
    let payload = entry.remove("payload")?;
    (entry.get("identity")?.as_str()? == key
        && entry.get("sha256")?.as_str()? == sha256(&json_bytes(&payload).ok()?))
    .then_some(payload)
}

pub fn write(directory: &Path, key: &str, payload: &Value) -> Result<()> {
    ensure!(source::hex(key, 64), "invalid conversion cache identity");
    source::no_links(directory)?;
    fs::create_dir_all(directory)?;
    let mut temporary = tempfile::NamedTempFile::new_in(directory)?;
    let bytes = json_bytes(payload)?;
    let header = json!({"identity":key, "sha256":sha256(&bytes)});
    let mut prefix = json_bytes(&header)?;
    prefix.truncate(prefix.len() - 2);
    temporary.write_all(&prefix)?;
    temporary.write_all(b",\"payload\":")?;
    temporary.write_all(&bytes[..bytes.len() - 1])?;
    temporary.write_all(b"}\n")?;
    temporary.as_file().sync_all()?;
    temporary.persist(directory.join(format!("{key}.json")))?;
    fs::File::open(directory)?.sync_all()?;
    Ok(())
}
