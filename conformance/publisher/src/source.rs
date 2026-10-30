// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Source identity and semantic admission, shared by validation and assembly.
use crate::{
    codec::sha256,
    history::{Capture, History, Release, expand_execution_bounded, fields, source_reference},
    yaml::{self, Node},
};
use anyhow::{Context, Result, ensure};
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::CString,
    fs::{self, File},
    io::{self, Read, Seek, SeekFrom},
    os::{
        fd::{AsRawFd, FromRawFd},
        unix::fs::MetadataExt,
    },
    path::{Component, Path, PathBuf},
    sync::{Arc, Mutex, OnceLock},
};

pub const STORE: &str = "conformance/fixtures-unified-v2";
pub const MAX_ARTIFACT: u64 = 1_000_000_000;
pub const MAX_ARCHIVE_COMPRESSED_BYTES: u64 = 32 * 1024 * 1024;
pub const MAX_ARCHIVE_MEMBER_BYTES: u64 = 32 * 1024 * 1024;
pub const MAX_ARCHIVE_EXTENSION_BYTES: u64 = 64 * 1024;
// These bounds keep retained member-name sets within the inventory reservation.
pub const MAX_ARCHIVE_MEMBERS: usize = 4096;
pub const MAX_ARCHIVE_MEMBER_PATH_BYTES: usize = 1024;
pub const ARCHIVE_INVENTORY_RESERVATION_BYTES: u64 = 16 * 1024 * 1024;

pub fn archive_reservation_bytes(archive: u64, member: u64) -> Result<u64> {
    archive
        .checked_add(member)
        .and_then(|bytes| bytes.checked_add(ARCHIVE_INVENTORY_RESERVATION_BYTES))
        .context("archive reservation overflow")
}

pub fn hex(value: &str, count: usize) -> bool {
    value.len() == count
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

pub fn relative(value: &str) -> Result<()> {
    ensure!(
        !value.is_empty()
            && !value.contains([':', '\\'])
            && value
                .split('/')
                .all(|s| !s.is_empty() && s != "." && s != ".."),
        "unsafe relative path: {value}"
    );
    ensure!(
        Path::new(value)
            .components()
            .all(|c| matches!(c, Component::Normal(_))),
        "unsafe relative path: {value}"
    );
    Ok(())
}

pub fn no_links(path: &Path) -> Result<()> {
    for ancestor in path.ancestors() {
        match fs::symlink_metadata(ancestor) {
            Ok(metadata) => ensure!(
                !metadata.file_type().is_symlink(),
                "symlink forbidden: {}",
                ancestor.display()
            ),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => (),
            Err(error) => return Err(error.into()),
        }
    }
    Ok(())
}

fn open(root: &Path, name: &str) -> Result<File> {
    relative(name)?;
    let path = root.join(name);
    ensure!(path.is_absolute(), "source root must be absolute");
    let mut file = File::open("/")?;
    let components = path
        .components()
        .filter_map(|c| {
            if let Component::Normal(name) = c {
                Some(name)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    for (index, name) in components.iter().enumerate() {
        let name = CString::new(name.as_encoded_bytes())?;
        let flags = libc::O_RDONLY
            | libc::O_NOFOLLOW
            | libc::O_CLOEXEC
            | libc::O_NONBLOCK
            | if index + 1 == components.len() {
                0
            } else {
                libc::O_DIRECTORY
            };
        let descriptor = unsafe { libc::openat(file.as_raw_fd(), name.as_ptr(), flags) };
        ensure!(
            descriptor >= 0,
            "cannot open regular source {}: {}",
            path.display(),
            std::io::Error::last_os_error()
        );
        file = unsafe { File::from_raw_fd(descriptor) };
    }
    let metadata = file.metadata()?;
    ensure!(
        metadata.is_file() && metadata.len() < MAX_ARTIFACT,
        "source must be a regular file smaller than 1 GB: {}",
        path.display()
    );
    Ok(file)
}

pub(crate) fn read_bounded(root: &Path, name: &str, limit: u64) -> Result<Vec<u8>> {
    let file = open(root, name)?;
    ensure!(
        file.metadata()?.len() <= limit,
        "source exceeds byte limit: {name}"
    );
    let mut bytes = Vec::new();
    file.take(limit + 1).read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= limit,
        "source grew beyond byte limit: {name}"
    );
    Ok(bytes)
}

pub fn read(root: &Path, name: &str) -> Result<Vec<u8>> {
    read_bounded(root, name, MAX_ARTIFACT - 1)
}

pub fn verified(root: &Path, descriptor: &Value) -> Result<Vec<u8>> {
    let scoped;
    let root = if let Some(prefix) = descriptor["dataset_root"].as_str() {
        relative(prefix)?;
        scoped = root.join(prefix);
        no_links(&scoped)?;
        &scoped
    } else {
        root
    };
    let name = descriptor
        .get("source_path")
        .unwrap_or(&descriptor["path"])
        .as_str()
        .context("resource has no path")?;
    let bytes = if let Some(archive) = descriptor.get("archive") {
        archived(root, name, archive)?
    } else {
        read(root, name)?
    };
    ensure!(
        descriptor["size"].as_u64() == Some(bytes.len() as u64)
            && descriptor["sha256"].as_str() == Some(sha256(&bytes).as_str()),
        "resource identity mismatch: {name}"
    );
    Ok(bytes)
}

pub fn dataset_member(dataset: &str, member: &str) -> Result<String> {
    relative(dataset)?;
    ensure!(
        !dataset.contains('/'),
        "dataset identity must be a single component"
    );
    relative(member)?;
    Ok(format!("conformance/history-datasets/{dataset}/{member}"))
}

pub fn original_sources(catalog: &Value) -> Result<Vec<Value>> {
    let mut sources = Vec::new();
    for kind in ["unified", "archives"] {
        sources.extend(
            catalog["sources"][kind]
                .as_array()
                .context("missing original inventory")?
                .iter()
                .cloned(),
        );
    }
    for dataset in catalog["sources"]["history_datasets"]
        .as_array()
        .into_iter()
        .flatten()
    {
        let identity = dataset["id"].as_str().context("missing dataset identity")?;
        for resource in dataset["resources"]
            .as_array()
            .context("missing dataset originals")?
        {
            let mut scoped = resource.clone();
            scoped["source_path"] = resource["path"].clone();
            scoped["path"] = Value::String(dataset_member(
                identity,
                resource["path"].as_str().context("missing original path")?,
            )?);
            sources.push(scoped);
        }
    }
    Ok(sources)
}

pub fn bind_record_requests(records: &mut Node, authored: &Node) -> Result<()> {
    let Node::Map(records) = records else {
        anyhow::bail!("expected resolved records")
    };
    for (case, wrapped) in records {
        let mut value = wrapped.field("value")?.clone();
        let mut stimulus = value.field("stimulus")?.clone();
        if stimulus.get("ref") == Some(&Node::Str("current".into())) {
            stimulus.remove("ref")?;
            stimulus.set(
                "inline",
                authored
                    .field("cases")?
                    .field(case.string()?)?
                    .field("request")?
                    .clone(),
            )?;
            value.set("stimulus", stimulus)?;
            wrapped.set("value", value)?;
        }
    }
    Ok(())
}

#[derive(Default)]
struct SourceArchives {
    users: usize,
    directories: BTreeMap<PathBuf, Arc<Mutex<Option<AdmittedArchive>>>>,
}

static SOURCE_ARCHIVES: OnceLock<Mutex<SourceArchives>> = OnceLock::new();

pub struct ArchiveSession;

pub fn archive_session() -> ArchiveSession {
    SOURCE_ARCHIVES
        .get_or_init(Default::default)
        .lock()
        .unwrap()
        .users += 1;
    ArchiveSession
}

impl Drop for ArchiveSession {
    fn drop(&mut self) {
        let mut cache = SOURCE_ARCHIVES.get().unwrap().lock().unwrap();
        cache.users -= 1;
        if cache.users == 0 {
            cache.directories.clear();
        }
    }
}

#[derive(Clone)]
struct AdmittedArchive {
    descriptor: Value,
    identity: (u64, u64, u64, i64, i64, i64, i64),
    directory: Arc<tempfile::TempDir>,
}

fn file_identity(file: &File) -> Result<(u64, u64, u64, i64, i64, i64, i64)> {
    let m = file.metadata()?;
    Ok((
        m.dev(),
        m.ino(),
        m.len(),
        m.mtime(),
        m.mtime_nsec(),
        m.ctime(),
        m.ctime_nsec(),
    ))
}

pub(crate) fn admitted_archive(root: &Path, descriptor: &Value) -> Result<Arc<tempfile::TempDir>> {
    let name = descriptor["path"]
        .as_str()
        .context("missing source archive path")?;
    Ok(admit_archive(root, name, Some(descriptor))?.1)
}

pub(crate) fn discover_archive(root: &Path, name: &str) -> Result<(Value, Arc<tempfile::TempDir>)> {
    admit_archive(root, name, None)
}

fn admit_archive(
    root: &Path,
    archive_name: &str,
    expected: Option<&Value>,
) -> Result<(Value, Arc<tempfile::TempDir>)> {
    relative(archive_name)?;
    let file_name = archive_name
        .strip_prefix(&format!("{STORE}/versions/"))
        .context("archive outside versions")?;
    ensure!(
        !file_name.contains('/') && file_name.ends_with(".tar.gz"),
        "invalid version archive filename"
    );
    if let Some(descriptor) = expected {
        ensure!(
            descriptor["sha256"]
                .as_str()
                .is_some_and(|hash| hex(hash, 64)),
            "invalid source archive hash"
        );
    }
    let slot = {
        let mut cache = SOURCE_ARCHIVES
            .get_or_init(Default::default)
            .lock()
            .unwrap();
        cache
            .directories
            .entry(root.join(archive_name))
            .or_default()
            .clone()
    };
    // Only readers of this archive wait for admission. Other archives decompress
    // concurrently, and failed admissions leave the slot empty for a later retry.
    let mut admitted = slot.lock().unwrap();
    let mut file = open(root, archive_name)?;
    let identity = file_identity(&file)?;
    ensure!(
        identity.2 <= MAX_ARCHIVE_COMPRESSED_BYTES,
        "source archive exceeds compressed limit"
    );
    if let Some(previous) = admitted.as_ref() {
        ensure!(
            previous.identity == identity,
            "source archive changed during operation"
        );
        if let Some(descriptor) = expected {
            ensure!(
                descriptor["size"] == previous.descriptor["size"]
                    && descriptor["sha256"] == previous.descriptor["sha256"],
                "source archive identity mismatch"
            );
        }
        return Ok((previous.descriptor.clone(), previous.directory.clone()));
    }
    let mut compressed = Vec::new();
    (&mut file)
        .take(MAX_ARCHIVE_COMPRESSED_BYTES + 1)
        .read_to_end(&mut compressed)?;
    let descriptor = serde_json::json!({
        "path":archive_name,"sha256":sha256(&compressed),"size":compressed.len()
    });
    ensure!(
        compressed.len() as u64 <= MAX_ARCHIVE_COMPRESSED_BYTES
            && descriptor["size"] == identity.2
            && expected.is_none_or(|expected| expected["sha256"] == descriptor["sha256"]
                && expected["size"] == descriptor["size"])
            && file_identity(&file)? == identity
            && file_identity(&open(root, archive_name)?)? == identity,
        "source archive identity mismatch"
    );
    let directory = tempfile::Builder::new()
        .prefix("unified-source-")
        .tempdir()?;
    let mut decoder = flate2::bufread::GzDecoder::new(compressed.as_slice());
    let decoded = (&mut decoder).take(512 * 1024 * 1024 + 1);
    let mut archive = tar::Archive::new(DeclaredArchive::new(decoded));
    let mut seen = BTreeSet::new();
    for entry in archive.entries()? {
        let mut entry = entry?;
        ensure!(
            entry.path_bytes().len() <= MAX_ARCHIVE_MEMBER_PATH_BYTES,
            "source archive member path exceeds limit"
        );
        let path = entry
            .path()?
            .to_str()
            .context("non UTF8 source member")?
            .to_string();
        relative(&path)?;
        let parts = path.split('/').collect::<Vec<_>>();
        ensure!(
            matches!(parts.as_slice(), ["families", _, file] if *file != "inputs_and_golden.yaml" && *file != "index.yaml")
                || matches!(parts.as_slice(), ["executions", _] | ["receipts", _]),
            "invalid Unified source member"
        );
        ensure!(
            seen.len() < MAX_ARCHIVE_MEMBERS,
            "source archive member count exceeds limit"
        );
        ensure!(
            path.ends_with(".yaml")
                && seen.insert(path.clone())
                && entry.header().entry_type().is_file()
                && entry.link_name()?.is_none()
                && entry.header().size()? == entry.size()
                && entry.size() <= MAX_ARCHIVE_MEMBER_BYTES,
            "invalid or duplicate source archive member"
        );
        let target = directory.path().join(&path);
        fs::create_dir_all(target.parent().unwrap())?;
        let mut output = File::create(target)?;
        std::io::copy(&mut entry, &mut output)?;
    }
    ensure!(!seen.is_empty(), "empty source archive");
    let mut declared = archive.into_inner();
    zero_padding(&mut declared)?;
    ensure!(
        declared.reader.limit() > 0,
        "invalid source archive tail or decoded limit"
    );
    ensure!(
        decoder.get_ref().is_empty(),
        "trailing source gzip bytes or members"
    );
    let directory = Arc::new(directory);
    *admitted = Some(AdmittedArchive {
        descriptor: descriptor.clone(),
        identity,
        directory: directory.clone(),
    });
    Ok((descriptor, directory))
}

// Inspect each declaration before tar can allocate GNU/PAX extension bodies or
// sparse maps. PAX sizes must agree with the physical header before extraction,
// so both readers keep the same record boundaries.
struct DeclaredArchive<R> {
    reader: R,
    header: [u8; 512],
    header_offset: usize,
    body_remaining: u64,
    extensions: u64,
    ended: bool,
}

impl<R: Read> DeclaredArchive<R> {
    fn new(reader: R) -> Self {
        Self {
            reader,
            header: [0; 512],
            header_offset: 512,
            body_remaining: 0,
            extensions: 0,
            ended: false,
        }
    }
}

impl<R: Read> Read for DeclaredArchive<R> {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        if bytes.is_empty() {
            return Ok(0);
        }
        if self.header_offset < 512 {
            let count = bytes.len().min(512 - self.header_offset);
            bytes[..count]
                .copy_from_slice(&self.header[self.header_offset..self.header_offset + count]);
            self.header_offset += count;
            return Ok(count);
        }
        if self.ended {
            return self.reader.read(bytes);
        }
        if self.body_remaining > 0 {
            let limit = bytes.len().min(self.body_remaining as usize);
            let count = self.reader.read(&mut bytes[..limit])?;
            self.body_remaining -= count as u64;
            return Ok(count);
        }
        self.reader.read_exact(&mut self.header)?;
        self.header_offset = 0;
        if self.header.iter().all(|byte| *byte == 0) {
            self.ended = true;
        } else {
            let header = tar::Header::from_byte_slice(&self.header);
            let kind = header.entry_type();
            let size = header.size()?;
            if kind.is_gnu_longname()
                || kind.is_gnu_longlink()
                || kind.is_pax_local_extensions()
                || kind.is_pax_global_extensions()
            {
                self.extensions = self
                    .extensions
                    .checked_add(size)
                    .ok_or_else(|| io::Error::other("source archive extension size overflow"))?;
                if self.extensions > MAX_ARCHIVE_EXTENSION_BYTES {
                    return Err(io::Error::other(
                        "source archive extension metadata exceeds limit",
                    ));
                }
            } else {
                if !kind.is_file() || size > MAX_ARCHIVE_MEMBER_BYTES {
                    return Err(io::Error::other(
                        "invalid source archive member declaration",
                    ));
                }
                self.extensions = 0;
            }
            self.body_remaining = size
                .checked_add(511)
                .ok_or_else(|| io::Error::other("source archive member size overflow"))?
                & !511;
        }
        self.read(bytes)
    }
}

fn zero_padding(reader: &mut impl Read) -> Result<()> {
    let mut buffer = [0; 8192];
    let mut count = 0u64;
    loop {
        let size = reader.read(&mut buffer)?;
        if size == 0 {
            break;
        }
        ensure!(
            buffer[..size].iter().all(|byte| *byte == 0),
            "invalid source archive tail or decoded limit"
        );
        count += size as u64;
    }
    ensure!(count >= 512, "invalid source archive tail or decoded limit");
    Ok(())
}

fn archived(root: &Path, name: &str, descriptor: &Value) -> Result<Vec<u8>> {
    let member = name
        .strip_prefix(&format!("{STORE}/"))
        .context("member outside source store")?;
    relative(member)?;
    let directory = admitted_archive(root, descriptor)?;
    read(directory.path(), member)
}

pub fn archived_inventory(root: &Path, representative: &Value) -> Result<BTreeSet<String>> {
    verified(root, representative)?;
    let directory = admitted_archive(root, &representative["archive"])?;
    let mut result = BTreeSet::new();
    walk_inventory(directory.path(), directory.path(), &mut result, false)?;
    Ok(result)
}

fn walk_inventory(
    root: &Path,
    dir: &Path,
    result: &mut BTreeSet<String>,
    include_directories: bool,
) -> Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        ensure!(
            kind.is_dir() || kind.is_file(),
            "nonregular inventory entry: {}",
            entry.path().display()
        );
        if include_directories || kind.is_file() {
            result.insert(
                entry
                    .path()
                    .strip_prefix(root)?
                    .to_str()
                    .context("non UTF8 path")?
                    .to_string(),
            );
        }
        if kind.is_dir() {
            walk_inventory(root, &entry.path(), result, include_directories)?;
        }
    }
    Ok(())
}

pub fn inventory(root: &Path) -> Result<BTreeSet<String>> {
    let mut result = BTreeSet::new();
    no_links(root)?;
    walk_inventory(root, root, &mut result, true)?;
    Ok(result)
}

pub fn source_size(root: &Path, name: &str) -> Result<u64> {
    Ok(open(root, name)?.metadata()?.len())
}

pub fn archive_source_size(root: &Path, name: &str) -> Result<u64> {
    let mut file = open(root, name)?;
    file.seek(SeekFrom::End(-4))?;
    let mut bytes = [0u8; 4];
    file.read_exact(&mut bytes)?;
    let size = u64::from(u32::from_le_bytes(bytes));
    ensure!(size < MAX_ARTIFACT, "decoded archive is at least 1 GB");
    Ok(size)
}

pub fn archive_decoded_size(bytes: &[u8]) -> Result<u64> {
    ensure!(bytes.len() >= 18, "truncated gzip archive");
    let size = u64::from(u32::from_le_bytes(bytes[bytes.len() - 4..].try_into()?));
    ensure!(size < MAX_ARTIFACT, "decoded archive is at least 1 GB");
    Ok(size)
}

/// Tar metadata must agree with the maintained inventory before an archive is wrapped.
pub fn archive(bytes: &[u8], descriptor: &Value) -> Result<()> {
    let size = archive_decoded_size(bytes)?;
    let decoded = crate::codec::gunzip(bytes, &sha256(bytes), usize::try_from(size)?)?;
    let mut archive = tar::Archive::new(std::io::Cursor::new(decoded.as_slice()));
    let mut members = Vec::new();
    let mut seen = BTreeSet::new();
    for entry in archive.entries()? {
        let entry = entry?;
        ensure!(
            entry.header().size()? == entry.size(),
            "PAX member size differs from tar header"
        );
        ensure!(
            entry.link_name()?.is_none(),
            "archive links are not supported"
        );
        let path = entry.path()?;
        let path = path.to_str().context("non UTF8 archive member")?;
        let kind = entry.header().entry_type();
        ensure!(
            kind.is_file() || kind.is_contiguous() || kind.is_dir(),
            "unsupported archive member type"
        );
        let path = if kind.is_dir() {
            path.trim_end_matches('/')
        } else {
            path
        };
        relative(path)?;
        ensure!(seen.insert(path.to_string()), "duplicate archive member");
        let size = entry.size();
        ensure!(size < MAX_ARTIFACT, "archive member is at least 1 GB");
        members.push(serde_json::json!({"path":path,"kind":if kind.is_dir() {"directory"} else {"file"},"size":size}));
    }
    let position = usize::try_from(archive.into_inner().position())?;
    ensure!(
        position >= 512
            && decoded.len() >= position + 512
            && decoded[position - 512..].iter().all(|b| *b == 0),
        "missing tar end markers or nonzero trailing tar bytes"
    );
    ensure!(
        descriptor["members"] == serde_json::json!(members),
        "archive inventory mismatch"
    );
    Ok(())
}

fn component(value: &str) -> Result<()> {
    ensure!(
        !value.is_empty() && !value.contains(['/', '\\', ':']) && value != "." && value != "..",
        "unsafe component"
    );
    Ok(())
}

fn request(node: &Node) -> Result<()> {
    fields(
        node,
        &["input", "init", "finish_reason", "tools", "chunks"],
        &[],
    )?;
    node.field("input")?.string()?;
    node.field("init")?.mapping()?;
    node.field("finish_reason")?.string()?;
    node.field("tools")?.array()?;
    for chunk in node.field("chunks")?.array()? {
        chunk.mapping()?;
    }
    Ok(())
}

fn request_json(node: &Node) -> Result<String> {
    Ok(match node {
        Node::Null => "null".into(),
        Node::Bool(v) => v.to_string(),
        Node::Int(v) => v.to_string(),
        Node::Float(v) if v.is_nan() => "NaN".into(),
        Node::Float(v) if v.is_infinite() => if v.is_sign_negative() {
            "-Infinity"
        } else {
            "Infinity"
        }
        .into(),
        Node::Float(v) => crate::yaml::float_repr(*v),
        Node::Str(v) => serde_json::to_string(v)?,
        Node::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(request_json)
                .collect::<Result<Vec<_>>>()?
                .join(",")
        ),
        Node::Map(values) => {
            let sorted = values
                .iter()
                .map(|(k, v)| Ok((k.string()?, v)))
                .collect::<Result<BTreeMap<_, _>>>()?;
            format!(
                "{{{}}}",
                sorted
                    .into_iter()
                    .map(|(k, v)| Ok(format!(
                        "{}:{}",
                        serde_json::to_string(k)?,
                        request_json(v)?
                    )))
                    .collect::<Result<Vec<_>>>()?
                    .join(",")
            )
        }
        _ => anyhow::bail!("non JSON request value"),
    })
}

fn request_digest(node: &Node) -> Result<String> {
    Ok(sha256(request_json(node)?.as_bytes()))
}

pub fn canonical_alias(alias: &str, family: &str, metadata: &Value) -> Result<String> {
    let original = alias.strip_prefix("UNIFIED.").unwrap_or(alias);
    let mut key = original.to_string();
    if let Some(value) = key.strip_suffix(&format!(".{family}")) {
        key = value.to_string();
    }
    if let Some(scenario) = metadata["family_scenarios"][family]
        .get(&key)
        .and_then(Value::as_str)
    {
        return Ok(metadata["canonical"][scenario]
            .as_str()
            .context("missing family alias owner")?
            .to_string());
    }
    if let Some(value) = metadata["canonical"].get(&key).and_then(Value::as_str) {
        return Ok(value.to_string());
    }
    if let Some(value) = metadata["families"][family]
        .get(&key)
        .and_then(Value::as_str)
    {
        key = value.to_string();
    }
    if let Some(letter) = key
        .strip_prefix("31.")
        .filter(|s| s.len() == 1 && (b'a'..=b'x').contains(&s.as_bytes()[0]))
    {
        key = format!("31-{}", letter.as_bytes()[0] - b'a' + 1);
    }
    if let Some(owners) = metadata["owners"].get(&key).and_then(Value::as_array) {
        ensure!(
            owners.len() <= 1,
            "conflicting Unified alias ownership: {alias}"
        );
        if let Some(owner) = owners.first().and_then(Value::as_str) {
            if metadata["owner_families"]
                .get(owner)
                .and_then(Value::as_str)
                .is_some_and(|f| f != family)
            {
                return Ok(original.to_string());
            }
            return Ok(metadata["canonical"][owner]
                .as_str()
                .context("missing canonical alias owner")?
                .to_string());
        }
    }
    Ok(original.to_string())
}

pub fn family(node: &Node, name: &str, alias_metadata: &Value) -> Result<()> {
    fields(node, &["input_document", "golden_document", "cases"], &[])?;
    ensure!(
        !name.is_empty()
            && name
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_'),
        "invalid family name"
    );
    for key in ["input_document", "golden_document"] {
        ensure!(
            node.field(key)?.field("family")?.string()? == name,
            "family identity mismatch"
        );
    }
    let mut aliases = BTreeSet::new();
    let mut canonical_owners = BTreeMap::new();
    for (id, case) in node.field("cases")?.mapping()? {
        id.string()?;
        fields(
            case,
            &[
                "lifecycle",
                "scenario",
                "description",
                "policy",
                "display_id",
                "historical_ids",
                "request",
                "golden",
            ],
            &[],
        )?;
        let active = match case.field("lifecycle")?.string()? {
            "active" => true,
            "retired" => false,
            _ => anyhow::bail!("invalid lifecycle"),
        };
        let description = case.field("description")?;
        ensure!(
            description == &Node::Null || matches!(description, Node::Str(_)),
            "invalid description"
        );
        if active {
            ensure!(
                !description.string()?.is_empty(),
                "active case needs description"
            );
            case.field("scenario")?.string()?;
            case.field("golden")?.mapping()?;
        }
        for tag in case.field("policy")?.array()? {
            tag.string()?;
        }
        if case.field("display_id")? != &Node::Null {
            let id = case.field("display_id")?.string()?;
            component(id)?;
            ensure!(aliases.insert(id), "duplicate display ID");
        } else {
            ensure!(!active, "active case needs display ID");
        }
        for id in case.field("historical_ids")?.array()? {
            let id = id.string()?;
            component(id)?;
            ensure!(aliases.insert(id), "duplicate alias");
        }
        for alias in std::iter::once(case.field("display_id")?)
            .chain(case.field("historical_ids")?.array()?.iter())
            .filter(|n| **n != Node::Null)
        {
            let canonical = canonical_alias(alias.string()?, name, alias_metadata)?;
            let previous = canonical_owners.insert(canonical, id.string()?);
            ensure!(
                previous.is_none_or(|owner| owner == id.string().unwrap()),
                "duplicate canonical ID"
            );
        }
        if active || case.field("request")? != &Node::Null {
            request(case.field("request")?)?;
        }
    }
    Ok(())
}

fn origin(node: &Node, version: Option<&str>) -> Result<()> {
    fields(
        node,
        &["crate_version", "source_sha256"],
        &["git_commit", "adapter_sha256", "job_sha256"],
    )?;
    let actual = node.field("crate_version")?.string()?;
    Release::parse_provenance(actual)?;
    if let Some(version) = version {
        Release::parse_provenance(version)?;
        let (release, source) = version
            .split_once("+source.")
            .map_or((version, None), |(release, source)| (release, Some(source)));
        ensure!(actual == release, "origin version mismatch");
        if let Some(source) = source {
            ensure!(
                node.field("source_sha256")?.string()? == source,
                "origin source identity mismatch"
            );
        }
    }
    for key in [
        "source_sha256",
        "adapter_sha256",
        "job_sha256",
        "git_commit",
    ] {
        if let Some(value) = node.get(key) {
            ensure!(
                hex(value.string()?, if key == "git_commit" { 40 } else { 64 }),
                "invalid origin hash"
            );
        }
    }
    ensure!(
        node.get("adapter_sha256").is_some() == node.get("job_sha256").is_some(),
        "origin needs both adapter and job identities"
    );
    Ok(())
}

fn stimulus(node: &Node, case: &Node) -> Result<()> {
    let states = ["ref", "inline", "partial", "error", "unavailable"];
    let selected = states
        .into_iter()
        .filter(|key| node.get(key).is_some())
        .collect::<Vec<_>>();
    ensure!(selected.len() == 1, "invalid stimulus state");
    fields(node, &[selected[0]], &["semantic_sha256"])?;
    let state = selected[0];
    let value = node.field(state)?;
    if state == "ref" || state == "inline" {
        let value = if state == "ref" {
            ensure!(value.string()? == "current", "invalid stimulus ref");
            case.field("request")?
        } else {
            value
        };
        request(value)?;
        if let Some(digest) = node.get("semantic_sha256").filter(|v| **v != Node::Null) {
            ensure!(
                digest.string()? == request_digest(value)?,
                "stimulus digest mismatch"
            );
        }
    } else {
        value.mapping()?;
        if state != "partial" {
            value.field("code")?.string()?;
        }
    }
    Ok(())
}

fn observation(node: &Node) -> Result<()> {
    ensure!(node.mapping()?.len() == 1, "invalid observation state");
    let (state, payload) = &node.mapping()?[0];
    let state = state.string()?;
    payload.mapping()?;
    if state == "value" {
        fields(payload, &[], &["assembled", "chunks"])?;
        ensure!(!payload.mapping()?.is_empty(), "empty observation value");
    } else {
        ensure!(
            ["error", "unavailable"].contains(&state),
            "invalid observation state"
        );
        payload.field("code")?.string()?;
    }
    Ok(())
}

pub fn capture(node: &Node, authored: &Node, implementation: &str, version: &str) -> Result<()> {
    capture_with_proof_requests(node, authored, implementation, version, None)
}

fn capture_with_proof_requests(
    node: &Node,
    authored: &Node,
    implementation: &str,
    version: &str,
    proof_requests: Option<&Node>,
) -> Result<()> {
    fields(
        node,
        &[
            "provenance",
            "document",
            "changes",
            "metadata_changes",
            "document_overrides",
        ],
        &["parent", "execution_proof"],
    )?;
    let provenance = node.field("provenance")?;
    match provenance.field("status")?.string()? {
        "captured" => {
            fields(provenance, &["status", "origin"], &[])?;
            origin(provenance.field("origin")?, Some(version))?;
        }
        "legacy" => {
            fields(provenance, &["status", "captured_with"], &[])?;
            let with = provenance.field("captured_with")?;
            ensure!(
                with.mapping()?.len() == 1 && with.field(implementation)?.string()? == version,
                "legacy provenance mismatch"
            );
        }
        _ => anyhow::bail!("invalid capture provenance"),
    }
    let document = node.field("document")?;
    document.mapping()?;
    ensure!(
        document.get("capture_origin").is_none() && document.get("captured_with").is_none(),
        "repeated derived provenance"
    );
    let cases = authored.field("cases")?;
    for (id, change) in node.field("changes")?.mapping()? {
        let case = cases.field(id.string()?)?;
        if change.mapping()?.len() == 1 && change.get("absent") == Some(&Node::Bool(true)) {
            continue;
        }
        fields(change, &[], &["case_key", "stimulus", "observation"])?;
        ensure!(!change.mapping()?.is_empty(), "empty change");
        if let Some(key) = change.get("case_key") {
            component(key.string()?)?;
        }
        if let Some(value) = change.get("stimulus") {
            stimulus(value, case)?;
        }
        if let Some(value) = change.get("observation") {
            observation(value)?;
        }
    }
    for key in ["metadata_changes", "document_overrides"] {
        for (id, metadata) in node.field(key)?.mapping()? {
            cases.field(id.string()?)?;
            metadata.mapping()?;
            if key == "document_overrides" {
                fields(metadata, &[], &["parser_path", "capture_provenance"])?;
                if let Some(parser) = metadata.get("parser_path").filter(|v| **v != Node::Null) {
                    parser.string()?;
                }
                if let Some(provenance) = metadata.get("capture_provenance") {
                    fields(
                        provenance,
                        &["captured_with", "capture_origin", "inherited_from"],
                        &[],
                    )?;
                    ensure!(
                        !provenance.field("captured_with")?.mapping()?.is_empty(),
                        "empty captured_with"
                    );
                    for (key, value) in provenance.field("captured_with")?.mapping()? {
                        key.string()?;
                        value.string()?;
                    }
                    if provenance.field("capture_origin")? != &Node::Null {
                        origin(provenance.field("capture_origin")?, None)?;
                    }
                    if provenance.field("inherited_from")? != &Node::Null {
                        provenance.field("inherited_from")?.string()?;
                    }
                }
            }
        }
    }
    if let Some(proof) = node.get("execution_proof") {
        fields(proof, &["receipt", "requests"], &[])?;
        let receipt = proof.field("receipt")?;
        ensure!(
            receipt.field("implementation")?.string()? == implementation
                && receipt.field("version")?.string()? == version,
            "proof identity mismatch"
        );
        ensure!(
            provenance.field("status")?.string()? == "captured",
            "legacy proof"
        );
        for (key, value) in provenance.field("origin")?.mapping()? {
            let key = key.string()?;
            if key != "crate_version" {
                ensure!(receipt.get(key) == Some(value), "proof origin mismatch");
            }
        }
        for key in [
            "source_sha256",
            "adapter_sha256",
            "job_sha256",
            "results_sha256",
        ] {
            ensure!(hex(receipt.field(key)?.string()?, 64), "unsealed proof");
        }
        let requests = proof.field("requests")?.mapping()?;
        ensure!(
            !requests.is_empty(),
            "execution proof requires sealed request digests"
        );
        for (_, digest) in requests {
            ensure!(
                hex(digest.string()?, 64),
                "execution proof requires sealed request digests"
            );
        }
        let mut expected = Node::map();
        for (id, case) in cases.mapping()? {
            if case.field("lifecycle")?.string()? != "retired" {
                expected.set(
                    id.string()?,
                    Node::Str(request_digest(case.field("request")?)?),
                )?;
            }
        }
        ensure!(
            proof_requests
                .unwrap_or(&expected)
                .equivalent(proof.field("requests")?),
            "proof request membership mismatch"
        );
    }
    Ok(())
}

pub fn resolved_snapshot<'a>(
    node: &'a Node,
    authored: &Node,
    family: &str,
    implementation: &str,
    version: &str,
) -> Result<&'a Node> {
    fields(
        node,
        &["schema_version", "provenance", "document", "records"],
        &[
            "execution_proof",
            "historical_evidence",
            "capture_selection",
        ],
    )?;
    ensure!(
        node.field("schema_version")? == &Node::Int(2.into()),
        "invalid Unified snapshot schema"
    );
    let provenance = node.field("provenance")?;
    match provenance.field("status")?.string()? {
        "captured" => {
            fields(provenance, &["status", "origin"], &[])?;
            origin(provenance.field("origin")?, Some(version))?;
        }
        "legacy" => {
            fields(provenance, &["status", "captured_with"], &[])?;
            let with = provenance.field("captured_with")?;
            ensure!(
                with.mapping()?.len() == 1 && with.field(implementation)?.string()? == version,
                "legacy provenance mismatch"
            );
        }
        _ => anyhow::bail!("invalid snapshot provenance"),
    }
    let document = node.field("document")?;
    document.mapping()?;
    ensure!(
        document.get("capture_origin").is_none() && document.get("captured_with").is_none(),
        "snapshot repeats derived provenance"
    );
    let cases = authored.field("cases")?;
    let records = node.field("records")?;
    for (case_id, wrapped) in records.mapping()? {
        let case_id = case_id.string()?;
        cases.field(case_id)?;
        fields(wrapped, &["value", "member"], &["suppress_inherited_from"])?;
        if let Some(flag) = wrapped.get("suppress_inherited_from") {
            ensure!(
                flag == &Node::Bool(true),
                "invalid snapshot inheritance projection"
            );
        }
        let member = wrapped.field("member")?.string()?;
        ensure!(
            member.starts_with(&format!("{STORE}/families/{family}/")) && member.ends_with(".yaml"),
            "invalid snapshot source member"
        );
        relative(member)?;
        let value = wrapped.field("value")?;
        fields(
            value,
            &["case_key", "stimulus", "observation", "document"],
            &[],
        )?;
        component(value.field("case_key")?.string()?)?;
        stimulus(value.field("stimulus")?, cases.field(case_id)?)?;
        observation(value.field("observation")?)?;
        let record_document = value.field("document")?;
        record_document.mapping()?;
        let source_name = member
            .strip_prefix(&format!("{STORE}/families/{family}/{implementation}-"))
            .and_then(|name| name.strip_suffix(".yaml"))
            .context("snapshot source member has inconsistent producer identity")?;
        let source_version = Release::parse(source_name)?;
        let producer_name = record_document
            .field("captured_with")?
            .field(implementation)?
            .string()?;
        // Document overrides describe the producer independently of observation lineage.
        ensure!(
            source_version <= Release::parse(version)?
                && Release::parse_provenance(producer_name)? <= Release::parse(version)?,
            "snapshot source or producer is newer than its checkpoint"
        );
        if let Some(record_origin) = record_document.get("capture_origin") {
            origin(record_origin, Some(producer_name))?;
        }
    }
    if let Some(proof) = node.get("execution_proof") {
        let mut checkpoint = Node::map();
        checkpoint.set("provenance", provenance.clone())?;
        checkpoint.set("document", document.clone())?;
        checkpoint.set("changes", Node::map())?;
        checkpoint.set("metadata_changes", Node::map())?;
        checkpoint.set("document_overrides", Node::map())?;
        checkpoint.set("execution_proof", proof.clone())?;
        // Receipt membership follows complete request bindings, independently
        // of older observation origins retained by deduplication.
        let mut executed = Node::map();
        for (id, wrapped) in records.mapping()? {
            let stimulus = wrapped.field("value")?.field("stimulus")?;
            ensure!(
                stimulus.get("ref").is_some() || stimulus.get("inline").is_some(),
                "execution proof requires a complete request"
            );
            executed.set(id.string()?, stimulus.field("semantic_sha256")?.clone())?;
        }
        capture_with_proof_requests(
            &checkpoint,
            authored,
            implementation,
            version,
            Some(&executed),
        )?;
    }
    if let Some(Node::Array(evidence_items)) = node.get("historical_evidence") {
        for evidence in evidence_items {
            fields(
                evidence,
                &[
                    "kind",
                    "schema_version",
                    "family",
                    "implementation",
                    "version",
                    "checkpoint",
                    "cases",
                    "defaults",
                    "exceptions",
                    "metadata_changes",
                    "receipt_membership",
                ],
                &["execution_proof"],
            )?;
            ensure!(
                evidence.field("kind")?.string()? == "execution_manifest"
                    && evidence.field("schema_version")? == &Node::Int(1.into())
                    && evidence.field("family")?.string()? == family
                    && evidence.field("implementation")?.string()? == implementation
                    && evidence.field("version")?.string()? == version,
                "historical execution evidence identity mismatch"
            );
            evidence.field("checkpoint")?.mapping()?;
            evidence.field("defaults")?.mapping()?;
            evidence.field("exceptions")?.mapping()?;
            let case_ids = evidence
                .field("cases")?
                .array()?
                .iter()
                .map(Node::string)
                .collect::<Result<Vec<_>>>()?;
            let unique_cases = case_ids.iter().copied().collect::<BTreeSet<_>>();
            ensure!(
                unique_cases.len() == case_ids.len(),
                "duplicate historical evidence case"
            );
            for case_id in &case_ids {
                authored.field("cases")?.field(case_id)?;
            }
            let metadata = evidence.field("metadata_changes")?.mapping()?;
            let metadata_cases = metadata
                .iter()
                .map(|(key, _)| key.string())
                .collect::<Result<BTreeSet<_>>>()?;
            ensure!(
                metadata_cases == unique_cases,
                "historical evidence metadata membership mismatch"
            );
            for (_, value) in metadata {
                value.mapping()?;
            }
            for receipt in evidence.field("receipt_membership")?.array()? {
                fields(receipt, &["members", "body"], &[])?;
                let members = receipt
                    .field("members")?
                    .array()?
                    .iter()
                    .map(Node::string)
                    .collect::<Result<Vec<_>>>()?;
                ensure!(
                    members.iter().collect::<BTreeSet<_>>().len() == members.len(),
                    "duplicate historical receipt case"
                );
                for case_id in members {
                    authored.field("cases")?.field(case_id)?;
                }
                receipt.field("body")?.mapping()?;
            }
        }
    } else if node.get("historical_evidence").is_some() {
        anyhow::bail!("historical_evidence must be a list");
    }
    Ok(records)
}

pub struct HistoryAdmission<'a> {
    pub limit: u64,
    pub index: &'a BTreeMap<String, &'a Value>,
}

pub fn load_history(
    root: &Path,
    authored: &Node,
    family: &str,
    implementation: &str,
    resources: &[&Value],
    batch: bool,
    admission: HistoryAdmission<'_>,
) -> Result<History> {
    let mut captures = Vec::new();
    let mut remaining = admission.limit;
    for resource in resources {
        let member = resource["path"].as_str().context("missing capture path")?;
        let version = resource["version"]
            .as_str()
            .context("missing capture version")?;
        ensure!(
            member == format!("{STORE}/families/{family}/{implementation}-{version}.yaml"),
            "capture path identity mismatch"
        );
        let bytes = verified(root, resource)?;
        let raw = yaml::load_str(std::str::from_utf8(&bytes)?)?;
        if raw.get("schema_version") == Some(&Node::Int(2.into())) {
            let records = resolved_snapshot(&raw, authored, family, implementation, version)
                .with_context(|| member.to_string())?;
            for (_, record) in records.mapping()? {
                let source_path = record.field("member")?.string()?;
                let source = admission
                    .index
                    .get(source_path)
                    .context("resolved snapshot source member is outside catalog inventory")?;
                let source_version = source["version"]
                    .as_str()
                    .context("resolved snapshot source has no version")?;
                ensure!(
                    source["kind"] == "history"
                        && source["family"] == family
                        && source["implementation"] == implementation
                        && Release::parse(source_version)? <= Release::parse(version)?,
                    "resolved snapshot source member has inconsistent identity"
                );
            }
            remaining = remaining
                .checked_sub(raw.heap_bytes())
                .context("expanded history exceeds its memory reservation")?;
            let records = records.clone();
            captures.push(Capture {
                id: format!("{implementation}-{version}"),
                version: version.to_string(),
                member: member.to_string(),
                document: raw,
                resolved_records: Some(records),
            });
            continue;
        }
        let mut document = if let Some(execution) = raw.get("execution") {
            fields(&raw, &["execution"], &["parent", "changes"])?;
            let digest = source_reference(execution.string()?, "executions")?;
            ensure!(
                resource["execution"] == digest,
                "execution identity mismatch"
            );
            let descriptor = admission
                .index
                .get(&format!(
                    "{STORE}/{}",
                    crate::history::source_path(digest, "executions")?
                ))
                .context("execution is outside catalog inventory")?;
            let bytes = verified(root, descriptor)?;
            let manifest = yaml::load_str(std::str::from_utf8(&bytes)?)?;
            let listed = manifest
                .get("receipts")
                .map(Node::mapping)
                .transpose()?
                .unwrap_or(&[])
                .iter()
                .map(|(k, _)| k.string())
                .collect::<Result<BTreeSet<_>>>()?;
            ensure!(
                descriptor["receipts"] == serde_json::json!(listed),
                "execution receipt inventory mismatch"
            );
            let mut receipts = BTreeMap::new();
            if let Some(members) = manifest.get("receipts") {
                for (digest, _) in members.mapping()? {
                    let digest = source_reference(digest.string()?, "receipts")?;
                    let descriptor = admission
                        .index
                        .get(&format!(
                            "{STORE}/{}",
                            crate::history::source_path(digest, "receipts")?
                        ))
                        .context("receipt is outside catalog inventory")?;
                    let bytes = verified(root, descriptor)?;
                    receipts.insert(
                        digest.to_string(),
                        yaml::load_str(std::str::from_utf8(&bytes)?)?,
                    );
                }
            }
            let mut expanded = expand_execution_bounded(
                &manifest,
                family,
                implementation,
                version,
                &receipts,
                remaining,
            )?;
            expanded.set(
                "changes",
                raw.get("changes").cloned().unwrap_or_else(Node::map),
            )?;
            expanded
        } else {
            let mut document = raw.clone();
            document.remove("parent")?;
            document
        };
        if let Some(parent) = raw.get("parent") {
            document.set("parent", parent.clone())?;
        }
        capture(&document, authored, implementation, version)
            .with_context(|| member.to_string())?;
        remaining = remaining
            .checked_sub(document.heap_bytes())
            .context("expanded history exceeds its memory reservation")?;
        captures.push(Capture {
            id: format!("{implementation}-{version}"),
            version: version.to_string(),
            member: member.to_string(),
            document,
            resolved_records: None,
        });
    }
    let mut history = History {
        implementation: implementation.to_string(),
        captures,
    };
    history.validate()?;
    if batch {
        let mut cursor = crate::history::Cursor::default();
        for (index, capture) in history.captures.iter().enumerate() {
            cursor.apply(&history, index)?;
            batch_records(&cursor.records(&history, &capture.version)?)?;
        }
    }
    Ok(history)
}

/// Batch producers preserve their final fields instead of claiming stream chunks.
pub fn batch_records(records: &Node) -> Result<()> {
    for (_, record) in records.mapping()? {
        let record = record.field("value")?;
        let document = record.field("document")?;
        let empty = Node::map();
        let origin = document.get("capture_origin").unwrap_or(&empty);
        let adapter = origin.get("adapter_sha256");
        let job = origin.get("job_sha256");
        ensure!(
            adapter.is_some() == job.is_some(),
            "adapter-backed batch observation needs adapter and job identities"
        );
        let metadata = document.get("record_metadata").unwrap_or(&empty);
        let final_fields = metadata.get("batch_final");
        if adapter.is_none() && final_fields.is_none() {
            continue;
        }
        if let Some(value) = record.field("observation")?.get("value") {
            let final_fields = final_fields.context("missing batch final fields")?;
            fields(
                final_fields,
                &["reasoning_text", "normal_text", "calls"],
                &[],
            )?;
            let mut events = Vec::new();
            for (key, kind) in [("reasoning_text", "reasoning"), ("normal_text", "text")] {
                let text = final_fields.field(key)?.string()?;
                if !text.is_empty() {
                    let mut event = Node::map();
                    event.set("kind", Node::Str(kind.into()))?;
                    event.set("text", Node::Str(text.into()))?;
                    events.push(event);
                }
            }
            for call in final_fields.field("calls")?.array()? {
                fields(call, &["name", "arguments"], &[])?;
                call.field("name")?.string()?;
                let mut event = Node::map();
                event.set("kind", Node::Str("tool_call".into()))?;
                event.update(call)?;
                events.push(event);
            }
            ensure!(
                value.get("chunks").is_none()
                    && value.field("assembled")?.equivalent(&Node::Array(events)),
                "batch history does not match final fields"
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn source_qualified_origin_binds_release_and_hash() -> anyhow::Result<()> {
        let record = super::yaml::load_str(&format!(
            "crate_version: 0.5.0\nsource_sha256: {}\n",
            "a".repeat(64)
        ))?;
        assert!(super::origin(&record, Some("0.5.0")).is_ok());
        assert!(super::origin(&record, Some(&format!("0.5.0+source.{}", "a".repeat(64)))).is_ok());
        assert!(super::origin(&record, Some(&format!("0.5.1+source.{}", "a".repeat(64)))).is_err());
        assert!(super::origin(&record, Some(&format!("0.5.0+source.{}", "b".repeat(64)))).is_err());
        Ok(())
    }

    use super::*;

    #[test]
    fn complete_snapshot_binds_stimulus_and_record_provenance() -> Result<()> {
        let authored = yaml::load_str(
            "cases:\n  id:\n    request: {input: sample, init: {}, finish_reason: stop, tools: [], chunks: []}\n",
        )?;
        let snapshot = "schema_version: 2\nprovenance: {status: legacy, captured_with: {dynamo_v2: '0.1.0'}}\ndocument: {}\nrecords:\n  id:\n    member: conformance/fixtures-unified-v2/families/sample/dynamo_v2-0.1.0.yaml\n    value:\n      case_key: UNIFIED.1\n      stimulus: {ref: current}\n      observation: {value: {assembled: []}}\n      document: {captured_with: {dynamo_v2: '0.1.0'}}\n";
        let valid = yaml::load_str(snapshot)?;
        resolved_snapshot(&valid, &authored, "sample", "dynamo_v2", "0.1.0")?;

        let overridden_producer = yaml::load_str(
            &snapshot
                .replace(
                    "provenance: {status: legacy, captured_with: {dynamo_v2: '0.1.0'}}",
                    "provenance: {status: legacy, captured_with: {dynamo_v2: '0.1.1'}}",
                )
                .replace(
                    "document: {captured_with: {dynamo_v2: '0.1.0'}}",
                    "document: {captured_with: {dynamo_v2: '0.1.1'}}",
                ),
        )?;
        resolved_snapshot(
            &overridden_producer,
            &authored,
            "sample",
            "dynamo_v2",
            "0.1.1",
        )?;

        let malformed_stimulus = yaml::load_str(&snapshot.replace("ref: current", "ref: wrong"))?;
        assert!(
            resolved_snapshot(
                &malformed_stimulus,
                &authored,
                "sample",
                "dynamo_v2",
                "0.1.0"
            )
            .is_err()
        );

        let mismatched_provenance = yaml::load_str(&snapshot.replace(
            "document: {captured_with: {dynamo_v2: '0.1.0'}}",
            "document: {captured_with: {dynamo_v2: '0.1.1'}}",
        ))?;
        assert!(
            resolved_snapshot(
                &mismatched_provenance,
                &authored,
                "sample",
                "dynamo_v2",
                "0.1.0"
            )
            .is_err()
        );

        let missing_provenance = yaml::load_str(&snapshot.replace(
            "document: {captured_with: {dynamo_v2: '0.1.0'}}",
            "document: {}",
        ))?;
        assert!(
            resolved_snapshot(
                &missing_provenance,
                &authored,
                "sample",
                "dynamo_v2",
                "0.1.0"
            )
            .is_err()
        );

        let malformed_origin = yaml::load_str(&snapshot.replace(
            "document: {captured_with: {dynamo_v2: '0.1.0'}}",
            "document: {captured_with: {dynamo_v2: '0.1.0'}, capture_origin: {crate_version: '0.1.0', source_sha256: bad}}",
        ))?;
        assert!(
            resolved_snapshot(&malformed_origin, &authored, "sample", "dynamo_v2", "0.1.0")
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn source_archives_preserve_member_bytes_and_reject_corruption() -> Result<()> {
        let _session = archive_session();
        let root = tempfile::tempdir()?;
        let name = "families/sample/dynamo_v2-0.1.0.yaml";
        let bytes = b"typed: [null, 'null', 1, 1.0, true]\r\n";
        let mut tar = tar::Builder::new(Vec::new());
        let mut header = tar::Header::new_ustar();
        header.set_size(bytes.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        tar.append_data(&mut header, name, bytes.as_slice())?;
        let compressed = crate::codec::gzip(&tar.into_inner()?)?;
        let path = format!("{STORE}/versions/dynamo-0.1.0.tar.gz");
        fs::create_dir_all(root.path().join(STORE).join("versions"))?;
        fs::write(root.path().join(&path), &compressed)?;
        let resource = serde_json::json!({"path":format!("{STORE}/{name}"),"sha256":sha256(bytes),"size":bytes.len(),
            "archive":{"path":path,"sha256":sha256(&compressed),"size":compressed.len()}});
        assert_eq!(verified(root.path(), &resource)?, bytes);
        assert_eq!(
            archived_inventory(root.path(), &resource)?,
            BTreeSet::from([name.to_string()])
        );
        let mut wrong = resource.clone();
        wrong["sha256"] = serde_json::json!("0".repeat(64));
        assert!(verified(root.path(), &wrong).is_err());
        wrong = resource.clone();
        wrong["path"] = serde_json::json!(format!("{STORE}/receipts/missing.yaml"));
        assert!(verified(root.path(), &wrong).is_err());
        for tail in [
            b"garbage".to_vec(),
            crate::codec::gzip(b"")?,
            crate::codec::gzip(&[0; 1024])?,
        ] {
            let corrupt = [&compressed[..], &tail[..]].concat();
            fs::write(root.path().join(&path), &corrupt)?;
            let mut repinned = resource.clone();
            repinned["archive"]["sha256"] = serde_json::json!(sha256(&corrupt));
            repinned["archive"]["size"] = serde_json::json!(corrupt.len());
            assert!(verified(root.path(), &repinned).is_err());
        }
        fs::write(root.path().join(&path), b"corrupt")?;
        assert!(
            verified(root.path(), &resource).is_err(),
            "spooled bytes must not hide a changed source archive"
        );
        Ok(())
    }

    fn extension_archive(extensions: &[(u8, Vec<u8>)], tail: usize) -> Result<Vec<u8>> {
        let mut tar = tar::Builder::new(Vec::new());
        for (kind, bytes) in extensions {
            let mut header = tar::Header::new_gnu();
            header.set_entry_type(tar::EntryType::new(*kind));
            header.set_size(bytes.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            tar.append_data(&mut header, "././@LongLink", bytes.as_slice())?;
        }
        let mut header = tar::Header::new_ustar();
        header.set_size(1);
        header.set_mode(0o644);
        header.set_cksum();
        tar.append_data(
            &mut header,
            "families/sample/dynamo_v2-0.1.0.yaml",
            &b"x"[..],
        )?;
        let mut raw = tar.into_inner()?;
        raw.resize(raw.len() + tail, 0);
        crate::codec::gzip(&raw)
    }

    fn pax_comment(size: usize) -> Vec<u8> {
        let prefix = format!("{size} comment=");
        assert!(prefix.len() < size);
        let mut bytes = prefix.into_bytes();
        bytes.resize(size - 1, b'x');
        bytes.push(b'\n');
        bytes
    }

    fn admit_test_archive(bytes: &[u8]) -> Result<Arc<tempfile::TempDir>> {
        let root = tempfile::tempdir()?;
        let path = format!("{STORE}/versions/dynamo-0.1.0.tar.gz");
        fs::create_dir_all(root.path().join(STORE).join("versions"))?;
        fs::write(root.path().join(&path), bytes)?;
        admitted_archive(
            root.path(),
            &serde_json::json!({
                "path":path,"sha256":sha256(bytes),"size":bytes.len()
            }),
        )
    }

    #[test]
    fn source_archives_bound_pax_metadata_before_allocation() -> Result<()> {
        for size in [64 * 1024, 64 * 1024 + 1, 1024 * 1024] {
            let compressed = extension_archive(&[(b'x', pax_comment(size))], 0)?;
            let result = admit_test_archive(&compressed);
            if size == 64 * 1024 {
                assert_eq!(
                    fs::read(result?.path().join("families/sample/dynamo_v2-0.1.0.yaml"))?,
                    b"x"
                );
            } else {
                let error = result.unwrap_err().to_string();
                assert!(
                    error.contains("extension metadata exceeds limit"),
                    "{error}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn source_archives_bound_gnu_metadata_before_allocation() -> Result<()> {
        let name = b"families/sample/dynamo_v2-0.1.0.yaml\0".to_vec();
        assert!(admit_test_archive(&extension_archive(&[(b'L', name.clone())], 0)?).is_ok());
        for size in [64 * 1024, 64 * 1024 + 1, 1024 * 1024] {
            let mut metadata = name.clone();
            metadata.resize(size, 0);
            let error = admit_test_archive(&extension_archive(&[(b'L', metadata)], 0)?)
                .unwrap_err()
                .to_string();
            let expected = if size == 64 * 1024 {
                "member path exceeds limit"
            } else {
                "extension metadata exceeds limit"
            };
            assert!(error.contains(expected), "{error}");
        }
        Ok(())
    }

    #[test]
    fn source_archives_bound_combined_extensions_per_logical_member() -> Result<()> {
        let name = b"families/sample/dynamo_v2-0.1.0.yaml\0".to_vec();
        let compressed = extension_archive(&[(b'L', name), (b'x', pax_comment(64 * 1024))], 0)?;
        assert!(
            admit_test_archive(&compressed)
                .unwrap_err()
                .to_string()
                .contains("extension metadata exceeds limit")
        );
        Ok(())
    }

    #[test]
    fn source_archive_declarations_reject_before_extension_body_reads() -> Result<()> {
        for kind in [b'L', b'K', b'x', b'g'] {
            let mut header = tar::Header::new_gnu();
            header.set_entry_type(tar::EntryType::new(kind));
            header.set_size(64 * 1024 + 1);
            header.set_cksum();
            let mut reader = DeclaredArchive::new(header.as_bytes().as_slice());
            let error = reader.read(&mut [0; 512]).unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("extension metadata exceeds limit")
            );
            assert!(reader.reader.is_empty());
        }
        Ok(())
    }

    #[test]
    fn source_archives_stream_excessive_zero_padding_and_reject_nonzero_tail() -> Result<()> {
        let compressed = extension_archive(&[], 4 * 1024 * 1024)?;
        assert!(admit_test_archive(&compressed).is_ok());
        let raw = crate::codec::gunzip(
            &compressed,
            &sha256(&compressed),
            (4 * 1024 * 1024 + 2048) as usize,
        )?;
        let mut corrupt = raw;
        *corrupt.last_mut().unwrap() = 1;
        assert!(admit_test_archive(&crate::codec::gzip(&corrupt)?).is_err());
        // The padding reader's memory and maximum read size do not grow with
        // decoded padding, including when called outside the gzip admission path.
        struct Zeros {
            remaining: usize,
            largest_read: usize,
        }
        impl Read for Zeros {
            fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
                self.largest_read = self.largest_read.max(bytes.len());
                let size = self.remaining.min(bytes.len());
                bytes[..size].fill(0);
                self.remaining -= size;
                Ok(size)
            }
        }
        let mut zeros = Zeros {
            remaining: 64 * 1024 * 1024,
            largest_read: 0,
        };
        zero_padding(&mut zeros)?;
        assert_eq!(zeros.largest_read, 8192);
        assert_eq!(zeros.remaining, 0);
        Ok(())
    }

    #[test]
    fn source_archive_discovery_shares_descriptor_admission_and_identity_checks() -> Result<()> {
        let _session = archive_session();
        let root = tempfile::tempdir()?;
        let path = format!("{STORE}/versions/dynamo-0.1.0.tar.gz");
        fs::create_dir_all(root.path().join(STORE).join("versions"))?;
        let mut bytes = extension_archive(&[], 0)?;
        fs::write(root.path().join(&path), &bytes)?;
        let (descriptor, directory) = discover_archive(root.path(), &path)?;
        assert_eq!(descriptor["sha256"], sha256(&bytes));
        assert_eq!(descriptor["size"], bytes.len());
        assert!(Arc::ptr_eq(
            &directory,
            &admitted_archive(root.path(), &descriptor)?
        ));
        assert!(Arc::ptr_eq(
            &directory,
            &discover_archive(root.path(), &path)?.1
        ));
        let mut wrong = descriptor.clone();
        wrong["sha256"] = serde_json::json!("0".repeat(64));
        assert!(admitted_archive(root.path(), &wrong).is_err());
        // A length change is observable even when filesystem timestamps are unchanged.
        bytes.push(0);
        fs::write(root.path().join(&path), &bytes)?;
        assert_eq!(
            discover_archive(root.path(), &path)
                .unwrap_err()
                .to_string(),
            "source archive changed during operation"
        );
        assert_eq!(
            admitted_archive(root.path(), &descriptor)
                .unwrap_err()
                .to_string(),
            "source archive changed during operation"
        );
        Ok(())
    }

    #[test]
    fn source_archives_reject_unbounded_member_inventory() -> Result<()> {
        let _session = archive_session();
        let root = tempfile::tempdir()?;
        let mut tar = tar::Builder::new(Vec::new());
        let bytes = b"kind: sample\n";
        for index in 0..4097 {
            let name = format!("families/f{index:05}/dynamo_v2-0.1.0.yaml");
            let mut header = tar::Header::new_ustar();
            header.set_size(bytes.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            tar.append_data(&mut header, name, bytes.as_slice())?;
        }
        let compressed = crate::codec::gzip(&tar.into_inner()?)?;
        let archive_path = format!("{STORE}/versions/dynamo-0.1.0.tar.gz");
        fs::create_dir_all(root.path().join(STORE).join("versions"))?;
        fs::write(root.path().join(&archive_path), &compressed)?;
        let resource = serde_json::json!({
            "path":format!("{STORE}/families/f00000/dynamo_v2-0.1.0.yaml"),
            "sha256":sha256(bytes),
            "size":bytes.len(),
            "archive":{"path":archive_path,"sha256":sha256(&compressed),"size":compressed.len()}
        });

        let error = verified(root.path(), &resource).unwrap_err().to_string();

        assert!(error.contains("source archive member count exceeds limit"));
        Ok(())
    }

    #[test]
    fn source_archive_members_support_parallel_reads() -> Result<()> {
        let _session = archive_session();
        let root = tempfile::tempdir()?;
        let members = [
            (
                "families/alpha/dynamo_v2-0.1.0.yaml",
                b"alpha: 1\n".as_slice(),
            ),
            (
                "families/beta/dynamo_v2-0.1.0.yaml",
                b"beta: 2\n".as_slice(),
            ),
        ];
        let mut tar = tar::Builder::new(Vec::new());
        for (name, bytes) in members {
            let mut header = tar::Header::new_ustar();
            header.set_size(bytes.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            tar.append_data(&mut header, name, bytes)?;
        }
        let compressed = crate::codec::gzip(&tar.into_inner()?)?;
        let archive_path = format!("{STORE}/versions/dynamo-0.1.0.tar.gz");
        fs::create_dir_all(root.path().join(STORE).join("versions"))?;
        fs::write(root.path().join(&archive_path), &compressed)?;
        let resources = members.map(|(name, bytes)| {
            serde_json::json!({
                "path": format!("{STORE}/{name}"),
                "sha256": sha256(bytes),
                "size": bytes.len(),
                "archive": {
                    "path": archive_path.clone(),
                    "sha256": sha256(&compressed),
                    "size": compressed.len(),
                }
            })
        });
        let root = root.path().to_path_buf();
        let readers = resources.clone().map(|resource| {
            let root = root.clone();
            std::thread::spawn(move || verified(&root, &resource))
        });
        let [left, right] = readers;
        assert_ne!(left.thread().id(), right.thread().id());
        assert_eq!(left.join().expect("reader panicked")?, b"alpha: 1\n");
        assert_eq!(right.join().expect("reader panicked")?, b"beta: 2\n");

        let resource = &resources[0];
        let mut independent = resource.clone();
        let independent_path = format!("{STORE}/versions/dynamo-0.2.0.tar.gz");
        fs::write(root.join(&independent_path), &compressed)?;
        independent["archive"]["path"] = serde_json::json!(independent_path);
        let blocked = SOURCE_ARCHIVES
            .get()
            .unwrap()
            .lock()
            .unwrap()
            .directories
            .get(&root.join(&archive_path))
            .unwrap()
            .clone();
        let admission = blocked.lock().unwrap();
        let references = Arc::strong_count(&blocked);
        let timeout = std::time::Duration::from_secs(5);
        let (completed, completion) = std::sync::mpsc::sync_channel(1);
        let (entered, result, retained) = std::thread::scope(|scope| {
            let retained = scope.spawn(|| verified(&root, resource));
            let deadline = std::time::Instant::now() + timeout;
            while Arc::strong_count(&blocked) == references && std::time::Instant::now() < deadline
            {
                std::thread::yield_now();
            }
            let entered = Arc::strong_count(&blocked) > references;
            scope.spawn(|| completed.send(verified(&root, &independent)).unwrap());
            // A reader waiting on one archive must not block another admission.
            let result = completion.recv_timeout(timeout);
            drop(admission);
            (entered, result, retained.join().expect("reader panicked"))
        });
        assert!(
            entered,
            "the retained reader never entered archive admission"
        );
        assert_eq!(
            result.context("an unrelated archive waited for the blocked admission")??,
            b"alpha: 1\n"
        );
        assert_eq!(retained?, b"alpha: 1\n");
        Ok(())
    }

    #[test]
    fn request_hashing_retains_big_integers_and_python_float_spelling() -> Result<()> {
        let value = yaml::load_str("z: 18446744073709551617\na: [1.0e-7, 1.0, -0.0]\n")?;
        assert_eq!(
            request_json(&value)?,
            "{\"a\":[1e-07,1.0,-0.0],\"z\":18446744073709551617}"
        );
        Ok(())
    }

    #[test]
    fn canonical_alias_collisions_are_not_literal_collisions() -> Result<()> {
        let metadata = serde_json::json!({"canonical":{"invalid_call":"31-guided_json_invalid_call"},"families":{},"owners":{"31-1":["invalid_call"]}});
        assert_eq!(
            canonical_alias("UNIFIED.31.a", "sample", &metadata)?,
            canonical_alias("UNIFIED.31-1", "sample", &metadata)?
        );
        Ok(())
    }

    #[test]
    fn batch_contract_rejects_chunks_missing_final_and_unpaired_receipts() -> Result<()> {
        let mut record = yaml::load_str(
            "one: {value: {observation: {value: {assembled: []}}, document: {record_metadata: {batch_final: {reasoning_text: '', normal_text: '', calls: []}}}}}\n",
        )?;
        batch_records(&record)?;
        let Node::Map(entries) = &mut record else {
            unreachable!()
        };
        let Node::Map(value) = &mut entries[0].1 else {
            unreachable!()
        };
        let mut inner = value[0].1.clone();
        let mut observation = inner.field("observation")?.clone();
        observation.set("value", yaml::load_str("chunks: []\nassembled: []\n")?)?;
        inner.set("observation", observation)?;
        value[0].1 = inner;
        assert!(batch_records(&record).is_err());
        Ok(())
    }
}

#[cfg(test)]
mod qualification_tests {
    use super::*;

    #[test]
    fn tar_termination_matches_browser_admission() -> Result<()> {
        let mut archive = tar::Builder::new(Vec::new());
        let mut header = tar::Header::new_gnu();
        header.set_mode(0o644);
        header.set_size(1);
        header.set_cksum();
        archive.append_data(&mut header, "family/source.yaml", &b"x"[..])?;
        let raw = archive.into_inner()?;
        let descriptor =
            serde_json::json!({"members":[{"path":"family/source.yaml","kind":"file","size":1}]});
        super::archive(&crate::codec::gzip(&raw)?, &descriptor)?;
        for mut corrupt in [
            raw[..raw.len() - 1024].to_vec(),
            raw[..raw.len() - 512].to_vec(),
            raw.clone(),
        ] {
            if corrupt.len() == raw.len() {
                corrupt.push(b'x');
            }
            assert!(super::archive(&crate::codec::gzip(&corrupt)?, &descriptor).is_err());
        }
        Ok(())
    }

    #[test]
    fn family_overrides_and_foreign_owner_exclusions_are_preserved() -> Result<()> {
        let metadata = serde_json::json!({"canonical":{"kimi_overlap":"kimi_k2-kimi_overlap","foreign":"other-foreign"},"family_scenarios":{"kimi_k2":{"1-2":"kimi_overlap"}},"families":{},"owners":{"foreign-old":["foreign"]},"owner_families":{"foreign":"other"}});
        assert_eq!(
            canonical_alias("UNIFIED.1-2", "kimi_k2", &metadata)?,
            "kimi_k2-kimi_overlap"
        );
        assert_eq!(
            canonical_alias("UNIFIED.foreign-old", "kimi_k2", &metadata)?,
            "foreign-old"
        );
        assert_eq!(
            canonical_alias("UNIFIED.unknown.kimi_k2", "kimi_k2", &metadata)?,
            "unknown.kimi_k2"
        );
        Ok(())
    }
}
