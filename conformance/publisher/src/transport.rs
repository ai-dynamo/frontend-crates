// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared, content-addressed transport for canonical report JSON and exact originals.

use crate::codec::{gunzip, gzip, sha256};
use anyhow::{Context, Result, ensure};
use fastcdc::v2020::FastCDC;
use rayon::prelude::*;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeSet, HashMap},
    fs,
    path::{Path, PathBuf},
};

pub const MIN_CHUNK: u32 = 8 * 1024;
pub const TARGET_CHUNK: u32 = 32 * 1024;
pub const MAX_CHUNK: u32 = 128 * 1024;
pub const MAX_PACK: usize = 1024 * 1024;
pub const VALIDATION_CACHE_BYTES: usize = 64 * MAX_PACK;

type ChunkKey = (String, usize);

struct Location {
    pack: usize,
    offset: usize,
}

struct Resource {
    sha256: String,
    size: usize,
    chunks: Vec<ChunkKey>,
}

/// Batches finalized packs within the publication pool; compressed bytes live on disk.
pub struct PackWriter {
    stage: PathBuf,
    pending: Vec<u8>,
    finalized: Vec<(usize, Vec<u8>)>,
    packs: Vec<Value>,
    chunks: HashMap<ChunkKey, Location>,
    resources: Vec<Resource>,
}

impl PackWriter {
    /// Create a writer in a private publication staging directory.
    pub fn new(stage: &Path) -> Self {
        Self {
            stage: stage.to_path_buf(),
            pending: Vec::new(),
            finalized: Vec::new(),
            packs: Vec::new(),
            chunks: HashMap::new(),
            resources: Vec::new(),
        }
    }

    /// Add exact logical bytes and return a recipe token. Flush before reading its recipe.
    pub fn add(&mut self, bytes: &[u8]) -> Result<usize> {
        let mut resource = Resource {
            sha256: sha256(bytes),
            size: bytes.len(),
            chunks: Vec::new(),
        };
        for chunk in FastCDC::new(bytes, MIN_CHUNK, TARGET_CHUNK, MAX_CHUNK) {
            let data = &bytes[chunk.offset..chunk.offset + chunk.length];
            let key = (sha256(data), data.len());
            if !self.chunks.contains_key(&key) {
                if self.pending.len() + data.len() > MAX_PACK {
                    self.seal_pending();
                    if self.finalized.len() >= rayon::current_num_threads() {
                        self.compress_finalized()?;
                    }
                }
                let location = Location {
                    pack: self.packs.len(),
                    offset: self.pending.len(),
                };
                self.pending.extend_from_slice(data);
                self.chunks.insert(key.clone(), location);
            }
            resource.chunks.push(key);
        }
        let token = self.resources.len();
        self.resources.push(resource);
        Ok(token)
    }

    /// Seal a dependency phase so later history/original bytes cannot enter its packs.
    /// Previously written chunks remain reusable by every subsequent phase.
    pub fn flush_phase(&mut self) -> Result<()> {
        self.seal_pending();
        self.compress_finalized()
    }

    fn seal_pending(&mut self) {
        if !self.pending.is_empty() {
            let index = self.packs.len();
            self.packs.push(Value::Null);
            self.finalized
                .push((index, std::mem::take(&mut self.pending)));
        }
    }

    fn compress_finalized(&mut self) -> Result<()> {
        // Reserve identities in input order before compression; scheduling must
        // never change chunk locations, phase boundaries or gzip level nine.
        let stage = &self.stage;
        let descriptors = self.finalized.par_iter().map(|(index, bytes)| {
            let encoded = gzip(bytes)?;
            let digest = sha256(&encoded);
            let path = format!("conformance/viewer-data/packs/{digest}.bin.gz");
            let target = stage.join(&path);
            fs::create_dir_all(target.parent().context("pack path has no parent")?)?;
            fs::write(target, &encoded)?;
            Ok((*index, json!({"path":path,"sha256":digest,"size":encoded.len(),"decoded_sha256":sha256(bytes),"decoded_size":bytes.len()})))
        }).collect::<Result<Vec<_>>>()?;
        for (index, descriptor) in descriptors {
            self.packs[index] = descriptor;
        }
        self.finalized.clear();
        Ok(())
    }

    /// Return a complete transport-v2 recipe, never an unfinished pack descriptor.
    pub fn recipe(&self, token: usize) -> Result<Value> {
        let resource = self.resources.get(token).context("invalid recipe token")?;
        let chunks = resource
            .chunks
            .iter()
            .map(|key| {
                let location = &self.chunks[key];
                let pack = self
                    .packs
                    .get(location.pack)
                    .filter(|pack| !pack.is_null())
                    .context("recipe requires a phase flush")?;
                Ok(json!({"sha256":key.0,"size":key.1,"pack":pack,"offset":location.offset}))
            })
            .collect::<Result<Vec<Value>>>()?;
        Ok(json!({"schema":2,"sha256":resource.sha256,"size":resource.size,"chunks":chunks}))
    }

    /// Flush remaining bytes and materialize every recipe in insertion order.
    pub fn finish(mut self) -> Result<Vec<Value>> {
        self.flush_phase()?;
        (0..self.resources.len())
            .map(|token| self.recipe(token))
            .collect()
    }

    /// Completed pack identities, used by the publication dependency inventory.
    pub fn packs(&self) -> &[Value] {
        &self.packs
    }
}

/// Reconstruct and validate a recipe without trusting paths, bounds or gzip headers.
pub fn reconstruct(stage: &Path, recipe: &Value) -> Result<Vec<u8>> {
    let mut result = Vec::new();
    visit_recipe(
        stage,
        recipe,
        &mut RecipeValidator::new(VALIDATION_CACHE_BYTES),
        |bytes| result.extend_from_slice(bytes),
    )?;
    Ok(result)
}

struct VerifiedPack {
    descriptor: Value,
    used: BTreeSet<(usize, usize)>,
}

struct DecodedPack {
    bytes: Vec<u8>,
    last_used: u64,
}

struct RecipeValidator {
    packs: HashMap<String, VerifiedPack>,
    decoded: HashMap<String, DecodedPack>,
    decoded_bytes: usize,
    limit: usize,
    clock: u64,
}

impl RecipeValidator {
    fn new(limit: usize) -> Self {
        assert!(limit >= MAX_PACK);
        Self {
            packs: HashMap::new(),
            decoded: HashMap::new(),
            decoded_bytes: 0,
            limit,
            clock: 0,
        }
    }

    fn load(&mut self, stage: &Path, path: &str, pack: &Value) -> Result<()> {
        if let Some(previous) = self.packs.get(path) {
            ensure!(
                previous.descriptor == *pack,
                "inconsistent shared pack identity"
            );
        } else {
            self.packs.insert(
                path.to_string(),
                VerifiedPack {
                    descriptor: pack.clone(),
                    used: BTreeSet::new(),
                },
            );
        }
        self.clock += 1;
        if let Some(cached) = self.decoded.get_mut(path) {
            cached.last_used = self.clock;
            return Ok(());
        }
        let decoded_size = number(pack, "decoded_size")?;
        // Keep topology metadata after eviction; bytes are revalidated on reload.
        while self.decoded_bytes + decoded_size > self.limit {
            let oldest = self
                .decoded
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(path, _)| path.clone())
                .context("missing eviction candidate")?;
            self.decoded_bytes -= self.decoded.remove(&oldest).unwrap().bytes.len();
        }
        let encoded = crate::source::read(stage, path)?;
        ensure!(encoded.len() == number(pack, "size")?, "pack size mismatch");
        let decoded = gunzip(
            &encoded,
            pack["sha256"].as_str().context("missing pack hash")?,
            decoded_size,
        )?;
        ensure!(
            decoded.len() == decoded_size
                && Some(sha256(&decoded).as_str()) == pack["decoded_sha256"].as_str(),
            "decoded pack identity mismatch"
        );
        self.decoded_bytes += decoded.len();
        self.decoded.insert(
            path.to_string(),
            DecodedPack {
                bytes: decoded,
                last_used: self.clock,
            },
        );
        Ok(())
    }
}

/// Validate all recipes with a bounded decoded-pack cache and streaming resource hashes.
/// Every decoded byte must belong to a referenced chunk, with no overlapping chunks.
pub fn validate_recipes(stage: &Path, recipes: &[Value]) -> Result<()> {
    let mut cache = RecipeValidator::new(VALIDATION_CACHE_BYTES);
    for recipe in recipes {
        visit_recipe(stage, recipe, &mut cache, |_| {})?;
    }
    for pack in cache.packs.values() {
        let mut end = 0;
        for (offset, length) in &pack.used {
            ensure!(*offset == end, "unreferenced or overlapping packed bytes");
            end = offset
                .checked_add(*length)
                .context("packed range overflow")?;
        }
        ensure!(
            end == number(&pack.descriptor, "decoded_size")?,
            "unreferenced trailing packed bytes"
        );
    }
    let directory = stage.join("conformance/viewer-data/packs");
    if directory.exists() {
        let files = crate::source::inventory(&directory)?;
        let expected = cache
            .packs
            .keys()
            .map(|path| {
                path.strip_prefix("conformance/viewer-data/packs/")
                    .unwrap()
                    .to_string()
            })
            .collect::<BTreeSet<_>>();
        ensure!(files == expected, "unreferenced pack files");
    }
    Ok(())
}

fn visit_recipe(
    stage: &Path,
    recipe: &Value,
    cache: &mut RecipeValidator,
    mut consume: impl FnMut(&[u8]),
) -> Result<()> {
    ensure!(recipe["schema"] == 2, "unsupported transport recipe");
    let size = number(recipe, "size")?;
    let mut seen = 0usize;
    let mut hash = Sha256::new();
    for chunk in recipe["chunks"]
        .as_array()
        .context("missing recipe chunks")?
    {
        let pack = &chunk["pack"];
        let path = pack["path"].as_str().context("missing pack path")?;
        let digest = pack["sha256"].as_str().context("missing pack hash")?;
        ensure!(
            digest.len() == 64
                && digest
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
            "invalid pack hash"
        );
        ensure!(
            path == format!("conformance/viewer-data/packs/{digest}.bin.gz"),
            "invalid pack path"
        );
        let decoded_size = number(pack, "decoded_size")?;
        ensure!(
            decoded_size > 0 && decoded_size <= MAX_PACK,
            "invalid decoded pack size"
        );
        cache.load(stage, path, pack)?;
        let cached = cache.decoded.get(path).context("missing verified pack")?;
        let offset = number(chunk, "offset")?;
        let length = number(chunk, "size")?;
        ensure!(
            length > 0 && length <= MAX_CHUNK as usize,
            "invalid chunk size"
        );
        let end = offset
            .checked_add(length)
            .context("chunk offset overflow")?;
        let data = cached
            .bytes
            .get(offset..end)
            .context("chunk outside pack")?;
        ensure!(
            Some(sha256(data).as_str()) == chunk["sha256"].as_str(),
            "chunk hash mismatch"
        );
        ensure!(
            seen.checked_add(length).context("resource size overflow")? <= size,
            "resource exceeds declared size"
        );
        seen += length;
        hash.update(data);
        consume(data);
        cache
            .packs
            .get_mut(path)
            .context("missing pack topology")?
            .used
            .insert((offset, length));
    }
    ensure!(
        seen == size
            && Some(format!("{:x}", hash.finalize()).as_str()) == recipe["sha256"].as_str(),
        "logical resource identity mismatch"
    );
    Ok(())
}

fn number(value: &Value, key: &str) -> Result<usize> {
    Ok(usize::try_from(
        value[key]
            .as_u64()
            .with_context(|| format!("invalid {key}"))?,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Temp(PathBuf);
    impl Temp {
        fn new() -> Result<Self> {
            Ok(Self(crate::transaction::unique(
                &std::env::temp_dir(),
                "transport-test-",
            )?))
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).expect("remove fixture-owned transport directory");
        }
    }

    fn payload(size: usize) -> Vec<u8> {
        let mut state = 7_u64;
        (0..size)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state as u8
            })
            .collect()
    }

    #[test]
    fn deterministic_bounded_packs_and_exact_roundtrip() -> Result<()> {
        let first = Temp::new()?;
        let second = Temp::new()?;
        let bytes = payload(3 * MAX_PACK);
        let build = |stage: &Path| -> Result<Vec<Value>> {
            let mut writer = PackWriter::new(stage);
            writer.add(&bytes)?;
            writer.add(b"\xef\xbb\xbfline\r\n\0\xff")?;
            writer.add(b"")?;
            writer.finish()
        };
        let one = rayon::ThreadPoolBuilder::new().num_threads(1).build()?;
        let many = rayon::ThreadPoolBuilder::new().num_threads(4).build()?;
        let recipes = one.install(|| build(&first.0))?;
        assert_eq!(recipes, many.install(|| build(&second.0))?);
        assert_eq!(reconstruct(&first.0, &recipes[0])?, bytes);
        assert_eq!(
            reconstruct(&first.0, &recipes[1])?,
            b"\xef\xbb\xbfline\r\n\0\xff"
        );
        assert_eq!(reconstruct(&first.0, &recipes[2])?, b"");
        for chunk in recipes[0]["chunks"].as_array().unwrap() {
            assert!(number(&chunk["pack"], "decoded_size")? <= MAX_PACK);
        }
        Ok(())
    }

    #[test]
    fn deduplication_survives_phase_boundaries_without_polluting_default_pack() -> Result<()> {
        let temp = Temp::new()?;
        let mut writer = PackWriter::new(&temp.0);
        let bytes = payload(200_000);
        let default = writer.add(&bytes)?;
        assert!(writer.recipe(default).is_err());
        writer.flush_phase()?;
        let original = writer.recipe(default)?;
        let default_packs = writer.packs().to_vec();
        let duplicate = writer.add(&bytes)?;
        let history = writer.add(b"unrelated historical bytes")?;
        writer.flush_phase()?;
        assert_eq!(original, writer.recipe(duplicate)?);
        assert_eq!(original, writer.recipe(default)?);
        assert_eq!(writer.packs().len(), default_packs.len() + 1);
        for chunk in original["chunks"].as_array().unwrap() {
            assert!(default_packs.contains(&chunk["pack"]));
        }
        let history = writer.recipe(history)?;
        assert!(!default_packs.contains(&history["chunks"][0]["pack"]));
        assert!(writer.recipe(usize::MAX).is_err());
        Ok(())
    }

    #[test]
    fn shared_validation_rejects_unreferenced_bytes_and_conflicting_cache_identity() -> Result<()> {
        let temp = Temp::new()?;
        let mut writer = PackWriter::new(&temp.0);
        writer.add(b"first exact resource")?;
        writer.add(b"second exact resource")?;
        let recipes = writer.finish()?;
        validate_recipes(&temp.0, &recipes)?;
        assert!(validate_recipes(&temp.0, &recipes[..1]).is_err());
        let mut conflicting = recipes.clone();
        conflicting[1]["chunks"][0]["pack"]["decoded_sha256"] =
            json!("conflicting cached identity");
        assert!(validate_recipes(&temp.0, &conflicting).is_err());
        let extra = temp
            .0
            .join("conformance/viewer-data/packs/unreferenced.bin.gz");
        fs::write(extra, gzip(b"orphan")?)?;
        assert!(validate_recipes(&temp.0, &recipes).is_err());
        Ok(())
    }

    #[test]
    fn validation_bounds_payload_cache_and_rechecks_evicted_pack() -> Result<()> {
        let temp = Temp::new()?;
        let mut writer = PackWriter::new(&temp.0);
        writer.add(&payload(3 * MAX_PACK))?;
        let recipes = writer.finish()?;
        let mut cache = RecipeValidator::new(MAX_PACK);
        visit_recipe(&temp.0, &recipes[0], &mut cache, |_| {})?;
        assert!(cache.decoded_bytes <= MAX_PACK);
        assert!(cache.packs.len() > cache.decoded.len());
        let path = recipes[0]["chunks"][0]["pack"]["path"].as_str().unwrap();
        assert!(!cache.decoded.contains_key(path));
        let mut conflicting = recipes[0].clone();
        conflicting["chunks"][0]["pack"]["decoded_sha256"] = json!("conflicting identity");
        assert!(visit_recipe(&temp.0, &conflicting, &mut cache, |_| {}).is_err());
        let target = temp.0.join(path);
        let mut corrupt = fs::read(&target)?;
        corrupt[12] ^= 1;
        fs::write(&target, corrupt)?;
        assert!(visit_recipe(&temp.0, &recipes[0], &mut cache, |_| {}).is_err());
        Ok(())
    }

    #[test]
    fn rejects_corrupt_missing_and_malformed_transport() -> Result<()> {
        let temp = Temp::new()?;
        let mut writer = PackWriter::new(&temp.0);
        writer.add(&payload(500_000))?;
        let recipe = writer.finish()?.remove(0);
        let mut variants = Vec::new();
        for (key, value) in [
            ("offset", json!(u64::MAX)),
            ("size", json!(0)),
            ("sha256", json!("incorrect")),
        ] {
            let mut bad = recipe.clone();
            bad["chunks"][0][key] = value;
            variants.push(bad);
        }
        for (key, value) in [
            ("decoded_size", json!(MAX_PACK + 1)),
            ("decoded_sha256", json!("incorrect")),
            ("size", json!(0)),
            ("path", json!("../outside")),
        ] {
            let mut bad = recipe.clone();
            bad["chunks"][0]["pack"][key] = value;
            variants.push(bad);
        }
        let mut reordered = recipe.clone();
        reordered["chunks"].as_array_mut().unwrap().swap(0, 1);
        variants.push(reordered);
        let mut truncated = recipe.clone();
        truncated["chunks"].as_array_mut().unwrap().pop();
        variants.push(truncated);
        for bad in variants {
            assert!(reconstruct(&temp.0, &bad).is_err());
        }
        let pack = temp
            .0
            .join(recipe["chunks"][0]["pack"]["path"].as_str().unwrap());
        let mut data = fs::read(&pack)?;
        data[12] ^= 1;
        fs::write(&pack, data)?;
        assert!(reconstruct(&temp.0, &recipe).is_err());
        fs::remove_file(&pack)?;
        assert!(reconstruct(&temp.0, &recipe).is_err());
        Ok(())
    }
}
