// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The Hugging Face backend must return what the `tokenizers` 0.23.2 backend returned.
//!
//! `tests/data/hf-expectations/<case>/expected.json` holds outputs frozen from that backend
//! before it was removed; see the README beside them. Each test replays a case's recorded
//! inputs through the current backend and compares every output. The few places where the
//! 1.x pipeline differs are listed in this file by case and field, never by editing a fixture.

use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

use dynamo_tokenizers::{
    DecodeStream, EncodeSegment, FastTokenizer, HuggingFaceTokenizer, TokenizerOptions,
    traits::Tokenizer,
};
use serde_json::{Value, json};

const DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data");

/// Configurations the legacy backend served and the 1.x pipeline cannot: the backend refuses
/// them at load with this error rather than change ids or text.
const REJECTED: &[(&str, &str)] = &[
    (
        "byte-fallback-bpe-partial-alphabet",
        "Byte fallback `<0x00>` not found in the vocabulary",
    ),
    (
        "byte-level-bpe-replace-decoder",
        "unsupported byte-level BPE decoder",
    ),
    (
        "byte-level-bpe-normalized-added-token",
        "unsupported added token \"CONTROL\"",
    ),
];

fn cases() -> Vec<(String, Value)> {
    let root = Path::new(DATA).join("hf-expectations");
    let mut dirs: Vec<PathBuf> = fs::read_dir(&root)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.is_dir())
        .collect();
    dirs.sort();
    dirs.into_iter()
        .map(|dir| {
            let name = dir.file_name().unwrap().to_str().unwrap().to_string();
            let expected = fs::read_to_string(dir.join("expected.json")).unwrap();
            (name, serde_json::from_str(&expected).unwrap())
        })
        .collect()
}

fn tokenizer_path(expected: &Value) -> String {
    format!(
        "{DATA}/{}/tokenizer.json",
        expected["model_dir"].as_str().unwrap()
    )
}

fn result_ids(result: dynamo_tokenizers::Result<dynamo_tokenizers::Encoding>) -> Value {
    match result {
        Ok(encoding) => json!({"ids": encoding.token_ids()}),
        Err(error) => json!({"error": error.to_string()}),
    }
}

fn decode_pair(tokenizer: &dyn Tokenizer, ids: &[u32]) -> Value {
    let decode = |skip| match tokenizer.decode(ids, skip) {
        Ok(text) => json!(String::from(text)),
        Err(error) => json!({"error": error.to_string()}),
    };
    json!([decode(false), decode(true)])
}

fn stream(tokenizer: &Arc<dyn Tokenizer>, prompt: &[u32], ids: &[u32], skip: bool) -> Value {
    let mut stream = DecodeStream::new(tokenizer.clone(), prompt, skip);
    let chunk = |result: dynamo_tokenizers::Result<Option<String>>| match result {
        Ok(text) => json!(text),
        Err(error) => json!({"error": error.to_string()}),
    };
    let steps: Vec<Value> = ids.iter().map(|id| chunk(stream.step(*id))).collect();
    json!({"steps": steps, "finish": chunk(stream.finish())})
}

fn stream_pair(tokenizer: &Arc<dyn Tokenizer>, ids: &[u32]) -> Value {
    json!([
        stream(tokenizer, &[], ids, false),
        stream(tokenizer, &[], ids, true)
    ])
}

fn ids_of(value: &Value) -> Vec<u32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|id| id.as_u64().unwrap() as u32)
        .collect()
}

/// Replay the inputs `expected` records and return the outputs in the same shape.
fn replay(expected: &Value) -> Value {
    let path = tokenizer_path(expected);
    let plain = HuggingFaceTokenizer::from_file(&path).unwrap();
    let special_tokens = plain.special_tokens().to_vec();
    let token_id_bound = plain.token_id_bound();
    let validate_prefix_cache = plain.validate_prefix_cache().is_ok();
    let validate_segment_prefix_cache = plain.validate_segment_prefix_cache().is_ok();
    let num_special_tokens_added = plain.num_special_tokens_added().unwrap();
    let plain: Arc<dyn Tokenizer> = Arc::new(plain);
    let with_special_tokens: Arc<dyn Tokenizer> = Arc::new(
        HuggingFaceTokenizer::from_file(&path)
            .unwrap()
            .with_options(TokenizerOptions {
                add_special_tokens: true,
            }),
    );
    let fast = FastTokenizer::from_file(&path).ok();
    if let Some(fast) = &fast {
        // The fastokens backend answers from its Hugging Face decoder.
        assert_eq!(fast.special_tokens(), special_tokens);
        assert_eq!(fast.token_id_bound(), token_id_bound);
        assert_eq!(fast.vocab_size(), plain.vocab_size());
        assert_eq!(
            fast.special_token_ids().unwrap(),
            plain.special_token_ids().unwrap()
        );
    }
    let fast: Option<Arc<dyn Tokenizer>> = fast.map(|fast| Arc::new(fast) as Arc<dyn Tokenizer>);
    let special_ids = plain.special_token_ids().unwrap();

    let token_to_id: serde_json::Map<String, Value> = expected["token_to_id"]
        .as_object()
        .unwrap()
        .keys()
        .map(|probe| (probe.clone(), json!(plain.token_to_id(probe).unwrap())))
        .collect();

    let added_tokens: Vec<Value> = expected["added_tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|token| {
            let content = token["content"].as_str().unwrap();
            let id = plain.token_to_id(content).unwrap();
            json!({
                "id": id,
                "content": content,
                "special": id.is_some_and(|id| special_ids.contains(&id)),
                "encode": result_ids(plain.encode(content)),
                "decode": id.map(|id| decode_pair(plain.as_ref(), &[id])),
            })
        })
        .collect();

    let texts: Vec<Value> = expected["texts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|recorded| {
            let text = recorded["text"].as_str().unwrap();
            let mut record = json!({"text": text});
            match plain.encode(text) {
                Ok(encoding) => {
                    let ids = encoding.token_ids();
                    record["ids"] = json!(ids);
                    record["decode"] = decode_pair(plain.as_ref(), ids);
                    record["stream"] = stream_pair(&plain, ids);
                }
                Err(error) => record["error"] = json!(error.to_string()),
            }
            match with_special_tokens.encode(text) {
                Ok(encoding) => {
                    let ids = encoding.token_ids();
                    record["special_ids"] = json!(ids);
                    record["special_decode"] = decode_pair(plain.as_ref(), ids);
                }
                Err(error) => record["special_error"] = json!(error.to_string()),
            }
            if let Some(fast) = &fast {
                record["fast"] = match fast.encode(text) {
                    Ok(encoding) => {
                        let ids = encoding.token_ids();
                        json!({
                            "ids": ids,
                            "decode": decode_pair(fast.as_ref(), ids),
                            "stream": stream_pair(fast, ids),
                        })
                    }
                    Err(error) => json!({"error": error.to_string()}),
                };
            }
            record
        })
        .collect();

    // A batch encodes each input as a single encode does.
    let inputs: Vec<&str> = expected["texts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|recorded| recorded["text"].as_str().unwrap())
        .collect();
    let batch = plain.encode_batch(&inputs).unwrap();
    assert_eq!(batch.len(), inputs.len());
    for (encoding, single) in batch.iter().zip(&texts) {
        assert_eq!(json!(encoding.token_ids()), single["ids"]);
    }
    assert!(plain.encode_batch(&[]).unwrap().is_empty());

    // The recorded ids are replayed, so a prefix decode is compared even where encoding differs.
    let prefixes: Vec<Value> = expected["prefixes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|recorded| {
            let ids = ids_of(&recorded["ids"]);
            let decoded: Vec<Value> = (1..=ids.len())
                .map(|end| {
                    let decoded = plain.decode(&ids[..end], false).unwrap();
                    json!({
                        "decode": decoded.as_str(),
                        "partial": decoded.is_partial(),
                        "unstable": plain.has_unstable_suffix(&ids[..end], false),
                    })
                })
                .collect();
            let streams: Vec<Value> = (0..ids.len())
                .map(|prompt| stream(&plain, &ids[..prompt], &ids[prompt..], false))
                .collect();
            json!({"text": recorded["text"], "ids": ids, "prefixes": decoded, "streams": streams})
        })
        .collect();

    let segments: Vec<Value> = expected["segments"]
        .as_array()
        .unwrap()
        .iter()
        .map(|recorded| {
            let owned: Vec<(String, bool)> =
                serde_json::from_value(recorded["segments"].clone()).unwrap();
            let borrowed: Vec<EncodeSegment<'_>> = owned
                .iter()
                .map(|(text, allow_special)| EncodeSegment::new(text, *allow_special))
                .collect();
            json!({
                "segments": owned,
                "plain": result_ids(plain.encode_segments(&borrowed)),
                "special": result_ids(with_special_tokens.encode_segments(&borrowed)),
                "fast": fast.as_ref().map(|fast| result_ids(fast.encode_segments(&borrowed))),
            })
        })
        .collect();

    let sequences: Vec<Value> = expected["sequences"]
        .as_array()
        .unwrap()
        .iter()
        .map(|recorded| {
            let ids = ids_of(&recorded["ids"]);
            json!({
                "ids": ids,
                "decode": decode_pair(plain.as_ref(), &ids),
                "unstable": [plain.has_unstable_suffix(&ids, false), plain.has_unstable_suffix(&ids, true)],
                "stream": stream_pair(&plain, &ids),
            })
        })
        .collect();

    json!({
        "model_dir": expected["model_dir"],
        "vocab_size": plain.vocab_size(),
        "token_id_bound": token_id_bound,
        "special_token_ids": special_ids,
        "special_tokens": special_tokens,
        "num_special_tokens_added": num_special_tokens_added,
        "validate_prefix_cache": validate_prefix_cache,
        "validate_segment_prefix_cache": validate_segment_prefix_cache,
        "fast_loads": fast.is_some(),
        "token_to_id": token_to_id,
        "added_tokens": added_tokens,
        "texts": texts,
        "prefixes": prefixes,
        "segments": segments,
        "sequences": sequences,
    })
}

/// Every leaf where `actual` differs from `expected`, as `path: expected X, got Y`.
fn differences(path: &str, expected: &Value, actual: &Value, out: &mut Vec<String>) {
    match (expected, actual) {
        (Value::Object(expected), Value::Object(actual)) => {
            for (key, value) in expected {
                let actual = actual.get(key).unwrap_or(&Value::Null);
                differences(&format!("{path}.{key}"), value, actual, out);
            }
            for key in actual.keys().filter(|key| !expected.contains_key(*key)) {
                out.push(format!("{path}.{key}: not recorded, got {}", actual[key]));
            }
        }
        (Value::Array(expected), Value::Array(actual)) if expected.len() == actual.len() => {
            // A record names itself by its input, which reads better than an index.
            for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
                let label = ["text", "content", "segments", "ids"]
                    .iter()
                    .find_map(|key| expected.get(key).filter(|_| expected.is_object()))
                    .map_or(index.to_string(), Value::to_string);
                differences(&format!("{path}[{label}]"), expected, actual, out);
            }
        }
        _ if expected != actual => out.push(format!("{path}: expected {expected}, got {actual}")),
        _ => {}
    }
}

#[test]
fn hugging_face_backend_matches_the_frozen_legacy_outputs() {
    let mut failures = Vec::new();
    let mut compared = 0;
    for (name, mut expected) in cases() {
        expected.as_object_mut().unwrap().remove("provenance");
        if expected.get("load_error").is_some()
            || REJECTED.iter().any(|(rejected, _)| *rejected == name)
        {
            continue;
        }
        if let Err(error) = HuggingFaceTokenizer::from_file(tokenizer_path(&expected)) {
            failures.push(format!("{name}: does not load: {error:#}"));
            continue;
        }
        let actual = replay(&expected);
        differences(&name, &expected, &actual, &mut failures);
        compared += 1;
    }
    assert!(compared >= 20, "only {compared} cases were compared");
    assert!(
        failures.is_empty(),
        "{} differences from the frozen legacy outputs:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn unsupported_configurations_are_rejected_at_load() {
    for (name, message) in REJECTED {
        let path = format!("{DATA}/hf-expectations/{name}/tokenizer.json");
        for error in [
            HuggingFaceTokenizer::from_file(&path).err(),
            FastTokenizer::from_file(&path).err(),
        ] {
            let error = format!("{:#}", error.expect("must not load"));
            assert!(
                error.contains(message) && error.contains(&path),
                "{name}: {error}"
            );
        }
    }
}

#[test]
fn configurations_the_legacy_backend_could_not_load() {
    // Neither file is a complete tokenizer.json, and the 1.x reader refuses both as legacy did.
    let reasons = [
        ("mock-special-tokens-only", "has no `single_word`"),
        ("mock-tokenizer-json-fallback", "the config has no `model`"),
    ];
    let unloadable: Vec<(String, Value)> = cases()
        .into_iter()
        .filter(|(_, expected)| expected.get("load_error").is_some())
        .collect();
    assert_eq!(unloadable.len(), reasons.len());
    for (name, expected) in unloadable {
        let reason = reasons
            .iter()
            .find(|(case, _)| *case == name)
            .unwrap_or_else(|| panic!("{name} has no recorded reason"))
            .1;
        let path = tokenizer_path(&expected);
        let error = HuggingFaceTokenizer::from_file(&path)
            .err()
            .unwrap_or_else(|| panic!("{name} must not load"));
        let error = format!("{error:#}");
        assert!(
            error.contains(reason) && error.contains(&path),
            "{name}: {error}"
        );
    }
}
