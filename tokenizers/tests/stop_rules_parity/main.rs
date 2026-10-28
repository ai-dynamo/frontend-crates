// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `eos_token_ids` and `StopTokenDecoder` against a copy of the Dynamo
//! frontend's code (`oracle.rs`).

mod oracle;

use std::path::{Path, PathBuf};

use dynamo_tokenizers::{
    StopTokenDecoder, StopTokenIds, StopTokenTrigger, Tokenizer, eos_token_ids,
};
use serde_json::{Value, json};

fn sample_models() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/sample-models")
}

/// The IDs, or `None` for an error, so the two sides compare.
fn ours(dir: &Path) -> Option<Vec<u32>> {
    eos_token_ids(dir).ok()
}

fn stock(dir: &Path) -> Option<Vec<u32>> {
    oracle::HFConfig::eos_token_ids(&dir.join("config.json")).ok()
}

/// Every checked-in model whose `config.json` the stock code reads.
#[test]
fn the_sample_models_resolve_as_the_frontend_resolves_them() {
    let mut compared = 0;
    for entry in std::fs::read_dir(sample_models()).expect("sample models") {
        let dir = entry.expect("entry").path();
        if !dir.join("config.json").exists() {
            continue;
        }
        let Some(expected) = stock(&dir) else {
            continue;
        };
        assert_eq!(ours(&dir), Some(expected), "{}", dir.display());
        compared += 1;
    }
    assert!(compared >= 4, "compared only {compared} models");
}

/// The Qwen3.5 case: the config names one ID, and the tokenizer config's
/// end token is added.
#[test]
fn the_tokenizer_configs_end_token_is_added() {
    let dir = sample_models().join("mock-qwen3.5-0.8B");
    let ids = ours(&dir).expect("ids");
    assert_eq!(ids.len(), 2, "{ids:?}");
    assert_eq!(Some(ids), stock(&dir));
}

const TOKEN: &str = "<|end|>";
const TOKEN_ID: u32 = 7;
const OTHER: &str = "<|other|>";
const OTHER_ID: u32 = 9;

/// Each file's variants: `None` leaves it out.
fn generation_configs() -> Vec<Option<Value>> {
    vec![
        None,
        Some(json!({"eos_token_id": 3})),
        Some(json!({"eos_token_id": [3, 4]})),
        Some(json!({"eos_token_id": [3, "x", 4]})),
        Some(json!({"eos_token_id": null})),
        Some(json!({"eos_token_id": "3"})),
        Some(json!({"bos_token_id": 1})),
    ]
}

fn configs() -> Vec<Value> {
    let base = json!({"architectures": ["X"], "model_type": "x"});
    let with = |extra: Value| {
        let mut config = base.clone();
        config
            .as_object_mut()
            .expect("object")
            .extend(extra.as_object().expect("object").clone());
        config
    };
    vec![
        base.clone(),
        with(json!({"eos_token_id": 5})),
        with(json!({"eos_token_id": [5, 6]})),
        with(json!({"eos_token_id": [5, "x"]})),
        with(json!({"eos_token_id": "5"})),
        with(json!({"eos_token_id": null, "text_config": {"eos_token_id": 6}})),
        with(json!({"text_config": {"eos_token_id": [6, 5]}})),
        with(json!({"eos_token_id": 5, "text_config": {"eos_token_id": 6}})),
        with(json!({"text_config": {"max_position_embeddings": 8}})),
    ]
}

fn added(content: &str, id: &str) -> Value {
    json!({ id: {"content": content, "special": true} })
}

fn tokenizer_configs() -> Vec<Option<Value>> {
    vec![
        None,
        Some(json!({})),
        Some(json!({"eos_token": TOKEN, "added_tokens_decoder": added(TOKEN, "7")})),
        Some(json!({"eos_token": {"content": TOKEN}, "added_tokens_decoder": added(TOKEN, "7")})),
        Some(json!({"eos_token": {"lstrip": false}})),
        Some(json!({"eos_token": OTHER, "added_tokens_decoder": added(TOKEN, "7")})),
        Some(json!({"eos_token": TOKEN, "added_tokens_decoder": added(TOKEN, "x7")})),
        Some(json!({"eos_token": TOKEN})),
        Some(json!({"eos_token": 7})),
        Some(json!({"eos_token": OTHER, "added_tokens_decoder": added(OTHER, "9")})),
    ]
}

fn special_tokens_maps() -> Vec<Option<Value>> {
    vec![
        None,
        Some(json!({"eos_token": TOKEN})),
        Some(json!({"eos_token": {"content": OTHER}})),
        Some(json!({"eos_token": "<|missing|>"})),
        Some(json!({"bos_token": TOKEN})),
    ]
}

fn tokenizer_jsons() -> Vec<Option<Value>> {
    vec![
        None,
        Some(json!({"added_tokens": [
            {"id": TOKEN_ID, "content": TOKEN},
            {"id": OTHER_ID, "content": OTHER},
        ]})),
    ]
}

fn write(dir: &Path, name: &str, value: &Option<Value>) {
    if let Some(value) = value {
        std::fs::write(dir.join(name), value.to_string()).expect("write");
    }
}

/// Every combination of the files the chain reads, each with its variants,
/// resolves as the stock code resolves it.
#[test]
fn every_combination_of_files_resolves_as_the_frontend_resolves_it() {
    let mut cases = 0;
    let mut differences = Vec::new();
    for generation in generation_configs() {
        for config in configs() {
            for tokenizer_config in tokenizer_configs() {
                for map in special_tokens_maps() {
                    for tokenizer in tokenizer_jsons() {
                        let dir = tempfile::tempdir().expect("tempdir");
                        write(dir.path(), "generation_config.json", &generation);
                        write(dir.path(), "config.json", &Some(config.clone()));
                        write(dir.path(), "tokenizer_config.json", &tokenizer_config);
                        write(dir.path(), "special_tokens_map.json", &map);
                        write(dir.path(), "tokenizer.json", &tokenizer);
                        let (ours, stock) = (ours(dir.path()), stock(dir.path()));
                        if ours != stock {
                            differences.push(format!(
                                "{generation:?} {config} {tokenizer_config:?} {map:?} \
                                 {tokenizer:?}: ours {ours:?}, stock {stock:?}"
                            ));
                        }
                        cases += 1;
                    }
                }
            }
        }
    }
    assert!(
        differences.is_empty(),
        "{} of {cases}: {differences:#?}",
        differences.len()
    );
}

/// No file names an end token: an error, as the stock code's.
#[test]
fn no_end_token_is_an_error() {
    let dir = tempfile::tempdir().expect("tempdir");
    write(
        dir.path(),
        "config.json",
        &Some(json!({"architectures": ["X"], "model_type": "x"})),
    );
    assert!(eos_token_ids(dir.path()).is_err());
    assert!(stock(dir.path()).is_none());
}

fn tokenizer() -> Tokenizer {
    let path = sample_models().join("TinyLlama_v1.1/tokenizer.json");
    Tokenizer::from_file(path.to_str().expect("path")).expect("tokenizer")
}

/// What a decoder emits for `ids`: the text and stop of each step, then the
/// finish's text.
type Run = (
    Vec<(Option<String>, Option<StopTokenTrigger>)>,
    Option<String>,
);

fn run_ours(
    tokenizer: &Tokenizer,
    ids: &[u32],
    stops: &StopTokenIds,
    min_tokens: u32,
    no_stop_trim: bool,
) -> Run {
    let stream = tokenizer.decode_stream(&[], true);
    let mut decoder = StopTokenDecoder::new(stream, stops.clone(), min_tokens, no_stop_trim);
    let mut steps = Vec::new();
    for &id in ids {
        let step = decoder.step(id).expect("step");
        let stopped = step.stop.is_some();
        steps.push((step.text, step.stop));
        if stopped {
            return (steps, None);
        }
    }
    (steps, decoder.finish().expect("finish"))
}

/// The stock decoder, stopped as its caller stops it: at the first trigger.
fn run_stock(
    tokenizer: &Tokenizer,
    ids: &[u32],
    stops: &StopTokenIds,
    min_tokens: u32,
    no_stop_trim: bool,
) -> Run {
    let stream = tokenizer.decode_stream(&[], true);
    let mut decoder = oracle::Decoder::new(
        stream,
        stops.user.clone(),
        stops.hidden.clone(),
        stops.visible.clone(),
        min_tokens,
        no_stop_trim,
    );
    let mut steps = Vec::new();
    for &id in ids {
        let step = decoder.step(id).expect("step");
        let stop = step.stop_trigger.map(|trigger| match trigger {
            oracle::StopTrigger::VisibleStopTokenDetected(id) => StopTokenTrigger::Visible(id),
            oracle::StopTrigger::UserStopTokenDetected(id) => StopTokenTrigger::User(id),
            oracle::StopTrigger::HiddenStopTokenDetected(id) => StopTokenTrigger::Hidden(id),
        });
        let stopped = stop.is_some();
        steps.push((step.released_text, stop));
        if stopped {
            return (steps, None);
        }
    }
    (steps, decoder.finish().expect("finish").released_text)
}

/// Each step's text and stop, and the finish, match the stock decoder: for
/// stop tokens at each place in the reply, in each set, from each
/// `min_tokens`, with and without `no_stop_trim`. The reply holds multi-byte
/// text, so a stop can flush a held byte sequence.
#[test]
fn every_step_decodes_and_stops_as_the_frontend_decoder() {
    let tokenizer = tokenizer();
    let mut ids = tokenizer
        .encode("héllo wörld, 你好! 🦀 end")
        .expect("encode")
        .token_ids()
        .to_vec();
    // The end-of-sequence token, twice, and one more after it.
    ids.extend([2, 2, 29871]);
    let mut candidates: Vec<u32> = ids.clone();
    candidates.sort_unstable();
    candidates.dedup();
    let mut cases = 0;
    for &stop in &candidates {
        for (user, hidden, visible) in [
            (vec![stop], vec![], vec![]),
            (vec![], vec![stop], vec![]),
            (vec![], vec![], vec![stop]),
            (vec![stop], vec![], vec![stop]),
            (vec![], vec![stop], vec![stop]),
            (vec![stop], vec![stop], vec![]),
            (vec![], vec![2], vec![stop]),
        ] {
            let stops = StopTokenIds {
                user,
                hidden,
                visible,
            };
            for min_tokens in 0..=ids.len() as u32 + 1 {
                for no_stop_trim in [false, true] {
                    let ours = run_ours(&tokenizer, &ids, &stops, min_tokens, no_stop_trim);
                    let stock = run_stock(&tokenizer, &ids, &stops, min_tokens, no_stop_trim);
                    assert_eq!(
                        ours, stock,
                        "stop {stop}, {stops:?}, min_tokens {min_tokens}, no_stop_trim {no_stop_trim}"
                    );
                    cases += 1;
                }
            }
        }
    }
    assert!(cases > 1000, "{cases} cases");
}

/// After a stop, later tokens add no text, and the finish adds none.
#[test]
fn nothing_decodes_after_a_stop() {
    let tokenizer = tokenizer();
    let ids = tokenizer
        .encode("a b")
        .expect("encode")
        .token_ids()
        .to_vec();
    let mut decoder = StopTokenDecoder::new(
        tokenizer.decode_stream(&[], true),
        StopTokenIds {
            hidden: vec![2],
            ..StopTokenIds::default()
        },
        0,
        false,
    );
    decoder.step(ids[0]).expect("step");
    let stop = decoder.step(2).expect("step");
    assert_eq!(stop.stop, Some(StopTokenTrigger::Hidden(2)));
    assert!(decoder.is_stopped());
    assert_eq!(decoder.step(ids[1]).expect("step"), Default::default());
    assert_eq!(decoder.finish().expect("finish"), None);
    assert_eq!(decoder.generated_tokens(), 2);
}
