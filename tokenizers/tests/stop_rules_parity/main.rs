// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `eos_token_ids` and `StopTokenDecoder` against a copy of the Dynamo
//! frontend's code (`oracle.rs`).

mod oracle;

use std::path::{Path, PathBuf};

use dynamo_tokenizers::{
    StopTokenDecoder, StopTokenIds, StopTokenTrigger, Tokenizer, eos_token_ids, lossless_token_text,
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

/// What a decoder emits for `ids`: the text, stop, and token text of each
/// step, then the finish's text.
type Run = (
    Vec<(Option<String>, Option<StopTokenTrigger>, Option<String>)>,
    Option<String>,
);

/// How a case runs: its stop tokens and `min_tokens`, `no_stop_trim`,
/// whether the decode skips special tokens, and whether the decoder gives
/// each token's own text.
#[derive(Debug, Clone)]
struct Case {
    stops: StopTokenIds,
    min_tokens: u32,
    no_stop_trim: bool,
    skip_special_tokens: bool,
    token_text: bool,
}

fn run_ours(tokenizer: &Tokenizer, ids: &[u32], case: &Case) -> Run {
    let stream = tokenizer.decode_stream(&[], case.skip_special_tokens);
    let mut decoder = StopTokenDecoder::new(
        stream,
        case.stops.clone(),
        case.min_tokens,
        case.no_stop_trim,
    );
    if case.token_text {
        decoder = decoder.with_token_text();
    }
    let mut steps = Vec::new();
    for &id in ids {
        let step = decoder.step(id).expect("step");
        let stopped = step.stop.is_some();
        steps.push((step.text, step.stop, step.token));
        if stopped {
            return (steps, None);
        }
    }
    (steps, decoder.finish().expect("finish"))
}

/// The stock decoder, stopped as its caller stops it: at the first trigger.
/// It decodes token text as the frontend's backend does: with the
/// tokenizer, and only when the request asks for log probabilities.
fn run_stock(tokenizer: &Tokenizer, ids: &[u32], case: &Case) -> Run {
    let stream = tokenizer.decode_stream(&[], case.skip_special_tokens);
    let mut decoder = oracle::Decoder::new(
        Some(tokenizer.clone()),
        case.skip_special_tokens,
        stream,
        case.stops.user.clone(),
        case.stops.hidden.clone(),
        case.stops.visible.clone(),
        case.min_tokens,
        case.no_stop_trim,
    );
    let mut steps = Vec::new();
    for &id in ids {
        let step = decoder
            .step_with_token_text(id, case.token_text)
            .expect("step");
        let stop = step.stop_trigger.map(|trigger| match trigger {
            oracle::StopTrigger::VisibleStopTokenDetected(id) => StopTokenTrigger::Visible(id),
            oracle::StopTrigger::UserStopTokenDetected(id) => StopTokenTrigger::User(id),
            oracle::StopTrigger::HiddenStopTokenDetected(id) => StopTokenTrigger::Hidden(id),
        });
        let stopped = stop.is_some();
        steps.push((step.released_text, stop, step.token));
        if stopped {
            return (steps, None);
        }
    }
    (steps, decoder.finish().expect("finish").released_text)
}

/// The reply of the parity cases: multi-byte text, so a stop can flush a
/// held byte sequence and a byte token decodes alone to U+FFFD; the
/// end-of-sequence token, twice; and one more token after it.
fn reply(tokenizer: &Tokenizer) -> Vec<u32> {
    let mut ids = tokenizer
        .encode("héllo wörld, 你好! 🦀 end")
        .expect("encode")
        .token_ids()
        .to_vec();
    ids.extend([2, 2, 29871]);
    ids
}

/// Every case of the parity tests: stop tokens at each place in the reply,
/// in each set, from each `min_tokens`, with and without `no_stop_trim`.
fn cases(ids: &[u32], skip_special_tokens: bool, token_text: bool) -> Vec<Case> {
    let mut candidates: Vec<u32> = ids.to_vec();
    candidates.sort_unstable();
    candidates.dedup();
    let mut cases = Vec::new();
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
                    cases.push(Case {
                        stops: stops.clone(),
                        min_tokens,
                        no_stop_trim,
                        skip_special_tokens,
                        token_text,
                    });
                }
            }
        }
    }
    cases
}

/// Each step's text and stop, and the finish, match the stock decoder, for
/// every case. With no token text asked for, each step's token text is
/// `None`, as the stock decoder's is.
#[test]
fn every_step_decodes_and_stops_as_the_frontend_decoder() {
    let tokenizer = tokenizer();
    let ids = reply(&tokenizer);
    let cases = cases(&ids, true, false);
    for case in &cases {
        let ours = run_ours(&tokenizer, &ids, case);
        let stock = run_stock(&tokenizer, &ids, case);
        assert_eq!(ours, stock, "{case:?}");
    }
    assert!(cases.len() > 1000, "{} cases", cases.len());
}

/// With token text asked for, each step also gives the token's own text as
/// the stock decoder gives it for a log probability: decoded alone, lossy,
/// and none for a hidden stop token. With and without special tokens.
#[test]
fn every_step_gives_the_token_text_of_the_frontend_decoder() {
    let tokenizer = tokenizer();
    let ids = reply(&tokenizer);
    let mut compared = 0;
    let mut replacement = 0;
    for skip_special_tokens in [true, false] {
        for case in cases(&ids, skip_special_tokens, true) {
            let ours = run_ours(&tokenizer, &ids, &case);
            let stock = run_stock(&tokenizer, &ids, &case);
            assert_eq!(ours, stock, "{case:?}");
            replacement += ours
                .0
                .iter()
                .filter(|step| step.2.as_deref().is_some_and(|t| t.contains('\u{fffd}')))
                .count();
            compared += 1;
        }
    }
    assert!(compared > 2000, "{compared} cases");
    // The control: a byte token's own text is U+FFFD, kept as the stock
    // decoder keeps it.
    assert!(replacement > 0, "no byte token decoded alone");
}

/// The text of a top candidate: a complete decode with no U+FFFD, as the
/// stock frontend fills a candidate's missing text, for every ID of the
/// vocabulary.
#[test]
fn every_candidate_text_is_the_frontend_s() {
    let tokenizer = tokenizer();
    let mut missing = 0;
    for skip_special_tokens in [true, false] {
        for id in 0..32000 {
            let ours = lossless_token_text(&**tokenizer, id, skip_special_tokens);
            let stock = oracle::top_logprob_text(&tokenizer, id, skip_special_tokens);
            assert_eq!(ours, stock, "id {id}, skip {skip_special_tokens}");
            missing += usize::from(ours.is_none());
        }
    }
    // The control: the byte tokens have no text.
    assert!(missing >= 2 * 128, "{missing} candidates with no text");
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
