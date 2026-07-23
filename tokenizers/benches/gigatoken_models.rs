// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    cell::Cell,
    env,
    hint::black_box,
    panic::{AssertUnwindSafe, catch_unwind},
    time::{Duration, Instant},
};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use dynamo_tokenizers::{
    Encoding, FastTokenizer, GigatokenTokenizer, HuggingFaceTokenizer, TikTokenTokenizer,
    traits::{DecodeResult, Decoder, Encoder},
};

const TARGET_TOKEN_COUNTS: &[usize] = &[128, 512, 2_048, 8_192, 32_768];
const WORKING_SET_SIZE: usize = 7;

const CORPUS_PARTS: &[&str] = &[
    include_str!("../README.md"),
    include_str!("../src/lib.rs"),
    include_str!("../src/hf.rs"),
    include_str!("../src/tiktoken.rs"),
    include_str!("../src/cache/mod.rs"),
    include_str!("../../renderer/src/template.rs"),
    include_str!("../../renderer/src/template/context.rs"),
    include_str!("../../renderer/src/template/oai.rs"),
    include_str!("../../protocols/src/types/chat.rs"),
    include_str!("../../protocols/src/types/responses/mod.rs"),
];

#[derive(Clone, Copy)]
enum TokenizerFormat {
    HuggingFace,
    TikToken { pretokenizer: &'static str },
}

struct ModelConfig {
    name: String,
    path: String,
    format: TokenizerFormat,
}

impl ModelConfig {
    fn from_env() -> Self {
        let name = env::var("TOKENIZER_BENCH_MODEL").unwrap_or_else(|_| "qwen3-0.6b".to_string());
        let path = env::var("TOKENIZER_PATH")
            .or_else(|_| env::var("QWEN_TOKENIZER"))
            .expect("set TOKENIZER_PATH=/path/to/tokenizer.json-or-tiktoken.model");
        let format = match name.as_str() {
            "kimi-k2.6" => TokenizerFormat::TikToken {
                pretokenizer: "kimi",
            },
            "qwen3-0.6b" | "glm-5.2" | "deepseek-v4" | "minimax-m3" => TokenizerFormat::HuggingFace,
            _ => panic!("unknown TOKENIZER_BENCH_MODEL {name:?}"),
        };
        Self { name, path, format }
    }

    fn reference_label(&self) -> &'static str {
        match self.format {
            TokenizerFormat::HuggingFace => "huggingface",
            TokenizerFormat::TikToken { .. } => "tiktoken",
        }
    }
}

// Keep the measured hot call as a direct enum dispatch rather than adding a
// backend-dependent heap indirection solely to equalize bench-only variant sizes.
#[allow(clippy::large_enum_variant)]
enum ReferenceTokenizer {
    HuggingFace(HuggingFaceTokenizer),
    TikToken(TikTokenTokenizer),
}

impl ReferenceTokenizer {
    fn load(config: &ModelConfig) -> dynamo_tokenizers::Result<Self> {
        match config.format {
            TokenizerFormat::HuggingFace => {
                HuggingFaceTokenizer::from_file(&config.path).map(Self::HuggingFace)
            }
            TokenizerFormat::TikToken { .. } => {
                TikTokenTokenizer::from_file_auto(&config.path).map(Self::TikToken)
            }
        }
    }
}

impl Encoder for ReferenceTokenizer {
    fn encode(&self, input: &str) -> dynamo_tokenizers::Result<Encoding> {
        match self {
            Self::HuggingFace(tokenizer) => tokenizer.encode(input),
            Self::TikToken(tokenizer) => tokenizer.encode(input),
        }
    }

    fn encode_batch(&self, inputs: &[&str]) -> dynamo_tokenizers::Result<Vec<Encoding>> {
        match self {
            Self::HuggingFace(tokenizer) => tokenizer.encode_batch(inputs),
            Self::TikToken(tokenizer) => tokenizer.encode_batch(inputs),
        }
    }
}

impl Decoder for ReferenceTokenizer {
    fn decode(
        &self,
        token_ids: &[dynamo_tokenizers::TokenIdType],
        skip_special_tokens: bool,
    ) -> dynamo_tokenizers::Result<DecodeResult> {
        match self {
            Self::HuggingFace(tokenizer) => tokenizer.decode(token_ids, skip_special_tokens),
            Self::TikToken(tokenizer) => tokenizer.decode(token_ids, skip_special_tokens),
        }
    }
}

struct InputSet {
    texts: Vec<String>,
    bytes: usize,
}

fn target_token_counts() -> Vec<usize> {
    match env::var("TOKENIZER_BENCH_TARGET") {
        Ok(target) => {
            let target = target
                .parse::<usize>()
                .expect("TOKENIZER_BENCH_TARGET must be an integer");
            assert!(
                TARGET_TOKEN_COUNTS.contains(&target),
                "unsupported TOKENIZER_BENCH_TARGET {target}"
            );
            vec![target]
        }
        Err(_) => TARGET_TOKEN_COUNTS.to_vec(),
    }
}

fn build_inputs(
    reference: &ReferenceTokenizer,
    target_token_counts: &[usize],
) -> Vec<(usize, InputSet)> {
    let mut variants = Vec::with_capacity(WORKING_SET_SIZE);
    // Rotation 3 triggers a reproducible Fastokens 0.2.0 Qwen panic at the
    // 8K-token prefix. Keep seven safe rotations for a shared cross-model set.
    for rotation in [0, 1, 2, 4, 5, 6, 7] {
        let mut corpus = String::new();
        for index in 0..CORPUS_PARTS.len() {
            let part = CORPUS_PARTS[(index + rotation) % CORPUS_PARTS.len()];
            // Keep the main performance matrix on the common exact-ID ASCII
            // support envelope; target-model tests cover multilingual parity.
            corpus.extend(part.chars().filter(char::is_ascii));
            corpus.push_str("\n\n");
        }
        variants.push(corpus);
    }

    let max_target = *target_token_counts.iter().max().unwrap();
    let mut all_inputs = target_token_counts
        .iter()
        .map(|&target| {
            (
                target,
                InputSet {
                    texts: Vec::with_capacity(WORKING_SET_SIZE),
                    bytes: 0,
                },
            )
        })
        .collect::<Vec<_>>();

    for corpus in variants {
        let encoded = reference.encode(&corpus).expect("encode benchmark corpus");
        assert!(
            encoded.token_ids().len() >= max_target,
            "benchmark corpus only produced {} tokens; need {max_target}",
            encoded.token_ids().len()
        );

        for (target, input) in &mut all_inputs {
            let decoded = reference
                .decode(&encoded.token_ids()[..*target], false)
                .expect("decode benchmark token prefix")
                .as_str()
                .to_owned();
            let roundtrip = reference
                .encode(&decoded)
                .expect("re-encode benchmark input");
            assert_eq!(
                roundtrip.token_ids(),
                &encoded.token_ids()[..*target],
                "decoded prefix did not re-encode to exactly {target} tokens"
            );
            input.bytes += decoded.len();
            input.texts.push(decoded);
        }
    }

    for (_, input) in &mut all_inputs {
        input.bytes /= input.texts.len();
    }
    all_inputs
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

fn validate_encoder<E: Encoder>(
    name: &str,
    encoder: &E,
    reference: &ReferenceTokenizer,
    inputs: &[(usize, InputSet)],
) -> Vec<usize> {
    let mut valid_targets = Vec::new();
    for (target_tokens, input) in inputs {
        let mut target_error = None;
        for (rotation, text) in input.texts.iter().enumerate() {
            let expected = match reference.encode(text) {
                Ok(encoding) => encoding,
                Err(error) => {
                    target_error = Some(format!(
                        "rotation={rotation} reference encode failed: {error:#}"
                    ));
                    break;
                }
            };
            let actual = match catch_unwind(AssertUnwindSafe(|| encoder.encode(text))) {
                Ok(Ok(encoding)) => encoding,
                Ok(Err(error)) => {
                    target_error = Some(format!("rotation={rotation} encode failed: {error:#}"));
                    break;
                }
                Err(payload) => {
                    target_error = Some(format!(
                        "rotation={rotation} panicked: {}",
                        panic_message(payload)
                    ));
                    break;
                }
            };
            if actual.token_ids() != expected.token_ids() {
                let first_difference = actual
                    .token_ids()
                    .iter()
                    .zip(expected.token_ids())
                    .position(|(actual, expected)| actual != expected)
                    .unwrap_or_else(|| actual.token_ids().len().min(expected.token_ids().len()));
                target_error = Some(format!(
                    "rotation={rotation} ID mismatch \
                     position={first_difference}: actual_len={} expected_len={}",
                    actual.token_ids().len(),
                    expected.token_ids().len()
                ));
                break;
            }
        }
        if let Some(error) = target_error {
            eprintln!(
                "backend_status backend={name} target={target_tokens} status=excluded \
                 reason={error:?}"
            );
        } else {
            eprintln!(
                "backend_status backend={name} target={target_tokens} status=exact_id_parity"
            );
            valid_targets.push(*target_tokens);
        }
    }
    valid_targets
}

fn load_gigatoken(config: &ModelConfig) -> dynamo_tokenizers::Result<GigatokenTokenizer> {
    match config.format {
        TokenizerFormat::HuggingFace => GigatokenTokenizer::from_file(&config.path),
        TokenizerFormat::TikToken { pretokenizer } => {
            GigatokenTokenizer::from_tiktoken_model(&config.path, pretokenizer)
        }
    }
}

fn benchmark_tokenizers(c: &mut Criterion) {
    let config = ModelConfig::from_env();
    let target_token_counts = target_token_counts();

    let started = Instant::now();
    let reference = ReferenceTokenizer::load(&config).expect("load reference tokenizer");
    let reference_load_ms = started.elapsed().as_secs_f64() * 1_000.0;

    let (fast, fast_load_ms, fast_load_error) = match config.format {
        TokenizerFormat::HuggingFace => {
            let started = Instant::now();
            let loaded = FastTokenizer::from_file(&config.path);
            let load_ms = started.elapsed().as_secs_f64() * 1_000.0;
            match loaded {
                Ok(tokenizer) => (Some(tokenizer), Some(load_ms), None),
                Err(error) => (None, Some(load_ms), Some(format!("{error:#}"))),
            }
        }
        TokenizerFormat::TikToken { .. } => (
            None,
            None,
            Some("official repository has no tokenizer.json".to_string()),
        ),
    };

    let started = Instant::now();
    let loaded_gigatoken = load_gigatoken(&config);
    let gigatoken_load_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let (gigatoken, gigatoken_load_error) = match loaded_gigatoken {
        Ok(tokenizer) => (Some(tokenizer), None),
        Err(error) => (None, Some(format!("{error:#}"))),
    };

    eprintln!(
        "model={} artifact={} reference={} reference_load_ms={reference_load_ms:.3} \
         fastokens_load_ms={} gigatoken_load_ms={gigatoken_load_ms:.3}",
        config.name,
        config.path,
        config.reference_label(),
        fast_load_ms
            .map(|value| format!("{value:.3}"))
            .unwrap_or_else(|| "unavailable".to_string())
    );
    if let Some(error) = &fast_load_error {
        eprintln!("backend_status backend=fastokens status=unavailable reason={error:?}");
    }
    if let Some(error) = &gigatoken_load_error {
        eprintln!("backend_status backend=gigatoken status=unavailable reason={error:?}");
    }

    let inputs = build_inputs(&reference, &target_token_counts);
    let fast_valid_targets = fast
        .as_ref()
        .map(|tokenizer| validate_encoder("fastokens", tokenizer, &reference, &inputs))
        .unwrap_or_default();
    let gigatoken_valid_targets = gigatoken
        .as_ref()
        .map(|tokenizer| validate_encoder("gigatoken", tokenizer, &reference, &inputs))
        .unwrap_or_default();

    if env::var_os("TOKENIZER_BENCH_PROBE_ONLY").is_some() {
        return;
    }

    for (target_tokens, input) in &inputs {
        let mut group = c.benchmark_group(format!("tokenizer_steady_state/{}", config.name));
        group.throughput(Throughput::Bytes(input.bytes as u64));

        let index = Cell::new(0usize);
        group.bench_with_input(
            BenchmarkId::new(config.reference_label(), target_tokens),
            input,
            |b, input| {
                b.iter(|| {
                    let current = index.get();
                    index.set((current + 1) % input.texts.len());
                    reference
                        .encode(black_box(&input.texts[current]))
                        .expect("reference benchmark encode")
                });
            },
        );

        if let Some(fast) = fast
            .as_ref()
            .filter(|_| fast_valid_targets.contains(target_tokens))
        {
            let index = Cell::new(0usize);
            group.bench_with_input(
                BenchmarkId::new("fastokens", target_tokens),
                input,
                |b, input| {
                    b.iter(|| {
                        let current = index.get();
                        index.set((current + 1) % input.texts.len());
                        fast.encode(black_box(&input.texts[current]))
                            .expect("Fastokens benchmark encode")
                    });
                },
            );
        }

        if let Some(gigatoken) = gigatoken
            .as_ref()
            .filter(|_| gigatoken_valid_targets.contains(target_tokens))
        {
            let index = Cell::new(0usize);
            group.bench_with_input(
                BenchmarkId::new("gigatoken", target_tokens),
                input,
                |b, input| {
                    b.iter(|| {
                        let current = index.get();
                        index.set((current + 1) % input.texts.len());
                        gigatoken
                            .encode(black_box(&input.texts[current]))
                            .expect("Gigatoken benchmark encode")
                    });
                },
            );
        }
        group.finish();
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(3))
        .measurement_time(Duration::from_secs(7))
        .sample_size(60);
    targets = benchmark_tokenizers
}
criterion_main!(benches);
