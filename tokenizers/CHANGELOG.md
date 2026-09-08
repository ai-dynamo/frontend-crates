# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
## [1.8.1](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-tokenizers-v1.8.0...dynamo-tokenizers-v1.8.1) - 2026-08-25

### Bug fixes

- *(tokenizers)* Avoid UTF-8 boundary panic in DecodeStream ([#187](https://github.com/ai-dynamo/frontend-crates/pull/187))

## [1.8.0](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-tokenizers-v1.7.0...dynamo-tokenizers-v1.8.0) - 2026-08-04

### Features

- *(tokenizers)* Support segmented encoding with fastokens 0.3.1 ([#171](https://github.com/ai-dynamo/frontend-crates/pull/171))

## [1.7.0](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-tokenizers-v1.6.0...dynamo-tokenizers-v1.7.0) - 2026-07-29

### Features

- Add support for Kimi-K3 ([#145](https://github.com/ai-dynamo/frontend-crates/pull/145))

## [1.6.0](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-tokenizers-v1.5.4...dynamo-tokenizers-v1.6.0) - 2026-07-28

### Features

- *(tokenizers)* Add basetenkenizer backend ([#154](https://github.com/ai-dynamo/frontend-crates/pull/154))

## [1.5.4](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-tokenizers-v1.5.3...dynamo-tokenizers-v1.5.4) - 2026-07-25

### Bug fixes

- *(tokenizers)* Support Kimi Linear tokenizer ([#134](https://github.com/ai-dynamo/frontend-crates/pull/134))

## [1.5.3](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-tokenizers-v1.5.2...dynamo-tokenizers-v1.5.3) - 2026-07-14

### Bug fixes

- *(tokenizers)* Reject incompatible prefix caching ([#115](https://github.com/ai-dynamo/frontend-crates/pull/115))

## [1.5.2](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-tokenizers-v1.5.1...dynamo-tokenizers-v1.5.2) - 2026-07-14

### Refactoring

- *(tokenizers)* Move tokenizer fixtures out of top-level llm/, detach from sync (part 2) ([#96](https://github.com/ai-dynamo/frontend-crates/pull/96))

## [1.5.1](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-tokenizers-v1.5.0...dynamo-tokenizers-v1.5.1) - 2026-07-13

### Bug fixes

- *(tokenizer)* Support tokenizer options ([#110](https://github.com/ai-dynamo/frontend-crates/pull/110))

## [1.5.0](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-tokenizers-v1.4.0...dynamo-tokenizers-v1.5.0) - 2026-07-09

### Features

- *(tokenizers)* Expose tiktoken special tokens for caching ([#103](https://github.com/ai-dynamo/frontend-crates/pull/103))

## [1.4.0](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-tokenizers-v1.3.2...dynamo-tokenizers-v1.4.0) - 2026-06-30

### Features

- *(tokenizers)* Expose cache token usage observability ([#90](https://github.com/ai-dynamo/frontend-crates/pull/90))

## [1.3.2](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-tokenizers-v1.3.1...dynamo-tokenizers-v1.3.2) - 2026-06-23

### Miscellaneous

- Update Cargo.toml dependencies

## [1.3.1](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-tokenizers-v1.3.0...dynamo-tokenizers-v1.3.1) - 2026-06-16

### Miscellaneous

- Sync from dynamo @ 628904d ([#30](https://github.com/ai-dynamo/frontend-crates/pull/30))
