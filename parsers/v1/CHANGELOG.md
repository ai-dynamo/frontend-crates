# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
## [6.0.3](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-parsers-v6.0.2...dynamo-parsers-v6.0.3) - 2026-07-28

### Bug fixes

- *(GUI)* Restore tooltip coloring + grammar popup; repair the fixture corpus and restore leak detection ([#142](https://github.com/ai-dynamo/frontend-crates/pull/142))

## [6.0.2](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-parsers-v6.0.1...dynamo-parsers-v6.0.2) - 2026-07-27

### Miscellaneous

- Update Cargo.lock dependencies

## [6.0.1](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-parsers-v6.0.0...dynamo-parsers-v6.0.1) - 2026-07-27

### Bug fixes

- *(parsers)* Harden MiniMax M3 parser ([#133](https://github.com/ai-dynamo/frontend-crates/pull/133))

## [6.0.0](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-parsers-v5.1.3...dynamo-parsers-v6.0.0) - 2026-07-27

### Features

- *(parsers)* Add Inkling tool-call and reasoning parsers ([#120](https://github.com/ai-dynamo/frontend-crates/pull/120))

## [5.1.3](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-parsers-v5.1.2...dynamo-parsers-v5.1.3) - 2026-07-24

### Performance

- *(parsers)* Switch Pythonic parser to Ruff ([#141](https://github.com/ai-dynamo/frontend-crates/pull/141))

## [5.1.2](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-parsers-v5.1.1...dynamo-parsers-v5.1.2) - 2026-07-22

### Miscellaneous

- Update Cargo.lock dependencies

## [5.1.1](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-parsers-v5.1.0...dynamo-parsers-v5.1.1) - 2026-07-21

### Miscellaneous

- Update Cargo.lock dependencies

## [5.1.0](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-parsers-v5.0.1...dynamo-parsers-v5.1.0) - 2026-07-17

### Features

- *(parsers-v2)* Streaming tool-call parsers for Gemma 4, GLM, Kimi K2, MiniMax M2/M3 (preserve surrounding text) ([#80](https://github.com/ai-dynamo/frontend-crates/pull/80))

## [5.0.1](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-parsers-v5.0.0...dynamo-parsers-v5.0.1) - 2026-07-14

### Miscellaneous

- Update Cargo.lock dependencies

## [5.0.0](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-parsers-v4.1.2...dynamo-parsers-v5.0.0) - 2026-07-13

### Bug fixes

- Add minimax m2 reasoning parser ([#108](https://github.com/ai-dynamo/frontend-crates/pull/108))

## [4.1.2](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-parsers-v4.1.1...dynamo-parsers-v4.1.2) - 2026-07-11

### Miscellaneous

- Update Cargo.toml dependencies

## [4.1.1](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-parsers-v4.1.0...dynamo-parsers-v4.1.1) - 2026-07-10

### Bug fixes

- *(jail)* Normalize terminal tool-call emissions ([#101](https://github.com/ai-dynamo/frontend-crates/pull/101))

## [4.1.0](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-parsers-v4.0.1...dynamo-parsers-v4.1.0) - 2026-07-08

### Features

- *(conformance)* Version toolcalling fixtures by peer parser version ([#93](https://github.com/ai-dynamo/frontend-crates/pull/93))

## [4.0.1](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-parsers-v4.0.0...dynamo-parsers-v4.0.1) - 2026-07-08

### Miscellaneous

- Upgrade Rust toolchain to 1.96.1 to match Dynamo ([#99](https://github.com/ai-dynamo/frontend-crates/pull/99))

## [4.0.0](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-parsers-v3.1.1...dynamo-parsers-v4.0.0) - 2026-07-07

### Performance

- Make tool-call jail completion incremental ([#94](https://github.com/ai-dynamo/frontend-crates/pull/94))

## [3.1.1](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-parsers-v3.1.0...dynamo-parsers-v3.1.1) - 2026-07-06

### Refactoring

- *(parsers)* Group v1/v2/v2-py under parsers/, stop publishing test-only binding (part 1) ([#95](https://github.com/ai-dynamo/frontend-crates/pull/95))

## [3.1.0](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-parsers-v3.0.0...dynamo-parsers-v3.1.0) - 2026-07-02

### Features

- *(parsers)* Move v1 tool-call jail into dynamo-parsers + sync #11045 (DIS-2296)

## [3.0.0](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-parsers-v2.1.2...dynamo-parsers-v3.0.0) - 2026-06-26

### Documentation

- Migrate parser docs from Dynamo into frontend-crates ([#82](https://github.com/ai-dynamo/frontend-crates/pull/82))

### Features

- Add MiniMax M3 tool-calling, reasoning, and conformance coverage ([#83](https://github.com/ai-dynamo/frontend-crates/pull/83))

## [2.1.2](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-parsers-v2.1.1...dynamo-parsers-v2.1.2) - 2026-06-23

### Bug fixes

- Stop granite reasoning parser leaking markers across spans and split chunks ([#75](https://github.com/ai-dynamo/frontend-crates/pull/75))
- Strip dangling reasoning end marker for non-ASCII delimiter families (Kimi unicode) ([#74](https://github.com/ai-dynamo/frontend-crates/pull/74))

## [2.1.1](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-parsers-v2.1.0...dynamo-parsers-v2.1.1) - 2026-06-23

### Bug fixes

- *(parsers)* Drop tool calls truncated mid-parameter-value ([#72](https://github.com/ai-dynamo/frontend-crates/pull/72))

## [2.1.0](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-parsers-v2.0.1...dynamo-parsers-v2.1.0) - 2026-06-23

### Features

- Jamba never-leaks tool-call markup on malformed input ([#69](https://github.com/ai-dynamo/frontend-crates/pull/69))

## [2.0.1](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-parsers-v2.0.0...dynamo-parsers-v2.0.1) - 2026-06-22

### Bug fixes

- *(parsers)* Align hermes tool-call parser to never leak markup ([#63](https://github.com/ai-dynamo/frontend-crates/pull/63))

## [2.0.0](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-parsers-v1.4.2...dynamo-parsers-v2.0.0) - 2026-06-17

### Bug fixes

- *(parsers)* Stop qwen25 tool-call parser from leaking <tool_call> markup ([#61](https://github.com/ai-dynamo/frontend-crates/pull/61))

## [1.4.2](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-parsers-v1.4.1...dynamo-parsers-v1.4.2) - 2026-06-17

### Bug fixes

- *(parsers)* Stop mistral parser leaking [TOOL_CALLS] into content ([#60](https://github.com/ai-dynamo/frontend-crates/pull/60))

## [1.4.1](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-parsers-v1.4.0...dynamo-parsers-v1.4.1) - 2026-06-16

### Miscellaneous

- Update Cargo.toml dependencies

## [1.4.0](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-parsers-v1.3.0...dynamo-parsers-v1.4.0) - 2026-06-12

### Features

- Add parser conformance capture workflow ([#42](https://github.com/ai-dynamo/frontend-crates/pull/42))
