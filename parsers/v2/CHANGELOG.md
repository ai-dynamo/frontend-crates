# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
## [0.4.1](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-parsers-v2-v0.4.0...dynamo-parsers-v2-v0.4.1) - 2026-09-02

### Chore

- Add Unified conformance coverage and explicit reporting for family-scoped scenarios.

## [0.4.0](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-parsers-v2-v0.3.4...dynamo-parsers-v2-v0.4.0) - 2026-09-01

### Features

- *(parsers)* Gemma4 on the v2 UnifiedParser — one ordered stream, with Dynamo request modes ([#166](https://github.com/ai-dynamo/frontend-crates/pull/166))

### Chore

- Retire obsolete Dynamo source sync ([#177](https://github.com/ai-dynamo/frontend-crates/pull/177))

## [0.3.3](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-parsers-v2-v0.3.2...dynamo-parsers-v2-v0.3.3) - 2026-08-25

### Features

- *(parsers)* Port kimi_k2 to the v2 parser ([#191](https://github.com/ai-dynamo/frontend-crates/pull/191))

## [0.3.2](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-parsers-v2-v0.3.1...dynamo-parsers-v2-v0.3.2) - 2026-08-21

### Features

- *(parsers)* Stream guided tool calls in v1 and v2 ([#194](https://github.com/ai-dynamo/frontend-crates/pull/194))

## [0.3.0](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-parsers-v2-v0.2.1...dynamo-parsers-v2-v0.3.0) - 2026-08-19

### Features

- *(parsers)* Shared scanner params, and guided tool-call fragment streaming ([#190](https://github.com/ai-dynamo/frontend-crates/pull/190))

## [0.2.1](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-parsers-v2-v0.2.0...dynamo-parsers-v2-v0.2.1) - 2026-08-18

### Features

- *(parsers)* Add Qwen3 unified request modes ([#174](https://github.com/ai-dynamo/frontend-crates/pull/174))

## [0.1.27](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-parsers-v2-v0.1.26...dynamo-parsers-v2-v0.1.27) - 2026-08-03

### Miscellaneous

- Update Cargo.lock dependencies

## [0.1.26](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-parsers-v2-v0.1.25...dynamo-parsers-v2-v0.1.26) - 2026-08-03

### Refactoring

- *(conformance)* One manifest row per unified family, plus popup pin/press — NO PARSER CHANGE ([#169](https://github.com/ai-dynamo/frontend-crates/pull/169))

## [0.1.25](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-parsers-v2-v0.1.24...dynamo-parsers-v2-v0.1.25) - 2026-07-30

### Miscellaneous

- *(conformance)* Simplify and remove old unused v1 HTML generator ([#167](https://github.com/ai-dynamo/frontend-crates/pull/167))

## [0.1.24](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-parsers-v2-v0.1.23...dynamo-parsers-v2-v0.1.24) - 2026-07-30

### Features

- *(parsers)* UnifiedParser for Qwen3 — one ordered reasoning/text/tool stream ([#151](https://github.com/ai-dynamo/frontend-crates/pull/151))

## [0.1.23](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-parsers-v2-v0.1.22...dynamo-parsers-v2-v0.1.23) - 2026-07-21

### Features

- *(conformance)* Parser-authoring docs + process — coverage taxonomy, CI lint, single-source markers (part 1) ([#127](https://github.com/ai-dynamo/frontend-crates/pull/127))

## [0.1.21](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-parsers-v2-v0.1.20...dynamo-parsers-v2-v0.1.21) - 2026-07-14

### Bug fixes

- *(GUI)* Conformance popup candidate chart + compare fixes (UI only) ([#109](https://github.com/ai-dynamo/frontend-crates/pull/109))

## [0.1.20](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-parsers-v2-v0.1.19...dynamo-parsers-v2-v0.1.20) - 2026-07-14

### Features

- *(parsers)* Add DYNAMO_PARSERS_DEBUG env-gated stderr instrumentation ([#106](https://github.com/ai-dynamo/frontend-crates/pull/106))

## [0.1.19](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-parsers-v2-v0.1.18...dynamo-parsers-v2-v0.1.19) - 2026-07-13

### Miscellaneous

- Updated the following local packages: dynamo-parsers

## [0.1.18](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-parsers-v2-v0.1.17...dynamo-parsers-v2-v0.1.18) - 2026-07-11

### Bug fixes

- Relax tokio pin =1.48.0 -> 1.48 so consumers can build on tokio 1.52.3+ ([#113](https://github.com/ai-dynamo/frontend-crates/pull/113))

## [0.1.17](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-parsers-v2-v0.1.16...dynamo-parsers-v2-v0.1.17) - 2026-07-10

### Miscellaneous

- Updated the following local packages: dynamo-parsers

## [0.1.16](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-parsers-v2-v0.1.15...dynamo-parsers-v2-v0.1.16) - 2026-07-08

### Features

- *(conformance)* Version toolcalling fixtures by peer parser version ([#93](https://github.com/ai-dynamo/frontend-crates/pull/93))

## [0.1.15](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-parsers-v2-v0.1.14...dynamo-parsers-v2-v0.1.15) - 2026-07-08

### Miscellaneous

- Updated the following local packages: dynamo-parsers

## [0.1.14](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-parsers-v2-v0.1.13...dynamo-parsers-v2-v0.1.14) - 2026-07-07

### Miscellaneous

- Updated the following local packages: dynamo-parsers

## [0.1.13](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-parsers-v2-v0.1.12...dynamo-parsers-v2-v0.1.13) - 2026-07-06

### Refactoring

- *(parsers)* Group v1/v2/v2-py under parsers/, stop publishing test-only binding (part 1) ([#95](https://github.com/ai-dynamo/frontend-crates/pull/95))

## [0.1.12](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-parsers-v2-v0.1.11...dynamo-parsers-v2-v0.1.12) - 2026-07-02

### Miscellaneous

- Updated the following local packages: dynamo-parsers

## [0.1.11](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-parsers-v2-v0.1.10...dynamo-parsers-v2-v0.1.11) - 2026-06-26

### Miscellaneous

- Updated the following local packages: dynamo-parsers

## [0.1.10](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-parsers-v2-v0.1.9...dynamo-parsers-v2-v0.1.10) - 2026-06-24

### Bug fixes

- *(parsers-v2)* Drop DSv4 tool calls truncated mid-call to match v1 batch ([#79](https://github.com/ai-dynamo/frontend-crates/pull/79))

## [0.1.9](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-parsers-v2-v0.1.8...dynamo-parsers-v2-v0.1.9) - 2026-06-23

### Miscellaneous

- Updated the following local packages: dynamo-parsers

## [0.1.8](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-parsers-v2-v0.1.7...dynamo-parsers-v2-v0.1.8) - 2026-06-23

### Miscellaneous

- Updated the following local packages: dynamo-parsers

## [0.1.7](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-parsers-v2-v0.1.6...dynamo-parsers-v2-v0.1.7) - 2026-06-23

### Features

- *(parsers-v2)* Qwen3-Coder v2 streaming tool-call parser ([#53](https://github.com/ai-dynamo/frontend-crates/pull/53))

## [0.1.6](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-parsers-v2-v0.1.5...dynamo-parsers-v2-v0.1.6) - 2026-06-23

### Miscellaneous

- Updated the following local packages: dynamo-parsers

## [0.1.5](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-parsers-v2-v0.1.4...dynamo-parsers-v2-v0.1.5) - 2026-06-23

### Features

- *(parsers-v2)* Emit DSv4 DSML tool name eagerly in streaming ([#66](https://github.com/ai-dynamo/frontend-crates/pull/66))

## [0.1.4](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-parsers-v2-v0.1.3...dynamo-parsers-v2-v0.1.4) - 2026-06-22

### Miscellaneous

- Updated the following local packages: dynamo-parsers

## [0.1.3](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-parsers-v2-v0.1.2...dynamo-parsers-v2-v0.1.3) - 2026-06-17

### Miscellaneous

- Updated the following local packages: dynamo-parsers

## [0.1.2](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-parsers-v2-v0.1.1...dynamo-parsers-v2-v0.1.2) - 2026-06-17

### Miscellaneous

- Updated the following local packages: dynamo-parsers

## [0.1.1](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-parsers-v2-v0.1.0...dynamo-parsers-v2-v0.1.1) - 2026-06-16

### Bug fixes

- Remove parser sync after frontend-crates cutover ([#59](https://github.com/ai-dynamo/frontend-crates/pull/59))

## [0.1.0](https://github.com/ai-dynamo/frontend-crates/releases/tag/dynamo-parsers-v2-v0.1.0) - 2026-06-12

### Features

- Add parser conformance capture workflow ([#42](https://github.com/ai-dynamo/frontend-crates/pull/42))
- *(parsers-v2)* Add v2 streaming tool-call parser (Harmony) ([#32](https://github.com/ai-dynamo/frontend-crates/pull/32))
