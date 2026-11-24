# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
## [0.2.16](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-multimodal-v0.2.15...dynamo-multimodal-v0.2.16) - 2026-11-24

### Features

- *(multimodal)* Add an experimental Qwen3-ASR profile (DIS-3555) ([#660](https://github.com/ai-dynamo/frontend-crates/pull/660))

## [0.2.15](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-multimodal-v0.2.14...dynamo-multimodal-v0.2.15) - 2026-11-20

### Features

- *(multimodal)* Add generic Rust audio preprocessing primitives (DIS-3521) ([#619](https://github.com/ai-dynamo/frontend-crates/pull/619))

## [0.2.14](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-multimodal-v0.2.13...dynamo-multimodal-v0.2.14) - 2026-11-19

### Features

- *(multimodal)* Add swscale RGB conversion and torchcodec average_fps ([#562](https://github.com/ai-dynamo/frontend-crates/pull/562))

## [0.2.13](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-multimodal-v0.2.12...dynamo-multimodal-v0.2.13) - 2026-11-18

### Features

- *(multimodal)* Add Pixtral image preprocessing for Mistral Large 4 ([#587](https://github.com/ai-dynamo/frontend-crates/pull/587))

## [0.2.12](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-multimodal-v0.2.11...dynamo-multimodal-v0.2.12) - 2026-11-18

### Features

- *(multimodal)* Add Nemotron 3.5 Super VL image preprocessing ([#588](https://github.com/ai-dynamo/frontend-crates/pull/588))

## [0.2.11](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-multimodal-v0.2.10...dynamo-multimodal-v0.2.11) - 2026-11-18

### Features

- *(multimodal)* Add Qwen3-VL video to the Qwen-VL processor ([#563](https://github.com/ai-dynamo/frontend-crates/pull/563))

## [0.2.10](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-multimodal-v0.2.9...dynamo-multimodal-v0.2.10) - 2026-11-17

### Features

- *(multimodal)* Add Kimi K3 image preprocessing ([#589](https://github.com/ai-dynamo/frontend-crates/pull/589))

## [0.2.9](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-multimodal-v0.2.8...dynamo-multimodal-v0.2.9) - 2026-11-17

### Features

- *(multimodal)* One parity table per model with an HF baseline ([#594](https://github.com/ai-dynamo/frontend-crates/pull/594))

## [0.2.8](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-multimodal-v0.2.7...dynamo-multimodal-v0.2.8) - 2026-11-17

### Features

- *(multimodal)* Add RGBA decode and a decode parity table ([#593](https://github.com/ai-dynamo/frontend-crates/pull/593))

## [0.2.7](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-multimodal-v0.2.6...dynamo-multimodal-v0.2.7) - 2026-11-13

### Features

- *(multimodal)* Add PyTorch's float32 antialias bilinear resize ([#561](https://github.com/ai-dynamo/frontend-crates/pull/561))

## [0.2.6](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-multimodal-v0.2.5...dynamo-multimodal-v0.2.6) - 2026-11-06

### Performance

- *(multimodal)* Decode GLM alpha images once ([#569](https://github.com/ai-dynamo/frontend-crates/pull/569))

## [0.2.5](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-multimodal-v0.2.4...dynamo-multimodal-v0.2.5) - 2026-11-06

### Features

- *(multimodal)* Add GLM-5.3-Flash image preprocessing ([#564](https://github.com/ai-dynamo/frontend-crates/pull/564))

## [0.2.4](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-multimodal-v0.2.3...dynamo-multimodal-v0.2.4) - 2026-10-27

### Bug fixes

- Preserve numeric tool arguments across parsers ([#339](https://github.com/ai-dynamo/frontend-crates/pull/339))

## [0.2.3](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-multimodal-v0.2.2...dynamo-multimodal-v0.2.3) - 2026-10-22

### Bug fixes

- *(multimodal)* Clearer NVDEC errors and docs; test the libvpx fallback rule ([#443](https://github.com/ai-dynamo/frontend-crates/pull/443))

## [0.2.2](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-multimodal-v0.2.1...dynamo-multimodal-v0.2.2) - 2026-10-20

### Features

- *(multimodal)* Add GLM-5.3-Flash image token planning ([#412](https://github.com/ai-dynamo/frontend-crates/pull/412))

## [0.2.1](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-multimodal-v0.2.0...dynamo-multimodal-v0.2.1) - 2026-10-15

### Bug fixes

- *(multimodal)* Reject overflowing Qwen image spec dimensions ([#400](https://github.com/ai-dynamo/frontend-crates/pull/400))

## [0.2.0](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-multimodal-v0.1.1...dynamo-multimodal-v0.2.0) - 2026-10-14

### Features

- *(multimodal)* Decode video on NVIDIA GPUs through NVDEC ([#378](https://github.com/ai-dynamo/frontend-crates/pull/378))

## [0.1.1](https://github.com/ai-dynamo/frontend-crates/compare/dynamo-multimodal-v0.1.0...dynamo-multimodal-v0.1.1) - 2026-10-13

### Features

- *(multimodal)* Expose opt-in EXIF-transposed image APIs ([#401](https://github.com/ai-dynamo/frontend-crates/pull/401))
- *(multimodal)* Register DSV4.1 metadata-only prompt planning ([#402](https://github.com/ai-dynamo/frontend-crates/pull/402))
- *(multimodal)* Add MediaFetcher, a protected http(s)/data: media fetcher ([#374](https://github.com/ai-dynamo/frontend-crates/pull/374))
- *(multimodal)* Add per-request video sampling and pixel-budget options ([#369](https://github.com/ai-dynamo/frontend-crates/pull/369))
- *(multimodal)* Add DSV4.1 image geometry planner ([#372](https://github.com/ai-dynamo/frontend-crates/pull/372))
- *(multimodal)* Add VideoFrames and JPEG encoding of sampled frames ([#373](https://github.com/ai-dynamo/frontend-crates/pull/373))
