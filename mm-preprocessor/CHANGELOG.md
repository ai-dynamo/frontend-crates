<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial crate: model-family multimodal preprocessing (`MmFamilyProcessor`
  seam, request driver, family registry, bit-exact PIL/ATen resize kernels,
  token layout, budgeted media fetch, optional rayon fan-out) with the
  Qwen-VL family (`smart_resize`, normalize/patchify, image-only M-RoPE),
  extracted from SGLang's `sglang-mm`.
