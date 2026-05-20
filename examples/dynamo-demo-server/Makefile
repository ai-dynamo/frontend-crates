# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# dynamo-demo-server -- build, run, test.

MODEL ?= Qwen/Qwen2.5-0.5B-Instruct

.DEFAULT_GOAL := build

.PHONY: build
build:
	cargo build --release

.PHONY: run
run:
	cargo run --release -- --model $(MODEL)

.PHONY: test
test:
	scripts/smoke.sh

.PHONY: clean
clean:
	cargo clean
