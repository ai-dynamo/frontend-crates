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
