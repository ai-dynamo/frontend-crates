#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends build-essential pkg-config libssl-dev libclang-dev protobuf-compiler libprotobuf-dev cmake git curl unzip patchelf python3-dev python3-venv
export RUSTUP_HOME=/work/rustup CARGO_HOME=/work/cargo
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /work/rustup.sh
sh /work/rustup.sh -y --profile minimal --default-toolchain 1.96.1 --no-modify-path
