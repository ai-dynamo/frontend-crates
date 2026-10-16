#!/usr/bin/env bash
set -euo pipefail
export RUSTUP_HOME=/work/rustup CARGO_HOME=/work/cargo
export PATH="/work/cargo/bin:$PATH"
mkdir -p /work/dynamo /work/frontend
 tar -xf /tmp/dynamo-source.tar -C /work/dynamo
 tar -xf /tmp/frontend-source.tar -C /work/frontend
cd /work/dynamo
patch -p1 < /tmp/dynamo-integration.patch
export CARGO_TARGET_DIR=/work/target CARGO_BUILD_JOBS=10 CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_DEV_OPT_LEVEL=1
export PYO3_PYTHON=/usr/bin/python3
cd lib/bindings/python
cargo build --no-default-features
mkdir -p /work/overlay/dynamo
cp -a src/dynamo/. /work/overlay/dynamo/
cp -a /work/dynamo/components/src/dynamo/. /work/overlay/dynamo/
cp /work/target/debug/lib_core.so /work/overlay/dynamo/_core.so
mkdir -p /work/overlay-v2/dynamo
cp -a /work/overlay/dynamo/. /work/overlay-v2/dynamo/
PYTHONPATH=/work/overlay python3 -m dynamo.frontend --help > /work/frontend-help.txt
printf 'BUILD_COMPLETE\n'
