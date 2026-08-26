# syntax=docker/dockerfile:1.7

FROM rust:1.96.1-bookworm AS builder
RUN apt-get update \
    && apt-get install -y --no-install-recommends protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
RUN --mount=type=cache,id=agent-rt-sandbox-cargo-registry,target=/usr/local/cargo/registry \
    --mount=type=cache,id=agent-rt-sandbox-target,target=/src/target \
    cargo build --locked --release -p agent-rt-sandbox-service \
    && cp /src/target/release/agent-rt-sandbox-service /agent-rt-sandbox-service

FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --home-dir /nonexistent --shell /usr/sbin/nologin agent-sandbox
COPY --from=builder /agent-rt-sandbox-service /usr/local/bin/agent-rt-sandbox-service
USER 10001:10001
ENTRYPOINT ["/usr/local/bin/agent-rt-sandbox-service"]
