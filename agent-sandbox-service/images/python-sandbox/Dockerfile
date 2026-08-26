# syntax=docker/dockerfile:1.7

# Replace this moving development tag with a reviewed digest in production.
ARG SANDBOXD_IMAGE=us-central1-docker.pkg.dev/k8s-staging-images/agent-sandbox/sandboxd:latest-main
FROM ${SANDBOXD_IMAGE} AS sandboxd

FROM python:3.12-slim-bookworm
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates procps \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /workspace \
    && chown 1000:1000 /workspace
COPY --from=sandboxd /usr/local/bin/sandboxd /usr/local/bin/sandboxd
WORKDIR /workspace
USER 1000:1000
EXPOSE 8080 9090
ENTRYPOINT ["/usr/local/bin/sandboxd"]
