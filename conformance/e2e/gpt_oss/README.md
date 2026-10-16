# GPT-OSS unified parser validation

`cases.yaml` is the original, unchanged 23-case live suite. Each deployment runs
both streaming modes and both reasoning axes (92 checks). Repeat with structural
tags on/off and worker stream intervals 1/20: 368 checks per implementation.
GPT-OSS does not support disabling reasoning; `none` must return HTTP 400.

```bash
uv run --with httpx --with jsonschema --with pyyaml python run.py \
  --implementation v2 --url http://localhost:8000 \
  --structural-tags on --stream-interval 1 \
  --provenance results/2026-10-16/environment.json --output results/new-run
```

Use `--implementation v1` or `sglang` for baselines. Native SGLang with tags also
needs `--tag-builder /path/to/harmony_structural_tag`, built with
`cargo build -p dynamo-parsers-v2 --example harmony_structural_tag`.
Use `--case ID` or `--reasoning enabled` for a diagnostic subset, in a separate
output directory. `--report-only --output DIR` regenerates HTML and scores.
Capture JSON contains complete requests, responses, SSE frames and timing.
Earlier failed iterations remain alongside the final runs.

`deployment/` pins the container and model revision, builds the local Rust
crates, and starts fresh servers for each configuration. Its Dynamo integration
patch records the exact serving changes required for these local crates:
GPT-OSS routing, unsupported-disable rejection, and required-tool JSON prompt
formatting when structural tags are off. Worker and v1 use a frozen core;
only the v2 frontend receives the final core. The initial build populates both
overlays; freeze `/work/overlay` and replace `/work/overlay-v2/dynamo/_core.so`
with a newly built core after stopping the v2 frontend. Verify the SHA before
restarting it. The recorded core hashes are in `environment.json`. Apply the patch to the recorded
Dynamo base, with these frontend crates available at `/work/frontend`.
Always select the intended Kubernetes context and an owned namespace explicitly.

The results HTML separates functional requests, unsupported-disable rejection,
and invalid-tool validation. Native SGLang's forced tool choice plus explicit
structural-tag response format is an HTTP 400 baseline failure and is retained.
