# Renderer conformance fixtures

This latest-only store owns renderer inputs, independently generated model-reference observations and recorded Dynamo observations. It does not contain model weights or tokenizer vocabularies. `manifest.json` pins the original repository, complete revision, source URL and SHA-256 of every small rendering asset. Assets retain their original bytes; each model's license is included beside its files.

The five model sources are [DeepSeek V4.1 Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash), [GLM 5.3](https://huggingface.co/zai-org/GLM-5.3), [Kimi K3](https://huggingface.co/moonshotai/Kimi-K3), [Qwen 3.8 27B](https://huggingface.co/Qwen/Qwen3.8-27B) and [MiniMax M3.1 Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash). Their model-specific license terms and attribution remain in `assets/<model>/LICENSE`; no ownership of those assets is claimed by this suite.

Each `families/<model>/inputs.json` owns concrete `RENDERER.*` case IDs, requests, constructor settings and independently authored metadata assertions. `reference.json` contains outputs from the pinned model author encoder or Transformers template environment. `dynamo.json` contains measured outputs from the current Rust runner. Both observations bind to the SHA-256 of the exact manifest and input-file bytes. Reference producer records also identify the adapter source hash and reference engine versions.

Reference adapters record their exact effective call and every bridge transformation. Reasoning segment joining is an independently authored renderer API contract, not a claim that model authors accept segment lists. Hugging Face templates receive decoded argument objects and image markers according to their rendering interface. K3's official `image_prompts` hook receives `<|media_pad|>` for each image. Its default effort is explicitly `max` to express the renderer API default. The pinned K3 encoder has no Partial Mode API, so that case remains visibly unsupported by the independent reference.

Prompt text and ordered K3 segment permissions come from executing the independent reference. `pending_segments` and source-message media order are explicitly labeled authored contract assertions where the reference exposes no such fields. In particular, the reversed K3 tool-image case expects source-message reordering rather than copying Dynamo's current metadata. Images use synthetic URLs and are never fetched.

Create a Python 3.12 environment and install `reference-requirements.txt` for a manual refresh, then run from the repository root:

```bash
python conformance/utils/src/renderer_reference.py \
  --implementation reference --output /absolute/path/to/reference-capture
python conformance/utils/src/renderer_reference.py \
  --implementation dynamo --output /absolute/path/to/dynamo-capture
```

Either command accepts `--model <manifest-id>`. Outputs appear under `families/<model>/reference.json` or `dynamo.json` in the required output directory. The directory must be separate from the source store. Inspect and deliberately copy reviewed captures into this store; the capture CLI cannot silently replace inputs or model assets. Dynamo dispatch uses the shared Rust capture example and never writes reference expectations.

Partial Dynamo refreshes must use the same renderer release as the other recorded models. After a renderer package version change, omit `--model` and refresh all five models together. The capture command rejects a partial cross-release refresh before writing results.

Routine checks run offline against the vendored assets. Reference differences remain visible warnings; live Dynamo output differing from its recorded observation, missing cases and invalid evidence fail. No normal test rewrites observations. Tokens, image fetching, model inference and HTTP API conversion are outside this MVP.
