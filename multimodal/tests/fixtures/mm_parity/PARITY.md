# Multimodal parity

`report.py` writes this file from the `<model>.json` tables. Do not edit it.

Each table's baseline is the checkpoint's own Hugging Face processor on the
CPU (`hf`). A cell compares one framework with the baseline on every field
the table records: `=` equal, `≠` differs on the named fields, `omits` the
framework does not give the named fields (it is equal on the others unless
`≠` names them), `rejects` the framework refuses the input, `n/a` the
framework cannot run the case, `gap` a known gap of this crate (README.md).
The crate is gated on the column marked *target*. Fields: `ids` the expanded
prompt, `offsets` the feature spans, `grid` the patch grid, `shapes` and
`features` the pixel values, `mrope` the M-RoPE positions and delta,
`accepts` a case the baseline refuses.

## zai-org/GLM-5.3-Flash

Revision `46a050bec13002ad29ee795c16e61cf4fa3c4a9c`. Baseline: `Glm5NextProcessor`, transformers 5.19.0, `torchvision` backend, `transformers.image_utils.load_image`. Target: `sglang`: SGLang serves M-RoPE positions (HF's model counts 1-D positions) and puts alpha over a gray or white chosen from the edges (HF's load_image drops it).

| framework | version | libraries | source | omits |
| --- | --- | --- | --- | --- |
| `hf` | 5.19.0 | transformers 5.19.0, torch 2.14.1+cu130, torchvision 0.29.2+cu132, pillow 12.3.0, numpy 2.3.5 | AutoProcessor(text, images=load_image(bytes)), and Qwen2-VL's get_rope_index when config.json has an mrope_section | - |
| `sglang` | 0.5.24.dev1101+ga2ffd2146 (a2ffd21469) | transformers 5.19.0, torch 2.14.1+cu130, torchvision 0.29.2+cu132, pillow 12.3.0, numpy 2.3.5 | process_mm_data_async(image bytes, token ids) on the CPU, PIL decode | - |
| `vllm` | 0.34.0rc1.dev156+g4e3463c87 | transformers 5.22.2, torch 2.14.0+cu130, torchvision 0.29.0+cu130, pillow 13.0.0, numpy 2.3.5 | vLLM MediaConnector.fetch_image (ImageMediaIO) on base64 data URLs, then MULTIMODAL_REGISTRY.create_processor(ModelConfig(dtype=float32, mm_device_do_normalize=False)).apply on the token prompt with per-request mm_processor_kwargs (pil backend: use_fast=False), then the model class's get_mrope_input_positions | - |
| `llm-multimodal` | 1.15.0 | llm-multimodal 1.15.0, harness_sha256 b5ae4f91ba1c36009d85235f9cddd2d8f02eee00bd1a4be91c68675966925286 | ModelRegistry::lookup -> glm53_flash.vision_processor(image processor config + image_kwargs) -> MediaConnector::fetch_image(InlineBytes, KeepAlpha) -> ImageFrame::image -> preprocess -> prompt_replacements_for + placeholder splice; glm53_flash uses resize_bicubic_pil_rgb for every backend; JPEG via image crate (no libturbojpeg.so.0), WebP via libwebp 1.3.2 | mrope_sha256, mrope_delta |
| `dynamo-multimodal` | 0.2.6 | - | tests/mm_parity.rs through the public API | - |

- `sglang`: 0 of 26 cases equal to HF (26 differ, 0 equal but omit fields, 0 reject, 0 n/a).
- `vllm`: 19 of 26 cases equal to HF (7 differ, 0 equal but omit fields, 0 reject, 0 n/a).
- `llm-multimodal`: 0 of 26 cases equal to HF (22 differ, 3 equal but omit fields, 1 reject, 0 n/a).
- `dynamo-multimodal`: 0 of 26 cases equal to HF (25 differ, 0 equal but omit fields, 1 reject, 0 n/a); 25 equal to the target `sglang`, 1 known gap.

| case | sglang (target) | vllm | llm-multimodal | dynamo-multimodal |
| --- | --- | --- | --- | --- |
| rgb_96x80.png | ≠ mrope | = | ≠ features; omits mrope | ≠ mrope |
| rgb_333x251.jpg | ≠ mrope | = | ≠ features; omits mrope | ≠ mrope |
| rgb_444_120x90.jpg | ≠ mrope | = | ≠ features; omits mrope | ≠ mrope |
| progressive_200x150.jpg | ≠ mrope | = | ≠ features; omits mrope | ≠ mrope |
| cmyk_80x64.jpg | ≠ mrope | = | ≠ features; omits mrope | ≠ mrope |
| exif_orientation6_64x48.jpg | ≠ mrope | = | ≠ features; omits mrope | ≠ mrope |
| gray_64x64.png | ≠ mrope | = | ≠ features; omits mrope | ≠ mrope |
| rgba_80x60.png | ≠ features, mrope | ≠ features | ≠ features; omits mrope | ≠ features, mrope |
| palette_transparency_70x50.png | ≠ features, mrope | ≠ features | omits mrope | ≠ features, mrope |
| gray16_40x30.png | ≠ mrope | = | ≠ features; omits mrope | gap |
| animated_48x40.gif | ≠ mrope | = | ≠ features; omits mrope | ≠ mrope |
| lossy_90x70.webp | ≠ mrope | = | ≠ features; omits mrope | ≠ mrope |
| lossless_alpha_60x40.webp | ≠ features, mrope | ≠ features | ≠ features; omits mrope | ≠ features, mrope |
| transparent_56x56.gif | ≠ features, mrope | ≠ features | omits mrope | ≠ features, mrope |
| rgba_raw_exif6_56x48.png | ≠ features, mrope | ≠ features | ≠ features, grid; omits mrope | ≠ features, mrope |
| rgb_50x40.bmp | ≠ mrope | = | ≠ features; omits mrope | ≠ mrope |
| rgba_bitfields_56x56.bmp | ≠ features, mrope | ≠ features | omits mrope | ≠ features, mrope |
| tiny_10x10.png | ≠ mrope | = | ≠ features; omits mrope | ≠ mrope |
| large_800x600.jpg | ≠ mrope | = | ≠ features; omits mrope | ≠ mrope |
| aspect_201_4020x20.png | ≠ mrope | = | rejects | ≠ mrope |
| tiny_10x10.png pil | ≠ mrope | = | ≠ features; omits mrope | ≠ mrope |
| rgb_333x251.jpg pil | ≠ mrope | = | ≠ features; omits mrope | ≠ mrope |
| rgba_80x60.png pil | ≠ features, mrope | ≠ features | ≠ features; omits mrope | ≠ features, mrope |
| large_800x600.jpg pil | ≠ mrope | = | ≠ features; omits mrope | ≠ mrope |
| two images | ≠ mrope | = | ≠ features; omits mrope | ≠ mrope |
| pixel overrides | ≠ mrope | = | ≠ features; omits mrope | ≠ mrope |

## nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16

Revision `fb0ee0bc6b3f70342afd8d84cb80dbadbaa8a1fe`. Baseline: `NemotronH_Omni_Reasoning_V3Processor`, transformers 5.19.0, `torchvision` backend, `transformers.image_utils.load_image`. Target: `sglang`.

| framework | version | libraries | source | omits |
| --- | --- | --- | --- | --- |
| `hf` | 5.19.0 | transformers 5.19.0, torch 2.14.1+cu130, torchvision 0.29.2+cu132, pillow 12.3.0, numpy 2.3.5 | AutoProcessor(trust_remote_code=True)(text, images=load_image(bytes)), with the image processor's max_model_len set to the case's; 1-D positions | - |
| `sglang` | 0.5.24.dev1101+ga2ffd2146 (a2ffd21469) | transformers 5.19.0, torch 2.14.1+cu130, torchvision 0.29.2+cu132, pillow 12.3.0, numpy 2.3.5 | NanoNemotronVLImageProcessor.process_mm_data_async(image bytes, token ids) on the CPU, PIL decode, ServerArgs(context_length=the case's max_model_len) | - |
| `vllm` | 0.34.0rc1.dev156+g4e3463c87 | transformers 5.22.2, torch 2.14.0+cu130, torchvision 0.29.0+cu130, pillow 13.0.0, numpy 2.3.5 | vLLM MediaConnector.fetch_image (ImageMediaIO) on base64 data URLs, then MULTIMODAL_REGISTRY.create_processor(ModelConfig(dtype=float32, mm_device_do_normalize=False)).apply on the token prompt with per-request mm_processor_kwargs (pil backend: use_fast=False), then the model class's get_mrope_input_positions | - |
| `llm-multimodal` | 1.15.0 | llm-multimodal 1.15.0, harness_sha256 b5ae4f91ba1c36009d85235f9cddd2d8f02eee00bd1a4be91c68675966925286 | llm-multimodal ModelRegistry (setup failed) | mrope_sha256, mrope_delta |
| `dynamo-multimodal` | 0.2.6 | - | tests/mm_parity.rs through the public API | - |

- `sglang`: 1 of 33 cases equal to HF (28 differ, 0 equal but omit fields, 4 reject, 0 n/a).
- `vllm`: 1 of 33 cases equal to HF (29 differ, 0 equal but omit fields, 3 reject, 0 n/a).
- `llm-multimodal`: 0 of 33 cases equal to HF (0 differ, 0 equal but omit fields, 0 reject, 33 n/a).
- `dynamo-multimodal`: 1 of 33 cases equal to HF (26 differ, 0 equal but omit fields, 6 reject, 0 n/a); 31 equal to the target `sglang`, 2 known gap.

| case | sglang (target) | vllm | llm-multimodal | dynamo-multimodal |
| --- | --- | --- | --- | --- |
| rgb_96x80.png | ≠ features, shapes, ids, offsets | ≠ features | n/a | ≠ features, shapes, ids, offsets |
| rgb_333x251.jpg | ≠ features, shapes, ids, offsets | ≠ features | n/a | ≠ features, shapes, ids, offsets |
| rgb_444_120x90.jpg | ≠ features | ≠ features | n/a | ≠ features |
| progressive_200x150.jpg | ≠ features, shapes, ids, offsets | ≠ features | n/a | ≠ features, shapes, ids, offsets |
| cmyk_80x64.jpg | ≠ features, shapes, ids, offsets | ≠ features | n/a | ≠ features, shapes, ids, offsets |
| exif_orientation6_64x48.jpg | ≠ features | ≠ features | n/a | ≠ features |
| gray_64x64.png | ≠ features | ≠ features | n/a | ≠ features |
| rgba_80x60.png | ≠ features, shapes, ids, offsets | ≠ features | n/a | ≠ features, shapes, ids, offsets |
| palette_transparency_70x50.png | ≠ features, shapes, ids, offsets | ≠ features | n/a | ≠ features, shapes, ids, offsets |
| gray16_40x30.png | ≠ features, shapes, ids, offsets | ≠ features | n/a | gap |
| animated_48x40.gif | ≠ features, shapes, ids, offsets | ≠ features | n/a | ≠ features, shapes, ids, offsets |
| lossy_90x70.webp | ≠ features, shapes, ids, offsets | ≠ features | n/a | ≠ features, shapes, ids, offsets |
| lossless_alpha_60x40.webp | ≠ features, shapes, ids, offsets | ≠ features | n/a | ≠ features, shapes, ids, offsets |
| transparent_56x56.gif | ≠ features | ≠ features | n/a | ≠ features |
| rgba_raw_exif6_56x48.png | ≠ features | ≠ features | n/a | ≠ features |
| rgb_50x40.bmp | ≠ features, shapes, ids, offsets | ≠ features | n/a | ≠ features, shapes, ids, offsets |
| rgba_bitfields_56x56.bmp | ≠ features | ≠ features | n/a | ≠ features |
| tiny_10x10.png | ≠ features | ≠ features | n/a | ≠ features |
| large_800x600.jpg | ≠ features | ≠ features | n/a | ≠ features |
| aspect_201_4020x20.png | ≠ features | ≠ features | n/a | ≠ features |
| la_40x30.png | ≠ features, shapes, ids, offsets | ≠ features | n/a | ≠ features, shapes, ids, offsets |
| huge_palette_14000x13000.png | ≠ accepts | rejects | n/a | gap |
| on_tile_512x512.png | ≠ features | ≠ features | n/a | ≠ features |
| tall_10x1100.png | ≠ features, shapes, ids, offsets | ≠ features | n/a | ≠ features, shapes, ids, offsets |
| lone image ctx 256 | ≠ features | ≠ features | n/a | ≠ features |
| text between images ctx 1024 | ≠ features, shapes, ids, offsets | ≠ features, shapes, ids, offsets | n/a | ≠ features, shapes, ids, offsets |
| text longer than context | ≠ features, shapes, ids, offsets | ≠ features, shapes, ids, offsets | n/a | ≠ features, shapes, ids, offsets |
| served over budget | ≠ features, shapes, ids, offsets | ≠ features, shapes, ids, offsets | n/a | ≠ features, shapes, ids, offsets |
| no image | = | = | n/a | = |
| reject two placeholders one image | rejects | ≠ accepts | n/a | rejects |
| reject no placeholder one image | rejects | rejects | n/a | rejects |
| reject one placeholder no image | rejects | ≠ offsets | n/a | rejects |
| reject corrupt bytes | rejects | rejects | n/a | rejects |

## Qwen/Qwen2-VL-2B-Instruct

Revision `895c3a49bc3fa70a340399125c650a463535e71c`. Baseline: `Qwen2VLProcessor`, transformers 5.19.0, `torchvision` backend, `transformers.image_utils.load_image`. Target: `hf`.

| framework | version | libraries | source | omits |
| --- | --- | --- | --- | --- |
| `hf` | 5.19.0 | transformers 5.19.0, torch 2.14.1+cu130, torchvision 0.29.2+cu132, pillow 12.3.0, numpy 2.3.5 | AutoProcessor(text, images=load_image(bytes)), and Qwen2-VL's get_rope_index when config.json has an mrope_section | - |
| `sglang` | 0.5.24.dev1101+ga2ffd2146 (a2ffd21469) | transformers 5.19.0, torch 2.14.1+cu130, torchvision 0.29.2+cu132, pillow 12.3.0, numpy 2.3.5 | process_mm_data_async(image bytes, token ids) on the CPU, PIL decode | - |
| `vllm` | 0.34.0rc1.dev156+g4e3463c87 | transformers 5.22.2, torch 2.14.0+cu130, torchvision 0.29.0+cu130, pillow 13.0.0, numpy 2.3.5 | vLLM MediaConnector.fetch_image (ImageMediaIO) on base64 data URLs, then MULTIMODAL_REGISTRY.create_processor(ModelConfig(dtype=float32, mm_device_do_normalize=False)).apply on the token prompt with per-request mm_processor_kwargs (pil backend: use_fast=False), then the model class's get_mrope_input_positions | - |
| `llm-multimodal` | 1.15.0 | llm-multimodal 1.15.0, harness_sha256 b5ae4f91ba1c36009d85235f9cddd2d8f02eee00bd1a4be91c68675966925286 | ModelRegistry::lookup -> qwen_vl.vision_processor(image processor config + image_kwargs) -> MediaConnector::fetch_image(InlineBytes, KeepAlpha) -> ImageFrame::image -> preprocess -> prompt_replacements_for + placeholder splice; qwen_vl uses resize_bicubic_torch_aa for every backend; JPEG via image crate (no libturbojpeg.so.0), WebP via libwebp 1.3.2 | mrope_sha256, mrope_delta |
| `dynamo-multimodal` | 0.2.6 | - | tests/mm_parity.rs through the public API | - |

- `sglang`: 5 of 5 cases equal to HF (0 differ, 0 equal but omit fields, 0 reject, 0 n/a).
- `vllm`: 5 of 5 cases equal to HF (0 differ, 0 equal but omit fields, 0 reject, 0 n/a).
- `llm-multimodal`: 0 of 5 cases equal to HF (2 differ, 3 equal but omit fields, 0 reject, 0 n/a).
- `dynamo-multimodal`: 5 of 5 cases equal to HF (0 differ, 0 equal but omit fields, 0 reject, 0 n/a); 5 equal to the target `hf`, 0 known gap.

| case | sglang | vllm | llm-multimodal | dynamo-multimodal |
| --- | --- | --- | --- | --- |
| aten_downscale | = | = | omits mrope | = |
| aten_multi | = | = | omits mrope | = |
| aten_upscale | = | = | omits mrope | = |
| pil_round | = | = | ≠ features; omits mrope | = |
| pil_tall | = | = | ≠ features; omits mrope | = |
