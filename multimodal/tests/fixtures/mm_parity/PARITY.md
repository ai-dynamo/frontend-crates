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

## moonshotai/Kimi-K3

Revision `f831ab66814297da540d832a5235f8e904f29d06`. Baseline: `KimiK3Processor`, transformers 5.19.0, `KimiK3VisionProcessor` backend, `media_utils.ensure_media_type (the checkpoint's remote code)`. Target: `sglang`.

| framework | version | libraries | source | omits |
| --- | --- | --- | --- | --- |
| `hf` | 5.19.0 | transformers 5.19.0, torch 2.14.1+cu130, torchvision 0.29.2+cu132, pillow 12.3.0, numpy 2.3.5 | KimiK3Processor(medias=[image bytes], text) with media_proc_cfg overrides, each <|media_pad|> expanded to media_tokens_calculator | - |
| `sglang` | 0.5.24.dev1101+ga2ffd2146 (a2ffd21469) | transformers 5.19.0, torch 2.14.1+cu130, torchvision 0.29.2+cu132, pillow 12.3.0, numpy 2.3.5 | process_mm_data_async(image bytes, token ids) on the CPU, PIL decode | - |
| `vllm` | 0.34.0rc1.dev156+g4e3463c87 | transformers 5.22.2, torch 2.14.0+cu130, torchvision 0.29.0+cu130, pillow 13.0.0, numpy 2.3.5 | vLLM MediaConnector.fetch_image (ImageMediaIO) on base64 data URLs, then MULTIMODAL_REGISTRY.create_processor(ModelConfig(dtype=float32, mm_device_do_normalize=False)).apply on the token prompt with per-request mm_processor_kwargs (pil backend: use_fast=False), then the model class's get_mrope_input_positions | - |
| `llm-multimodal` | 1.15.0 | llm-multimodal 1.15.0, harness_sha256 b5ae4f91ba1c36009d85235f9cddd2d8f02eee00bd1a4be91c68675966925286 | llm-multimodal ModelRegistry (setup failed) | mrope_sha256, mrope_delta |
| `dynamo-multimodal` | 0.2.6 | - | tests/mm_parity.rs through the public API | - |

- `sglang`: 62 of 65 cases equal to HF (0 differ, 0 equal but omit fields, 3 reject, 0 n/a).
- `vllm`: 9 of 65 cases equal to HF (14 differ, 0 equal but omit fields, 1 reject, 41 n/a).
- `llm-multimodal`: 0 of 65 cases equal to HF (0 differ, 0 equal but omit fields, 0 reject, 65 n/a).
- `dynamo-multimodal`: 53 of 65 cases equal to HF (0 differ, 0 equal but omit fields, 12 reject, 0 n/a); 56 equal to the target `sglang`, 9 known gap.

| case | sglang (target) | vllm | llm-multimodal | dynamo-multimodal |
| --- | --- | --- | --- | --- |
| rgb_96x80.png | = | = | n/a | = |
| rgba_80x60.png | = | ≠ features | n/a | = |
| palette_transparency_70x50.png | = | = | n/a | = |
| gray16_40x30.png | = | = | n/a | gap |
| lossless_alpha_60x40.webp | = | ≠ features | n/a | = |
| tiny_10x10.png | = | = | n/a | = |
| aspect_201_4020x20.png | = | = | n/a | = |
| alpha_noise_84x60.png | = | ≠ features | n/a | = |
| gray_alpha_noise_84x60.png | = | ≠ features | n/a | = |
| rgb_trns_64x48.png | = | ≠ features | n/a | = |
| gray_trns_64x48.png | = | ≠ features | n/a | = |
| bilevel_trns_64x48.png | = | ≠ features | n/a | = |
| transparent_56x56.gif | = | ≠ features | n/a | = |
| lossy_alpha_60x40.webp | = | ≠ features | n/a | = |
| rgba_bitfields_56x56.bmp | = | ≠ features | n/a | = |
| wide_300x24.png | = | = | n/a | = |
| wide_alpha_300x24.png | = | ≠ features | n/a | = |
| tall_2x1000.png | = | = | n/a | = |
| tall_alpha_3x700.png | = | ≠ features | n/a | = |
| one_1x1.png | = | = | n/a | = |
| rgb_3x5.png | = | = | n/a | = |
| truncated_96x80.png | rejects | rejects | n/a | rejects |
| rgb_96x80.png small | = | n/a | n/a | = |
| rgb_333x251.jpg small | = | n/a | n/a | = |
| rgb_444_120x90.jpg small | = | n/a | n/a | = |
| progressive_200x150.jpg small | = | n/a | n/a | = |
| cmyk_80x64.jpg small | = | n/a | n/a | gap |
| exif_orientation6_64x48.jpg small | = | n/a | n/a | = |
| gray_64x64.png small | = | n/a | n/a | = |
| rgba_80x60.png small | = | n/a | n/a | = |
| palette_transparency_70x50.png small | = | n/a | n/a | gap |
| animated_48x40.gif small | = | n/a | n/a | gap |
| lossy_90x70.webp small | = | n/a | n/a | = |
| lossless_alpha_60x40.webp small | = | n/a | n/a | = |
| rgb_50x40.bmp small | = | n/a | n/a | = |
| large_800x600.jpg small | = | n/a | n/a | = |
| aspect_201_4020x20.png small | = | n/a | n/a | = |
| alpha_noise_84x60.png small | = | n/a | n/a | = |
| gray_alpha_noise_84x60.png small | = | n/a | n/a | = |
| rgb_trns_64x48.png small | = | n/a | n/a | = |
| gray_trns_64x48.png small | = | n/a | n/a | gap |
| palette_64x48.png small | = | n/a | n/a | gap |
| bilevel_64x48.png small | = | n/a | n/a | gap |
| bilevel_trns_64x48.png small | = | n/a | n/a | gap |
| gray4_64x48.png small | = | n/a | n/a | = |
| transparent_56x56.gif small | = | n/a | n/a | gap |
| lossy_alpha_60x40.webp small | = | n/a | n/a | = |
| rgbx_56x40.bmp small | = | n/a | n/a | = |
| rgba_bitfields_56x56.bmp small | = | n/a | n/a | = |
| gray_90x70.jpg small | = | n/a | n/a | = |
| wide_300x24.png small | = | n/a | n/a | = |
| wide_alpha_300x24.png small | = | n/a | n/a | = |
| tall_2x1000.png small | = | n/a | n/a | = |
| tall_alpha_3x700.png small | = | n/a | n/a | = |
| libvpx_filter_above63_96x80.webp small | = | n/a | n/a | = |
| libvpx_filter_in_range_96x80.webp small | = | n/a | n/a | = |
| gray_alpha_noise_84x60.png medium | = | n/a | n/a | = |
| side cap | = | n/a | n/a | = |
| tall | = | n/a | n/a | = |
| two images | = | ≠ features | n/a | = |
| two images medium | = | n/a | n/a | = |
| background | = | n/a | n/a | = |
| clip normalize | = | n/a | n/a | = |
| odd side limit | rejects | n/a | n/a | rejects |
| two placeholders one image | rejects | ≠ accepts | n/a | rejects |

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
